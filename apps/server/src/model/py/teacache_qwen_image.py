import math
import torch
import comfy.model_management as mm

from torch import Tensor
from typing import Optional, Union, Tuple
from diffusers.models.modeling_outputs import Transformer2DModelOutput


# QwenImage TeaCache coefficients - 这些值需要根据实际测试调整
SUPPORTED_MODELS_COEFFICIENTS = {
    "qwen_image": [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
    "qwen_image_edit": [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
    "qwen_image_lightning": [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
}

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result

def teacache_qwenimage_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        img_shapes: Optional[list] = None,
        txt_seq_lens: Optional[list] = None,
        guidance: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[dict] = None,
        controlnet_block_samples=None,
        return_dict: bool = True,
        transformer_options={},
    ) -> Union[torch.Tensor, Transformer2DModelOutput, Tuple[torch.Tensor, ...]]:
    """
    TeaCache加速的QwenImage前向传播函数
    
    参数:
        hidden_states: 图像流输入，形状为(batch_size, image_sequence_length, in_channels)
        encoder_hidden_states: 文本流输入，形状为(batch_size, text_sequence_length, joint_attention_dim)
        encoder_hidden_states_mask: 编码器隐藏状态掩码
        timestep: 时间步张量
        img_shapes: 图像形状列表，用于旋转嵌入
        txt_seq_lens: 文本序列长度列表
        guidance: 引导张量
        attention_kwargs: 注意力参数
        controlnet_block_samples: ControlNet块样本
        return_dict: 是否返回字典
        transformer_options: 变换器选项，包含TeaCache参数
        
    返回:
        处理后的张量或Transformer2DModelOutput
    """
    # 获取TeaCache参数
    rel_l1_thresh = transformer_options.get("rel_l1_thresh")
    coefficients = transformer_options.get("coefficients")
    enable_teacache = transformer_options.get("enable_teacache", True)
    cache_device = transformer_options.get("cache_device")
    cond_or_uncond = transformer_options.get("cond_or_uncond", [0])
    
    device = hidden_states.device
    if self.offload:
        self.offload_manager.set_device(device)

    # 预处理输入
    hidden_states = self.img_in(hidden_states)
    if timestep is not None:
        timestep = timestep.to(hidden_states.dtype)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    if timestep is not None:
        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )
    else:
        # 如果没有timestep，创建一个默认的时间嵌入
        temb = torch.zeros_like(hidden_states[:, :1])  # 简化处理

    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

    # TeaCache实现 - 参考teacache_flux_forward的简化实现
    # 使用第一个transformer块的调制输入作为缓存判断依据
    modulated_inp = temb.to(cache_device)
    
    # 参数验证
    if coefficients is None:
        # 从transformer_options获取model_type
        model_type = transformer_options.get("model_type", "qwen_image")
        coefficients = SUPPORTED_MODELS_COEFFICIENTS.get(model_type, [4.98651651e+02, -2.83781631e+02, 5.58554382e+01, -3.82021401e+00, 2.64230861e-01])
    
    # 初始化缓存状态 - 参考teacache_flux_forward的简化实现
    if not hasattr(self, 'accumulated_rel_l1_distance'):
        should_calc = True
        self.accumulated_rel_l1_distance = 0
    else:
        try:
            # 计算相对L1距离 - 与teacache_flux_forward保持一致
            if hasattr(self, 'previous_modulated_input'):
                rel_l1_distance = ((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean())
                self.accumulated_rel_l1_distance += poly1d(coefficients, rel_l1_distance).abs()
                if self.accumulated_rel_l1_distance < rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        except:
            should_calc = True
            self.accumulated_rel_l1_distance = 0

    self.previous_modulated_input = modulated_inp

    if not enable_teacache:
        should_calc = True

    # 根据缓存状态决定是否计算
    if not should_calc:
        # 使用缓存的残差 - 参考teacache_flux_forward的简化实现
        hidden_states += self.previous_residual.to(hidden_states.device)
    else:
        # 执行完整的transformer计算
        ori_hidden_states = hidden_states.to(cache_device)
        
        compute_stream = torch.cuda.current_stream()
        if self.offload:
            self.offload_manager.initialize(compute_stream)
            
        for block_idx, block in enumerate(self.transformer_blocks):
            with torch.cuda.stream(compute_stream):
                if self.offload:
                    block = self.offload_manager.get_block(block_idx)

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        encoder_hidden_states_mask,
                        temb,
                        image_rotary_emb,
                    )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )

                # controlnet残差 - 与diffusers QwenImageTransformer2DModel相同的逻辑
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(math.ceil(interval_control))
                    hidden_states = hidden_states + controlnet_block_samples[block_idx // interval_control]

            if self.offload:
                self.offload_manager.step(compute_stream)

        # 缓存残差 - 参考teacache_flux_forward的简化实现
        self.previous_residual = hidden_states.to(cache_device) - ori_hidden_states

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    torch.cuda.empty_cache()

    # 与demo222标准实现保持一致：直接返回处理后的张量
    return output


class TeaCacheQwenImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The QwenImage diffusion model the TeaCache will be applied to."}),
                "model_type": (["qwen_image", "qwen_image_edit", "qwen_image_lightning"], {"default": "qwen_image", "tooltip": "Supported QwenImage diffusion model."}),
                "rel_l1_thresh": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The start percentage of the steps that will apply TeaCache."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The end percentage of the steps that will apply TeaCache."}),
                "cache_device": (["cuda", "cpu"], {"default": "cuda", "tooltip": "Device where the cache will reside."}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "TeaCache"
    TITLE = "TeaCache for QwenImage"
    
    def apply_teacache(self, model, model_type: str, rel_l1_thresh: float, start_percent: float, end_percent: float, cache_device: str):
        """
        应用TeaCache加速到QwenImage模型
        
        重构要点：
        1. 移除unittest.mock依赖，使用直接方法替换
        2. 简化缓存状态管理逻辑
        3. 优化时间步处理算法
        4. 增强错误处理和边界检查
        """
        if rel_l1_thresh == 0:
            return (model,)

        # 参数验证
        if start_percent < 0 or start_percent > 1 or end_percent < 0 or end_percent > 1:
            raise ValueError("start_percent and end_percent must be between 0 and 1")
        
        if start_percent > end_percent:
            raise ValueError("start_percent must be less than or equal to end_percent")

        new_model = model.clone()
        
        # 获取扩散模型
        diffusion_model = new_model.get_model_object("diffusion_model")

        # 检查是否是QwenImage模型
        if not hasattr(diffusion_model, 'transformer_blocks'):
            raise ValueError("The provided model is not a QwenImage model")

        # 保存原始forward方法
        original_forward = diffusion_model.forward
        
        # 创建TeaCache参数
        teacache_params = {
            "rel_l1_thresh": rel_l1_thresh,
            "coefficients": SUPPORTED_MODELS_COEFFICIENTS[model_type],
            "model_type": model_type,
            "cache_device": mm.get_torch_device() if cache_device == "cuda" else torch.device("cpu"),
            "start_percent": start_percent,
            "end_percent": end_percent
        }

        def teacache_forward_wrapper(self, *args, **kwargs):
            """TeaCache包装的forward方法"""
            # 获取transformer_options
            transformer_options = kwargs.get('transformer_options', {})
            
            # 设置TeaCache参数
            for key, value in teacache_params.items():
                transformer_options[key] = value
            
            # 时间步处理逻辑
            timestep = kwargs.get('timestep')
            if timestep is not None:
                sigmas = transformer_options.get("sample_sigmas", [])
                
                if sigmas:
                    # 优化时间步匹配算法
                    current_step_index = self._find_timestep_index(timestep[0], sigmas)
                    
                    # 重置缓存状态（如果是第一步）
                    if current_step_index == 0:
                        if hasattr(self, 'teacache_state'):
                            delattr(self, 'teacache_state')
                    
                    # 确定是否启用TeaCache
                    current_percent = current_step_index / (len(sigmas) - 1)
                    transformer_options["current_percent"] = current_percent
                    transformer_options["enable_teacache"] = (
                        teacache_params["start_percent"] <= current_percent <= teacache_params["end_percent"]
                    )
            
            kwargs['transformer_options'] = transformer_options
            
            # 调用TeaCache forward方法
            return teacache_qwenimage_forward(self, *args, **kwargs)
        
        # 添加辅助方法到模型实例
        def _find_timestep_index(timestep_value, sigmas):
            """查找时间步对应的索引"""
            # 精确匹配
            matched_indices = (sigmas == timestep_value).nonzero(as_tuple=True)[0]
            if len(matched_indices) > 0:
                return matched_indices[0].item()
            
            # 区间匹配（优化算法）
            for i in range(len(sigmas) - 1):
                if (sigmas[i] - timestep_value) * (sigmas[i + 1] - timestep_value) <= 0:
                    return i
            
            # 默认返回0
            return 0
        
        # 将辅助方法绑定到模型实例
        diffusion_model._find_timestep_index = _find_timestep_index.__get__(diffusion_model)
        
        # 替换forward方法
        diffusion_model.forward = teacache_forward_wrapper.__get__(diffusion_model, diffusion_model.__class__)

        return (new_model,)
