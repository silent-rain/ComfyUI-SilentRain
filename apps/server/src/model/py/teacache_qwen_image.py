import math
import torch
import comfy.ldm.common_dit
import comfy.model_management as mm

from torch import Tensor
from einops import repeat
from typing import Optional
from unittest.mock import patch

from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput


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
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[list] = None,
        txt_seq_lens: Optional[list] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[dict] = None,
        controlnet_block_samples=None,
        return_dict: bool = True,
        transformer_options={},
    ) -> torch.Tensor:
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
    timestep = timestep.to(hidden_states.dtype)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states)
    )

    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

    # TeaCache实现
    # 使用第一个transformer块的调制输入作为缓存判断依据
    modulated_inp = temb.to(cache_device)
    
    # 初始化缓存状态
    if not hasattr(self, 'teacache_state'):
        self.teacache_state = {
            0: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None},
            1: {'should_calc': True, 'accumulated_rel_l1_distance': 0, 'previous_modulated_input': None, 'previous_residual': None}
        }

    def update_cache_state(cache, modulated_inp):
        if cache['previous_modulated_input'] is not None:
            try:
                cache['accumulated_rel_l1_distance'] += poly1d(coefficients, ((modulated_inp-cache['previous_modulated_input']).abs().mean() / cache['previous_modulated_input'].abs().mean()))
                if cache['accumulated_rel_l1_distance'] < rel_l1_thresh:
                    cache['should_calc'] = False
                else:
                    cache['should_calc'] = True
                    cache['accumulated_rel_l1_distance'] = 0
            except:
                cache['should_calc'] = True
                cache['accumulated_rel_l1_distance'] = 0
        cache['previous_modulated_input'] = modulated_inp
        
    # 处理批处理中的不同条件
    b = int(len(hidden_states) / len(cond_or_uncond))
    
    for i, k in enumerate(cond_or_uncond):
        update_cache_state(self.teacache_state[k], modulated_inp[i*b:(i+1)*b])

    if enable_teacache:
        should_calc = False
        for k in cond_or_uncond:
            should_calc = (should_calc or self.teacache_state[k]['should_calc'])
    else:
        should_calc = True

    # 根据缓存状态决定是否计算
    if not should_calc:
        # 使用缓存的残差
        for i, k in enumerate(cond_or_uncond):
            hidden_states[i*b:(i+1)*b] += self.teacache_state[k]['previous_residual'].to(hidden_states.device)
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

        # 缓存残差
        for i, k in enumerate(cond_or_uncond):
            self.teacache_state[k]['previous_residual'] = (hidden_states.to(cache_device) - ori_hidden_states)[i*b:(i+1)*b]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    torch.cuda.empty_cache()

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


class TeaCacheQwenImage:
    @classmethod
    def INPUT_TYPES(s):
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
        if rel_l1_thresh == 0:
            return (model,)

        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
        new_model.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        new_model.model_options["transformer_options"]["coefficients"] = SUPPORTED_MODELS_COEFFICIENTS[model_type]
        new_model.model_options["transformer_options"]["model_type"] = model_type
        new_model.model_options["transformer_options"]["cache_device"] = mm.get_torch_device() if cache_device == "cuda" else torch.device("cpu")
        
        diffusion_model = new_model.get_model_object("diffusion_model")

        # 检查是否是QwenImage模型
        if not hasattr(diffusion_model, 'transformer_blocks'):
            raise ValueError("The provided model is not a QwenImage model")
            
        # 使用patch替换原始forward方法
        context = patch.multiple(
            diffusion_model,
            forward=teacache_qwenimage_forward.__get__(diffusion_model, diffusion_model.__class__)
        )
        
        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            
            # 获取当前步数
            sigmas = c["transformer_options"]["sample_sigmas"]
            matched_step_index = (sigmas == timestep[0]).nonzero()
            if len(matched_step_index) > 0:
                current_step_index = matched_step_index.item()
            else:
                current_step_index = 0
                for i in range(len(sigmas) - 1):
                    if (sigmas[i] - timestep[0]) * (sigmas[i + 1] - timestep[0]) <= 0:
                        current_step_index = i
                        break
            
            # 重置缓存状态
            if current_step_index == 0:
                if hasattr(diffusion_model, 'teacache_state'):
                    delattr(diffusion_model, 'teacache_state')
            
            # 确定是否启用TeaCache
            current_percent = current_step_index / (len(sigmas) - 1)
            c["transformer_options"]["current_percent"] = current_percent
            if start_percent <= current_percent <= end_percent:
                c["transformer_options"]["enable_teacache"] = True
            else:
                c["transformer_options"]["enable_teacache"] = False
                
            with context:
                return model_function(input, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)
