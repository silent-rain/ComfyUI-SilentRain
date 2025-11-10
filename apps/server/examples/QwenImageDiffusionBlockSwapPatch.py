import torch
import torch.nn as nn
from types import MethodType
import gc

class QwenImageDiffusionBlockSwapPatch:
    def __init__(self, model, transformer_blocks_cuda_size=6, diffusion_blocks_cuda_size=4):
        """
        Qwen-Image扩散模型Block Swap Patch
        
        Args:
            model: Qwen-Image扩散模型
            transformer_blocks_cuda_size: Transformer块在GPU上的数量
            diffusion_blocks_cuda_size: 扩散块在GPU上的数量
        """
        self.model = model
        self.transformer_blocks_cuda_size = transformer_blocks_cuda_size
        self.diffusion_blocks_cuda_size = diffusion_blocks_cuda_size
        
        self._apply_patch()
    
    def _apply_patch(self):
        """应用block swap patch到扩散模型"""
        # 注册模型级别的pre-hook
        self.model.model.diffusion_model.register_forward_pre_hook(
            self._pre_diffusion_forward_hook
        )
        
        # 为Transformer块注册hooks
        if hasattr(self.model.model.diffusion_model, 'transformer_blocks'):
            transformer_blocks = self.model.model.diffusion_model.transformer_blocks
            for i in range(0, len(transformer_blocks), self.transformer_blocks_cuda_size):
                block_size = min(self.transformer_blocks_cuda_size, len(transformer_blocks) - i)
                transformer_blocks[i].register_forward_pre_hook(
                    self._generate_transformer_forward_hook(i, block_size)
                )
        
        # 为扩散块注册hooks
        if hasattr(self.model.model.diffusion_model, 'down_blocks'):
            down_blocks = self.model.model.diffusion_model.down_blocks
            for i in range(0, len(down_blocks), self.diffusion_blocks_cuda_size):
                block_size = min(self.diffusion_blocks_cuda_size, len(down_blocks) - i)
                down_blocks[i].register_forward_pre_hook(
                    self._generate_down_block_forward_hook(i, block_size)
                )
        
        if hasattr(self.model.model.diffusion_model, 'up_blocks'):
            up_blocks = self.model.model.diffusion_model.up_blocks
            for i in range(0, len(up_blocks), self.diffusion_blocks_cuda_size):
                block_size = min(self.diffusion_blocks_cuda_size, len(up_blocks) - i)
                up_blocks[i].register_forward_pre_hook(
                    self._generate_up_block_forward_hook(i, block_size)
                )
        
        # 处理mid_block（如果存在）
        if hasattr(self.model.model.diffusion_model, 'mid_block'):
            self.model.model.diffusion_model.mid_block.register_forward_pre_hook(
                self._mid_block_forward_hook
            )
    
    def _pre_diffusion_forward_hook(self, module, inp):
        """扩散模型前向传播前的hook"""
        # 将所有blocks移到CPU
        self._move_all_blocks_to_cpu()
        
        # 将输入处理层移到GPU
        self._move_input_layers_to_cuda()
        
        return inp
    
    def _move_all_blocks_to_cpu(self):
        """将所有transformer和扩散blocks移到CPU"""
        if hasattr(self.model.model.diffusion_model, 'transformer_blocks'):
            self.model.model.diffusion_model.transformer_blocks.to('cpu')
        
        if hasattr(self.model.model.diffusion_model, 'down_blocks'):
            self.model.model.diffusion_model.down_blocks.to('cpu')
        
        if hasattr(self.model.model.diffusion_model, 'up_blocks'):
            self.model.model.diffusion_model.up_blocks.to('cpu')
        
        if hasattr(self.model.model.diffusion_model, 'mid_block'):
            self.model.model.diffusion_model.mid_block.to('cpu')
        
        torch.cuda.empty_cache()
    
    def _move_input_layers_to_cuda(self):
        """将输入处理层移到GPU"""
        diffusion_model = self.model.model.diffusion_model
        
        # 时间嵌入
        if hasattr(diffusion_model, 'time_embed'):
            diffusion_model.time_embed.to('cuda')
        
        # 输入卷积层
        if hasattr(diffusion_model, 'conv_in'):
            diffusion_model.conv_in.to('cuda')
        
        # 类别嵌入
        if hasattr(diffusion_model, 'class_embed'):
            diffusion_model.class_embed.to('cuda')
        
        # 标签嵌入
        if hasattr(diffusion_model, 'label_emb'):
            diffusion_model.label_emb.to('cuda')
    
    def _generate_transformer_forward_hook(self, layer_start, layer_size):
        """生成Transformer块forward hook"""
        def transformer_forward_hook(module, inp):
            # 将之前的Transformer块移到CPU
            if layer_start > 0:
                self.model.model.diffusion_model.transformer_blocks[:layer_start].to('cpu')
            
            # 将当前需要的Transformer块移到GPU
            self.model.model.diffusion_model.transformer_blocks[
                layer_start:layer_start + layer_size
            ].to('cuda')
            
            torch.cuda.empty_cache()
            return inp
        
        return transformer_forward_hook
    
    def _generate_down_block_forward_hook(self, layer_start, layer_size):
        """生成下采样块forward hook"""
        def down_block_forward_hook(module, inp):
            # 将之前的下采样块移到CPU
            if layer_start > 0:
                self.model.model.diffusion_model.down_blocks[:layer_start].to('cpu')
            
            # 将当前需要的下采样块移到GPU
            self.model.model.diffusion_model.down_blocks[
                layer_start:layer_start + layer_size
            ].to('cuda')
            
            torch.cuda.empty_cache()
            return inp
        
        return down_block_forward_hook
    
    def _generate_up_block_forward_hook(self, layer_start, layer_size):
        """生成上采样块forward hook"""
        def up_block_forward_hook(module, inp):
            # 将之前的上采样块移到CPU
            if layer_start > 0:
                self.model.model.diffusion_model.up_blocks[:layer_start].to('cpu')
            
            # 将当前需要的上采样块移到GPU
            self.model.model.diffusion_model.up_blocks[
                layer_start:layer_start + layer_size
            ].to('cuda')
            
            torch.cuda.empty_cache()
            return inp
        
        return up_block_forward_hook
    
    def _mid_block_forward_hook(self, module, inp):
        """中间块forward hook"""
        # 确保中间块在GPU上
        module.to('cuda')
        torch.cuda.empty_cache()
        return inp


# ComfyUI节点实现
class QwenImageDiffusionLoader_BlockSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "transformer_blocks_cuda_size": ("INT", {"min": 1, "max": 24, "default": 6}),
                "down_blocks_cuda_size": ("INT", {"min": 1, "max": 12, "default": 3}),
                "up_blocks_cuda_size": ("INT", {"min": 1, "max": 12, "default": 3}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/model"
    
    def load_unet(self, model_name, transformer_blocks_cuda_size=6, 
                   down_blocks_cuda_size=3, up_blocks_cuda_size=3):
        # 加载原始扩散模型
        model_path = folder_paths.get_full_path("diffusion_models", model_name)
        model = comfy.utils.load_unet_model(model_path)
        
        # 应用block swap patch
        patched_model = QwenImageDiffusionBlockSwapPatch(
            model,
            transformer_blocks_cuda_size=transformer_blocks_cuda_size,
            diffusion_blocks_cuda_size=min(down_blocks_cuda_size, up_blocks_cuda_size)
        )
        
        return (patched_model.model,)


# 针对Qwen-Image的更精确实现
class QwenImageDiTBlockSwapPatch:
    """专门针对Qwen-Image DiT架构的Block Swap"""
    
    def __init__(self, model, dit_blocks_cuda_size=6):
        self.model = model
        self.dit_blocks_cuda_size = dit_blocks_cuda_size
        self._apply_patch()
    
    def _apply_patch(self):
        """应用DiT block swap"""
        dit_model = self.model.model.diffusion_model
        
        # 注册模型级别hook
        dit_model.register_forward_pre_hook(self._pre_dit_forward_hook)
        
        # 为DiT blocks注册hooks
        if hasattr(dit_model, 'blocks'):
            blocks = dit_model.blocks
            for i in range(0, len(blocks), self.dit_blocks_cuda_size):
                block_size = min(self.dit_blocks_cuda_size, len(blocks) - i)
                blocks[i].register_forward_pre_hook(
                    self._generate_dit_block_forward_hook(i, block_size)
                )
    
    def _pre_dit_forward_hook(self, module, inp):
        """DiT模型前向传播前的hook"""
        # 将所有blocks移到CPU
        if hasattr(module, 'blocks'):
            module.blocks.to('cpu')
        
        # 将输入层移到GPU
        self._move_dit_inputs_to_cuda(module)
        
        torch.cuda.empty_cache()
        return inp
    
    def _move_dit_inputs_to_cuda(self, dit_model):
        """将DiT输入层移到GPU"""
        if hasattr(dit_model, 'x_embedder'):
            dit_model.x_embedder.to('cuda')
        if hasattr(dit_model, 't_embedder'):
            dit_model.t_embedder.to('cuda')
        if hasattr(dit_model, 'y_embedder'):
            dit_model.y_embedder.to('cuda')
        if hasattr(dit_model, 'pos_embed'):
            dit_model.pos_embed.to('cuda')
    
    def _generate_dit_block_forward_hook(self, layer_start, layer_size):
        """生成DiT块forward hook"""
        def dit_block_forward_hook(module, inp):
            dit_model = self.model.model.diffusion_model
            
            # 将之前的块移到CPU
            if layer_start > 0:
                dit_model.blocks[:layer_start].to('cpu')
            
            # 将当前需要的块移到GPU
            dit_model.blocks[layer_start:layer_start + layer_size].to('cuda')
            
            torch.cuda.empty_cache()
            return inp
        
        return dit_block_forward_hook
