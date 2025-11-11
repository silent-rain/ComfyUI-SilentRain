"""
Flux.1-Dev 扩散模型 Block Swap Patch（适配纯Transformer架构）
"""
import torch

class Flux1BlockSwapPatch:
    def __init__(self, model, double_blocks_cuda_size=6, single_blocks_cuda_size=6):
        """
        Flux.1-Dev扩散模型Block Swap Patch（适配纯Transformer架构）
        
        Args:
            model: Flux.1-Dev扩散模型
            double_blocks_cuda_size: double_blocks在GPU上的数量
            single_blocks_cuda_size: single_blocks在GPU上的数量
        """
        self.model = model
        self.double_blocks_cuda_size = double_blocks_cuda_size
        self.single_blocks_cuda_size = single_blocks_cuda_size
        
        self._apply_patch()
    
    def _apply_patch(self):
        """应用block swap patch到Flux.1-Dev模型"""
        # 注册模型级别的pre-hook
        self.model.model.diffusion_model.register_forward_pre_hook(
            self._pre_diffusion_forward_hook
        )
        
        # 为double_blocks注册hooks
        double_blocks_depth = len(self.model.model.diffusion_model.double_blocks)
        for i in range(0, double_blocks_depth, self.double_blocks_cuda_size):
            block_size = min(self.double_blocks_cuda_size, double_blocks_depth - i)
            self.model.model.diffusion_model.double_blocks[i].register_forward_pre_hook(
                self._generate_double_blocks_forward_hook(i, block_size)
            )
        
        # 为single_blocks注册hooks
        single_blocks_depth = len(self.model.model.diffusion_model.single_blocks)
        for i in range(0, single_blocks_depth, self.single_blocks_cuda_size):
            block_size = min(self.single_blocks_cuda_size, single_blocks_depth - i)
            self.model.model.diffusion_model.single_blocks[i].register_forward_pre_hook(
                self._generate_single_blocks_forward_hook(i, block_size)
            )
    
    def _pre_diffusion_forward_hook(self, module, inp):
        """扩散模型前向传播前的hook"""
        # 将所有blocks移到CPU
        self._double_blocks_to_cpu()
        self._single_blocks_to_cpu()
        
        # 将输入处理层移到GPU
        self._move_other_layers_to_cuda()
        
        return inp
    
    def _double_blocks_to_cpu(self, layer_start=0, layer_size=-1):
        """将double_blocks移到CPU"""
        if layer_size == -1:
            self.model.model.diffusion_model.double_blocks.to("cpu")
        else:
            self.model.model.diffusion_model.double_blocks[
                layer_start:layer_start + layer_size
            ].to("cpu")
        
        torch.cuda.empty_cache()
    
    def _double_blocks_to_cuda(self, layer_start=0, layer_size=-1):
        """将double_blocks移到GPU"""
        if layer_size == -1:
            self.model.model.diffusion_model.double_blocks.to("cuda")
        else:
            self.model.model.diffusion_model.double_blocks[
                layer_start:layer_start + layer_size
            ].to("cuda")
    
    def _single_blocks_to_cpu(self, layer_start=0, layer_size=-1):
        """将single_blocks移到CPU"""
        if layer_size == -1:
            self.model.model.diffusion_model.single_blocks.to("cpu")
        else:
            self.model.model.diffusion_model.single_blocks[
                layer_start:layer_start + layer_size
            ].to("cpu")
        
        torch.cuda.empty_cache()
    
    def _single_blocks_to_cuda(self, layer_start=0, layer_size=-1):
        """将single_blocks移到GPU"""
        if layer_size == -1:
            self.model.model.diffusion_model.single_blocks.to("cuda")
        else:
            self.model.model.diffusion_model.single_blocks[
                layer_start:layer_start + layer_size
            ].to("cuda")
    
    def _move_other_layers(self, device):
        """将其他层移到指定设备"""
        diffusion_model = self.model.model.diffusion_model
        
        # 定义需要移动的层列表
        other_layers = [
            'img_in',           # 图像输入层
            'time_in',          # 时间输入层
            'guidance_in',      # 引导输入层
            'vector_in',        # 向量输入层
            'txt_in',           # 文本输入层
            'pe_embedder'       # 位置编码嵌入层
        ]
        
        # 循环移动所有层到指定设备
        for layer_name in other_layers:
            if hasattr(diffusion_model, layer_name):
                getattr(diffusion_model, layer_name).to(device)
        
        # 如果是移动到CPU，清理GPU缓存
        if device == 'cpu':
            torch.cuda.empty_cache()
    
    def _move_other_layers_to_cpu(self):
        """将其他层移到CPU"""
        self._move_other_layers('cpu')
    
    def _move_other_layers_to_cuda(self):
        """将其他层移到GPU"""
        self._move_other_layers('cuda')
    
    def _generate_double_blocks_forward_hook(self, layer_start, layer_size):
        """生成double_blocks forward hook"""
        def double_blocks_forward_hook(module, inp):
            # 将其他层移到CPU
            self._move_other_layers_to_cpu()
            
            # 将之前的double_blocks移到CPU
            if layer_start > 0:
                self._double_blocks_to_cpu(layer_start=0, layer_size=layer_start)
            
            # 将当前需要的double_blocks移到GPU
            self._double_blocks_to_cuda(layer_start=layer_start, layer_size=layer_size)
            
            torch.cuda.empty_cache()
            return inp
        
        return double_blocks_forward_hook
    
    def _generate_single_blocks_forward_hook(self, layer_start, layer_size):
        """生成single_blocks forward hook"""
        def single_blocks_forward_hook(module, inp):
            # 将double_blocks移到CPU
            self._double_blocks_to_cpu()
            
            # 将之前的single_blocks移到CPU
            if layer_start > 0:
                self._single_blocks_to_cpu(layer_start=0, layer_size=layer_start)
            
            # 将当前需要的single_blocks移到GPU
            self._single_blocks_to_cuda(layer_start=layer_start, layer_size=layer_size)
            
            torch.cuda.empty_cache()
            return inp
        
        return single_blocks_forward_hook


# ComfyUI节点实现
class Flux1BlockSwapPatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", ),
                "double_blocks": ("INT", {"min": 0, "max": 19, "default": 7, "tooltip": "double blocks 19"}),
                "single_blocks": ("INT", {"min": 0, "max": 38, "default": 7, "tooltip": "single blocks 38"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "advanced/model"
    
    def apply_patch(self, model, double_blocks=6, single_blocks=6):
        # 应用block swap patch
        patched_model = Flux1BlockSwapPatch(
            model,
            double_blocks_cuda_size=double_blocks,
            single_blocks_cuda_size=single_blocks
        )
        
        return (patched_model.model,)
