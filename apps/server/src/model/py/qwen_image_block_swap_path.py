"""
Qwen-Image扩散模型 Block Swap Patch（适配纯Transformer架构）
"""

import torch


class QwenImageBlockSwapPatch:
    def __init__(self, model, transformer_blocks_cuda_size=6):
        """
        Qwen-Image扩散模型Block Swap Patch（适配纯Transformer架构）

        Args:
            model: Qwen-Image扩散模型
            transformer_blocks_cuda_size: Transformer块在GPU上的数量
        """
        self.model = model
        self.transformer_blocks_cuda_size = transformer_blocks_cuda_size

        self._apply_patch()

    def _apply_patch(self):
        """应用block swap patch到Qwen-Image模型"""
        # 注册模型级别的pre-hook
        self.model.model.diffusion_model.register_forward_pre_hook(
            self._pre_diffusion_forward_hook
        )

        # 为Transformer块注册hooks transformer_blocks
        if self.transformer_blocks_cuda_size > 0 and hasattr(self.model.model.diffusion_model, "transformer_blocks"):
            transformer_blocks = self.model.model.diffusion_model.transformer_blocks
            double_blocks_depth = len(transformer_blocks)
            steps = self.transformer_blocks_cuda_size
            for i in range(0, double_blocks_depth, steps):
                block_size = steps
                if i + block_size > double_blocks_depth:
                    block_size = double_blocks_depth - i
                transformer_blocks[i].register_forward_pre_hook(
                    self._generate_transformer_forward_hook(i, block_size)
                )

    def _pre_diffusion_forward_hook(self, module, inp):
        """扩散模型前向传播前的hook"""
        # 将所有blocks移到CPU
        if self.transformer_blocks_cuda_size > 0:
            self._transformer_blocks_to_cpu()

        # 将输入处理层移到GPU
        self._move_input_layers_to_cuda()

        return inp

    def _transformer_blocks_to_cpu(self, layer_start=0, layer_size=-1):
        """将transformer blocks移到CPU"""
        if hasattr(self.model.model.diffusion_model, "transformer_blocks"):
            if layer_size == -1:
                self.model.model.diffusion_model.transformer_blocks.to("cpu")
            else:
                self.model.model.diffusion_model.transformer_blocks[
                    layer_start : layer_start + layer_size
                ].to("cpu")

        torch.cuda.empty_cache()

    def _transformer_blocks_to_cuda(self, layer_start=0, layer_size=-1):
        """将transformer blocks移到GPU"""
        if hasattr(self.model.model.diffusion_model, "transformer_blocks"):
            if layer_size == -1:
                self.model.model.diffusion_model.transformer_blocks.to("cuda")
            else:
                self.model.model.diffusion_model.transformer_blocks[
                    layer_start : layer_start + layer_size
                ].to("cuda")

    def _move_input_layers(self, device):
        """将输入处理层移到指定设备"""
        diffusion_model = self.model.model.diffusion_model

        # 定义需要移动的层列表
        input_layers = [
            "time_text_embed",  # 时间嵌入层
            "img_in",  # 图像输入层
            "txt_in",  # 文本输入层
            "txt_norm",  # 文本归一化层
            "norm_out",  # 输出归一化层
            "proj_out",  # 输出投影层
        ]

        # 循环移动所有层到指定设备
        for layer_name in input_layers:
            if hasattr(diffusion_model, layer_name):
                getattr(diffusion_model, layer_name).to(device)

        # 如果是移动到CPU，清理GPU缓存
        if device == "cpu":
            torch.cuda.empty_cache()

    def _move_input_layers_to_cpu(self):
        """将输入处理层移到CPU"""
        self._move_input_layers("cpu")

    def _move_input_layers_to_cuda(self):
        """将输入处理层移到GPU"""
        self._move_input_layers("cuda")

    def _generate_transformer_forward_hook(self, layer_start, layer_size):
        """生成Transformer块forward hook"""

        def transformer_forward_hook(module, inp):
            # 将输入层移到CPU
            self._move_input_layers_to_cpu()

            # 将之前的Transformer块移到CPU
            if layer_start > 0:
                self._transformer_blocks_to_cpu(layer_start=0, layer_size=layer_start)

            # 将当前需要的Transformer块移到GPU
            self._transformer_blocks_to_cuda(
                layer_start=layer_start, layer_size=layer_size
            )

            torch.cuda.empty_cache()
            return inp

        return transformer_forward_hook


# ComfyUI节点实现 - 这个类名与上面的类冲突，需要重命名或删除
# 由于Rust代码会直接调用上面的QwenImageBlockSwapPatch类，这个ComfyUI节点实现可以删除
# 或者重命名为QwenImageBlockSwapPatchNode
class QwenImageBlockSwapPatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "blocks_cuda_size": (
                    "INT",
                    {
                        "min": 1,
                        "max": 60,
                        "default": 6,
                        "tooltip": "QwenImage transformer_blocks 60",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/model"

    def load_unet(self, model, blocks_cuda_size=6):
        # 应用block swap patch
        patched_model = QwenImageBlockSwapPatch(
            model, transformer_blocks_cuda_size=blocks_cuda_size
        )

        return (patched_model.model,)
