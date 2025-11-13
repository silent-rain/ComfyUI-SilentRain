"""
Nunchaku SDXL UNet Loader for ComfyUI

This module provides a ComfyUI node for loading Nunchaku-optimized SDXL UNet models.
It supports quantized inference with SVDQ quantization and integrates with ComfyUI workflows.
"""

import gc
import logging
import os

import comfy.model_management
import comfy.model_patcher
import torch
from torch import nn
import torch.nn.functional as F
from comfy.supported_models import SDXL
from folder_paths import get_filename_list

# Set the latent_format for SDXL models
# This is required for ComfyUI to properly handle latent images
from comfy import latent_formats

from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel

# 使用绝对导入导入ComfySDXLUNetWrapper
# 注意：这个模块应该已经在Rust代码中被添加到sys.modules中
try:
    from comfy_sdxl_unet_wrapper import ComfySDXLUNetWrapper
except ImportError as e:
    # 如果导入失败，抛出更详细的错误信息
    raise ImportError(
        f"无法导入ComfySDXLUNetWrapper: {e}. 请确保comfy_sdxl_unet_wrapper模块已在Rust代码中正确加载并添加到sys.modules中。"
    )


# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NunchakuSDXLUNetLoader:
    """
    Loader for Nunchaku SDXL UNet models.

    This class manages model loading, device selection, and configuration
    for efficient SDXL inference with Nunchaku quantization.

    Attributes
    ----------
    unet : :class:`~nunchaku.models.unets.unet_sdxl.NunchakuSDXLUNet2DConditionModel` or None
        The loaded UNet model.
    metadata : dict or None
        Metadata associated with the loaded model.
    model_path : str or None
        Path to the loaded model.
    device : torch.device or None
        Device on which the model is loaded.
    data_type : str or None
        Data type used for inference.
    patcher : object or None
        ComfyUI model patcher instance.
    """

    def __init__(self):
        """
        Initialize the NunchakuSDXLUNetLoader.

        Sets up internal state and selects the default torch device.
        """
        self.unet = None
        self.metadata = None
        self.model_path = None
        self.device = None
        self.data_type = None
        self.patcher = None
        self.device = comfy.model_management.get_torch_device()

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for the node interface.
        """
        safetensor_files = get_filename_list("diffusion_models")

        ngpus = torch.cuda.device_count()

        # Determine supported data types based on GPU capabilities
        if ngpus > 0:
            # Check if all GPUs support bfloat16
            all_support_bfloat16 = True
            for i in range(torch.cuda.device_count()):
                if (
                    torch.cuda.get_device_capability(i)[0] < 8
                ):  # Ampere or newer required for bfloat16
                    all_support_bfloat16 = False
                    break

            if all_support_bfloat16:
                dtype_options = ["bfloat16", "float16"]
            else:
                dtype_options = ["float16"]
        else:
            dtype_options = ["float16"]

        return {
            "required": {
                "model_path": (
                    safetensor_files,
                    {"tooltip": "The Nunchaku SDXL UNet model."},
                ),
                "device_id": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": ngpus - 1 if ngpus > 0 else 0,
                        "step": 1,
                        "display": "number",
                        "lazy": True,
                        "tooltip": "The GPU device ID to use for the model.",
                    },
                ),
                "data_type": (
                    dtype_options,
                    {
                        "default": dtype_options[0],
                        "tooltip": "Specifies the model's data type. Default is `bfloat16` for compatible GPUs, otherwise `float16`.",
                    },
                ),
            },
            "optional": {
                "cpu_offload": (
                    ["enable", "disable"],
                    {
                        "default": "disable",
                        "tooltip": "Whether to enable CPU offload for the UNet model. Note: SDXL UNet does not support offload.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Nunchaku"
    TITLE = "Nunchaku SDXL UNet Loader"

    def load_model(
        self,
        model_path: str,
        device_id: int,
        data_type: str,
        cpu_offload: str = "disable",
    ):
        """
        Load a Nunchaku SDXL UNet model with the specified configuration.

        Parameters
        ----------
        model_path : str
            Path to the model directory or safetensors file.
        device_id : int
            GPU device ID to use.
        data_type : str
            Data type for inference ("bfloat16" or "float16").
        cpu_offload : str
            Whether to enable CPU offload ("enable" or "disable").

        Returns
        -------
        tuple
            A tuple containing the loaded and patched model.

        Raises
        ------
        ValueError
            If the device_id is invalid or CPU offload is requested (not supported).
        """
        # Set device
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")

        # model_path = get_full_path_or_raise("diffusion_models", model_path)

        # Check if CPU offload is requested (not supported for SDXL UNet)
        cpu_offload_bool = cpu_offload == "enable"
        if cpu_offload_bool:
            logger.warning(
                "CPU offload is not supported for Nunchaku SDXL UNet. Disabling offload."
            )

        # Check if the device_id is valid
        # if device_id >= torch.cuda.device_count() and torch.cuda.is_available():
        #     raise ValueError(f"Invalid device_id: {device_id}. Only {torch.cuda.device_count()} GPUs available.")

        # Get GPU properties if available
        if torch.cuda.is_available():
            gpu_properties = torch.cuda.get_device_properties(device_id)
            gpu_memory = gpu_properties.total_memory / (1024**2)  # Convert to MiB
            gpu_name = gpu_properties.name
            logger.debug(f"GPU {device_id} ({gpu_name}) Memory: {gpu_memory} MiB")

        # Determine torch dtype
        if data_type == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16

        # Check if we need to reload the model
        if (
            self.model_path != model_path
            or self.device != device
            or self.data_type != data_type
        ):
            # Clean up previous model if exists
            if self.unet is not None:
                model_size = comfy.model_management.module_size(self.unet)
                unet = self.unet
                self.unet = None
                unet.to("cpu")
                del unet
                gc.collect()
                comfy.model_management.cleanup_models_gc()
                comfy.model_management.soft_empty_cache()
                if torch.cuda.is_available():
                    comfy.model_management.free_memory(model_size, device)

            # Load the Nunchaku SDXL UNet model
            logger.info(f"Loading Nunchaku SDXL UNet from: {model_path}")
            logger.info(f"device: {device}")
            logger.info(f"torch_dtype: {torch_dtype}")
            logger.info(f"cpu_offload_bool: {cpu_offload_bool}")

            try:
                self.unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(
                    model_path,
                    device=device,
                    torch_dtype=torch_dtype,
                    offload=cpu_offload_bool,
                )

                self.model_path = model_path
                self.device = device
                self.data_type = data_type

                logger.info("Nunchaku SDXL UNet loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Nunchaku SDXL UNet: {e}")
                # 确保异常被正确传播，而不是返回空值
                raise RuntimeError(f"Failed to load Nunchaku SDXL UNet: {e}") from e

        # Create SDXL model configuration optimized for Nunchaku SDXL UNet
        unet_config = {
            "image_size": 128,
            "in_channels": 4,
            "model_channels": 320,
            "out_channels": 4,
            "num_res_blocks": [2, 2, 2],
            "dropout": 0,
            "channel_mult": [1, 2, 4],
            "conv_resample": True,
            "dims": 2,
            "num_classes": "sequential",
            "use_checkpoint": False,
            "dtype": "torch.float16",
            "num_heads": -1,
            "num_head_channels": 64,
            "num_heads_upsample": -1,
            "use_scale_shift_norm": False,
            "resblock_updown": False,
            "use_new_attention_order": False,
            "use_spatial_transformer": True,
            "transformer_depth": [0, 0, 2, 2, 10, 10],
            "context_dim": 2048,
            "n_embed": None,
            "legacy": False,
            "disable_self_attentions": None,
            "num_attention_blocks": None,
            "disable_middle_self_attn": False,
            "use_linear_in_transformer": True,
            "adm_in_channels": 2816,
            "transformer_depth_middle": 10,
            "transformer_depth_output": [
                0,
                0,
                0,
                2,
                2,
                2,
                10,
                10,
                10,
            ],
            "use_temporal_resblock": False,
            "use_temporal_attention": False,
            "time_context_dim": None,
            "extra_ff_mix_layer": False,
            "use_spatial_context": False,
            "merge_strategy": None,
            "merge_factor": 0.0,
            "video_kernel_size": None,
            "disable_temporal_crossattention": False,
            "max_ddpm_temb_period": 10000,
            "attn_precision": None,
        }

        # Create SDXL model configuration
        model_config = SDXL(unet_config)
        model_config.set_inference_dtype(torch_dtype, None)
        model_config.custom_operations = None

        # Create the model and wrap the UNet with ComfySDXLUNetWrapper
        model = model_config.get_model({})

        # Replace the diffusion_model with our wrapped Nunchaku model
        # This ensures that the model has the correct interface for ComfyUI
        model.diffusion_model = ComfySDXLUNetWrapper(self.unet)
        logger.info(
            f"Set model.diffusion_model to ComfySDXLUNetWrapper, type: {type(model.diffusion_model)}"
        )

        # Reset any existing LoRA state
        if not (
            hasattr(model.diffusion_model, "reset_lora")
            and hasattr(model.diffusion_model, "model")
            and hasattr(model.diffusion_model, "loras")
        ):
            logger.error("model.diffusion_model is missing required attributes")
            raise RuntimeError("ComfySDXLUNetWrapper initialization failed")
        else:
            model.diffusion_model.reset_lora()
            logger.info("Reset LoRA state")

        # Ensure the model has the correct dtype
        model.diffusion_model = model.diffusion_model.to(dtype=torch_dtype)
        logger.info(
            f"After dtype conversion, model.diffusion_model type: {type(model.diffusion_model)}"
        )

        model.latent_format = latent_formats.SDXL()

        # Create model patcher with proper device management
        model = comfy.model_patcher.ModelPatcher(model, device, device_id)
        return (model,)
