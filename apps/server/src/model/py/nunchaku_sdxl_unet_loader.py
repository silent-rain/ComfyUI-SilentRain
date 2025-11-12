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
from folder_paths import get_filename_list, get_full_path_or_raise

from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
from nunchaku.utils import load_state_dict_in_safetensors


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

                # 添加模型参数信息
                logger.info(f"Model dtype: {next(self.unet.parameters()).dtype}")
                logger.info(f"Model device: {next(self.unet.parameters()).device}")

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

        # Reset any existing LoRA state
        model.diffusion_model.reset_lora()

        # Ensure the model has the correct dtype
        model.diffusion_model = model.diffusion_model.to(dtype=torch_dtype)

        # Set the latent_format for SDXL models
        # This is required for ComfyUI to properly handle latent images
        from comfy import latent_formats

        model.latent_format = latent_formats.SDXL()

        # Create model patcher with proper device management
        model = comfy.model_patcher.ModelPatcher(model, device, device_id)
        return (model,)


class ComfySDXLUNetWrapper(nn.Module):
    """
    Wrapper for NunchakuSDXLUNet2DConditionModel to support ComfyUI workflows.

    This wrapper adapts the diffusers UNet2DConditionModel interface to ComfyUI's UNetModel interface
    by mapping the parameter names and handling the different calling conventions.

    Parameters
    ----------
    model : NunchakuSDXLUNet2DConditionModel
        The underlying Nunchaku SDXL UNet model to wrap.
    """

    def __init__(self, model):
        super(ComfySDXLUNetWrapper, self).__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.loras = []  # List of (lora_path, lora_strength) tuples

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        """
        Forward pass for the wrapped SDXL UNet model.

        This method adapts ComfyUI's UNetModel interface to diffusers' UNet2DConditionModel interface.

        Parameters
        ----------
        x : torch.Tensor
            Input latent tensor of shape (batch, channels, height, width).
        timesteps : torch.Tensor or None
            Diffusion timestep tensor.
        context : torch.Tensor or None
            Text embeddings for conditioning.
        y : torch.Tensor or None
            Additional conditioning (for SDXL, this contains pooled text embeddings).
        control : dict or None
            ControlNet inputs.
        transformer_options : dict
            Additional transformer options.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        torch.Tensor
            Output tensor of the same spatial size as the input.
        """
        # Ensure inputs are on the same device as the model
        device = next(self.model.parameters()).device

        # Convert inputs to the correct dtype if needed
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        # Handle timesteps - ComfyUI may pass a tensor or a float
        if timesteps is not None:
            if isinstance(timesteps, torch.Tensor):
                # If it's a tensor, use it directly
                pass
            else:
                # If it's a scalar, convert to tensor
                timesteps = torch.tensor(
                    [timesteps], dtype=torch.float32, device=device
                )

        # Handle context (text embeddings)
        if context is not None:
            if context.dtype != self.dtype:
                context = context.to(self.dtype)

        # Handle y (pooled embeddings for SDXL)
        if y is not None:
            if y.dtype != self.dtype:
                y = y.to(self.dtype)

        # Prepare encoder_hidden_states for the model
        # In SDXL, we concatenate the text embeddings and pooled embeddings
        if context is not None and y is not None:
            # For SDXL, the model expects concatenated context and y
            encoder_hidden_states = context
            # The pooled embeddings are passed separately in the model call
        elif context is not None:
            encoder_hidden_states = context
        else:
            encoder_hidden_states = None

        # Prepare added_cond_kwargs for SDXL model
        # SDXL models expect text_embeds and time_ids in added_cond_kwargs
        added_cond_kwargs = None
        if y is not None:
            # y contains pooled embeddings and time_ids for SDXL
            # Split y into text_embeds and time_ids
            # In SDXL, y is typically a concatenation of pooled embeddings and time_ids
            # The first part is pooled embeddings (1280 dims), the rest is time_ids (6 dims)
            if y.shape[-1] >= 1286:  # 1280 (pooled) + 6 (time_ids)
                text_embeds = y[:, :1280]
                time_ids = y[:, 1280:1286]
                added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
            else:
                # Fallback if y doesn't have the expected dimensions
                added_cond_kwargs = {
                    "text_embeds": y,
                    "time_ids": torch.zeros(
                        (y.shape[0], 6), dtype=y.dtype, device=y.device
                    ),
                }

        # Apply LoRAs if any are loaded
        if self.loras:
            try:
                # Store original weights if not already stored
                if not hasattr(self, "_original_weights"):
                    self._original_weights = {}

                # Apply each LoRA
                for lora_path, lora_strength in self.loras:
                    if abs(lora_strength) < 1e-5:
                        continue  # Skip LoRAs with negligible strength

                    # Load LoRA state dict
                    lora_sd = load_state_dict_in_safetensors(lora_path)

                    # Apply LoRA weights to the model
                    for key, value in lora_sd.items():
                        # Skip non-weight keys
                        if not any(suffix in key for suffix in [".weight", ".bias"]):
                            continue

                        # Apply strength to LoRA weights
                        if lora_strength != 1.0:
                            value = value * lora_strength

                        # Get the target module and parameter
                        if "." in key:
                            module_name, param_name = key.rsplit(".", 1)

                            # Skip if we've already processed this LoRA for this module
                            cache_key = f"{lora_path}:{module_name}:{param_name}"
                            if cache_key in self._original_weights:
                                continue

                            try:
                                module = self.model
                                for part in module_name.split("."):
                                    module = getattr(module, part)

                                # Store original weight if not already stored
                                if hasattr(module, param_name):
                                    orig_weight = getattr(module, param_name).clone()
                                    self._original_weights[cache_key] = orig_weight

                                    # Apply LoRA weight
                                    if "lora_down" in key or "lora_A" in key:
                                        # Down projection LoRA
                                        setattr(module, param_name, orig_weight + value)
                                    elif "lora_up" in key or "lora_B" in key:
                                        # Up projection LoRA
                                        setattr(module, param_name, orig_weight + value)
                                    else:
                                        # Other types of LoRA
                                        setattr(module, param_name, orig_weight + value)
                            except Exception as e:
                                logger.warning(
                                    f"Failed to apply LoRA weight for {key}: {e}"
                                )
            except Exception as e:
                logger.warning(f"Failed to apply LoRAs: {e}")

        # Call the underlying Nunchaku model
        # The NunchakuSDXLUNet2DConditionModel follows the diffusers interface
        # Note: We don't pass pooled_projections as it's not supported by this model
        # Instead, pooled embeddings are handled through added_cond_kwargs
        output = self.model(
            sample=x,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )

        # Return the sample from the output
        return output.sample

    def reset_lora(self):
        """
        Reset all LoRA modifications to restore original model weights.
        """
        if hasattr(self, "_original_weights"):
            for cache_key, orig_weight in self._original_weights.items():
                try:
                    _, module_name, param_name = cache_key.split(":", 2)
                    module = self.model
                    for part in module_name.split("."):
                        module = getattr(module, part)

                    if hasattr(module, param_name):
                        setattr(module, param_name, orig_weight)
                except Exception as e:
                    logger.warning(f"Failed to reset LoRA weight for {cache_key}: {e}")

            # Clear the stored weights
            self._original_weights = {}
