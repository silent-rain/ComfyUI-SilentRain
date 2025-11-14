"""
This module provides the :class:`NunchakuSDXLLoraLoader` node
for applying LoRA weights to Nunchaku SDXL models within ComfyUI.
"""

import copy
import logging
import os

import torch
from nunchaku.utils import load_state_dict_in_safetensors
from folder_paths import get_filename_list, get_full_path_or_raise

# Import the new converter
from nunchaku_sdxl_lora_converter import to_diffusers, convert_comfyui_to_nunchaku_sdxl_keys


# 使用绝对导入导入ComfySDXLUNetWrapper
# 注意：这个模块应该已经在Rust代码中被添加到sys.modules中
try:
    from comfy_sdxl_wrapper import ComfySDXLWrapper
except ImportError as e:
    # 如果导入失败，抛出更详细的错误信息
    raise ImportError(
        f"无法导入ComfySDXLUNetWrapper: {e}. 请确保comfy_sdxl_unet_wrapper模块已在Rust代码中正确加载并添加到sys.modules中。"
    )


# The convert_comfyui_to_nunchaku_sdxl_keys function is now imported from nunchaku_sdxl_lora_converter


# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class NunchakuSDXLLoraLoader:
    """
    Node for loading and applying a LoRA to a Nunchaku SDXL model.

    Attributes
    ----------
    RETURN_TYPES : tuple
        The return type of the node ("MODEL",).
    OUTPUT_TOOLTIPS : tuple
        Tooltip for the output.
    FUNCTION : str
        The function to call ("load_lora").
    TITLE : str
        Node title.
    CATEGORY : str
        Node category.
    DESCRIPTION : str
        Node description.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types and tooltips for the node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and their descriptions for node interface.
        """
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model LoRA will be applied to. "
                        "Make sure model is loaded by `Nunchaku SDXL UNet Loader`."
                    },
                ),
                "lora_name": (
                    get_filename_list("loras"),
                    {"tooltip": "The file name of the LoRA."},
                ),
                "lora_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model.",)
    FUNCTION = "load_lora"
    TITLE = "Nunchaku SDXL LoRA Loader"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "LoRAs are used to modify the diffusion model, "
        "altering the way in which latents are denoised such as applying styles. "
        "You can link multiple LoRA nodes."
    )

    def load_lora(self, model, lora_name: str, lora_strength: float):
        """
        Apply a LoRA to a Nunchaku SDXL diffusion model.

        Parameters
        ----------
        model : object
            The diffusion model to modify.
        lora_name : str
            The name of the LoRA to apply.
        lora_strength : float
            The strength with which to apply the LoRA.

        Returns
        -------
        tuple
            A tuple containing the modified diffusion model.
        """
        if abs(lora_strength) < 1e-5:
            return (model,)  # If the strength is too small, return the original model

        model_wrapper = model.model.diffusion_model

        # 安全地检查类型，使用类名比较而不是isinstance
        expected_class_name = "ComfySDXLWrapper"
        actual_class_name = model_wrapper.__class__.__name__
        
        if actual_class_name != expected_class_name:
            logger.error(f"Expected {expected_class_name} but got {actual_class_name}")
            return (model,)  # 返回原始模型而不是抛出错误

        unet = model_wrapper.model
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)  # copy everything except the model
        ret_model_wrapper = ret_model.model.diffusion_model

        # 安全地检查返回模型的包装器类型
        ret_actual_class_name = ret_model_wrapper.__class__.__name__
        if ret_actual_class_name != expected_class_name:
            logger.error(
                f"Expected {expected_class_name} for ret_model but got {ret_actual_class_name}"
            )
            # 恢复原始模型并返回
            model_wrapper.model = unet
            return (model,)

        model_wrapper.model = unet
        ret_model_wrapper.model = unet

        lora_path = get_full_path_or_raise("loras", lora_name)

        # Add the LoRA to the existing list (append instead of replace)
        ret_model_wrapper.loras.append((lora_path, lora_strength))

        # Load the LoRA state dict and convert keys to Nunchaku SDXL format
        lora_state_dict = load_state_dict_in_safetensors(lora_path, device="cpu")
        
        # Log LoRA format information
        logger.info(f"Loading LoRA from {lora_path}")
        logger.info(f"LoRA has {len(lora_state_dict)} keys")
        
        # Check LoRA format
        if any(k.startswith("lora_unet_") for k in lora_state_dict.keys()):
            logger.info("Detected ComfyUI format LoRA")
        elif any(k.startswith("base_model.model.") for k in lora_state_dict.keys()):
            logger.info("Detected PEFT format LoRA")
        else:
            logger.info("Unknown LoRA format, attempting conversion")
        
        # Convert to PEFT format for Nunchaku
        converted_lora = to_diffusers(lora_state_dict)
        logger.info(f"Converted LoRA has {len(converted_lora)} keys")
        
        # Log a few converted keys for debugging
        sample_keys = list(converted_lora.keys())[:3]
        logger.info(f"Sample converted keys: {sample_keys}")
        
        # Apply the LoRA to the model
        # Note: The actual application of LoRA weights will be handled by the Nunchaku model
        # during inference, similar to how it's done in the Flux implementation
        
        # Store the LoRA information for later application
        ret_model_wrapper.lora_state_dict = converted_lora
        ret_model_wrapper.lora_strength = lora_strength
        
        logger.info(f"LoRA {lora_name} loaded with strength {lora_strength}")

        return (ret_model,)
