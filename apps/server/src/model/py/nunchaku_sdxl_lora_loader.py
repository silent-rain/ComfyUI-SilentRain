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
    def INPUT_TYPES(s):
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
        assert isinstance(model_wrapper, ComfySDXLUNetWrapper)

        unet = model_wrapper.model
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)  # copy everything except the model
        ret_model_wrapper = ret_model.model.diffusion_model
        assert isinstance(ret_model_wrapper, ComfySDXLUNetWrapper)

        model_wrapper.model = unet
        ret_model_wrapper.model = unet

        lora_path = get_full_path_or_raise("loras", lora_name)

        # Reset any existing LoRAs before applying new ones
        ret_model_wrapper.reset_lora()
        ret_model_wrapper.loras = [(lora_path, lora_strength)]

        # Load the LoRA state dict to check for any special handling
        try:
            sd = load_state_dict_in_safetensors(lora_path)

            # Check if the LoRA modifies the input channels
            # SDXL LoRA keys might have different naming conventions
            input_keys = [k for k in sd.keys() if "input_blocks" in k and "lora_A" in k]
            if input_keys:
                # Find the first input block LoRA key
                first_key = input_keys[0]
                new_in_channels = sd[first_key].shape[1]
                old_in_channels = ret_model.model.model_config.unet_config.get(
                    "in_channels", 4
                )

                # Update input channels if needed
                if new_in_channels > old_in_channels:
                    ret_model.model.model_config.unet_config["in_channels"] = (
                        new_in_channels
                    )
                    logger.info(
                        f"Updated input channels from {old_in_channels} to {new_in_channels}"
                    )
        except Exception as e:
            logger.warning(f"Failed to process LoRA state dict: {e}")

        return (ret_model,)


class NunchakuSDXLLoraStack:
    """
    Node for loading and applying multiple LoRAs to a Nunchaku SDXL model with dynamic input.

    This node allows you to configure multiple LoRAs with their respective strengths
    in a single node, providing the same effect as chaining multiple LoRA nodes.

    Attributes
    ----------
    RETURN_TYPES : tuple
        The return type of the node ("MODEL",).
    OUTPUT_TOOLTIPS : tuple
        Tooltip for the output.
    FUNCTION : str
        The function to call ("load_lora_stack").
    TITLE : str
        Node title.
    CATEGORY : str
        Node category.
    DESCRIPTION : str
        Node description.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the LoRA stack node.

        Returns
        -------
        dict
            A dictionary specifying the required inputs and optional LoRA inputs.
        """
        # Base inputs
        inputs = {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model LoRAs will be applied to. "
                        "Make sure model is loaded by `Nunchaku SDXL UNet Loader`."
                    },
                ),
            },
            "optional": {},
        }

        # Add fixed number of LoRA inputs (10 slots)
        for i in range(1, 11):  # Support up to 10 LoRAs
            inputs["optional"][f"lora_name_{i}"] = (
                ["None"] + get_filename_list("loras"),
                {
                    "tooltip": f"The file name of LoRA {i}. Select 'None' to skip this slot."
                },
            )
            inputs["optional"][f"lora_strength_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": f"Strength for LoRA {i}. This value can be negative.",
                },
            )

        return inputs

    RETURN_TYPES = ("MODEL",)
    OUTPUT_TOOLTIPS = ("The modified diffusion model with all LoRAs applied.",)
    FUNCTION = "load_lora_stack"
    TITLE = "Nunchaku SDXL LoRA Stack"

    CATEGORY = "Nunchaku"
    DESCRIPTION = (
        "Apply multiple LoRAs to a diffusion model in a single node. "
        "Equivalent to chaining multiple LoRA nodes but more convenient for managing many LoRAs. "
        "Supports up to 10 LoRAs simultaneously. Set unused slots to 'None' to skip them."
    )

    def load_lora_stack(self, model, **kwargs):
        """
        Apply multiple LoRAs to a Nunchaku SDXL diffusion model.

        Parameters
        ----------
        model : object
            The diffusion model to modify.
        **kwargs
            Dynamic LoRA name and strength parameters.

        Returns
        -------
        tuple
            A tuple containing the modified diffusion model.
        """
        # Collect LoRA information to apply
        loras_to_apply = []

        for i in range(1, 11):  # Check all 10 LoRA slots
            lora_name = kwargs.get(f"lora_name_{i}")
            lora_strength = kwargs.get(f"lora_strength_{i}", 1.0)

            # Skip unset or None LoRAs
            if lora_name is None or lora_name == "None" or lora_name == "":
                continue

            # Skip LoRAs with zero strength
            if abs(lora_strength) < 1e-5:
                continue

            loras_to_apply.append((lora_name, lora_strength))

        # If no LoRAs need to be applied, return the original model
        if not loras_to_apply:
            return (model,)

        model_wrapper = model.model.diffusion_model
        assert isinstance(model_wrapper, ComfySDXLUNetWrapper)

        unet = model_wrapper.model
        model_wrapper.model = None
        ret_model = copy.deepcopy(model)  # copy everything except the model
        ret_model_wrapper = ret_model.model.diffusion_model
        assert isinstance(ret_model_wrapper, ComfySDXLUNetWrapper)

        model_wrapper.model = unet
        ret_model_wrapper.model = unet

        # Reset any existing LoRAs and clear the list
        ret_model_wrapper.reset_lora()
        ret_model_wrapper.loras = []

        # Track the maximum input channels needed
        max_in_channels = ret_model.model.model_config.unet_config.get("in_channels", 4)

        # Add all LoRAs
        for lora_name, lora_strength in loras_to_apply:
            lora_path = get_full_path_or_raise("loras", lora_name)
            ret_model_wrapper.loras.append((lora_path, lora_strength))

            # Check if input channels need to be updated
            try:
                sd = load_state_dict_in_safetensors(lora_path)
                # SDXL LoRA keys might have different naming conventions
                input_keys = [
                    k for k in sd.keys() if "input_blocks" in k and "lora_A" in k
                ]
                if input_keys:
                    # Find the first input block LoRA key
                    first_key = input_keys[0]
                    new_in_channels = sd[first_key].shape[1]
                    max_in_channels = max(max_in_channels, new_in_channels)
            except Exception as e:
                logger.warning(
                    f"Failed to process LoRA state dict for {lora_name}: {e}"
                )

        # Update the model's input channels
        if max_in_channels > ret_model.model.model_config.unet_config.get(
            "in_channels", 4
        ):
            ret_model.model.model_config.unet_config["in_channels"] = max_in_channels
            logger.info(f"Updated input channels to {max_in_channels}")

        return (ret_model,)
