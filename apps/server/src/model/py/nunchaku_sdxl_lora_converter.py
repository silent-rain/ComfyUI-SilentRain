"""
This module implements the functions to convert SDXL LoRA weights from various formats
to the Diffusers format, which will later be converted to Nunchaku format.
"""

import logging
import os

import torch
from diffusers.loaders.lora_pipeline import LoraLoaderMixin
from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
from safetensors.torch import save_file

from nunchaku.utils import load_state_dict_in_safetensors

# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def handle_kohya_lora(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert Kohya LoRA format keys to Diffusers format for SDXL.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        LoRA weights, possibly in Kohya format.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in Diffusers format.
    """
    # Check if the state_dict is in the kohya format
    # SDXL Kohya format typically starts with "lora_unet_"
    if any([not k.startswith("lora_unet_") for k in state_dict.keys()]):
        return state_dict
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("lora_unet_", "unet.")
            
            # Convert underscores to dots for hierarchical structure
            new_k = new_k.replace("_", ".")
            
            # Convert lora_down/lora_up to lora_A/lora_B
            new_k = new_k.replace(".lora_down.", ".lora_A.")
            new_k = new_k.replace(".lora_up.", ".lora_B.")
            
            new_state_dict[new_k] = v
        return new_state_dict


def convert_comfyui_to_nunchaku_sdxl_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert ComfyUI LoRA format keys to Nunchaku SDXL format.

    This function handles the conversion from ComfyUI's lora_unet_* format
    to the format expected by Nunchaku SDXL models.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        LoRA weights in ComfyUI format.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in Nunchaku SDXL format.
    """
    converted_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Convert ComfyUI format to Diffusers format first
        if key.startswith("lora_unet_"):
            # Remove lora_unet_ prefix
            new_key = key.replace("lora_unet_", "")
            
            # Convert underscores to dots for hierarchical structure
            new_key = new_key.replace("_", ".")
            
            # Convert lora_down/lora_up to lora_A/lora_B
            new_key = new_key.replace(".lora_down.", ".lora_A.")
            new_key = new_key.replace(".lora_up.", ".lora_B.")
            
            # Add unet. prefix
            new_key = f"unet.{new_key}"
        
        converted_dict[new_key] = value
        
        if key != new_key:
            logger.debug(f"Converted: {key} → {new_key}")
    
    return converted_dict


def convert_nunchaku_to_comfyui_sdxl_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert Nunchaku SDXL LoRA format keys to ComfyUI format.

    This function handles the conversion from Nunchaku SDXL format
    to ComfyUI's lora_unet_* format.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        LoRA weights in Nunchaku SDXL format.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in ComfyUI format.
    """
    converted_dict = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Convert Nunchaku format to ComfyUI format
        if key.startswith("unet."):
            # Remove unet. prefix
            new_key = key.replace("unet.", "")
            
            # Convert dots to underscores for ComfyUI format
            new_key = new_key.replace(".", "_")
            
            # Convert lora_A/lora_B to lora_down/lora_up
            new_key = new_key.replace("lora_A", "lora_down")
            new_key = new_key.replace("lora_B", "lora_up")
            
            # Add lora_unet_ prefix
            new_key = f"lora_unet_{new_key}"
        
        converted_dict[new_key] = value
        
        if key != new_key:
            logger.debug(f"Converted: {key} → {new_key}")
    
    return converted_dict


def to_diffusers(input_lora: str | dict[str, torch.Tensor], output_path: str | None = None) -> dict[str, torch.Tensor]:
    """
    Convert LoRA weights to Diffusers format, which will later be converted to Nunchaku format.

    Parameters
    ----------
    input_lora : str or dict[str, torch.Tensor]
        Path to a safetensors file or a LoRA weight dictionary.
    output_path : str, optional
        If given, save the converted weights to this path.

    Returns
    -------
    dict[str, torch.Tensor]
        LoRA weights in Diffusers format.
    """
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = {k: v for k, v in input_lora.items()}

    # Check if this is ComfyUI format (lora_unet_*) and convert to Diffusers format
    if any(k.startswith("lora_unet_") for k in tensors.keys()):
        logger.info("Converting ComfyUI format to Diffusers format")
        tensors = convert_comfyui_to_nunchaku_sdxl_keys(tensors)
        
        # Convert to PEFT format for Nunchaku
        new_tensors, alphas = LoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)
        
        # Add base_model.model. prefix to all keys for Nunchaku format
        # Also convert lora.down/lora.up to lora_A/lora_B
        peft_tensors = {}
        for key, value in new_tensors.items():
            # Convert lora.down/lora.up to lora_A/lora_B
            key = key.replace(".lora.down.", ".lora_A.")
            key = key.replace(".lora.up.", ".lora_B.")
            
            if not key.startswith("base_model.model."):
                peft_tensors[f"base_model.model.{key}"] = value
            else:
                peft_tensors[key] = value
        
        # Apply alpha values if present
        if alphas is not None and len(alphas) > 0:
            for k, v in alphas.items():
                key_A = k.replace(".alpha", ".lora_A.weight")
                key_B = k.replace(".alpha", ".lora_B.weight")
                # Add base_model.model. prefix for alpha keys
                if not key_A.startswith("base_model.model."):
                    key_A = f"base_model.model.{key_A}"
                if not key_B.startswith("base_model.model."):
                    key_B = f"base_model.model.{key_B}"
                    
                if key_A in peft_tensors and key_B in peft_tensors:
                    rank = peft_tensors[key_A].shape[0]
                    if peft_tensors[key_B].shape[1] == rank:
                        peft_tensors[key_A] = peft_tensors[key_A] * v / rank
        
        # Instead of pre-computing updates, keep the original lora_A and lora_B weights
        # This allows for proper application in ComfySDXLWrapper
        tensors = peft_tensors
    else:
        # Handle other formats (Kohya, PEFT, etc.)
        tensors = handle_kohya_lora(tensors)
        
        # Convert FP8 tensors to BF16
        for k, v in tensors.items():
            if v.dtype not in [torch.float64, torch.float32, torch.bfloat16, torch.float16]:
                tensors[k] = v.to(torch.bfloat16)

        # Apply SDXL-specific key conversion for PEFT format
        if any(k.startswith("base_model.model.") for k in tensors.keys()):
            logger.info("Already in PEFT format")
        else:
            # Convert to PEFT format
            new_tensors, alphas = LoraLoaderMixin.lora_state_dict(tensors, return_alphas=True)
            new_tensors = convert_unet_state_dict_to_peft(new_tensors)
            
            # Apply alpha values if present
            if alphas is not None and len(alphas) > 0:
                for k, v in alphas.items():
                    key_A = k.replace(".alpha", ".lora_A.weight")
                    key_B = k.replace(".alpha", ".lora_B.weight")
                    if key_A in new_tensors and key_B in new_tensors:
                        rank = new_tensors[key_A].shape[0]
                        if new_tensors[key_B].shape[1] == rank:
                            new_tensors[key_A] = new_tensors[key_A] * v / rank
            
            tensors = new_tensors

    if output_path is not None:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(tensors, output_path)

    return tensors