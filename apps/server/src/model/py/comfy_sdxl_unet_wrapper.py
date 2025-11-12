"""
Nunchaku SDXL UNet Wrapper for ComfyUI

This module provides a wrapper for the Nunchaku SDXL UNet model to support ComfyUI workflows.
"""

import logging
import os

import torch
from torch import nn

from nunchaku.utils import load_state_dict_in_safetensors


# Get log level from environment variable (default to INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
