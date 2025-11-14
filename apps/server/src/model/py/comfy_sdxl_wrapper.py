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


class ComfySDXLWrapper(nn.Module):
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
        super(ComfySDXLWrapper, self).__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.loras = []  # List of (lora_path, lora_strength) tuples
        self.lora_state_dict = None  # Store the converted LoRA state dict
        self.lora_strength = 0.0  # Store the LoRA strength

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

        # Apply LoRA if available
        if hasattr(self, 'lora_state_dict') and self.lora_state_dict is not None and abs(self.lora_strength) > 1e-5:
            # Apply LoRA weights to the model
            # This implementation follows the standard LoRA application pattern
            try:
                # Store original weights if not already stored
                if not hasattr(self, '_original_weights'):
                    self._original_weights = {}
                
                # Group LoRA weights by module
                lora_groups = {}
                for key, value in self.lora_state_dict.items():
                    # Extract the module path and parameter name from the key
                    # For SDXL PEFT format, keys are in format like "base_model.model.unet.down_blocks.0.resnets.0.conv1.lora_A.weight"
                    if key.startswith("base_model.model.unet."):
                        # Check if this is a lora_A or lora_B weight
                        if ".lora_A.weight" in key:
                            # Extract module path before .lora_A.weight
                            param_name = "lora_A.weight"
                            module_path = key.replace("base_model.model.unet.", "").replace(".lora_A.weight", "")
                        elif ".lora_B.weight" in key:
                            # Extract module path before .lora_B.weight
                            param_name = "lora_B.weight"
                            module_path = key.replace("base_model.model.unet.", "").replace(".lora_B.weight", "")
                        else:
                            # For other parameters (like alpha), just use the full path
                            param_name = key.split(".")[-1]
                            module_path = key.replace("base_model.model.unet.", "").replace(f".{param_name}", "")
                        
                        # Group by module path
                        if module_path not in lora_groups:
                            lora_groups[module_path] = {}
                        lora_groups[module_path][param_name] = value
                    elif key.startswith("unet."):
                        # Handle keys that start with "unet." (Diffusers format)
                        # Check if this is a lora_A or lora_B weight
                        if ".lora_A.weight" in key:
                            # Extract module path before .lora_A.weight
                            param_name = "lora_A.weight"
                            module_path = key.replace("unet.", "").replace(".lora_A.weight", "")
                        elif ".lora_B.weight" in key:
                            # Extract module path before .lora_B.weight
                            param_name = "lora_B.weight"
                            module_path = key.replace("unet.", "").replace(".lora_B.weight", "")
                        else:
                            # For other parameters (like alpha), just use the full path
                            param_name = key.split(".")[-1]
                            module_path = key.replace("unet.", "").replace(f".{param_name}", "")
                        
                        # Group by module path
                        if module_path not in lora_groups:
                            lora_groups[module_path] = {}
                        lora_groups[module_path][param_name] = value
                    else:
                        # Try to handle other formats
                        logger.warning(f"Unexpected LoRA key format: {key}")
                        continue
                
                logger.info(f"Applying LoRA to {len(lora_groups)} modules with strength {self.lora_strength}")
                
                # Apply LoRA weights to each module
                for module_path, lora_weights in lora_groups.items():
                    try:
                        # Skip modules that don't exist in the model
                        # Nunchaku models might have a different structure than standard UNet
                        if not self._module_exists(self.model, module_path):
                            logger.debug(f"Module path {module_path} not found in Nunchaku model, skipping")
                            continue
                        
                        # Get the module
                        module = self.model
                        for part in module_path.split("."):
                            module = getattr(module, part)
                        
                        # Check if module has a weight attribute
                        if not hasattr(module, 'weight'):
                            logger.warning(f"Module {module_path} has no weight attribute, skipping")
                            continue
                        
                        # Store original weight if not already stored
                        cache_key = f"unet:{module_path}:weight"
                        if cache_key not in self._original_weights:
                            self._original_weights[cache_key] = module.weight.clone()
                        
                        # Apply LoRA if both lora_A and lora_B are present
                        if "lora_A.weight" in lora_weights and "lora_B.weight" in lora_weights:
                            lora_A = lora_weights["lora_A.weight"]
                            lora_B = lora_weights["lora_B.weight"]
                            
                            # Get original weight
                            original_weight = self._original_weights[cache_key]
                            
                            # Apply LoRA: W' = W + (B @ A) * strength
                            # For Conv2D: reshape A and B appropriately
                            if len(original_weight.shape) == 4:  # Conv2D
                                # For Conv2D, A is [rank, in_channels, kernel_h, kernel_w]
                                # and B is [out_channels, rank, 1, 1]
                                # We need to compute B @ A efficiently
                                in_channels = original_weight.shape[1]
                                kernel_h, kernel_w = original_weight.shape[2], original_weight.shape[3]
                                out_channels = original_weight.shape[0]
                                rank = lora_A.shape[0]
                                
                                # Reshape A to [rank, in_channels * kernel_h * kernel_w]
                                lora_A_reshaped = lora_A.view(rank, in_channels * kernel_h * kernel_w)
                                
                                # Reshape B to [out_channels, rank]
                                lora_B_reshaped = lora_B.view(out_channels, rank)
                                
                                # Compute LoRA update: B @ A
                                lora_update_flat = torch.matmul(lora_B_reshaped, lora_A_reshaped)
                                # Reshape to the original Conv2D weight shape
                                lora_update = lora_update_flat.view(out_channels, in_channels, kernel_h, kernel_w)
                            else:  # Linear
                                # For Linear, A is [in_features, rank] and B is [rank, out_features]
                                # We need to compute B @ A
                                in_features, rank_A = lora_A.shape
                                rank_B, out_features = lora_B.shape
                                
                                # Check if ranks match
                                if rank_A != rank_B:
                                    # If ranks don't match, try the other way around
                                    # A might be [rank, in_features] and B might be [out_features, rank]
                                    try:
                                        lora_update = torch.matmul(lora_B, lora_A)
                                    except:
                                        # If that doesn't work either, skip this LoRA
                                        logger.warning(f"Skipping LoRA with incompatible shapes: A={lora_A.shape}, B={lora_B.shape}")
                                        continue
                                else:
                                    # Compute LoRA update: B @ A
                                    lora_update = torch.matmul(lora_B, lora_A)
                            
                            # Apply LoRA with strength
                            modified_weight = original_weight + lora_update * self.lora_strength
                            module.weight.data = modified_weight
                            logger.debug(f"Applied LoRA to module {module_path} with strength {self.lora_strength}")
                        else:
                            logger.warning(f"Missing lora_A.weight or lora_B.weight for module {module_path}")
                    except AttributeError as e:
                        logger.warning(f"Module path {module_path} not found in model: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error applying LoRA to module {module_path}: {e}")
                        continue
            except Exception as e:
                logger.error(f"Failed to apply LoRA: {e}")
                import traceback
                traceback.print_exc()
            except Exception as e:
                logger.warning(f"Failed to apply LoRA: {e}")

        # Call the underlying Nunchaku model
        # The NunchakuSDXLUNet2DConditionModel follows the diffusers interface
        # Note: We don't pass pooled_projections as it's not supported by this model
        # Instead, pooled embeddings are handled through added_cond_kwargs
        
        # Note: LoRA weights are already applied in the code above
        
        output = self.model(
            sample=x,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )

        # Return the sample from the output
        return output.sample

    def _module_exists(self, model, module_path):
        """
        Check if a module path exists in the model.
        
        Parameters
        ----------
        model : torch.nn.Module
            The model to check.
        module_path : str
            The module path to check (e.g., "down_blocks.0.resnets.0.conv1").
            
        Returns
        -------
        bool
            True if the module path exists, False otherwise.
        """
        try:
            module = model
            for part in module_path.split("."):
                module = getattr(module, part)
            return True
        except AttributeError:
            return False

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
        
        # Clear LoRA state dict and strength
        self.lora_state_dict = None
        self.lora_strength = 0.0
        
        # Clear applied LoRA cache
        if hasattr(self, "_applied_loras"):
            self._applied_loras.clear()
