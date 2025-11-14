#!/usr/bin/env python3
"""
Test script to verify LoRA application to a model.
"""

import sys
import os
sys.path.append('/home/one/code/ComfyUI-SilentRain/apps/server/src/model/py')

# Set up environment
os.environ["LOG_LEVEL"] = "INFO"

# Import our modules
from nunchaku_sdxl_lora_converter import to_diffusers
from safetensors import safe_open
import torch

def main():
    print("Testing LoRA application to model...")
    
    # Load the LoRA file
    lora_path = '/home/one/code/ComfyUI/models/loras/HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors'
    
    with safe_open(lora_path, framework='pt', device='cpu') as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    
    print(f"Original LoRA has {len(state_dict)} keys")
    
    # Convert LoRA to Nunchaku format
    converted_lora = to_diffusers(state_dict)
    
    print(f"Converted LoRA has {len(converted_lora)} keys")
    
    # Check if keys have the expected prefix
    if any(key.startswith('base_model.model.unet.') for key in converted_lora.keys()):
        print("\n✓ Successfully converted to Nunchaku format (base_model.model.unet.*)")
    else:
        print("\n✗ Failed to convert to Nunchaku format")
        return False
    
    # Create a mock model state dict with the same structure as the LoRA
    print("\nCreating mock model state dict...")
    model_state_dict = {}
    
    # Get a sample of LoRA keys to understand the structure
    sample_keys = list(converted_lora.keys())[:5]
    print(f"Sample LoRA keys: {sample_keys}")
    
    # Create mock weights for each LoRA key
    for key in converted_lora.keys():
        # Extract the base module path from the LoRA key
        # For example, "base_model.model.unet.down.blocks.0.resnets.0.conv1" -> "base_model.model.unet.down.blocks.0.resnets.0.conv1"
        base_key = key
        
        # Create a mock weight tensor with the same shape as the LoRA update
        lora_update = converted_lora[key]
        mock_weight = torch.zeros_like(lora_update)
        
        # Add the mock weight to the model state dict
        model_state_dict[base_key] = mock_weight
    
    print(f"Created mock model with {len(model_state_dict)} weights")
    
    # Apply LoRA weights to the model state dict
    print("\nApplying LoRA weights to model...")
    strength = 1.0
    
    for key, lora_update in converted_lora.items():
        if key in model_state_dict:
            # Apply LoRA: W' = W + lora_update * strength
            model_state_dict[key] = model_state_dict[key] + lora_update * strength
        else:
            print(f"Warning: LoRA key {key} not found in model state dict")
    
    print(f"Applied LoRA to {len(converted_lora)} weights")
    
    # Verify that the weights have been updated
    print("\nVerifying weight updates...")
    updated_count = 0
    for key, weight in model_state_dict.items():
        if torch.any(weight != 0):  # Check if any element is non-zero
            updated_count += 1
    
    print(f"Updated {updated_count} out of {len(model_state_dict)} weights")
    
    if updated_count > 0:
        print("\n✓ LoRA application test passed!")
        return True
    else:
        print("\n✗ LoRA application test failed!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)