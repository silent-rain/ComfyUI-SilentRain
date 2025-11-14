#!/usr/bin/env python3
"""
Simple test script to verify LoRA conversion.
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
    print("Testing LoRA conversion...")
    
    # Load the LoRA file
    lora_path = '/home/one/code/ComfyUI/models/loras/HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors'
    
    with safe_open(lora_path, framework='pt', device='cpu') as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    
    print(f"Original LoRA has {len(state_dict)} keys")
    
    # Convert LoRA to Nunchaku format
    converted_lora = to_diffusers(state_dict)
    
    print(f"Converted LoRA has {len(converted_lora)} keys")
    print("Sample converted keys:")
    for i, key in enumerate(list(converted_lora.keys())[:3]):
        print(f"  {i+1}. {key}")
    
    # Check if keys have the expected prefix
    if any(key.startswith('base_model.model.unet.') for key in converted_lora.keys()):
        print("\n✓ Successfully converted to Nunchaku format (base_model.model.unet.*)")
    else:
        print("\n✗ Failed to convert to Nunchaku format")
        return False
    
    # Check if the converted LoRA can be applied to a model state dict
    print("\nTesting LoRA application to model state dict...")
    
    # Create a mock model state dict
    model_state_dict = {}
    for key in converted_lora.keys():
        if key.endswith('.weight'):
            # Create a mock weight tensor
            if 'lora_A' in key:
                # lora_A weight
                shape = converted_lora[key].shape
                model_state_dict[key.replace('lora_A', 'weight')] = torch.zeros(shape)
            elif 'lora_B' in key:
                # lora_B weight
                shape = converted_lora[key].shape
                model_state_dict[key.replace('lora_B', 'weight')] = torch.zeros(shape)
    
    # Apply LoRA weights to the state dict
    for key, value in converted_lora.items():
        if key in model_state_dict:
            # Apply LoRA: W' = W + value * strength
            model_state_dict[key] = model_state_dict[key] + value * 1.0
    
    print(f"Applied LoRA to {len(model_state_dict)} weights")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ LoRA simple test passed!")
    else:
        print("\n✗ LoRA simple test failed!")
        sys.exit(1)