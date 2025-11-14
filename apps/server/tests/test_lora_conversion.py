#!/usr/bin/env python3
"""
Test script to verify LoRA loading and conversion for Nunchaku SDXL.
"""

import sys
import os
sys.path.append('/home/one/code/ComfyUI-SilentRain/apps/server/src/model/py')

# Set up environment
os.environ["LOG_LEVEL"] = "INFO"

# Import our modules
from nunchaku_sdxl_lora_converter import to_diffusers, convert_comfyui_to_nunchaku_sdxl_keys
from safetensors import safe_open

def test_lora_conversion():
    """Test LoRA conversion from ComfyUI format to PEFT format."""
    # Path to the Hyper-SDXL LoRA
    lora_path = '/home/one/code/ComfyUI/models/loras/HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors'
    
    print(f"Testing LoRA conversion with: {lora_path}")
    
    # Load the LoRA
    with safe_open(lora_path, framework='pt', device='cpu') as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}
    
    print(f"Original LoRA has {len(state_dict)} keys")
    
    # Check format
    if any(key.startswith('lora_unet_') for key in state_dict.keys()):
        print("✓ Detected ComfyUI format LoRA")
    else:
        print("✗ Not ComfyUI format LoRA")
        return False
    
    # Convert to PEFT format
    converted_lora = to_diffusers(state_dict)
    
    print(f"Converted LoRA has {len(converted_lora)} keys")
    
    # Check if conversion was successful
    if any(key.startswith('base_model.model.unet.') for key in converted_lora.keys()):
        print("✓ Successfully converted to PEFT format")
    elif any(key.startswith('unet.') for key in converted_lora.keys()):
        print("✓ Converted to Diffusers format (unet.*)")
    else:
        print("✗ Failed to convert to expected format")
        return False
    
    # Print some sample keys
    print("\nSample original keys:")
    for i, key in enumerate(list(state_dict.keys())[:3]):
        print(f"  {i+1}. {key}")
    
    print("\nSample converted keys:")
    for i, key in enumerate(list(converted_lora.keys())[:3]):
        print(f"  {i+1}. {key}")
    
    # Check key prefixes
    prefixes = set()
    for key in converted_lora.keys():
        if '.' in key:
            prefix = key.split('.')[0]
            prefixes.add(prefix)
    
    print(f"\nKey prefixes in converted LoRA: {prefixes}")
    
    return True

if __name__ == "__main__":
    success = test_lora_conversion()
    if success:
        print("\n✓ LoRA conversion test passed!")
    else:
        print("\n✗ LoRA conversion test failed!")
        sys.exit(1)