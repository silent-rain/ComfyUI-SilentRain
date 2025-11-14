#!/usr/bin/env python3
"""
Test script to verify LoRA loading and application for Nunchaku SDXL.
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

def test_lora_loading():
    """Test LoRA loading and conversion."""
    # Path to the Hyper-SDXL LoRA
    lora_path = '/home/one/code/ComfyUI/models/loras/HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors'
    
    print(f"Testing LoRA loading with: {lora_path}")
    
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
        key_prefix = "base_model.model.unet."
    elif any(key.startswith('unet.') for key in converted_lora.keys()):
        print("✓ Converted to Diffusers format (unet.*)")
        key_prefix = "unet."
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
    
    # Test LoRA application logic
    print("\nTesting LoRA application logic...")
    
    # Group LoRA weights by module (similar to ComfySDXLWrapper)
    lora_groups = {}
    for key, value in converted_lora.items():
        if key.startswith(key_prefix):
            # Remove prefix
            module_path = key[len(key_prefix):]
            
            # Check if this is a lora_A or lora_B weight
            if ".lora_A.weight" in key:
                # Extract module path before .lora_A.weight
                param_name = "lora_A.weight"
                module_path = key.replace(key_prefix, "").replace(".lora_A.weight", "")
            elif ".lora_B.weight" in key:
                # Extract module path before .lora_B.weight
                param_name = "lora_B.weight"
                module_path = key.replace(key_prefix, "").replace(".lora_B.weight", "")
            else:
                # For other parameters (like alpha), just use the full path
                param_name = key.split(".")[-1]
                module_path = key.replace(key_prefix, "").replace(f".{param_name}", "")
            
            # Group by module path
            if module_path not in lora_groups:
                lora_groups[module_path] = {}
            lora_groups[module_path][param_name] = value
    
    print(f"Grouped LoRA weights into {len(lora_groups)} modules")
    
    # Check a few modules
    sample_modules = list(lora_groups.keys())[:3]
    for module_path in sample_modules:
        lora_weights = lora_groups[module_path]
        has_lora_A = "lora_A.weight" in lora_weights
        has_lora_B = "lora_B.weight" in lora_weights
        print(f"  Module {module_path}: lora_A={has_lora_A}, lora_B={has_lora_B}")
        
        if has_lora_A and has_lora_B:
            lora_A = lora_weights["lora_A.weight"]
            lora_B = lora_weights["lora_B.weight"]
            print(f"    lora_A shape: {lora_A.shape}, lora_B shape: {lora_B.shape}")
    
    return True

if __name__ == "__main__":
    success = test_lora_loading()
    if success:
        print("\n✓ LoRA loading test passed!")
    else:
        print("\n✗ LoRA loading test failed!")
        sys.exit(1)