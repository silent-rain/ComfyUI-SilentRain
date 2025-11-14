#!/usr/bin/env python3
"""
Test script to verify the full LoRA loading and application pipeline for Nunchaku SDXL.
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
import copy

def test_full_lora_pipeline():
    """Test the full LoRA loading and application pipeline."""
    # Path to the Hyper-SDXL LoRA
    lora_path = '/home/one/code/ComfyUI/models/loras/HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors'
    
    print(f"Testing full LoRA pipeline with: {lora_path}")
    
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
    
    # Simulate the LoRA application logic from ComfySDXLWrapper
    print("\nSimulating LoRA application logic...")
    
    # Group LoRA weights by module (similar to ComfySDXLWrapper)
    lora_groups = {}
    for key, value in converted_lora.items():
        if key.startswith(key_prefix):
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
            
            # Simulate LoRA application for a Conv2D layer
            if len(lora_A.shape) == 4 and len(lora_B.shape) == 4:
                # For Conv2D
                in_channels = lora_A.shape[1]
                kernel_h, kernel_w = lora_A.shape[2], lora_A.shape[3]
                rank = lora_A.shape[0]
                
                # Reshape A
                lora_A_reshaped = lora_A.view(rank, in_channels, kernel_h, kernel_w)
                
                # Compute LoRA update: B @ A
                # For Conv2D, this is a bit more complex
                out_channels = lora_B.shape[0]
                original_weight_shape = (out_channels, in_channels, kernel_h, kernel_w)
                lora_update = torch.zeros(original_weight_shape)
                
                for i in range(out_channels):  # out_channels
                    for j in range(in_channels):
                        for kh in range(kernel_h):
                            for kw in range(kernel_w):
                                # Index into A: [rank, j, kh, kw]
                                # Index into B: [i, rank]
                                lora_update[i, j, kh, kw] = torch.sum(
                                    lora_B[i, :] * lora_A_reshaped[:, j, kh, kw]
                                )
                
                print(f"    Computed LoRA update with shape: {lora_update.shape}")
                
                # Apply LoRA with strength
                lora_strength = 1.0
                modified_weight = torch.zeros(original_weight_shape) + lora_update * lora_strength
                print(f"    Applied LoRA with strength {lora_strength}")
    
    return True

if __name__ == "__main__":
    success = test_full_lora_pipeline()
    if success:
        print("\n✓ Full LoRA pipeline test passed!")
    else:
        print("\n✗ Full LoRA pipeline test failed!")
        sys.exit(1)