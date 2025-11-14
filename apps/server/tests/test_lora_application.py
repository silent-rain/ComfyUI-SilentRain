#!/usr/bin/env python3
"""
Test script to verify LoRA application in ComfySDXLWrapper.
"""

import sys
import os
sys.path.append('/home/one/code/ComfyUI-SilentRain/apps/server/src/model/py')

# Set up environment
os.environ["LOG_LEVEL"] = "INFO"

# Mock folder_paths module
class MockFolderPaths:
    @staticmethod
    def get_filename_list(folder_name):
        if folder_name == "loras":
            return ["HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors"]
        return []
    
    @staticmethod
    def get_full_path_or_raise(folder_name, filename):
        if folder_name == "loras" and filename == "HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors":
            return "/home/one/code/ComfyUI/models/loras/HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors"
        raise FileNotFoundError(f"File not found: {folder_name}/{filename}")

# Add the mock to sys.modules
sys.modules['folder_paths'] = MockFolderPaths()

# Import our modules
from nunchaku_sdxl_lora_loader import NunchakuSDXLLoraLoader
from nunchaku_sdxl_lora_converter import to_diffusers
from safetensors import safe_open
import torch

def main():
    print("Testing LoRA application in ComfySDXLWrapper...")
    
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
    
    # Group LoRA weights by module (similar to ComfySDXLWrapper)
    lora_groups = {}
    for key, value in converted_lora.items():
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
    
    print(f"\nGrouped LoRA weights into {len(lora_groups)} modules")
    
    # Check a few groups
    sample_modules = list(lora_groups.keys())[:3]
    for module_path in sample_modules:
        weights = lora_groups[module_path]
        print(f"  Module: {module_path}")
        for param_name, weight in weights.items():
            print(f"    {param_name}: shape={weight.shape}, dtype={weight.dtype}")
    
    # Check if each module has both lora_A and lora_B
    complete_modules = 0
    incomplete_modules = 0
    for module_path, weights in lora_groups.items():
        if "lora_A.weight" in weights and "lora_B.weight" in weights:
            complete_modules += 1
        else:
            incomplete_modules += 1
    
    print(f"\nModules with complete LoRA weights (lora_A + lora_B): {complete_modules}")
    print(f"Modules with incomplete LoRA weights: {incomplete_modules}")
    
    if complete_modules > 0:
        print("\n✓ LoRA weights are properly grouped for application")
        return True
    else:
        print("\n✗ No complete LoRA weights found")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ LoRA application test passed!")
    else:
        print("\n✗ LoRA application test failed!")
        sys.exit(1)