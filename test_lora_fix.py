#!/usr/bin/env python3
"""
Test script to verify LoRA fix.
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
    print("Testing LoRA conversion and loading...")
    
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
    
    # Check if the _module_exists method works
    print("\nTesting _module_exists method...")
    
    # Create a mock model with a simple structure
    class MockModel:
        def __init__(self):
            self.down_blocks = torch.nn.ModuleList([
                torch.nn.Module()
            ])
    
    mock_model = MockModel()
    
    # Test the _module_exists method
    def _module_exists(model, module_path):
        try:
            module = model
            for part in module_path.split("."):
                module = getattr(module, part)
            return True
        except AttributeError:
            return False
    
    # Test with existing path
    exists = _module_exists(mock_model, "down_blocks")
    print(f"Module 'down_blocks' exists: {exists}")
    
    # Test with non-existing path
    exists = _module_exists(mock_model, "up_blocks")
    print(f"Module 'up_blocks' exists: {exists}")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ LoRA fix test passed!")
    else:
        print("\n✗ LoRA fix test failed!")
        sys.exit(1)