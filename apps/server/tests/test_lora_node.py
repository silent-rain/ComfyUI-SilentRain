#!/usr/bin/env python3
"""
Test script to verify the NunchakuSDXLLoraLoader node.
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

def test_lora_node():
    """Test the NunchakuSDXLLoraLoader node."""
    print("Testing NunchakuSDXLLoraLoader node...")
    
    # Create a mock model object
    class MockModel:
        def __init__(self):
            self.model = MockDiffusionModel()
    
    class MockDiffusionModel:
        def __init__(self):
            self.diffusion_model = MockWrapper()
    
    class MockWrapper:
        def __init__(self):
            self.model = None
            self.loras = []
            self.lora_state_dict = None
            self.lora_strength = 0.0
            self.__class__.__name__ = "ComfySDXLWrapper"
    
    # Create the LoRA loader
    lora_loader = NunchakuSDXLLoraLoader()
    
    # Create a mock model
    model = MockModel()
    
    # Test with the Hyper-SDXL LoRA
    lora_name = "HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors"
    lora_strength = 1.0
    
    print(f"Loading LoRA: {lora_name} with strength: {lora_strength}")
    
    try:
        # Call the load_lora method
        result_model = lora_loader.load_lora(model, lora_name, lora_strength)
        
        # Check the result
        if result_model and len(result_model) == 1:
            result_model = result_model[0]
            wrapper = result_model.model.diffusion_model
            
            print(f"✓ LoRA loaded successfully")
            print(f"  - Number of LoRAs: {len(wrapper.loras)}")
            print(f"  - LoRA strength: {wrapper.lora_strength}")
            print(f"  - LoRA state dict keys: {len(wrapper.lora_state_dict) if wrapper.lora_state_dict else 0}")
            
            if wrapper.lora_state_dict:
                # Check a few keys
                sample_keys = list(wrapper.lora_state_dict.keys())[:3]
                print(f"  - Sample keys: {sample_keys}")
            
            return True
        else:
            print("✗ LoRA loading failed: Invalid result")
            return False
    except Exception as e:
        print(f"✗ LoRA loading failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_lora_node()
    if success:
        print("\n✓ NunchakuSDXLLoraLoader node test passed!")
    else:
        print("\n✗ NunchakuSDXLLoraLoader node test failed!")
        sys.exit(1)