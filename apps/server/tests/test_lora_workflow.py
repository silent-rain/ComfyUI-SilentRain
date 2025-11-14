#!/usr/bin/env python3
"""
Test script to simulate a complete ComfyUI workflow with LoRA.
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
    print("Testing complete ComfyUI workflow with LoRA...")
    
    # Step 1: Load LoRA file
    lora_path = '/home/one/code/ComfyUI/models/loras/HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors'
    
    with safe_open(lora_path, framework='pt', device='cpu') as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    
    print(f"Original LoRA has {len(state_dict)} keys")
    
    # Step 2: Convert LoRA to Nunchaku format
    converted_lora = to_diffusers(state_dict)
    
    print(f"Converted LoRA has {len(converted_lora)} keys")
    print("Sample converted keys:")
    for i, key in enumerate(list(converted_lora.keys())[:3]):
        print(f"  {i+1}. {key}")
    
    # Step 3: Check if keys have expected prefix
    if any(key.startswith('base_model.model.unet.') for key in converted_lora.keys()):
        print("\n✓ Successfully converted to Nunchaku format (base_model.model.unet.*)")
    else:
        print("\n✗ Failed to convert to Nunchaku format")
        return False
    
    # Step 4: Create a mock model wrapper
    class MockModel:
        def __init__(self):
            self.loras = []
            self.lora_state_dict = None
            self.lora_strength = 0.0
    
    class MockModelWrapper:
        def __init__(self):
            self.model = MockModel()
            self.loras = []
            self.lora_state_dict = None
            self.lora_strength = 0.0
    
    class MockComfyModel:
        def __init__(self):
            self.model = type('obj', (object,), {'diffusion_model': MockModelWrapper()})()
    
    # Step 5: Test NunchakuSDXLLoraLoader
    lora_loader = NunchakuSDXLLoraLoader()
    mock_model = MockComfyModel()
    
    # Test load_lora method
    result = lora_loader.load_lora(mock_model, "HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors", 1.0)
    
    # Check result
    if result is not None and len(result) == 1:
        print("\n✓ LoRA loader returned a result")
        
        # Check if LoRA was applied
        ret_model = result[0]
        if hasattr(ret_model.model.diffusion_model, 'lora_state_dict'):
            print("✓ LoRA state dict was set")
            
            # Check if LoRA strength was set
            if hasattr(ret_model.model.diffusion_model, 'lora_strength'):
                print("✓ LoRA strength was set")
                
                # Check if LoRA list was updated
                if len(ret_model.model.diffusion_model.loras) > 0:
                    print("✓ LoRA list was updated")
                    
                    # Check LoRA details
                    lora_path, lora_strength = ret_model.model.diffusion_model.loras[0]
                    if lora_path == '/home/one/code/ComfyUI/models/loras/HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors':
                        print("✓ LoRA path is correct")
                    else:
                        print(f"✗ LoRA path is incorrect: {lora_path}")
                    
                    if abs(lora_strength - 1.0) < 1e-5:
                        print("✓ LoRA strength is correct")
                    else:
                        print(f"✗ LoRA strength is incorrect: {lora_strength}")
                else:
                    print("✗ LoRA list was not updated")
            else:
                print("✗ LoRA strength was not set")
        else:
            print("✗ LoRA state dict was not set")
    else:
        print("✗ LoRA loader did not return a valid result")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ Complete LoRA workflow test passed!")
    else:
        print("\n✗ Complete LoRA workflow test failed!")
        sys.exit(1)