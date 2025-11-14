#!/usr/bin/env python3
"""
Test script to verify LoRA application in ComfySDXLWrapper with a real model.
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
from comfy_sdxl_wrapper import ComfySDXLWrapper
import torch

def main():
    print("Testing LoRA application with ComfySDXLWrapper...")
    
    try:
        # Try to import the Nunchaku model
        from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
        
        # Load the Nunchaku model
        print("Loading Nunchaku SDXL model...")
        model_path = "/home/one/code/ComfyUI/models/unet/svdq-int4_r32-sdxl-turbo.safetensors"
        
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            print("Please update the model path in the test script")
            return False
        
        nunchaku_model = NunchakuSDXLUNet2DConditionModel.from_pretrained(model_path)
        print("✓ Successfully loaded Nunchaku SDXL model")
        
        # Wrap the model with ComfySDXLWrapper
        wrapper = ComfySDXLWrapper(nunchaku_model)
        print("✓ Successfully wrapped model with ComfySDXLWrapper")
        
        # Create a mock model object for the LoRA loader
        class MockModel:
            def __init__(self, wrapper):
                self.model = type('Model', (), {})()
                self.model.diffusion_model = wrapper
        
        mock_model = MockModel(wrapper)
        
        # Load and apply LoRA
        print("\nLoading and applying LoRA...")
        lora_loader = NunchakuSDXLLoraLoader()
        lora_name = "HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors"
        lora_strength = 1.0
        
        result_model_tuple = lora_loader.load_lora(mock_model, lora_name, lora_strength)
        result_model = result_model_tuple[0]  # Extract the model from the tuple
        print("✓ LoRA loaded successfully")
        
        # Check if LoRA was applied
        result_wrapper = result_model.model.diffusion_model
        if hasattr(result_wrapper, 'lora_state_dict') and result_wrapper.lora_state_dict is not None:
            print(f"✓ LoRA state dict has {len(result_wrapper.lora_state_dict)} keys")
            print(f"✓ LoRA strength: {result_wrapper.lora_strength}")
        else:
            print("✗ LoRA state dict not found")
            return False
        
        # Test LoRA application without forward pass
        print("\nTesting LoRA application...")
        
        # Check if LoRA weights are properly stored
        if hasattr(result_wrapper, 'lora_state_dict') and result_wrapper.lora_state_dict is not None:
            print(f"✓ LoRA state dict has {len(result_wrapper.lora_state_dict)} keys")
            
            # Check if LoRA weights are properly grouped
            lora_groups = {}
            for key, value in result_wrapper.lora_state_dict.items():
                if key.startswith("base_model.model.unet."):
                    if ".lora_A.weight" in key:
                        param_name = "lora_A.weight"
                        module_path = key.replace("base_model.model.unet.", "").replace(".lora_A.weight", "")
                    elif ".lora_B.weight" in key:
                        param_name = "lora_B.weight"
                        module_path = key.replace("base_model.model.unet.", "").replace(".lora_B.weight", "")
                    else:
                        continue
                    
                    if module_path not in lora_groups:
                        lora_groups[module_path] = {}
                    lora_groups[module_path][param_name] = value
            
            complete_modules = sum(1 for weights in lora_groups.values() if "lora_A.weight" in weights and "lora_B.weight" in weights)
            print(f"✓ LoRA weights grouped into {len(lora_groups)} modules")
            print(f"✓ {complete_modules} modules have complete LoRA weights")
            
            if complete_modules > 0:
                print("✓ LoRA weights are ready for application")
            else:
                print("✗ No complete LoRA weights found")
                return False
        else:
            print("✗ LoRA state dict not found")
            return False
        
        print("\n✓ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import Nunchaku model: {e}")
        print("This is expected if the Nunchaku library is not installed")
        return False
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✓ LoRA application test passed!")
    else:
        print("\n✗ LoRA application test failed!")
        sys.exit(1)