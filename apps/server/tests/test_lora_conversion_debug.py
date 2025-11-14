#!/usr/bin/env python3
"""
Debug script to check LoRA conversion in detail.
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
    print("Testing LoRA conversion in detail...")
    
    # Load the LoRA file
    lora_path = '/home/one/code/ComfyUI/models/loras/HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors'
    
    with safe_open(lora_path, framework='pt', device='cpu') as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    
    print(f"Original LoRA has {len(state_dict)} keys")
    print("Sample original keys:")
    for i, key in enumerate(list(state_dict.keys())[:3]):
        print(f"  {i+1}. {key}")
    
    # Step 1: Convert ComfyUI format to Diffusers format
    from nunchaku_sdxl_lora_converter import convert_comfyui_to_nunchaku_sdxl_keys
    diffusers_format = convert_comfyui_to_nunchaku_sdxl_keys(state_dict)
    
    print("\nAfter ComfyUI to Diffusers conversion:")
    print(f"  Keys: {len(diffusers_format)}")
    print("Sample keys:")
    for i, key in enumerate(list(diffusers_format.keys())[:3]):
        print(f"  {i+1}. {key}")
    
    # Step 2: Convert Diffusers format to PEFT format
    from diffusers.loaders.lora_pipeline import LoraLoaderMixin
    from diffusers.utils.state_dict_utils import convert_unet_state_dict_to_peft
    
    peft_format, alphas = LoraLoaderMixin.lora_state_dict(diffusers_format, return_alphas=True)
    peft_format = convert_unet_state_dict_to_peft(peft_format)
    
    print("\nAfter Diffusers to PEFT conversion:")
    print(f"  Keys: {len(peft_format)}")
    print("Sample keys:")
    for i, key in enumerate(list(peft_format.keys())[:3]):
        print(f"  {i+1}. {key}")
    
    # Check if keys have the expected prefix
    if any(key.startswith('base_model.model.unet.') for key in peft_format.keys()):
        print("\n✓ Successfully converted to PEFT format (base_model.model.unet.*)")
    else:
        print("\n✗ Failed to convert to PEFT format")
        print("Key prefixes found:")
        prefixes = set()
        for key in peft_format.keys():
            if '.' in key:
                prefix = key.split('.')[0]
                prefixes.add(prefix)
        print(f"  {prefixes}")
    
    # Step 3: Apply alpha values if present
    if alphas is not None and len(alphas) > 0:
        print(f"\nFound {len(alphas)} alpha values")
        for k, v in list(alphas.items())[:3]:
            print(f"  {k}: {v}")
        
        for k, v in alphas.items():
            key_A = k.replace(".alpha", ".lora_A.weight")
            key_B = k.replace(".alpha", ".lora_B.weight")
            if key_A in peft_format and key_B in peft_format:
                rank = peft_format[key_A].shape[0]
                if peft_format[key_B].shape[1] == rank:
                    peft_format[key_A] = peft_format[key_A] * v / rank
    
    # Final check
    print("\nFinal format after applying alphas:")
    print(f"  Keys: {len(peft_format)}")
    print("Sample keys:")
    for i, key in enumerate(list(peft_format.keys())[:3]):
        print(f"  {i+1}. {key}")
    
    # Compare with to_diffusers function
    print("\nUsing to_diffusers function:")
    converted_lora = to_diffusers(state_dict)
    print(f"  Keys: {len(converted_lora)}")
    print("Sample keys:")
    for i, key in enumerate(list(converted_lora.keys())[:3]):
        print(f"  {i+1}. {key}")
    
    # Check if the results match
    if set(converted_lora.keys()) == set(peft_format.keys()):
        print("\n✓ to_diffusers produces the same result as manual conversion")
    else:
        print("\n✗ to_diffusers produces a different result")
        print("Keys in to_diffusers but not in manual conversion:")
        for key in set(converted_lora.keys()) - set(peft_format.keys()):
            print(f"  {key}")
        print("Keys in manual conversion but not in to_diffusers:")
        for key in set(peft_format.keys()) - set(converted_lora.keys()):
            print(f"  {key}")

if __name__ == "__main__":
    main()