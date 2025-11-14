#!/usr/bin/env python3
"""
Debug script to examine LoRA weight shapes.
"""

import sys
import os
sys.path.append('/home/one/code/ComfyUI-SilentRain/apps/server/src/model/py')

# Set up environment
os.environ["LOG_LEVEL"] = "INFO"

# Import our modules
from safetensors import safe_open
import torch

def main():
    print("Examining LoRA weight shapes...")
    
    # Load the LoRA file
    lora_path = '/home/one/code/ComfyUI/models/loras/HyperSD/SDXL/Hyper-SDXL-8steps-lora.safetensors'
    
    with safe_open(lora_path, framework='pt', device='cpu') as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    
    print(f"Original LoRA has {len(state_dict)} keys")
    
    # Find a Conv2D LoRA pair
    conv_lora_keys = [k for k in state_dict.keys() if 'lora_down' in k and len(state_dict[k].shape) == 4]
    
    if conv_lora_keys:
        # Get the first Conv2D LoRA
        down_key = conv_lora_keys[0]
        up_key = down_key.replace('lora_down', 'lora_up')
        
        print(f"\nExamining Conv2D LoRA pair:")
        print(f"Down key: {down_key}")
        print(f"Up key: {up_key}")
        
        down_weight = state_dict[down_key]
        up_weight = state_dict[up_key]
        
        print(f"Down weight shape: {down_weight.shape}")
        print(f"Up weight shape: {up_weight.shape}")
        
        # Check if these are Conv2D weights
        if len(down_weight.shape) == 4 and len(up_weight.shape) == 4:
            print("These are Conv2D LoRA weights")
            
            # For Conv2D LoRA, the typical shapes are:
            # down: [rank, in_channels, kernel_h, kernel_w]
            # up: [out_channels, rank, 1, 1]
            
            rank = down_weight.shape[0]
            in_channels = down_weight.shape[1]
            kernel_h, kernel_w = down_weight.shape[2], down_weight.shape[3]
            out_channels = up_weight.shape[0]
            
            print(f"Rank: {rank}")
            print(f"In channels: {in_channels}")
            print(f"Kernel size: {kernel_h}x{kernel_w}")
            print(f"Out channels: {out_channels}")
            
            # Try to compute the LoRA update
            print("\nTrying to compute LoRA update...")
            
            # Reshape down weight to [rank, in_channels * kernel_h * kernel_w]
            down_reshaped = down_weight.view(rank, in_channels * kernel_h * kernel_w)
            print(f"Down reshaped: {down_reshaped.shape}")
            
            # Reshape up weight to [out_channels, rank]
            up_reshaped = up_weight.view(out_channels, rank)
            print(f"Up reshaped: {up_reshaped.shape}")
            
            # Compute LoRA update: up @ down
            try:
                lora_update_flat = torch.matmul(up_reshaped, down_reshaped)
                print(f"LoRA update flat shape: {lora_update_flat.shape}")
                
                # Reshape to [out_channels, in_channels, kernel_h, kernel_w]
                lora_update = lora_update_flat.view(out_channels, in_channels, kernel_h, kernel_w)
                print(f"LoRA update shape: {lora_update.shape}")
                print("✓ Successfully computed LoRA update")
            except Exception as e:
                print(f"✗ Error computing LoRA update: {e}")
        else:
            print("These are not Conv2D LoRA weights")
    else:
        print("No Conv2D LoRA found")
    
    # Find a Linear LoRA pair
    linear_lora_keys = [k for k in state_dict.keys() if 'lora_down' in k and len(state_dict[k].shape) == 2]
    
    if linear_lora_keys:
        # Get the first Linear LoRA
        down_key = linear_lora_keys[0]
        up_key = down_key.replace('lora_down', 'lora_up')
        
        print(f"\nExamining Linear LoRA pair:")
        print(f"Down key: {down_key}")
        print(f"Up key: {up_key}")
        
        down_weight = state_dict[down_key]
        up_weight = state_dict[up_key]
        
        print(f"Down weight shape: {down_weight.shape}")
        print(f"Up weight shape: {up_weight.shape}")
        
        # For Linear LoRA, the typical shapes are:
        # down: [in_features, rank]
        # up: [rank, out_features]
        
        print(f"Down weight shape: {down_weight.shape}")
        print(f"Up weight shape: {up_weight.shape}")
        
        # Check actual shapes
        in_features, rank = down_weight.shape
        rank_up, out_features = up_weight.shape
        
        print(f"Rank (from down): {rank}")
        print(f"Rank (from up): {rank_up}")
        print(f"In features: {in_features}")
        print(f"Out features: {out_features}")
        
        # Try to compute the LoRA update
        print("\nTrying to compute LoRA update...")
        
        try:
            # Compute LoRA update: up @ down
            # Option 1: up @ down
            lora_update1 = torch.matmul(up_weight, down_weight)
            print(f"LoRA update (up @ down) shape: {lora_update1.shape}")
            
            # Option 2: up.t() @ down.t()
            lora_update2 = torch.matmul(up_weight.t(), down_weight.t())
            print(f"LoRA update (up.t() @ down.t()) shape: {lora_update2.shape}")
            
            # Option 3: down @ up
            lora_update3 = torch.matmul(down_weight, up_weight)
            print(f"LoRA update (down @ up) shape: {lora_update3.shape}")
            
            # Option 4: down.t() @ up.t()
            lora_update4 = torch.matmul(down_weight.t(), up_weight.t())
            print(f"LoRA update (down.t() @ up.t()) shape: {lora_update4.shape}")
            
            print("✓ Successfully computed LoRA update")
        except Exception as e:
            print(f"✗ Error computing LoRA update: {e}")
    else:
        print("No Linear LoRA found")

if __name__ == "__main__":
    main()