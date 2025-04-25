#!/bin/bash
# 本脚本用于快速启动comfyui服务

# ComfyUI 部署路径
ComfyUI=/home/one/code/ComfyUI

# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"

# switch ComfyUI dir
cd ${ComfyUI}

# switch venv
echo "switch venv ..."
source .venv/bin/activate

# run server
python main.py --cpu --cpu-va
