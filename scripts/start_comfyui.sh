#!/bin/bash
# 本脚本用于快速启动comfyui服务

# ComfyUI 部署路径
ComfyUI=~/code/ComfyUI

# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"

# switch ComfyUI dir
cd ${ComfyUI}

# switch venv
echo "switch venv ..."
source .venv/bin/activate

# 更新 comfyui-frontend-package
# pip install --upgrade comfyui-frontend-package

# run server
python main.py --cpu --cpu-va
