#!/bin/bash
# Description: Publishing the release on ComfyUI website.


# 任何命令失败后立即退出
set -e


# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"



# 项目名称, 一般不需要调整
App=comfyui_silentrain
# ComfyUI 节点目录
nodesDir=${root}/nodes


# switch venv
echo "switch venv ..."
source .venv/bin/activate


# build
./scripts/build.sh


cd ${nodesDir}

# publishing
uv run comfy node publish
