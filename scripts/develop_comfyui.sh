#!/bin/bash
# 本脚本用于将测试环境, 可以快速将调试的包导入到本地的ComfyUI自定义节点目录下, 用于快速验证节点.

# 项目名称, 一般不需要调整
App=comfyui_silentrain
# ComfyUI 部署路径
ComfyUI=~/code/ComfyUI/custom_nodes

# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"


# build
./scripts/build.sh


# 将 ${App} 拷贝到 ${ComfyUI} 的自定义节点目录下
echo "develop ${App} to ComfyUI path ..."
\cp -r ./nodes/${App} ${ComfyUI}/


# 检查是否安装成功
echo "ls comfyui custom nodes ..."
ls -hl ${ComfyUI}/${App}

echo -e "\n"

echo "tree comfyui custom nodes ..."
tree ${ComfyUI}/${App}


echo -e "\nDevelop Done"
