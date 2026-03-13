#!/bin/bash
# 本脚本用于将测试环境, 可以快速将调试的包导入到本地的ComfyUI自定义节点目录下, 用于快速验证节点.

# 任何命令失败后立即退出
set -e


# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"


# 项目名称, 一般不需要调整
App=comfyui_silentrain
# ComfyUI 部署路径
ComfyUIDir=/data/ComfyUI/custom_nodes
# ComfyUI 节点目录
nodesDir=${root}/nodes
# 编译目录
buildDir=./target/wheels


# build
# ./scripts/build.sh
./scripts/build_vulkan.sh
./scripts/build_web.sh


# 将 ${App} 拷贝到 ComfyUI 的自定义节点目录下
# 删除之前解压目录, 避免出现残留文件.
rm -rf ${ComfyUIDir}/${App}


if [ ! -d ${ComfyUIDir}/${App} ]; then 
    echo "mkdir ${ComfyUIDir}/${App}"
    mkdir ${ComfyUIDir}/${App}
fi


# 移动到nodes目录.
echo "copy ${App} to ${ComfyUIDir} ..."
\cp -r ${nodesDir}/${App} ${ComfyUIDir}/${App}/
if [ -d ${nodesDir}/${App}.libs ]; then 
    \cp -r ${nodesDir}/${App}.libs ${ComfyUIDir}/${App}/
fi
\cp -r ${nodesDir}/${App}-*.dist-info ${ComfyUIDir}/${App}/
\cp -r ${nodesDir}/__init__.py ${ComfyUIDir}/${App}/
\cp -r ${nodesDir}/locales ${ComfyUIDir}/${App}/
\cp -r ${nodesDir}/web ${ComfyUIDir}/${App}/
echo -e "\n"


# 检查是否安装成功
echo "ls comfyui custom nodes ..."
ls -hl ${ComfyUIDir}/${App}

echo -e "\n"

echo "tree comfyui custom nodes ..."
tree ${ComfyUIDir}/${App}


echo -e "\nDevelop Done"
