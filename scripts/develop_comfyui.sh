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
ComfyUIDir=~/code/ComfyUI/custom_nodes
# ComfyUI 节点目录
nodesDir=${root}/nodes
# 编译目录
buildDir=./target/wheels

# nodes 目录
if [ ! -d ${nodesDir} ]; then 
    echo "mkdir ${nodesDir}"
    mkdir ${nodesDir}
fi


# build
./scripts/build.sh


# 获取whl文件路径
echo "find whl file ..."
whl=$(find ${buildDir} -name "${App}-*.whl" | head -n 1)


# 解压Python包
echo "extract whl file ..."
unzip -o ${whl} -d ${nodesDir}
echo -e "\n"

# 将 ${App} 拷贝到 ComfyUI 的自定义节点目录下
# 删除之前解压目录, 避免出现残留文件.
rm -rf ${ComfyUIDir}/${App}


if [ ! -d ${ComfyUIDir}/${App} ]; then 
    echo "mkdir ${ComfyUIDir}/${App}"
    mkdir ${ComfyUIDir}/${App}
fi


# 移动到nodes目录.
echo "mv ${App} to ${ComfyUIDir} ..."
\mv -f ${nodesDir}/${App} ${ComfyUIDir}/${App}/
if [ -d ${nodesDir}/${App} ]; then 
    \mv -f ${nodesDir}/${App}.libs ${ComfyUIDir}/${App}/
fi
\mv -f ${nodesDir}/${App}-*.dist-info ${ComfyUIDir}/${App}/
echo "from .${App} import *" > ${ComfyUIDir}/${App}/__init__.py
echo -e "\n"


# 检查是否安装成功
echo "ls comfyui custom nodes ..."
ls -hl ${ComfyUIDir}/${App}

echo -e "\n"

echo "tree comfyui custom nodes ..."
tree ${ComfyUIDir}/${App}


echo -e "\nDevelop Done"
