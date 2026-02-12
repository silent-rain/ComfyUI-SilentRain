#!/bin/bash

# 任何命令失败后立即退出
set -e

# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"


# 项目名称, 一般不需要调整
App=comfyui_silentrain
# 编译目录
buildDir=${root}/target/wheels
# 节点目录
nodesDir=${root}/nodes
# Python Version
PythonVersion=3.12


cd apps/server


# 检查虚拟环境
echo "check venv ..."
if [ ! -d .venv ]; then
    echo "create venv ..."
    # 安装
    pipx install uv

    # 安装Python版本
    uv python install ${PythonVersion}

    # 同步虚拟环境
    uv sync
fi


# switch venv
echo "switch venv ..."
source /data/ComfyUI/.venv/bin/activate
# source .venv/bin/activate

# GUP 编译参数
export NVCC_FLAGS="-D__CORRECT_ISO_CPP_MATH_H_PROTO"

# build
echo "build release whl ..."
uv tool run maturin build -r --features vulkan


# del old dir
if [ -d ${nodesDir}/comfyui_silentrain.libs ]; then 
    rm -rf ${nodesDir}/comfyui_silentrain.libs/*
fi


# 获取whl文件路径
echo "find whl file ..."
whl=$(find ${buildDir} -name "${App}-*.whl" | head -n 1)
echo "whl file: ${whl}"

# 覆盖解压whl包
echo "extract whl file ..."
unzip -o ${whl} -d ${nodesDir}
echo -e "\n"


tree -sh ${nodesDir}

echo -e "\n"
echo "total size:"
du -sh ${nodesDir}

echo -e "\nBuild Done"

echo -e "\n"
