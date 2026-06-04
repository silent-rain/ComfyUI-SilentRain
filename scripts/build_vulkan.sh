#!/bin/bash

# 任何命令失败后立即退出
set -e

# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"


# 项目名称, 一般不需要调整
App=comfyui_silentrain
AppV3=comfyui_silentrain_v3
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

# build v1
echo "build release whl for v1 ..."
uv tool run maturin build -r --features vulkan

# build v3
echo "build release whl for v3 ..."
cd ../server_v3
uv tool run maturin build -r

# 回到 nodes 处理
cd ${root}

# del old dir
if [ -d ${nodesDir}/${App}.libs ]; then 
    rm -rf ${nodesDir}/${App}.libs/*
    rm -rf ${nodesDir}/${AppV3}.libs/*
fi


# 获取whl文件路径
echo "find whl files ..."
whl=$(find ${buildDir} -name "${App}-*.whl" | head -n 1)
echo "v1 whl file: ${whl}"

whl_v3=$(find ${buildDir} -name "${AppV3}-*.whl" | head -n 1)
echo "v3 whl file: ${whl_v3}"

# 覆盖解压 whl 包到 nodes 目录
echo "extract whl file ..."
unzip -o ${whl} -d ${nodesDir}
unzip -o ${whl_v3} -d ${nodesDir}


echo -e "\n"


tree -sh ${nodesDir}

echo -e "\n"
echo "total size:"
du -sh ${nodesDir}

echo -e "\nBuild Done"

echo -e "\n"
