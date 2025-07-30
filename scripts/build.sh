#!/bin/bash

# 任何命令失败后立即退出
set -e

# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"


# 项目名称, 一般不需要调整
App=comfyui_silentrain
# 编译目录
buildDir=./target/wheels
# 节点目录
nodesDir=${root}/nodes
# Python Version
PythonVersion=3.12


# 检查虚拟环境
echo "check venv ..."
if [ ! -d .venv ]; then
    echo "create venv ..."
    # 安装
    pipx install uv

    # 安装Python版本
    uv python install ${PythonVersion}

    # 创建虚拟环境
    uv venv --python ${PythonVersion}
fi


# switch venv
echo "switch venv ..."
source .venv/bin/activate

# build
echo "build release whl ..."
maturin build -r


# 获取whl文件路径
echo "find whl file ..."
whl=$(find ${buildDir} -name "${App}-*.whl" | head -n 1)


# 覆盖解压whl包
echo "extract whl file ..."
unzip -o ${whl} -d ${nodesDir}
echo -e "\n"


echo -e "\nBuild Done"

echo -e "\n"
