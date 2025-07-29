#!/bin/bash

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


echo -e "\nBuild Done"

echo -e "\n"
