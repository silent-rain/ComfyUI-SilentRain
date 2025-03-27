#!/bin/bash

APP=comfyui_silentrain
ComfyUI=~/code/ComfyUI/custom_nodes

# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"

# switch venv
echo "switch venv ..."
source .venv/bin/activate

# build
echo "build whl ..."
maturin build

cd target/wheels

# 解压Python包
echo "unzip ${APP} whl ..."
unzip ${APP}-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl

# 将${APP}移动到ComfyUI的自定义节点目录下
echo "develop ${APP} to ComfyUI path ..."
\cp -r ./${APP} ${ComfyUI}/

# 检查是否安装成功
echo "print comfyui new node ..."
tree ${ComfyUI}/${APP}

# 删除解压的文件
rm -rf ${APP} ${APP}-*.dist-info

echo "Done"