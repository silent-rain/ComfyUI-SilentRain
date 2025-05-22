#!/bin/bash
# 本脚本用于将测试环境, 可以快速将调试的包导入到本地的ComfyUI自定义节点目录下, 用于快速验证节点.

# 项目名称, 一般不需要调整
App=comfyui_silentrain
# ComfyUI 部署路径
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
echo "unzip ${App} whl ..."
unzip ${App}-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl

# 将 ${App} 移动到 ${ComfyUI} 的自定义节点目录下
echo "develop ${App} to ComfyUI path ..."
\cp -r ./${App} ${ComfyUI}/

# 检查是否安装成功
echo "comfyui new node ..."
tree ${ComfyUI}/${App}

echo "comfyui node dir list ..."
ls -hl ${ComfyUI}/${App}

# 删除解压的文件
rm -rf ${App} ${App}-*.dist-info

echo "Done"