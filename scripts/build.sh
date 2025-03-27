#!/bin/bash
# 本脚本用于将生成环境, 打包为 release 包, 同时将whl文件解构到当前项目下, 用于提交到git仓库.


# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"

App=comfyui_silentrain
Project=${root}/nodes


# nodes 目录
if [ ! -d ${Project} ]; then 
    echo "mkdir ${Project}"
    mkdir ${Project}
fi


# switch venv
echo "switch venv ..."
source .venv/bin/activate

# build
echo "build release whl ..."
maturin build -r

cd target/wheels

# 解压Python包
echo "unzip ${App} whl ..."
unzip ${App}-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl

# 将 ${App} 移动到 ${Project} 的自定义节点目录下
echo "develop ${App} to Project path ..."
\cp -r ./${App} ${Project}/

# 检查是否安装成功
echo "print comfyui new node ..."
tree ${Project}/${App}

# 删除解压的文件
rm -rf ${App} ${App}-*.dist-info

echo "Done"