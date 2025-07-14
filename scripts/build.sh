#!/bin/bash
# 本脚本用于将生成环境, 打包为 release 包, 同时将whl文件解构到当前项目下, 用于提交到git仓库.


# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"

App=comfyui_silentrain
nodesDir=${root}/nodes
PythonVersion=3.12

# App 版本号, 从 Cargo.toml 提取包名和版本号
AppVersion=$(grep '^version =' Cargo.toml | cut -d '"' -f2  | tr -d '\n')
echo "App version: ${AppVersion}"

# nodes 目录
if [ ! -d ${nodesDir} ]; then 
    echo "mkdir ${nodesDir}"
    mkdir ${nodesDir}
fi

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

cd target/wheels

# 解压Python包
echo "unzip ${App} whl ..."
# zipFile=${App}-${AppVersion}-cp312-cp312-manylinux_2_34_x86_64.whl
zipFile=$(find . -name "${App}-*.whl" | head -n 1)
echo "zip file: ${zipFile}"

unzip ${zipFile}

# 将 ${App} 移动到 ${nodesDir} 的自定义节点目录下
echo "develop ${App} to ${nodesDir} ..."
rm -rf ${nodesDir}/${App} ${nodesDir}/${App}.libs ${nodesDir}/${App}-*.dist-info

\mv -f ./${App} ${nodesDir}/
\mv -f ./${App}.libs ${nodesDir}/
\mv -f ./${App}-*.dist-info ${nodesDir}/

echo -e "\n"

echo "ls nodes dir ..."
ls -hl ${nodesDir}/${App}

echo -e "\n"

echo "tree nodes dir ..."
tree ${nodesDir}/

echo -e "\n"


# 删除解压的文件
rm -rf ${App} ${App}.libs ${App}-*.dist-info

echo -e "\nBuild Done"

echo -e "\n"
