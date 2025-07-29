#!/bin/bash
# Description: Publishing the release on ComfyUI website.


# 任何命令失败后立即退出
set -e


# get dir, file path
root=$(cd "$(dirname ${0})/../";pwd)
echo "====================== root:${root} ======================"



# 项目名称, 一般不需要调整
App=comfyui_silentrain
# ComfyUI 节点目录
nodesDir=${root}/nodes
# 编译目录
buildDir=./target/wheels


# switch venv
echo "switch venv ..."
source .venv/bin/activate


# 获取whl文件路径
echo "find whl file ..."
whl=$(find ${buildDir} -name "${App}-*.whl" | head -n 1)
cp ${whl} ${nodesDir}/

nodeWhl=$(find ${nodesDir} -name "${App}-*.whl" | head -n 1)
newNodewhl=$(echo ${nodeWhl} | sed 's/-[0-9.]*\(-cp312\)/\1/')
echo $newNodewhl
mv ${nodeWhl} ${newNodewhl}


# publishing
uv run comfy node publish
