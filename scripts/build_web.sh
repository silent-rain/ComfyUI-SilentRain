#!/bin/bash

# 任何命令失败后立即退出
set -e

OUT_DIR=../../nodes/web/pkg

cd apps/web

# Web 端
echo "build web ..."
wasm-pack build --release --target web --out-dir ${OUT_DIR}

tree -sh ${nodesDir}

echo -e "\n"
echo "total size:"
du -sh ${OUT_DIR}

echo -e "\n"

echo -e "\nBuild Done"

# tmp
cp -rf ${OUT_DIR} /home/one/code/ComfyUI/custom_nodes/comfyui_silentrain/web/