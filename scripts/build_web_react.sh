#!/bin/bash
# 构建 React + TS + Vite 前端（所有 React 节点 UI 的统一打包）
# 输出到 nodes/web/dist/silentrain.bundle.js

set -e

root=$(cd "$(dirname ${0})/../";pwd)
ComfyUI=/data/ComfyUI

echo "====================== root:${root} ======================"


cd "${root}/apps/web-react"

# 选择包管理器
if command -v pnpm >/dev/null 2>&1; then
    PM=pnpm
elif command -v bun >/dev/null 2>&1; then
    PM=bun
elif command -v npm >/dev/null 2>&1; then
    PM=npm
else
    echo "Error: no npm/pnpm/bun found in PATH"
    exit 1
fi

echo "package manager: ${PM}"

# 安装依赖
if [ ! -d node_modules ]; then
    echo "install deps ..."
    ${PM} install
fi

# 构建
echo "build react bundle ..."
# 使用 TypeScript 配置文件 vite.config.ts（vite 会自动加载）
${PM} run build

# 将 vite 构建产物从前端项目内 dist 拷贝到 nodes/web/dist
SRC_DIST="${root}/apps/web-react/dist"
DEST_DIST="${root}/nodes/web/dist"
mkdir -p "${DEST_DIST}"
cp -rf "${SRC_DIST}/." "${DEST_DIST}/"
echo "copied: ${SRC_DIST} -> ${DEST_DIST}"

echo -e "\noutput:"
ls -lh "${DEST_DIST}/" || true

echo -e "\nBuild Done"

# 同步到 ComfyUI 实际安装目录（与 build_web.sh 保持一致行为）
TARGET_DIR=${ComfyUI}/custom_nodes/comfyui_silentrain/web/dist
if [ -d "$(dirname ${TARGET_DIR})" ]; then
    mkdir -p "${TARGET_DIR}"
    cp -rf "${DEST_DIST}/." "${TARGET_DIR}/"
    echo "synced to ${TARGET_DIR}"
fi
