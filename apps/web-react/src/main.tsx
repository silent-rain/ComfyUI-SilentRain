// ComfyUI-SilentRain React Bundle 入口
//
// 通过 nodes/web/main.js 在 ComfyUI 环境中加载，
// 注册所有 React 节点扩展。
//
// ⚠️ 注意：本 bundle 不再需要一个全局 React 根节点。
// React UI 通过 defineReactNode + mountReactWidget 直接嵌入到
// 每个 LiteGraph 节点的 DOM widget 中。

import App from './App';

App();
