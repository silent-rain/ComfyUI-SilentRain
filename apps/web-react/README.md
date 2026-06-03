# Web React UI

ComfyUI-SilentRain 所有需要复杂动态 UI 的节点的统一 React 前端工程。

## 技术栈

- React 18 + TypeScript（严格模式）
- Vite 6（IIFE 单文件 + CSS 内联打包）
- mitt（事件总线，跨节点通信）
- uuid（参数稳定 ID）

## 目录结构

```
apps/web-react/
├── src/
│   ├── main.ts                       # 入口：从 nodes/index.ts 收集所有扩展并注册
│   ├── styles/
│   │   └── common.css                # 通用控件样式（input/select/button 等）
│   ├── core/                         # 通用基础设施（与具体节点无关）
│   │   ├── index.ts                  # 统一出口
│   │   ├── bus.ts                    # 全局事件总线
│   │   ├── lg-bridge.ts              # LiteGraph React 适配层（mountReactWidget/hideWidget）
│   │   ├── react-node.ts             # 通用 ReactNodeExtension 注册器（defineReactNode）
│   │   └── types/comfy.d.ts          # ComfyUI / LiteGraph 类型声明
│   └── nodes/
│       ├── index.ts                  # ★ 节点总表 —— 新增节点只改这里
│       ├── _shared/                  # 跨节点共享的协议、工具
│       │   └── param-protocol.ts
│       ├── param-hub/                # 一节点一目录
│       │   ├── index.tsx             # 节点扩展定义
│       │   ├── HubPanel.tsx          # UI 组件
│       │   └── HubPanel.css          # 节点专有样式
│       └── param-port/
│           ├── index.tsx
│           ├── PortPanel.tsx
│           ├── PortPanel.css
│           └── protocol.ts           # 节点专有协议
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## 构建

```bash
bash scripts/build_web_react.sh
```

或：

```bash
cd apps/web-react
npm install
npm run build       # 一次性构建
npm run dev         # vite build --watch
```

构建产物：`nodes/web/dist/silentrain.bundle.js`（单 IIFE，含 React 与 CSS）。

ComfyUI 通过 `WEB_DIRECTORY = "./web"` 加载 `nodes/web/main.js`，
该入口会在 wasm 启动后动态 `import` `dist/silentrain.bundle.js`。

## 新增一个节点

1. 在 `src/nodes/<your-node>/` 下创建：
   - `index.tsx` —— 调用 `defineReactNode({...})` 并默认导出
   - `XxxPanel.tsx` —— React UI 组件
   - `XxxPanel.css` —— 节点专有样式
   - 可选：`protocol.ts` —— 节点专有协议
2. 在 `src/nodes/index.ts` 中加入导出：

```ts
import { YourNodeExtension } from './your-node';

export const NODE_EXTENSIONS = [
  // ...
  YourNodeExtension,
];
```

1. `bash scripts/build_web_react.sh` 即可生效。

### `defineReactNode` 模板

```tsx
import { defineReactNode } from '../../core';
import { Panel } from './Panel';

export const YourNodeExtension = defineReactNode({
  comfyClass: 'YourNode',                 // Python 端 NODE_CLASS_MAPPINGS 的 key
  extensionName: 'SilentRain.YourNode',
  minHeight: 60,

  onCreate({ node }) {
    // 隐藏序列化用 widget 等
  },

  render({ node }) {
    return <Panel node={node} />;
  },

  onConfigure({ node }) {
    // 工作流加载完成后同步状态
  },

  onConnectionsChange({ node, link, connected }) {
    // 连线变化
  },

  onRemoved({ node }) {
    // 清理事件监听等
  },
});
```

## 设计要点

- **无后端注册表**：UI 状态完全持久化到 widgets_values，跨电脑导入零信息丢失。
- **稳定 UUID 关联**：参数重命名不失联（用于 ParamHub ↔ ParamPort）。
- **一节点一目录**：每个节点独立的 UI / 协议 / 样式，互不干扰。
- **统一打包**：所有节点编译到单个 `silentrain.bundle.js`，自动按需触发渲染。
