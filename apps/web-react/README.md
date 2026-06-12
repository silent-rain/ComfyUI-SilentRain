# Web React UI

ComfyUI-SilentRain 所有需要复杂动态 UI 的节点的统一 React 前端工程。

## 技术栈

- React 18 + TypeScript（严格模式）
- Vite 8（IIFE 单文件 + CSS 内联打包）
- zustand（轻量级状态管理）
- @vitejs/plugin-react + react-refresh（快速热更新）
- vite-plugin-css-injected-by-js（CSS 运行时自动注入）
- SCSS（组件级样式，使用 `*.module.scss`）

## 目录结构

```text
apps/web-react/
├── src/
│   ├── main.tsx                    # 入口：初始化并注册所有节点扩展
│   ├── App.tsx                     # 应用初始化逻辑
│   ├── core/                       # 通用基础设施（与具体节点无关）
│   │   ├── index.ts                # 统一出口
│   │   └── lg-bridge.ts           # LiteGraph React 适配层（mountReactWidget/hideWidget）
│   ├── nodes/                      # 节点扩展（一节点一目录或文件）
│   ├── store/                      # zustand 状态管理
│   ├── constant/                   # 常量定义
│   ├── enums/                      # 枚举类型
│   ├── hook/                       # 自定义 React Hooks
│   ├── types/                      # TypeScript 类型声明
│   └── utils/                      # 工具函数
├── package.json
├── tsconfig.json
├── vite.config.ts
└── eslint.config.js
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

构建产物：`dist/silentrain.bundle.js`（单 IIFE，含 React 与 CSS）。

ComfyUI 通过 `WEB_DIRECTORY = "./web"` 加载 `nodes/web/main.js`，
该入口会在 wasm 启动后动态 `import` `dist/silentrain.bundle.js`。

## 添加新节点

1. 在 `src/nodes/` 下创建节点目录或文件（如 `my-node/index.tsx`）
2. 导出默认函数，返回 `ComfyExtension` 对象
3. 在 `src/nodes/index.tsx` 中导入并注册到 `NODE_EXTENSIONS` 数组

示例：

```tsx
// src/nodes/my-node/index.tsx
import type { ComfyExtension } from '@comfyorg/comfyui-frontend-types';

export default function MyNode(): ComfyExtension {
  return {
    name: 'SilentRain.MyNode',
    async setup(app) {
      // 注册节点扩展逻辑
    },
  };
}
```

## 代码规范

- 使用 ESLint + Prettier 进行代码格式化
- 支持 lint-staged + Husky 预提交检查
- TypeScript 严格模式

```bash
npm run lint          # ESLint 检查
npm run format        # Prettier 格式化
npm run type-check    # TypeScript 类型检查
```

## 参考

- [ComfyUI 自定义节点前端开发](https://docs.comfy.org/zh/custom-nodes/js/javascript_overview)
- [Vite 官方文档](https://vite.dev/)
- [zustand 状态管理](https://zustand.docs.pmnd.rs/)
