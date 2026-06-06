import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import cssInjectedByJsPlugin from 'vite-plugin-css-injected-by-js';
import { resolve } from 'node:path';

// 构建为单个 IIFE 文件，输出到当前项目下的 dist/silentrain.bundle.js
// 然后由 scripts/build_web_react.sh 拷贝到 nodes/web/dist/。
//
// ComfyUI 通过 WEB_DIRECTORY 静态托管 nodes/web 目录，
// 由 nodes/web/main.js 通过 dynamic import 加载。
//
// CSS 通过 cssInjectedByJsPlugin 内联到 JS，运行时自动注入到 <head>。
//
// 设计为通用 bundle —— 未来所有 React 节点都打包到同一个文件中。
export default defineConfig({
  plugins: [react(), cssInjectedByJsPlugin()],
  // 浏览器环境下没有 process 全局变量；React 生产构建会引用 process.env.NODE_ENV
  // 通过 define 静态替换这些引用，避免运行时报 "process is not defined"
  define: {
    'process.env.NODE_ENV': JSON.stringify('production'),
    'process.env': '{}',
    'process.platform': JSON.stringify('browser'),
    'process.version': JSON.stringify(''),
    global: 'globalThis',
  },
  resolve: {
    alias: {
      '@/*': resolve(__dirname, 'src/*'),
    },
  },
  build: {
    outDir: resolve(__dirname, 'dist'),
    emptyOutDir: true,
    target: 'es2020',
    minify: 'esbuild',
    sourcemap: false,
    cssCodeSplit: false,
    lib: {
      entry: resolve(__dirname, 'src/main.tsx'),
      name: 'SilentRainUI',
      formats: ['iife'],
      fileName: () => 'silentrain.bundle.js',
    },
    rollupOptions: {
      external: [],
      output: {
        inlineDynamicImports: true,
        assetFileNames: 'silentrain.bundle.[ext]',
      },
    },
  },
});
