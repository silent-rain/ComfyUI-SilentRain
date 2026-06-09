import type { ComfyApp } from '@comfyorg/comfyui-frontend-types';
import { NODE_EXTENSIONS } from './nodes';

/**
 * SilentRain React Bundle 入口函数。
 *
 * 注册所有 React 节点扩展到 ComfyUI 中。
 * 注意：本函数不再返回 JSX（不再挂载全局 React 根节点），
 * 各节点的 React UI 通过 defineReactNode 内部自行挂载到 DOM widget。
 */
export default async function App() {
  // 动态导入 app.js（该文件是动态生成的，无法静态导入）
  let app: ComfyApp;
  try {
    // @ts-ignore: 动态导入的文件
    const module = await import('../../../scripts/app.js');
    app = module.app || window.app;
  } catch (e) {
    console.warn('[SilentRain] failed to dynamically import app.js, using window.app', e);
    app = window.app!;
  }

  if (!app) {
    console.error('[SilentRain] app not ready');
    return;
  }

  console.log('[SilentRain] app ready', app);
  const names: string[] = [];
  for (const ext of NODE_EXTENSIONS) {
    try {
      // 注册节点扩展
      app.registerExtension(ext);
      names.push(ext.name);
    } catch (e) {
      console.error(`[SilentRain] failed to register ${ext.name}`, e);
    }
  }

  console.log(`[SilentRain] React extensions registered: ${names.join(', ')}`);
}
