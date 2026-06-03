// ComfyUI-SilentRain React Bundle 入口
//
// 通过 nodes/web/main.js 在 ComfyUI 环境中加载，
// 注册所有 React 节点扩展。

import './styles/common.css';
import { NODE_EXTENSIONS } from './nodes';
import { fitNodeHeight, backfillStripOptionalInputs, getStripPatterns } from './core';
import type { ComfyApp, ComfyExtension, LGraphNode } from './core/types/comfy';

function ready(cb: () => void) {
  if (typeof window === 'undefined') return;
  if (window.app) {
    cb();
    return;
  }
  let attempts = 0;
  const timer = setInterval(() => {
    attempts++;
    if (window.app) {
      clearInterval(timer);
      cb();
    } else if (attempts > 200) {
      clearInterval(timer);
      console.error('[SilentRain] window.app not ready after 20s, abort');
    }
  }, 100);
}

/**
 * 注册扩展之后，遍历当前 graph 已存在的节点，补调用 nodeCreated。
 *
 * 背景：ComfyUI 启动顺序大致是
 *   1) 加载官方扩展
 *   2) 加载 workflow → 创建节点 → 触发 nodeCreated / loadedGraphNode
 *   3) nodes/web/main.js 异步加载（await wasm init → import bundle）
 *   4) bundle 内 registerExtension(...)
 *
 * 步骤 3-4 通常晚于步骤 2（尤其 wasm 初始化耗时较长时），
 * 这就导致已经创建的 ParamHub/ParamPort 节点不会再触发 nodeCreated 钩子，
 * 其 React 面板与隐藏 widget 全部失效，节点退化为原始 32 个预留槽 + 裸 JSON 文本框。
 *
 * 这里在注册完成后立即扫描一次现有节点，对每个匹配的节点手动调用 nodeCreated，
 * 这样无论加载顺序如何都能保证扩展正确接管。
 */
function backfillExistingNodes(app: ComfyApp, ext: ComfyExtension): LGraphNode[] {
  const nodes = app.graph?._nodes;
  const touched: LGraphNode[] = [];
  if (!Array.isArray(nodes) || nodes.length === 0) return touched;
  for (const node of nodes as LGraphNode[]) {
    try {
      ext.nodeCreated?.(node, app);
      touched.push(node);
    } catch (e) {
      console.error(`[SilentRain] backfill nodeCreated failed for ${ext.name}`, e);
    }
  }
  return touched;
}

ready(() => {
  const app = window.app!;
  const names: string[] = [];
  const touchedAll: LGraphNode[] = [];
  for (const ext of NODE_EXTENSIONS) {
    try {
      app.registerExtension(ext);
      // 由于本 bundle 是 await init() 后才异步加载，扩展注册时机一定晚于
      // ComfyUI 的 beforeRegisterNodeDef 阶段——那里的钩子已经错过。
      // 这里立即对已注册的 nodeData 和已存在的节点做一次回溯剥离，
      // 把 ParamHub 上"被预塞的 32 个 param_* 输入槽"清掉，并强制重置高度。
      const { comfyClass, patterns } = getStripPatterns(ext);
      if (comfyClass && patterns?.length) {
        backfillStripOptionalInputs(app, comfyClass, patterns);
      }
      const touched = backfillExistingNodes(app, ext);
      touchedAll.push(...touched);
      names.push(ext.name);
    } catch (e) {
      console.error(`[SilentRain] failed to register ${ext.name}`, e);
    }
  }
  // 兜底：等 React widget 完成首次布局后再统一收敛一次。
  //   主链路是 lg-bridge 中的 ResizeObserver + syncFromMeta/refreshPort 末尾的 fitNodeHeight，
  //   这里只是兜底（处理 RO 早期某些浏览器 bug 或边界场景）。
  const matched = touchedAll.filter((n) => {
    const flags = n as unknown as Record<string, unknown>;
    return Object.keys(flags).some((k) => k.startsWith('__sr_bound_') && flags[k] === true);
  });
  if (matched.length > 0) {
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        for (const n of matched) fitNodeHeight(n);
      }),
    );
  }
  console.log(`[SilentRain] React extensions registered: ${names.join(', ')}`);
});
