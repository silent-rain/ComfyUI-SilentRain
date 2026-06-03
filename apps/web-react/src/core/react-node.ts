/**
 * 通用「React 节点扩展」注册器。
 *
 * 把 ParamHub / ParamPort 这类「在节点上挂 React UI」的样板逻辑抽出来，
 * 让每个节点的实现只需关注：
 *   - 节点类名（comfyClass）
 *   - 渲染函数（拿到 node + 一些工具，返回 ReactNode）
 *   - 可选的 onConnections / onConfigure 钩子
 *
 * 一个节点对应一个文件 / 一个目录，调用一次 `defineReactNode(...)` 即可。
 */

import type { ComfyApp, ComfyExtension, LGraphNode, LLink } from './types/comfy';
import { mountReactWidget, type ReactWidgetHandle } from './lg-bridge';
import type { ReactNode } from 'react';

/**
 * 渲染函数能拿到的上下文
 */
export interface ReactNodeContext {
  /** 当前节点 */
  node: LGraphNode;
  /** ComfyUI 应用 */
  app: ComfyApp;
  /** 触发节点重绘 */
  redraw: () => void;
  /** 强制重新渲染 React 内容（一般不需要，状态由 React 内部管理） */
  rerender: () => void;
}

/**
 * 节点的连接变化回调上下文
 */
export interface ReactNodeConnectionContext extends ReactNodeContext {
  type: number;
  index: number;
  connected: boolean;
  link: LLink | null;
}

export interface DefineReactNodeOptions {
  /** ComfyUI 中注册的节点类名（对应 Python 端 NODE_CLASS_MAPPINGS 的 key） */
  comfyClass: string;

  /** 扩展名，注册到 ComfyUI；建议使用 `SilentRain.<NodeName>` */
  extensionName: string;

  /** widget 显示名（节点上挂载的 DOM widget 名字，仅用于内部识别） */
  widgetName?: string;

  /** UI 最小高度 */
  minHeight?: number;

  /**
   * 渲染函数（可选）。
   *
   * 仅当确实需要在节点上展示 React UI 时才提供 render；
   * 否则像 ParamPort 这类「纯副作用、靠 input/output 通信」的节点，
   * 应该省略 render，避免挂 DOM widget 导致节点底部出现灰色占位。
   *
   * 这一点对齐 cg-use-everywhere 的设计思路：节点不挂 widget，
   * 高度由 inputs/outputs 自然决定。
   */
  render?: (ctx: ReactNodeContext) => ReactNode;

  /** 节点首次创建时的钩子（在 React 挂载前） */
  onCreate?: (ctx: ReactNodeContext) => void;

  /** 节点配置（加载工作流）后的钩子，发生在 widgets_values 已填充时 */
  onConfigure?: (ctx: ReactNodeContext, info: unknown) => void;

  /** 连接变化钩子 */
  onConnectionsChange?: (ctx: ReactNodeConnectionContext) => void;

  /** 节点移除前的钩子（清理事件监听等） */
  onRemoved?: (ctx: ReactNodeContext) => void;

  /**
   * 自定义节点右键菜单项。
   * 返回的数组会被追加到 LiteGraph 默认菜单项之后。
   */
  onNodeContextMenu?: (ctx: ReactNodeContext) => Array<{
    content: string;
    callback?: () => void;
    has_submenu?: boolean;
    submenu?: Array<{ content: string; callback?: () => void }>;
  }>;

  /**
   * 在 ComfyUI 注册节点定义时，从 nodeData.input.optional 中剥离匹配本数组中
   * 任意正则的 input 名。
   */
  stripOptionalInputs?: RegExp[];
}

/**
 * 定义并返回一个可注册到 ComfyUI 的扩展。
 *
 * 用法：
 * ```tsx
 * export const ParamHubExtension = defineReactNode({
 *   comfyClass: 'ParamHub',
 *   extensionName: 'SilentRain.ParamHub',
 *   render: ({ node }) => <HubRoot node={node} />,
 * });
 *
 * // src/nodes/index.ts
 * export const NODE_EXTENSIONS = [ParamHubExtension, ...];
 * ```
 */
export function defineReactNode(opts: DefineReactNodeOptions): ComfyExtension {
  const widgetName = opts.widgetName ?? `sr_${opts.comfyClass}_ui`;
  const boundFlag = `__sr_bound_${opts.comfyClass}`;

  function isMatch(node: LGraphNode): boolean {
    return node.comfyClass === opts.comfyClass || node.type === opts.comfyClass;
  }

  function bind(node: LGraphNode, app: ComfyApp): void {
    const flagged = node as unknown as Record<string, boolean | undefined>;
    if (flagged[boundFlag]) return;
    flagged[boundFlag] = true;

    const redraw = () => {
      (node.graph as unknown as { setDirtyCanvas?: (a: boolean, b: boolean) => void })
        ?.setDirtyCanvas?.(true, true);
    };

    let handle: ReactWidgetHandle | null = null;

    const ctx: ReactNodeContext = {
      node,
      app,
      redraw,
      rerender: () => {
        if (!handle) return;
        const next = opts.render?.(ctx);
        if (next == null) return;
        handle.rerender(next);
      },
    };

    opts.onCreate?.(ctx);

    // 仅当 render 存在且返回非 null 时才挂载 DOM widget。
    // 这样像 ParamPort 这种「只用 React 跑副作用、不渲染 UI」的节点，
    // 不会产生空白的 widget 灰色占位区。
    const initial = opts.render?.(ctx);
    if (initial != null) {
      handle = mountReactWidget(node, widgetName, initial, {
        minHeight: opts.minHeight ?? undefined,
      });
    }

    // 包装 onConfigure
    const origConfigure = node.onConfigure;
    node.onConfigure = function (info: unknown) {
      const r = origConfigure?.call(this, info);
      queueMicrotask(() => opts.onConfigure?.(ctx, info));
      return r;
    };

    // 包装 onConnectionsChange
    if (opts.onConnectionsChange) {
      const origConn = node.onConnectionsChange;
      node.onConnectionsChange = function (
        type: number,
        index: number,
        connected: boolean,
        link: LLink | null,
        slot: any
      ) {
        origConn?.call(this, type, index, connected, link, slot);
        queueMicrotask(() =>
          opts.onConnectionsChange!({
            ...ctx,
            type,
            index,
            connected,
            link,
          })
        );
      };
    }

    // 包装 onRemoved
    const origRemoved = node.onRemoved;
    node.onRemoved = function () {
      try {
        opts.onRemoved?.(ctx);
      } finally {
        handle?.unmount();
        handle = null;
      }
      origRemoved?.call(this);
    };
  }

  const ext: ComfyExtension = {
    name: opts.extensionName,
    beforeRegisterNodeDef(_nodeType: any, nodeData: any) {
      if (!opts.stripOptionalInputs?.length) return;
      if (nodeData?.name !== opts.comfyClass) return;
      const optional = nodeData?.input?.optional;
      if (!optional || typeof optional !== 'object') return;
      let removed = 0;
      for (const key of Object.keys(optional)) {
        if (opts.stripOptionalInputs.some(re => re.test(key))) {
          delete optional[key];
          removed++;
        }
      }
      if (removed > 0) {
        console.log(
          `[SilentRain] stripped ${removed} optional inputs from ${opts.comfyClass}`,
        );
      }
    },
    setup() {
      if (!opts.onNodeContextMenu) return;
      const LG = (window as unknown as { LiteGraph?: any }).LiteGraph;
      if (!LG || !LG.LGraphCanvas) return;

      const proto = LG.LGraphCanvas.prototype;
      const origGetMenuOptions = proto.getMenuOptions;
      proto.getMenuOptions = function () {
        const items = origGetMenuOptions.apply(this, arguments);
        // 当右键的是节点时，LiteGraph 会在 options 中注入 node 引用
        const node = (this as unknown as { current_node?: LGraphNode }).current_node;
        if (!node || !isMatch(node)) return items;

        const ctx: ReactNodeContext = {
          node,
          app: (window as unknown as { app?: ComfyApp }).app!,
          redraw: () => {
            (node.graph as unknown as { setDirtyCanvas?: (a: boolean, b: boolean) => void })
              ?.setDirtyCanvas?.(true, true);
          },
          rerender: () => {
            // 右键菜单触发时不需要 rerender，占位
          },
        };

        const custom = opts.onNodeContextMenu!(ctx);
        if (custom && custom.length > 0) {
          items.push(null); // separator
          for (const item of custom) {
            items.push({
              content: item.content,
              callback: item.callback,
              has_submenu: item.has_submenu,
              submenu: item.submenu,
            });
          }
        }
        return items;
      };
    },
    nodeCreated(node: LGraphNode, app: ComfyApp) {
      if (!isMatch(node)) return;
      bind(node, app);
    },
    loadedGraphNode(node: LGraphNode, app: ComfyApp) {
      if (!isMatch(node)) return;
      // 兼容性：某些 ComfyUI 版本不会先调 nodeCreated 再调 loadedGraphNode
      bind(node, app);
    },
  };
  // 把 strip 配置作为元数据挂在 ext 上，main.ts 在 bundle 加载完成后会
  // 用它做回溯 backfillStripOptionalInputs（解决我们扩展注册晚于
  // beforeRegisterNodeDef 阶段的问题）
  if (opts.stripOptionalInputs?.length) {
    (ext as unknown as Record<string, unknown>).__sr_strip_class__ = opts.comfyClass;
    (ext as unknown as Record<string, unknown>).__sr_strip_patterns__ = opts.stripOptionalInputs;
  }
  return ext;
}

/* -------------------------------------------------------------------------- */
/* 辅助：对已注册节点定义 + 已存在节点回溯剥离 optional inputs                  */
/* -------------------------------------------------------------------------- */

/**
 * 在 bundle 加载完成后，对已注册节点定义和已存在节点回溯剥离 optional inputs。
 *
 * 由于 SilentRain bundle 通过 `await init()` 异步加载，本扩展的注册时机一定
 * 晚于 ComfyUI 的 `beforeRegisterNodeDef` 阶段。这意味着：
 *   1. 节点定义里 optional 没被剥离 → LiteGraph 创建新节点时仍会预塞 N 个 input
 *   2. 工作流里已存在的节点 reload 时也会被预塞
 *
 * 所以必须在 bundle 加载完成后执行回溯：
 *   - 修剪 `LiteGraph.registered_node_types[comfyClass]` 上挂的 nodeData
 *   - 对 graph 中现有节点，移除未连线且匹配 strip 正则的 input 槽
 *   - 然后 setSize(computeSize()) 强制重置高度
 */
export function backfillStripOptionalInputs(
  app: ComfyApp,
  comfyClass: string,
  patterns: RegExp[]
): void {
  if (!patterns?.length) return;

  // 1. 修剪已注册的 nodeData，使后续新建的节点不再被预塞
  const LiteGraph = (window as unknown as { LiteGraph?: any }).LiteGraph;
  const registered = LiteGraph?.registered_node_types?.[comfyClass];
  const candidateData = [
    registered?.nodeData,
    registered?.prototype?.nodeData,
    (app as unknown as { nodeOutputs?: Record<string, any> })?.nodeOutputs?.[comfyClass],
  ];
  let defStripped = 0;
  for (const nodeData of candidateData) {
    const optional = nodeData?.input?.optional;
    if (!optional || typeof optional !== 'object') continue;
    for (const key of Object.keys(optional)) {
      if (patterns.some(re => re.test(key))) {
        delete optional[key];
        defStripped++;
      }
    }
  }

  // 2. 对 graph 中已存在的节点，移除未连线且匹配的 input 槽
  const nodes = app.graph?._nodes;
  if (!Array.isArray(nodes)) {
    if (defStripped > 0) {
      console.log(
        `[SilentRain] backfill stripped ${defStripped} optional inputs from ${comfyClass} nodeData`,
      );
    }
    return;
  }

  let nodesAffected = 0;
  let inputsRemoved = 0;
  for (const node of nodes) {
    if (node.comfyClass !== comfyClass && node.type !== comfyClass) continue;
    const inputs = node.inputs;
    if (!Array.isArray(inputs)) continue;

    const cap = node as unknown as {
      removeInput?: (idx: number) => void;
      setSize?: (size: [number, number]) => void;
      computeSize?: () => [number, number];
    };
    if (typeof cap.removeInput !== 'function') continue;

    // 倒序删除，避免下标漂移；只删未连线（link 为空）且名字匹配的
    let localRemoved = 0;
    for (let i = inputs.length - 1; i >= 0; i--) {
      const slot = inputs[i];
      if (!slot) continue;
      if (slot.link != null) continue; // 已连线的不动
      if (!patterns.some(re => re.test(slot.name))) continue;
      cap.removeInput.call(node, i);
      localRemoved++;
    }
    if (localRemoved === 0) continue;

    nodesAffected++;
    inputsRemoved += localRemoved;

    // 强制重置节点高度：不取 max，让节点能从历史保存的"超大 size"缩回
    try {
      const min = cap.computeSize?.();
      if (Array.isArray(min) && min.length >= 2 && Array.isArray(node.size)) {
        node.size[0] = Math.max(node.size[0] || 0, min[0]);
        node.size[1] = min[1];
      }
    } catch {
      /* ignore */
    }
  }

  if (defStripped > 0 || nodesAffected > 0) {
    console.log(
      `[SilentRain] backfill ${comfyClass}: nodeData -${defStripped} input(s); ` +
        `${nodesAffected} node(s) trimmed -${inputsRemoved} input(s)`,
    );
    (app.graph as unknown as { setDirtyCanvas?: (a: boolean, b: boolean) => void })
      ?.setDirtyCanvas?.(true, true);
  }
}

/**
 * 从 ComfyExtension 中读取 stripOptionalInputs 配置（由 defineReactNode 写入）
 */
export function getStripPatterns(ext: ComfyExtension): { comfyClass?: string | undefined; patterns?: RegExp[] | undefined } {
  const e = ext as unknown as {
    __sr_strip_class__?: string;
    __sr_strip_patterns__?: RegExp[];
  };
  return {
    comfyClass: e.__sr_strip_class__,
    patterns: e.__sr_strip_patterns__,
  };
}
