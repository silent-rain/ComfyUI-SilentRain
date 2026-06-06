import { createRoot, type Root } from 'react-dom/client';
import type { ReactNode } from 'react';

export interface ReactWidgetHandle {
  widget: IWidget;
  root: Root;
  container: HTMLDivElement;
  unmount: () => void;
  rerender: (children: ReactNode) => void;
}

/**
 * 把节点高度收敛到当前内容真实需要的高度。
 *
 * 直接覆盖 size[1]（不取 max），让节点能从历史保存的"超大 size"
 * 缩回到真正需要的尺寸；同时把宽度向下取 max 保证不变窄。
 */
export function fitNodeHeight(node: LGraphNode): void {
  if (typeof node.computeSize !== 'function') return;
  if (!Array.isArray(node.size)) return;
  let min: number[] | undefined;
  try {
    min = node.computeSize();
  } catch {
    return;
  }
  if (!Array.isArray(min) || min.length < 2) return;
  let dirty = false;
  const minW = min[0]!;
  const minH = min[1]!;
  const w = Math.max(node.size[0] || 0, minW);
  if (node.size[0] !== w) {
    node.size[0] = w;
    dirty = true;
  }
  if (node.size[1] !== minH) {
    if (typeof console !== 'undefined' && (window as unknown as { __SR_DEBUG__?: boolean }).__SR_DEBUG__) {
      console.log('[SilentRain] fitNodeHeight', node.type, 'from', node.size[1], 'to', minH);
    }
    node.size[1] = minH;
    dirty = true;
  }
  if (dirty) {
    (node.graph as unknown as {
      setDirtyCanvas?: (a: boolean, b: boolean) => void;
    } | null)?.setDirtyCanvas?.(true, true);
  }
}

/**
 * 将一个 React 组件挂载到 LiteGraph 节点上，作为一个 DOM widget。
 *
 * 优先使用 ComfyUI 提供的 addDOMWidget（推荐方式），
 * 回退方案直接创建一个 widget 并塞入 element 字段。
 */
export function mountReactWidget(
  node: LGraphNode,
  name: string,
  initial: ReactNode,
  options?: { minHeight?: number | undefined }
): ReactWidgetHandle {
  const minH = options?.minHeight ?? 0;

  const container = document.createElement('div');
  container.classList.add('sr-react-widget');
  container.style.width = '100%';
  container.style.boxSizing = 'border-box';
  container.style.padding = '4px 6px';
  container.style.fontSize = '12px';
  container.style.color = 'var(--input-text)';
  if (minH > 0) container.style.minHeight = `${minH}px`;

  // 让 widget 高度跟随真实内容，避免 ComfyUI 把 DOM widget 撑得过大
  const measure = (): number => {
    // scrollHeight 会随 React 内容变化
    const h = container.scrollHeight || container.offsetHeight || 0;
    return Math.max(h, minH);
  };

  let widget: IWidget;

  if (typeof node.addDOMWidget === 'function') {
    widget = node.addDOMWidget(name, 'sr-react', container, {
      serialize: false,
      // 新版 ComfyUI 的 DOM widget 会调用这两个回调来确定占位高度
      getHeight: () => measure(),
      getMinHeight: () => minH,
      getMaxHeight: () => measure(),
      hideOnZoom: false,
    });
    // 兜底：旧版/不识别上述回调时，computeSize 仍然生效
    widget.computeSize = () => [node.size?.[0] ?? 200, measure()];
  } else {
    widget = node.addWidget('div', name, '', null, {
      serialize: false,
    });
    (widget as { element?: HTMLElement }).element = container;
    widget.computeSize = () => [node.size?.[0] ?? 200, measure()];
  }

  const root = createRoot(container);
  root.render(initial as any);

  // 监听 widget 容器的真实布局尺寸：
  //   - 首次 React commit 完成后会触发一次（解决"刷新后节点底部一大片灰色"的核心问题）
  //   - 后续 reconcile 导致行数增减时也会触发，节点能自适应缩放
  //
  // 用 ResizeObserver 比用 rAF/setTimeout 猜测时序更可靠，
  // 也避免了 ComfyUI 在 widget 挂载完成后再次写回 size 的覆盖问题。
  let ro: ResizeObserver | null = null;
  if (typeof ResizeObserver !== 'undefined') {
    let pending = false;
    ro = new ResizeObserver(() => {
      // 把同步多次回调合并到下一帧，避免布局抖动
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        fitNodeHeight(node);
      });
    });
    ro.observe(container);
  }

  return {
    widget,
    root,
    container,
    unmount: () => {
      try {
        ro?.disconnect();
      } catch {
        /* ignore */
      }
      try {
        root.unmount();
      } catch {
        /* ignore */
      }
      container.remove();
    },
    rerender: (children: ReactNode) => {
      root.render(children as any);
    },
  };
}

/**
 * 在 widgets 中按 name 查找一个 widget
 */
export function findWidget(node: LGraphNode, name: string): IWidget | undefined {
  return node.widgets?.find((w) => w.name === name);
}

/**
 * 隐藏一个 widget（用于把序列化用 widget 藏起来）
 *
 * 关键点：保持 serializeValue 行为，确保 widget 的值仍然写入到 widgets_values，
 * 但在 UI 上不显示也不占据高度。
 *
 * 兼容多个 ComfyUI 版本：
 *   - 旧版 LiteGraph：依赖 type 前缀 `hidden_` 来跳过绘制
 *   - 新版 ComfyUI（Vue 重构后）：识别 widget.hidden / widget.computedHeight
 *   - 极端情况：直接覆盖 widget.draw 阻止绘制
 */
export function hideWidget(widget: IWidget) {
  const w = widget as IWidget & {
    origType?: string;
    hidden?: boolean;
    computedHeight?: number;
    draw?: (...args: unknown[]) => void;
    onMouseDown?: (...args: unknown[]) => boolean;
    element?: HTMLElement;
  };
  if (!w.type.startsWith('hidden_')) {
    w.origType = w.type;
    w.type = 'hidden_' + w.type;
  }
  w.hidden = true;
  w.computedHeight = 0;
  w.computeSize = () => [0, -4];
  // 拦截绘制与交互，避免新版 ComfyUI 仍然把字符串/JSON 画到节点上
  w.draw = () => { };
  w.onMouseDown = () => false;
  // 如果 widget 之前作为 DOM widget 挂了 element，也一并隐藏
  if (w.element) {
    w.element.style.display = 'none';
  }
}
