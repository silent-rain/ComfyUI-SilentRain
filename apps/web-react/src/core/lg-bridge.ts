import { createRoot, type Root } from 'react-dom/client';
import type { ReactNode } from 'react';

type SrNode = any; // LiteGraph 节点，运行时来自 ComfyUI 全局

export interface ReactWidgetHandle {
  widget: any;
  root: Root;
  container: HTMLDivElement;
  unmount: () => void;
  rerender: (children: ReactNode) => void;
}

/**
 * 将 React 组件挂载到 LiteGraph 节点的 DOM widget 上。
 *
 * 会在节点内创建一个 DOM container，并用 ReactDOM.createRoot 渲染内容。
 * 返回的 handle 包含 unmount / rerender 方法，以及 widget 引用。
 *
 * 用法示例（在 ComfyExtension 的 loadedGraphNode / nodeCreated 中调用）：
 *
 * ```ts
 * const handle = mountReactWidget(node, 'my_panel', <MyPanel />);
 * // 需要更新时
 * handle.rerender(<MyPanel newProps />);
 * // 节点移除时
 * handle.unmount();
 * ```
 */
export function mountReactWidget(
  node: SrNode,
  name: string,
  initial: ReactNode,
  options?: { minHeight?: number | undefined },
): ReactWidgetHandle {
  const minH = options?.minHeight ?? 0;

  const container = document.createElement('div');
  container.className = 'sr-react-widget';
  container.style.width = '100%';
  container.style.boxSizing = 'border-box';
  container.style.padding = '0';
  container.style.margin = '0';
  container.style.fontSize = '12px';
  container.style.color = 'var(--input-text)';
  container.style.overflow = 'hidden';
  container.style.height = 'fit-content';
  if (minH > 0) container.style.minHeight = `${minH}px`;

  const measure = (): number => {
    const h = container.scrollHeight || container.offsetHeight || 0;
    return Math.max(h, minH);
  };

  let widget: any;

  if (typeof node.addDOMWidget === 'function') {
    widget = node.addDOMWidget(name, 'sr-react', container, {
      serialize: false,
      getHeight: () => measure(),
      getMinHeight: () => minH,
      getMaxHeight: () => measure(),
      hideOnZoom: false,
    });
    widget.computeSize = () => {
      const height = measure();
      return [node.size?.[0] ?? 200, height];
    };
  } else {
    widget = node.addWidget('div', name, '', null, { serialize: false });
    widget.element = container;
    widget.computeSize = () => {
      const height = measure();
      return [node.size?.[0] ?? 200, height];
    };
  }

  // 使用 ResizeObserver 监听内容变化，自动调整节点大小
  let resizeObserver: ResizeObserver | null = null;
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(() => {
      if (node.setSize && widget.computeSize) {
        const [width, height] = widget.computeSize();
        // 只调整高度，保持宽度不变
        node.setSize([width, height]);
        // 触发节点重绘
        if (node.graph) {
          node.graph.setDirtyCanvas(true, true);
        }
      }
    });
    resizeObserver.observe(container);
  }

  const root = createRoot(container);
  root.render(initial as any);

  return {
    widget,
    root,
    container,
    unmount: () => {
      // 清理 ResizeObserver
      if (resizeObserver) {
        resizeObserver.disconnect();
        resizeObserver = null;
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
 * 按 name 查找节点上的 widget
 */
export function findWidget(node: SrNode, name: string): any | undefined {
  return node.widgets?.find((w: any) => w.name === name);
}
