/**
 * Workflow Image - Core utilities for canvas export
 *
 * Analyzed and adapted from:
 * - pythongosssss/ComfyUI-Custom-Scripts workflowImage.js
 * - BobRandomNumber/ComfyUI-QoL-Pack QoL_WorkflowImage.js
 *
 * Key fix for empty text in exported images:
 * The new ComfyUI frontend (React-based) uses DOM widgets (textarea) for text input.
 * These DOM elements are overlaid on the canvas and NOT drawn via widget.draw().
 * When we modify the canvas transform for export, the DOM elements become invisible
 * or misaligned, and the canvas only shows empty widget backgrounds.
 *
 * Solution: After drawCanvas(), we iterate all nodes and their widgets,
 * then manually draw the text content of DOM widgets and text-value widgets
 * directly onto the canvas at their correct positions.
 */

/** Canvas state snapshot for save/restore */
export interface CanvasState {
  scale: number;
  offset: [number, number];
  width: number;
  height: number;
  transform: DOMMatrix;
}

/**
 * Wrap text to fit within a maximum width.
 * Adapted from https://codepen.io/peterhry/pen/nbMaYg
 * Handles long words that exceed maxWidth by breaking them character by character.
 */
function wrapText(
  context: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  maxWidth: number,
  lineHeight: number,
): void {
  if (!text) return;

  const words = text.split(' ');
  let line = '';

  for (let i = 0; i < words.length; i++) {
    let test = words[i]!;
    let metrics = context.measureText(test);

    // Break long words that exceed maxWidth
    while (metrics.width > maxWidth && test.length > 1) {
      test = test.substring(0, test.length - 1);
      metrics = context.measureText(test);
    }
    if (words[i] !== test) {
      words.splice(i + 1, 0, words[i]!.substring(test.length));
      words[i] = test;
    }

    const testLine = line + words[i] + ' ';
    metrics = context.measureText(testLine);

    if (metrics.width > maxWidth && i > 0) {
      context.fillText(line, x, y);
      line = words[i] + ' ';
      y += lineHeight;
    } else {
      line = testLine;
    }
  }

  context.fillText(line, x, y);
}

/**
 * Wait for all DOM widgets to be visible and properly positioned.
 *
 * In the new ComfyUI frontend, text widgets (textarea/customtext) use DOM elements
 * overlaid on the canvas. After we call updateView() and drawCanvas(), these DOM
 * elements need time to:
 * 1. Be repositioned by the canvas DragAndScale system
 * 2. Complete any pending CSS transitions/animations
 * 3. Finish loading any remote resources (e.g., model preview images)
 *
 * This function polls the DOM widgets until they are all visible and positioned,
 * or until the timeout is reached.
 *
 * @param timeout - Maximum time to wait in milliseconds (default: 3000ms)
 */
export async function waitForDomWidgetsReady(timeout: number = 3000): Promise<void> {
  const app = window.app;
  if (!app) return;

  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    const nodes = app.graph?._nodes;
    if (!nodes) break;

    let allReady = true;

    for (const node of nodes) {
      if (!node.widgets) continue;

      for (const widget of node.widgets as any[]) {
        if (widget.hidden) continue;

        const domEl = widget.element ?? widget.inputEl;
        if (!domEl) continue;

        // Only check DOM widgets (textarea/customtext)
        if (widget.type !== 'textarea' && widget.type !== 'customtext') continue;

        const el = domEl as HTMLElement;

        // Check if the element is visible and has non-zero dimensions
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) {
          allReady = false;
          break;
        }

        // Check if the element is inside the canvas viewport
        // The dom-widget wrapper should be positioned within the canvas
        const wrapper = el.closest('.dom-widget') as HTMLElement;
        if (wrapper) {
          const wrapperRect = wrapper.getBoundingClientRect();
          // If the wrapper has zero size, it's not ready yet
          if (wrapperRect.width === 0 || wrapperRect.height === 0) {
            allReady = false;
            break;
          }
        }
      }

      if (!allReady) break;
    }

    if (allReady) break;

    // Wait a short period before polling again
    await new Promise<void>(resolve => setTimeout(resolve, 100));
  }
}

/**
 * Draw text content of all widgets onto the canvas.
 *
 * This is the core fix for empty text in exported images.
 * In the new ComfyUI frontend, text widgets (textarea/customtext) use DOM elements
 * overlaid on the canvas. When the canvas transform is modified for export,
 * these DOM elements are no longer visible at the correct position.
 *
 * This function iterates through all nodes and their widgets, and for any
 * widget that has text content (DOM widgets with element/inputEl, or widgets
 * with string values), it draws the text directly onto the canvas.
 *
 * Must be called AFTER drawCanvas() and BEFORE canvas.toBlob()/toDataURL().
 *
 * @param bounds - The bounding box used for updateView (to calculate offsets)
 */
export function drawWidgetTextOnCanvas(bounds: [number, number, number, number]): void {
  const app = window.app;
  if (!app) return;

  const canvasEl = app.canvasEl;
  const ctx = canvasEl.getContext('2d');
  if (!ctx) return;

  const nodes = app.graph._nodes;
  if (!nodes) return;

  for (const node of nodes) {
    if (!node.widgets) continue;

    for (const widget of node.widgets) {
      // Skip hidden widgets
      if (widget.hidden) continue;

      drawSingleWidgetText(ctx, widget, node, bounds);
    }
  }
}

/**
 * Draw text for a single widget onto the canvas.
 * Handles both DOM widgets (textarea/customtext) and standard canvas widgets.
 */
function drawSingleWidgetText(
  ctx: CanvasRenderingContext2D,
  widget: any,
  node: any,
  bounds: [number, number, number, number],
): void {
  const domEl = widget.element ?? widget.inputEl;

  // DOM widget (textarea/customtext) - these are the primary source of empty text
  if (domEl && (widget.type === 'textarea' || widget.type === 'customtext')) {
    drawDomWidgetText(ctx, domEl as HTMLElement, widget, node, bounds);
    return;
  }

  // Standard text-value widgets (combo, string, text) that are drawn on canvas
  // but may need their value text re-rendered if the draw method doesn't show value
  if (widget.type === 'combo' || widget.type === 'string' || widget.type === 'text') {
    drawCanvasWidgetValue(ctx, widget, node, bounds);
    return;
  }
}

/**
 * Draw text content of a DOM widget (textarea/customtext) onto the canvas.
 *
 * These widgets use HTML elements overlaid on the canvas for editing.
 * During export, the DOM element position doesn't match the canvas rendering,
 * so we need to draw the text content directly on the canvas.
 */
function drawDomWidgetText(
  ctx: CanvasRenderingContext2D,
  domEl: HTMLElement,
  widget: any,
  node: any,
  _bounds: [number, number, number, number],
): void {
  const value = widget.value ?? (domEl as HTMLInputElement & HTMLElement).value;
  if (value === undefined || value === null) return;

  const text = String(value);
  if (!text) return;

  // Calculate widget position from the DOM element style
  // In the new frontend, the dom-widget wrapper is positioned relative to the canvas
  const domWrapper = (domEl.closest('.dom-widget') ?? domEl) as HTMLElement;

  // Parse CSS value robustly:
  // - Handles "123px", "calc(...)", pure numbers, etc.
  // - Returns 0 for unparseable values (like "auto", "inherit", etc.)
  function parseCssValue(val: string | undefined, fallback: number): number {
    if (!val) return fallback;
    // Remove calc() wrapper if present
    let s = val.trim();
    if (s.startsWith('calc(') && s.endsWith(')')) {
      s = s.slice(5, -1).trim();
    }
    // Remove "px" suffix
    s = s.replace(/px$/i, '');
    // Try to evaluate simple arithmetic in calc expressions
    try {
      // Replace CSS calc tokens with JS operators
      const expr = s.replace(/(\d+(?:\.\d+)?)\s*%/g, (_, num) => `${parseFloat(num) / 100}`);
      // Safety: only allow numbers, +, -, *, /, (, ), and whitespace
      if (/^[\d\s+\-*/().]+$/.test(expr)) {
        const result = new Function(`return (${expr})`)();
        if (isFinite(result)) return result;
      }
    } catch {
      // Fall through
    }
    // Fallback: try parseInt
    const n = parseInt(s);
    return isNaN(n) ? fallback : n;
  }

  const domLeft = parseCssValue(domWrapper.style.left, 0);
  const domTop = parseCssValue(domWrapper.style.top, 0);
  const domWidth = parseCssValue(domWrapper.style.width, (node.size?.[0] ?? 200) - 20);
  const domHeight = parseCssValue(domWrapper.style.height, 100);

  // Determine x, y position - prefer DOM style, fallback to getBoundingClientRect
  let x = domLeft;
  let y = domTop;
  let positionFromStyle = false;

  if (domLeft !== 0 || domTop !== 0) {
    x = domLeft;
    y = domTop;
    positionFromStyle = true;
  } else {
    // Fall back to getBoundingClientRect for actual rendered position
    try {
      const rect = domEl.getBoundingClientRect();
      const canvasRect = (window.app?.canvasEl as HTMLCanvasElement)?.getBoundingClientRect();
      if (canvasRect) {
        // Convert screen coordinates to canvas coordinates (undo the DragAndScale transform)
        const app = window.app!;
        const ds = app.canvas.ds as any;
        const canvasScale = ds.scale;
        const canvasOffset = ds.offset as [number, number];
        x = (rect.left - canvasRect.left) / canvasScale + canvasOffset[0];
        y = (rect.top - canvasRect.top) / canvasScale + canvasOffset[1];
      } else {
        positionFromStyle = false;
      }
    } catch {
      positionFromStyle = false;
    }
  }

  // Final fallback: calculate from node position and widget y/last_y
  if (!positionFromStyle) {
    if (widget.y !== undefined) {
      x = node.pos[0] + 10;
      y = node.pos[1] + widget.y;
    } else if (widget.last_y !== undefined) {
      x = node.pos[0] + 10;
      y = node.pos[1] + widget.last_y;
    }
  }

  // Reset transform to identity before drawing text.
  // The positions from DOM style/left/top are already in canvas coordinates,
  // matching the reference implementations' "resetTransform: true" approach.
  // Using resetTransform instead of setTransform(scale,...) avoids
  // double-scaling the coordinates.
  ctx.save();
  ctx.resetTransform();

  // Get computed styles from the actual DOM element for proper text rendering
  let bgColor = '#222';
  let textColor = '#fff';
  let font = '12px sans-serif';
  let lineHeight = 14;

  try {
    const style = window.getComputedStyle(domEl, null);
    bgColor = style.getPropertyValue('background-color') || '#222';
    textColor = style.getPropertyValue('color') || '#fff';
    font = style.getPropertyValue('font') || '12px sans-serif';
    const fontSize = parseFloat(style.getPropertyValue('font-size')) || 12;
    lineHeight = fontSize * 1.2;
  } catch {
    // Use defaults if getComputedStyle fails
  }

  // Draw background (covers the empty area left by the hidden DOM element)
  ctx.fillStyle = bgColor;
  ctx.fillRect(x, y, domWidth, domHeight);

  // Draw text content
  ctx.fillStyle = textColor;
  ctx.font = font;

  const lines = text.split('\n');
  let startY = y + lineHeight;
  const maxWidth = domWidth - 8;

  for (const line of lines) {
    wrapText(ctx, line, x + 4, startY, maxWidth, lineHeight);
    startY += lineHeight;
  }

  ctx.restore();
}

/**
 * Draw the value of a standard canvas widget (combo/string/text) onto the canvas.
 *
 * These widgets are normally drawn by the canvas renderer, but in some cases
 * the value text may not be rendered correctly after the canvas transform is reset.
 * This function ensures the value text is visible in the exported image.
 */
function drawCanvasWidgetValue(
  ctx: CanvasRenderingContext2D,
  widget: any,
  node: any,
  _bounds: [number, number, number, number],
): void {
  const value = widget.value;
  if (value === undefined || value === null || value === '') return;

  const text = String(value);
  if (!text) return;

  // Calculate widget position
  // widget.y is the y position relative to the node
  let x: number;
  let y: number;

  if (widget.y !== undefined) {
    x = node.pos[0] + 10;
    y = node.pos[1] + widget.y;
  } else if (widget.last_y !== undefined) {
    x = node.pos[0] + 10;
    y = node.pos[1] + widget.last_y;
  } else {
    // Try getBoundingClientRect for widgets without y/last_y
    try {
      const widgetEl = widget.element ?? widget.inputEl;
      if (widgetEl) {
        const rect = widgetEl.getBoundingClientRect();
        const canvasEl = window.app?.canvasEl as HTMLCanvasElement;
        if (canvasEl) {
          const canvasRect = canvasEl.getBoundingClientRect();
          const app = window.app!;
          const ds = app.canvas.ds as any;
          const canvasScale = ds.scale;
          const canvasOffset = ds.offset as [number, number];
          x = (rect.left - canvasRect.left) / canvasScale + canvasOffset[0];
          y = (rect.top - canvasRect.top) / canvasScale + canvasOffset[1];
        } else {
          return;
        }
      } else {
        return;
      }
    } catch {
      return;
    }
  }

  ctx.save();
  // Reset transform to identity before drawing text.
  // The positions we calculated (widget.y/last_y or getBoundingClientRect) are
  // already in canvas coordinates, so we draw directly without additional scaling.
  ctx.resetTransform();

  // Use a consistent style that matches ComfyUI's default widget rendering
  const widgetWidth = (node.size?.[0] ?? 200) - 20;
  const widgetHeight = widget.computedHeight ?? 20;

  // Draw value text for combo/string/text widgets
  if (widget.type === 'combo') {
    // Combo widgets: draw background and value
    ctx.fillStyle = '#222';
    ctx.fillRect(x, y, widgetWidth, widgetHeight);
    ctx.fillStyle = '#fff';
    ctx.font = '12px sans-serif';
    ctx.fillText(text, x + 4, y + 14);
  } else if (widget.type === 'string' || widget.type === 'text') {
    // String/text widgets: draw value text directly
    ctx.fillStyle = '#fff';
    ctx.font = '12px sans-serif';
    const lines = text.split('\n');
    let startY = y + 14;
    for (const line of lines) {
      wrapText(ctx, line, x + 4, startY, widgetWidth - 8, 14);
      startY += 14;
    }
  }

  ctx.restore();
}

/**
 * Initialize ComfyWidgets.STRING override for canvas text rendering.
 *
 * This is kept for backward compatibility with old ComfyUI frontend versions
 * that still use customtext widget type with widget.draw().
 *
 * In the new React-based frontend, this override may not find ComfyWidgets.STRING
 * or the widget type may be 'textarea' instead of 'customtext'. In that case,
 * the drawWidgetTextOnCanvas() approach handles text rendering.
 */
export function initComfyWidgetsForExport(): void {
  const ComfyWidgets = (window as any).ComfyWidgets;
  if (!ComfyWidgets?.STRING) {
    // New frontend may not expose ComfyWidgets.STRING - that's OK,
    // we use drawWidgetTextOnCanvas() instead
    return;
  }

  const stringWidget = ComfyWidgets.STRING;

  ComfyWidgets.STRING = function (this: any, ...args: any[]): any {
    const w = stringWidget.apply(this, args);

    // Override draw for both 'customtext' (old) and 'textarea' (new) types
    if (w?.widget) {
      const wt = w.widget;
      if (wt.type === 'customtext' || wt.type === 'textarea') {
        const originalDraw = wt.draw?.bind(wt);
        wt.draw = function (ctx: CanvasRenderingContext2D, ...drawArgs: any[]) {
          // Call original draw first
          if (originalDraw) {
            originalDraw(ctx, ...drawArgs);
          }

          const inputEl = wt.inputEl ?? wt.element;
          if (!inputEl || inputEl.hidden) return;

          // Only draw text when __sr_getDrawTextConfig is set (during export)
          // This global is set by the old-style export flow
          if ((window as any).__sr_getDrawTextConfig) {
            const config = (window as any).__sr_getDrawTextConfig(ctx, wt);
            if (!config) return;

            const t = ctx.getTransform();
            ctx.save();

            if (config.resetTransform) {
              ctx.resetTransform();
            }

            const style = window.getComputedStyle(inputEl, null);
            const x = config.x;
            const y = config.y;
            const domWrapper = (inputEl.closest('.dom-widget') ?? inputEl) as HTMLElement;
            let w = parseInt(domWrapper.style.width);
            if (w === 0) {
              w = (wt.node?.size?.[0] || 200) - 20;
            }
            const h = parseInt(domWrapper.style.height) || 100;

            ctx.fillStyle = style.getPropertyValue('background-color') || '#222';
            ctx.fillRect(x, y, w, h);

            ctx.fillStyle = style.getPropertyValue('color') || '#fff';
            ctx.font = style.getPropertyValue('font') || '12px sans-serif';

            const line = (t.d || 1) * 12;
            const split = ((inputEl as HTMLInputElement).value ?? '').split('\n');
            let start = y;
            for (const l of split) {
              start += line;
              wrapText(ctx, l, x + 4, start, w - 8, line);
            }

            ctx.restore();
          }
        };
      }
    }
    return w;
  };
}

/**
 * Get the ComfyApp instance from the global window
 */
function getApp(): NonNullable<typeof window.app> {
  const app = window.app;
  if (!app) throw new Error('ComfyApp not available');
  return app;
}

/**
 * Calculate bounding box of all nodes in the current graph
 * Returns [minX, minY, maxX, maxY] with 100px padding
 */
export function getBounds(): [number, number, number, number] {
  const app = getApp();
  const nodes = app.graph._nodes;

  const bounds: [number, number, number, number] = [99999, 99999, -99999, -99999];

  for (const n of nodes) {
    const x = n.pos[0];
    const y = n.pos[1];
    const b = n.getBounding();
    const r = x + b[2];
    const btm = y + b[3];

    if (x < bounds[0]) bounds[0] = x;
    if (y < bounds[1]) bounds[1] = y;
    if (r > bounds[2]) bounds[2] = r;
    if (btm > bounds[3]) bounds[3] = btm;
  }

  // Padding: 200px to ensure node title bars, widget labels, and other
  // elements that extend beyond the node bounding box are captured
  bounds[0] -= 200;
  bounds[1] -= 200;
  bounds[2] += 200;
  bounds[3] += 200;

  return bounds;
}

/**
 * Save the current canvas viewport state
 */
export function saveCanvasState(): CanvasState {
  const app = getApp();
  const canvas = app.canvas;
  const ds = canvas.ds as any;
  const canvasEl = app.canvasEl;

  return {
    scale: ds.scale as number,
    offset: [...(ds.offset as [number, number])] as [number, number],
    width: canvasEl.width,
    height: canvasEl.height,
    transform: canvasEl.getContext('2d')!.getTransform(),
  };
}

/**
 * Restore a previously saved canvas state
 */
export function restoreCanvasState(state: CanvasState): void {
  const app = getApp();
  const canvas = app.canvas;
  const ds = canvas.ds as any;
  const canvasEl = app.canvasEl;

  ds.scale = state.scale;
  canvasEl.width = state.width;
  canvasEl.height = state.height;
  ds.offset = state.offset;
  canvasEl.getContext('2d')!.setTransform(state.transform);
}

/**
 * Update the canvas view to render the full workflow within given bounds
 */
export function updateView(bounds: [number, number, number, number]): void {
  const app = getApp();
  const canvas = app.canvas;
  const ds = canvas.ds as any;
  const canvasEl = app.canvasEl;

  const scale = window.devicePixelRatio || 1;
  // Set scale to 1 so all nodes render at 1:1 scale
  // (without this, some nodes may be outside the viewport due to zoom)
  ds.scale = 1;
  canvasEl.width = (bounds[2] - bounds[0]) * scale;
  canvasEl.height = (bounds[3] - bounds[1]) * scale;
  ds.offset = [-bounds[0], -bounds[1]];
  canvasEl.getContext('2d')!.setTransform(scale, 0, 0, scale, 0, 0);
}

/**
 * Trigger a full canvas redraw
 */
export function drawCanvas(clearCanvas = true, drawForeground = true): void {
  const app = getApp();
  app.canvas.draw(clearCanvas, drawForeground);
}

/**
 * Serialize the current graph to a JSON string
 */
export function getWorkflowJson(): string {
  const app = getApp();
  return JSON.stringify(app.graph.serialize());
}

/**
 * Download a Blob as a file
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  Object.assign(a, {
    href: url,
    download: filename,
    style: 'display: none',
  });
  document.body.append(a);
  a.click();
  setTimeout(() => {
    a.remove();
    URL.revokeObjectURL(url);
  }, 0);
}

/**
 * Convert a canvas to a PNG Blob
 */
export async function canvasToPngBlob(): Promise<Blob> {
  const app = getApp();
  return new Promise(resolve => {
    app.canvasEl.toBlob(blob => {
      if (!blob) throw new Error('Failed to create PNG blob');
      resolve(blob);
    }, 'image/png');
  });
}
