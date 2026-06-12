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

        const w = widget as any;
        const domEl = w.element ?? w.inputEl;
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

        // Check if any images within the element have finished loading
        // This ensures model preview images and other remote resources
        // are fully loaded before we capture the screenshot.
        const images = el.querySelectorAll('img');
        for (const img of images) {
          if (!img.complete || (img.naturalWidth === 0 && img.src)) {
            allReady = false;
            break;
          }
        }
        if (!allReady) break;

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

  // Hide all DOM widget elements before drawing text on canvas.
  // This prevents double-rendering: once by DOM elements overlaid on the canvas,
  // and once by our manual canvas drawing below.
  // After the canvas is captured (toBlob/toDataURL), DOM elements are not
  // included in the output, so hiding them doesn't affect the final image.
  // However, during the capture process, visible DOM elements cause visual
  // artifacts (double text) and may interfere with the canvas rendering.
  const hiddenDomElements: { el: HTMLElement; origVisibility: string }[] = [];

  for (const node of nodes) {
    if (!node.widgets) continue;

    for (const widget of node.widgets) {
      if (widget.hidden) continue;

      const domEl = (widget as any).element ?? (widget as any).inputEl;
      if (!domEl) continue;

      // Only process DOM widgets (textarea/customtext)
      if (widget.type !== 'textarea' && widget.type !== 'customtext') continue;

      // Get the dom-widget wrapper element
      const wrapper = (domEl.closest('.dom-widget') ?? domEl) as HTMLElement;

      // Save original visibility and hide the wrapper
      const origVisibility = wrapper.style.visibility;
      hiddenDomElements.push({ el: wrapper, origVisibility });
      wrapper.style.visibility = 'hidden';

      drawSingleWidgetText(ctx, widget, node, bounds);
    }
  }

  // Restore DOM widget element visibility after drawing
  for (const { el, origVisibility } of hiddenDomElements) {
    el.style.visibility = origVisibility;
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
 *
 * Position calculation follows the reference implementations:
 * - pythongosssss/SvgWorkflowImage: parseInt(domWrapper.style.left/top), resetTransform=true
 * - pythongosssss/PngWorkflowImage: x=10, y=widget.last_y+10, resetTransform=false
 *
 * We use a two-strategy approach:
 * 1. If domWrapper.style.left/top are available → use them with resetTransform=true
 *    (these are in canvas pixel coordinates, identity transform)
 * 2. Otherwise, fall back to widget.last_y with resetTransform=false
 *    (these are in logical coordinates, drawn under the scale transform)
 */
/**
 * Draw text content of a DOM widget (textarea/customtext) onto the canvas.
 *
 * These widgets use HTML elements overlaid on the canvas for editing.
 * During export, the DOM element position doesn't match the canvas rendering,
 * so we need to draw the text content directly on the canvas.
 *
 * Position calculation follows the reference implementations:
 * - pythongosssss/SvgWorkflowImage: parseInt(domWrapper.style.left/top), resetTransform=true
 * - pythongosssss/PngWorkflowImage: x=10, y=widget.last_y+10, resetTransform=false
 */
function drawDomWidgetText(
  ctx: CanvasRenderingContext2D,
  domEl: HTMLElement,
  widget: any,
  _node: any,
  _bounds: [number, number, number, number],
): void {
  const value = widget.value ?? (domEl as HTMLInputElement & HTMLElement).value;
  if (value === undefined || value === null) return;

  const text = String(value);
  if (!text) return;

  // Get the dom-widget wrapper
  const domWrapper = (domEl.closest('.dom-widget') ?? domEl) as HTMLElement;

  // Use getBoundingClientRect on both the wrapper and the canvas to convert
  // viewport-relative CSS pixel coordinates into canvas-relative logical
  // coordinates. This avoids issues with CSS transforms, iframes, or scroll.
  const wrapperRect = domWrapper.getBoundingClientRect();
  const canvasEl = (window as any).app?.canvasEl;
  const canvasRect = canvasEl?.getBoundingClientRect?.() ?? { left: 0, top: 0 };

  // canvasX = wrapper CSS-X minus canvas CSS-X
  let x = wrapperRect.left - canvasRect.left;
  let y = wrapperRect.top - canvasRect.top;

  // Get widget dimensions from DOM style
  const domWidth = wrapperRect.width;

  // Get the current canvas transform to determine the scale factor t.d
  const t = (window as any).app?.canvasEl?.getContext('2d')?.getTransform();
  const scale = t?.d || 1;

  // Line height follows pythongosssss' formula: line = t.d * 12
  // This accounts for the canvas transform scale factor
  const line = scale * 12;

  // Calculate domHeight based on actual text lines to ensure the background
  // rectangle covers all text content. The DOM element's style.height is
  // preferred, but if unavailable, we estimate from the number of text lines.
  const textLines = text.split('\n');
  const domHeight = wrapperRect.height || Math.max(textLines.length * line + 10, 20);

  // Get computed styles from the actual DOM element for proper text rendering
  let bgColor = '#222';
  let textColor = '#ffffff';
  let font = '12px sans-serif';

  try {
    const style = window.getComputedStyle(domEl, null);
    bgColor = style.getPropertyValue('background-color') || '#222';
    textColor = style.getPropertyValue('color') || '#ffffff';
    font = style.getPropertyValue('font') || '12px sans-serif';
  } catch {
    // Use defaults if getComputedStyle fails
  }

  // Draw background (covers the empty area left by the hidden DOM element)
  ctx.fillStyle = bgColor;
  ctx.fillRect(x, y, domWidth, domHeight);

  // Draw text content
  ctx.fillStyle = textColor;
  ctx.font = font;

  const maxWidth = domWidth - 8;
  const split = text.split('\n');
  let start = y;
  for (const l of split) {
    start += line;
    wrapText(ctx, l, x + 4, start, maxWidth, line);
  }
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

  // Position calculation: we're drawing after drawCanvas(), so the canvas
  // transform is setTransform(dpr, 0, 0, dpr, 0, 0) — there's no per-node
  // translation applied. We must convert node-relative coordinates to canvas
  // logical coordinates using the LiteGraph coordinate conversion method:
  //   node.last_mouse_x = node.pos[0] + (x * ds.scale) + ds.offset[0]
  //   node.last_mouse_y = node.pos[1] + (y * ds.scale) + ds.offset[1]
  // After updateView(): ds.scale=1, ds.offset = [-bounds[0], -bounds[1]],
  // so: canvasLogicalX = nodePos.x + x - bounds[0]
  //     canvasLogicalY = nodePos.y + y - bounds[1]
  //
  // For widgets, the LiteGraph reordering formula is:
  //   last_y = y + w.last_y - w.last_y + height
  //   y = last_y - w.last_y + w.last_y - height ???
  //   Actually: last_y = y + widget.last_y - widget.last_y + height
  //   so: y = last_y ??
  //   Better: use the formula from pythongosssss PngWorkflowImage:
  //   x=10, y=(widget.last_y ?? widget.y ?? 0) + LiteGraph.NODE_TITLE_HEIGHT + 20
  const ds = (window as any).app?.canvas?.ds;
  const offsetX = ds?.offset?.[0] ?? 0;
  const offsetY = ds?.offset?.[1] ?? 0;
  const nodePosX = node.pos?.[0] ?? 0;
  const nodePosY = node.pos?.[1] ?? 0;

  const x = 10 + nodePosX + offsetX;
  // Note: widget.last_y is relative to the NODE CONTENT AREA (below title bar).
  // We add 20 for padding, similar to pythongosssss approach.
  const y = (widget.last_y ?? widget.y ?? 0) + 20 + nodePosY + offsetY;

  // Get widget dimensions
  const widgetWidth = (node.size?.[0] ?? 240) - 20;

  // Use scale-aware line height: line = t.d * 12
  const t = (window as any).app?.canvasEl?.getContext('2d')?.getTransform();
  const scale = t?.d || 1;
  const line = scale * 12;

  // Calculate widgetHeight based on actual text lines
  const textLines = text.split('\n');
  const widgetHeight = Math.max(textLines.length * line + 10, 20);

  // Save current state
  ctx.save();

  // Draw value text for combo/string/text widgets
  if (widget.type === 'combo') {
    // Combo widgets: draw background and value
    ctx.fillStyle = '#222';
    ctx.fillRect(x, y, widgetWidth, widgetHeight);
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px sans-serif';
    const fontSize = 12;
    const baselineY = y + (widgetHeight + fontSize) / 2;
    ctx.fillText(text, x + 4, baselineY);
  } else if (widget.type === 'string' || widget.type === 'text') {
    // String/text widgets: draw value text directly
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px sans-serif';
    const split = text.split('\n');
    let start = y;
    for (const l of split) {
      start += line;
      wrapText(ctx, l, x + 4, start, widgetWidth - 8, line);
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
    // This is a legacy path for old ComfyUI frontends that use widget.draw()
    // to render customtext widgets. In the new React-based frontend, text
    // widgets use DOM elements (textarea) overlaid on the canvas, and the
    // primary text rendering mechanism is drawWidgetTextOnCanvas().
    if (w?.widget) {
      const wt = w.widget;
      if (wt.type === 'customtext') {
        // Only override for 'customtext' type (old frontend)
        // Do NOT override for 'textarea' type (new frontend) because:
        // 1. In the new frontend, textarea widgets are DOM elements that
        //    are not drawn via widget.draw() at all
        // 2. Overriding draw() for textarea would interfere with the DOM
        //    widget lifecycle and cause rendering artifacts
        // 3. drawWidgetTextOnCanvas() handles textarea text rendering
        //    after drawCanvas(), which is the correct approach
        const originalDraw = wt.draw?.bind(wt);
        wt.draw = function (ctx: CanvasRenderingContext2D, ...drawArgs: any[]) {
          // Call original draw first
          if (originalDraw) {
            originalDraw(ctx, ...drawArgs);
          }
          // For 'customtext' in old frontend, the widget.draw() method
          // is the sole renderer, so no additional action is needed here.
          // Text rendering during export is handled by drawWidgetTextOnCanvas().
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
