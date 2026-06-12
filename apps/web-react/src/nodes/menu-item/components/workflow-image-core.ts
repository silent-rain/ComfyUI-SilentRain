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
 * Capture workflow canvas after setting up export view and drawing widgets.
 * Returns a cleanup function that restores the original canvas state.
 */
export async function captureWorkflowCanvas(): Promise<{
  canvasEl: HTMLCanvasElement;
  restore: () => void;
}> {
  const savedState = saveCanvasState();
  const bounds = getBounds();
  updateView(bounds);
  drawCanvas(true, true);

  await new Promise<void>(resolve => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => setTimeout(resolve, 1000));
    });
  });
  await waitForDomWidgetsReady(3000);
  drawWidgetTextOnCanvas(bounds);

  return {
    canvasEl: getApp().canvasEl,
    restore: () => {
      restoreCanvasState(savedState);
      drawCanvas(true, true);
    },
  };
}

/**
 * Unified error handler for export operations.
 */
export function handleExportError(error: unknown, context: string): void {
  const msg = error instanceof Error ? error.message : String(error);
  console.error(`[WorkflowImage] ${context} failed:`, error);
  alert(`Export failed: ${context}\n${msg}`);
}

/**
 * Draw text for a single widget onto the canvas.
 */
function drawSingleWidgetText(
  ctx: CanvasRenderingContext2D,
  widget: any,
  _node: any,
  _bounds: [number, number, number, number],
): void {
  const domEl = widget.element ?? widget.inputEl;
  if (!domEl) return;
  drawDomWidgetText(ctx, domEl as HTMLElement, widget);
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
 */
function drawDomWidgetText(ctx: CanvasRenderingContext2D, domEl: HTMLElement, widget: any): void {
  const value = widget.value ?? (domEl as HTMLInputElement & HTMLElement).value;
  if (value === undefined || value === null || value === '') return;

  const text = String(value);

  // Get the dom-widget wrapper
  const domWrapper = (domEl.closest('.dom-widget') ?? domEl) as HTMLElement;

  // Convert viewport-relative CSS pixel coordinates into canvas-relative logical coordinates
  const wrapperRect = domWrapper.getBoundingClientRect();
  const canvasEl = (window as any).app?.canvasEl;
  const canvasRect = canvasEl?.getBoundingClientRect?.() ?? { left: 0, top: 0 };

  const x = wrapperRect.left - canvasRect.left;
  let y = wrapperRect.top - canvasRect.top;
  const domWidth = wrapperRect.width;

  // Use cached scale from canvas transform
  const scale = ctx.getTransform().d || 1;
  const line = scale * 12;

  const textLines = text.split('\n');
  const domHeight = wrapperRect.height || Math.max(textLines.length * line + 10, 20);

  let bgColor = '#222';
  let textColor = '#ffffff';
  let font = '12px sans-serif';

  try {
    const style = window.getComputedStyle(domEl, null);
    bgColor = style.getPropertyValue('background-color') || '#222';
    textColor = style.getPropertyValue('color') || '#ffffff';
    font = style.getPropertyValue('font') || '12px sans-serif';
  } catch {
    // Use defaults
  }

  ctx.save();
  ctx.fillStyle = bgColor;
  ctx.fillRect(x, y, domWidth, domHeight);
  ctx.fillStyle = textColor;
  ctx.font = font;

  const maxWidth = domWidth - 8;
  for (let i = 0; i < textLines.length; i++) {
    y += line;
    wrapText(ctx, textLines[i]!, x + 4, y, maxWidth, line);
  }
  ctx.restore();
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
export async function canvasToPngBlob(canvasEl: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvasEl.toBlob(
      blob => (blob ? resolve(blob) : reject(new Error('Failed to create PNG blob'))),
      'image/png',
    );
  });
}
