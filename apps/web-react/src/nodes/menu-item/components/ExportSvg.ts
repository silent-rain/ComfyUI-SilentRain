/**
 * SVG Workflow Image Export
 *
 * Analyzed from:
 * - pythongosssss/ComfyUI-Custom-Scripts: SvgWorkflowImage
 *   Uses canvas2svg (C2S) library to create a true vector SVG
 *   Replaces canvas context with a C2S context, renders the graph,
 *   then serializes to SVG. Embeds workflow in a <desc> tag.
 *   Also overrides ComfyWidgets.STRING to render text in SVG properly.
 *
 * - BobRandomNumber/ComfyUI-QoL-Pack: No SVG export support
 *
 * Our simplified approach:
 *   Since canvas2svg is an external dependency and complex to integrate,
 *   we use the simpler approach of rendering the canvas to a PNG data URL
 *   and embedding it inside an SVG wrapper. The workflow JSON is embedded
 *   in a <desc> element for later import.
 *
 * For true vector SVG export, canvas2svg integration would be needed,
 * which is left as a future enhancement.
 */

import {
  type CanvasState,
  getBounds,
  saveCanvasState,
  restoreCanvasState,
  updateView,
  drawCanvas,
  drawWidgetTextOnCanvas,
  waitForDomWidgetsReady,
  getWorkflowJson,
  downloadBlob,
} from './workflow-image-core';

/**
 * Escape XML special characters for safe embedding in SVG
 */
function escapeXml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

/**
 * Unescape XML entities (used when importing SVG with embedded workflow)
 */
export function unescapeXml(s: string): string {
  return s
    .replace(/&apos;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&gt;/g, '>')
    .replace(/&lt;/g, '<')
    .replace(/&amp;/g, '&');
}

/**
 * Export the current workflow as an SVG image
 *
 * @param includeWorkflow - Whether to embed the workflow JSON in the SVG <desc> tag
 */
export async function exportSvg(includeWorkflow: boolean): Promise<void> {
  let savedState: CanvasState | undefined;

  try {
    // Save current canvas state
    savedState = saveCanvasState();

    // Update view to fit all nodes and redraw
    const bounds = getBounds();
    updateView(bounds);
    drawCanvas(true, true);

    // Wait for rendering to complete:
    // 1. Double rAF ensures the browser has painted at least one frame
    // 2. Additional 1000ms delay ensures remote/complex nodes finish rendering
    //    (increased from 500ms because complex nodes with images/models need more time)
    // 3. Poll DOM widgets to ensure they are visible and positioned
    await new Promise<void>(resolve => {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          setTimeout(resolve, 1000);
        });
      });
    });

    // Additional wait: poll until all DOM widgets are rendered or timeout
    await waitForDomWidgetsReady(3000);

    // Draw widget text content directly on canvas.
    // This is the key fix for empty text in exported images:
    // In the new ComfyUI frontend, text widgets (textarea) use DOM elements
    // overlaid on the canvas, which are invisible when we modify the transform.
    drawWidgetTextOnCanvas(bounds);

    const app = window.app!;
    const canvasEl = app.canvasEl;
    const width = canvasEl.width;
    const height = canvasEl.height;

    // Convert canvas to PNG data URL for embedding in SVG
    const dataUrl = canvasEl.toDataURL('image/png');

    // Build SVG content
    const descTag = includeWorkflow ? `\n  <desc>${escapeXml(getWorkflowJson())}</desc>` : '';

    const svgContent =
      `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">` +
      `\n  <image width="${width}" height="${height}" href="${dataUrl}"/>` +
      `${descTag}` +
      `\n</svg>`;

    // Create blob and download
    const blob = new Blob([svgContent], { type: 'image/svg+xml' });
    downloadBlob(blob, 'workflow.svg');
  } finally {
    // Always restore canvas state
    if (savedState) {
      restoreCanvasState(savedState);
      drawCanvas(true, true);
    }
  }
}
