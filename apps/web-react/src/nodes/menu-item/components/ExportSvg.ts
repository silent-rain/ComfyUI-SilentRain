/**
 * SVG Workflow Image Export
 */
import {
  captureWorkflowCanvas,
  getWorkflowJson,
  downloadBlob,
  handleExportError,
} from './workflow-image-core';

function escapeXml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

export function unescapeXml(s: string): string {
  return s
    .replace(/&apos;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&gt;/g, '>')
    .replace(/&lt;/g, '<')
    .replace(/&amp;/g, '&');
}

export async function exportSvg(includeWorkflow: boolean): Promise<void> {
  const { canvasEl, restore } = await captureWorkflowCanvas();

  try {
    const width = canvasEl.width;
    const height = canvasEl.height;
    const dataUrl = canvasEl.toDataURL('image/png');

    const descTag = includeWorkflow ? `\n  <desc>${escapeXml(getWorkflowJson())}</desc>` : '';
    const svgContent =
      `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">` +
      `\n  <image width="${width}" height="${height}" href="${dataUrl}"/>` +
      `${descTag}` +
      `\n</svg>`;

    downloadBlob(new Blob([svgContent], { type: 'image/svg+xml' }), 'workflow.svg');
  } catch (err) {
    handleExportError(err, 'SVG export');
  } finally {
    restore();
  }
}
