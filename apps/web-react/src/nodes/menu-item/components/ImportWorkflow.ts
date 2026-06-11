/**
 * Workflow Import
 *
 * Analyzed from:
 * - pythongosssss/ComfyUI-Custom-Scripts: WorkflowImage.import()
 *   Creates a hidden file input, sets accept types based on registered formats,
 *   and delegates to app.handleFile() for loading.
 *
 *   Also supports importing from:
 *   - SVG files: extracts <desc> tag content and parses as JSON
 *   - JPEG files: reads EXIF UserComment for embedded workflow
 *   - Standard PNG: ComfyUI's built-in handleFile reads tEXt chunks
 *
 * Our implementation supports:
 * - PNG with embedded workflow (handled by ComfyUI's built-in handleFile)
 * - SVG with embedded workflow (custom parsing)
 * - JSON workflow files (handled by ComfyUI's built-in handleFile)
 */

import { unescapeXml } from './ExportSvg';

/** Supported file types for workflow import */
const IMPORT_ACCEPT = '.png,.svg,.json,image/png,image/svg+xml,application/json';

let fileInput: HTMLInputElement | null = null;

/**
 * Import a workflow from a file (PNG, SVG, or JSON)
 *
 * This creates a hidden file input dialog and processes the selected file:
 * - PNG/JSON: delegates to ComfyUI's built-in app.handleFile()
 * - SVG: parses <desc> tag for embedded workflow JSON
 */
export function importWorkflow(): void {
  if (!fileInput) {
    fileInput = document.createElement('input');
    Object.assign(fileInput, {
      type: 'file',
      accept: IMPORT_ACCEPT,
      style: 'display: none',
    });
    fileInput.onchange = () => {
      if (!fileInput?.files?.length) return;
      const file = fileInput.files![0];

      if (file!.type === 'image/svg+xml' || file!.name.endsWith('.svg')) {
        handleSvgFile(file!);
      } else {
        // PNG, JSON, etc. - use ComfyUI's built-in handler
        const app = window.app;
        if (app) {
          app.handleFile(file!);
        }
      }
    };
    document.body.append(fileInput);
  }

  // Reset value so same file can be re-imported
  fileInput.value = '';
  fileInput.click();
}

/**
 * Handle SVG file import by extracting workflow from <desc> tag
 */
function handleSvgFile(file: File): void {
  const reader = new FileReader();
  reader.onload = () => {
    const svgContent = reader.result as string;
    const descEnd = svgContent.lastIndexOf('</desc>');
    if (descEnd !== -1) {
      const descStart = svgContent.lastIndexOf('<desc>', descEnd);
      if (descStart !== -1) {
        const jsonStr = svgContent.substring(descStart + 6, descEnd);
        try {
          const workflowData = JSON.parse(unescapeXml(jsonStr));
          const app = window.app;
          if (app) {
            app.loadGraphData(workflowData);
          }
        } catch (e) {
          console.error('[WorkflowImage] Failed to parse SVG embedded workflow:', e);
        }
      }
    }
  };
  reader.readAsText(file);
}
