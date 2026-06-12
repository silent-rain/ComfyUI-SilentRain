/**
 * PNG Workflow Image Export
 */
import {
  captureWorkflowCanvas,
  getWorkflowJson,
  downloadBlob,
  handleExportError,
  canvasToPngBlob,
} from './workflow-image-core';
import { embedWorkflowInPng } from './png-utils';

/**
 * Export the current workflow as a PNG image.
 *
 * @param includeWorkflow - Whether to embed the workflow JSON in the PNG tEXt chunk
 */
export async function exportPng(includeWorkflow: boolean): Promise<void> {
  const { canvasEl, restore } = await captureWorkflowCanvas();

  try {
    let blob = await canvasToPngBlob(canvasEl);

    if (includeWorkflow) {
      const arrayBuffer = await blob.arrayBuffer();
      const pngWithWorkflow = embedWorkflowInPng(new Uint8Array(arrayBuffer), getWorkflowJson());
      blob = new Blob([pngWithWorkflow.buffer as ArrayBuffer], { type: 'image/png' });
    }

    downloadBlob(blob, 'workflow.png');
  } catch (err) {
    handleExportError(err, 'PNG export');
  } finally {
    restore();
  }
}
