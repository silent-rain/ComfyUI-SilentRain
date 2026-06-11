/**
 * JSON Workflow Export
 *
 * Export the current workflow as a standalone JSON file.
 * This is a convenience feature not present in the original
 * workflowImage.js implementations, but commonly needed.
 */

import { getWorkflowJson, downloadBlob } from './workflow-image-core';

/**
 * Export the current workflow as a JSON file
 */
export function exportJson(): void {
  const workflow = getWorkflowJson();
  const blob = new Blob([workflow], { type: 'application/json' });
  downloadBlob(blob, 'workflow.json');
}
