/**
 * Export Image + Workflow
 *
 * Simultaneously export a PNG image (with embedded workflow) and a JSON workflow file.
 * Shows a prompt dialog for the user to name the files.
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
 * Show a custom prompt dialog for entering the filename.
 * Returns the entered name or null if cancelled.
 */
function showFilenamePrompt(): Promise<string | null> {
  return new Promise(resolve => {
    // Create overlay
    const overlay = document.createElement('div');
    overlay.style.cssText = `
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 10000;
      font-family: system-ui, -apple-system, sans-serif;
    `;

    // Create dialog
    const dialog = document.createElement('div');
    dialog.style.cssText = `
      background: #2b2b2b;
      border-radius: 12px;
      padding: 24px;
      width: 340px;
      color: #e0e0e0;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    `;

    // Title
    const title = document.createElement('div');
    title.textContent = '导出工作流';
    title.style.cssText = `
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    `;

    // Close button
    const closeBtn = document.createElement('span');
    closeBtn.textContent = '✕';
    closeBtn.style.cssText = `
      cursor: pointer;
      font-size: 20px;
      color: #888;
      line-height: 1;
    `;
    closeBtn.onclick = () => {
      cleanup();
      resolve(null);
    };
    title.appendChild(closeBtn);

    // Label
    const label = document.createElement('div');
    label.textContent = '输入文件名：';
    label.style.cssText = `
      font-size: 14px;
      color: #999;
      margin-bottom: 8px;
    `;

    // Input
    const input = document.createElement('input');
    input.type = 'text';
    input.value = 'Unsaved Workflow';
    input.style.cssText = `
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: none;
      background: #1e1e1e;
      color: #fff;
      font-size: 14px;
      outline: none;
      box-sizing: border-box;
      margin-bottom: 16px;
    `;
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter') confirmBtn.click();
    });

    // Confirm button
    const confirmBtn = document.createElement('button');
    confirmBtn.textContent = '确认';
    confirmBtn.style.cssText = `
      width: 100%;
      padding: 10px;
      border-radius: 8px;
      border: none;
      background: #3a3a3a;
      color: #fff;
      font-size: 14px;
      cursor: pointer;
      font-weight: 500;
    `;
    confirmBtn.onmouseenter = () => {
      confirmBtn.style.background = '#4a4a4a';
    };
    confirmBtn.onmouseleave = () => {
      confirmBtn.style.background = '#3a3a3a';
    };
    confirmBtn.onclick = () => {
      const name = input.value.trim();
      cleanup();
      resolve(name || null);
    };

    dialog.appendChild(title);
    dialog.appendChild(label);
    dialog.appendChild(input);
    dialog.appendChild(confirmBtn);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // Auto focus and select
    setTimeout(() => {
      input.focus();
      input.select();
    }, 10);

    function cleanup() {
      overlay.remove();
      document.removeEventListener('keydown', handleEscape);
    }

    function handleEscape(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        cleanup();
        resolve(null);
      }
    }
    document.addEventListener('keydown', handleEscape);
  });
}

/**
 * Export both PNG image (with embedded workflow) and JSON workflow file.
 * Prompts the user for a base filename.
 */
export async function exportImageAndWorkflow(): Promise<void> {
  const filename = await showFilenamePrompt();
  if (!filename) return;

  const { canvasEl, restore } = await captureWorkflowCanvas();

  try {
    const blob = await canvasToPngBlob(canvasEl);

    const arrayBuffer = await blob.arrayBuffer();
    const workflow = getWorkflowJson();
    const pngWithWorkflow = embedWorkflowInPng(new Uint8Array(arrayBuffer), workflow);

    downloadBlob(
      new Blob([pngWithWorkflow.buffer as ArrayBuffer], { type: 'image/png' }),
      `${filename}.png`,
    );
    downloadBlob(new Blob([workflow], { type: 'application/json' }), `${filename}.json`);
  } catch (err) {
    handleExportError(err, 'Image+Workflow export');
  } finally {
    restore();
  }
}
