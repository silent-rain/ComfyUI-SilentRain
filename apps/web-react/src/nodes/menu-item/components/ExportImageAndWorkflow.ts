/**
 * Export Image + Workflow
 *
 * Simultaneously export a PNG image (with embedded workflow) and a JSON workflow file.
 * Shows a prompt dialog for the user to name the files.
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
  canvasToPngBlob,
} from './workflow-image-core';

/** CRC32 lookup table (lazy initialized) */
let crcTable: Uint32Array | null = null;

function getCrcTable(): Uint32Array {
  if (crcTable) return crcTable;

  crcTable = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    crcTable[n] = c;
  }
  return crcTable;
}

function crc32(data: Uint8Array): number {
  const table = getCrcTable();
  let crc = 0 ^ -1;
  for (let i = 0; i < data.byteLength; i++) {
    crc = (crc >>> 8) ^ table[(crc ^ data[i]!) & 0xff]!;
  }
  return (crc ^ -1) >>> 0;
}

function n2b(n: number): Uint8Array {
  return new Uint8Array([(n >> 24) & 0xff, (n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]);
}

function concat(...bufs: Uint8Array[]): Uint8Array {
  const totalSize = bufs.reduce((total, buf) => total + buf.byteLength, 0);
  const result = new Uint8Array(totalSize);
  let offset = 0;
  for (const buf of bufs) {
    result.set(buf, offset);
    offset += buf.byteLength;
  }
  return result;
}

async function embedWorkflowInPng(pngBlob: Blob, workflow: string): Promise<Blob> {
  const buffer = await pngBlob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);

  const chunkData = new TextEncoder().encode(`tEXtworkflow\0${workflow}`);
  const chunkDataLen = chunkData.byteLength - 4;
  const chunkCrc = crc32(chunkData);
  const chunk = concat(n2b(chunkDataLen), chunkData, n2b(chunkCrc));

  const ihdrDataLen = view.getUint32(8);
  const insertPos = 8 + 4 + 4 + ihdrDataLen + 4;

  const result = concat(bytes.subarray(0, insertPos), chunk, bytes.subarray(insertPos));

  return new Blob([result.buffer as ArrayBuffer], { type: 'image/png' });
}

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
      if (e.key === 'Enter') {
        confirmBtn.click();
      }
      if (e.key === 'Escape') {
        cleanup();
        resolve(null);
      }
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
    //    (increased from single rAF because complex nodes need more time)
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

    // Get PNG blob from canvas
    let pngBlob = await canvasToPngBlob();

    // Embed workflow data into PNG
    const workflow = getWorkflowJson();
    pngBlob = await embedWorkflowInPng(pngBlob, workflow);

    // Download PNG
    downloadBlob(pngBlob, `${filename}.png`);

    // Download JSON workflow
    const jsonBlob = new Blob([workflow], { type: 'application/json' });
    downloadBlob(jsonBlob, `${filename}.json`);
  } finally {
    // Always restore canvas state
    if (savedState) {
      restoreCanvasState(savedState);
      drawCanvas(true, true);
    }
  }
}
