/**
 * PNG Workflow Image Export
 *
 * Analyzed from:
 * - pythongosssss/ComfyUI-Custom-Scripts: PngWorkflowImage
 *   Uses canvas.toBlob() -> embeds workflow via PNG tEXt chunk with keyword "workflow"
 *   Inserts the tEXt chunk right after the IHDR chunk (before IDAT)
 *
 * - BobRandomNumber/ComfyUI-QoL-Pack: QoL_PngWorkflowImage
 *   Nearly identical implementation, same tEXt chunk embedding approach
 *   Always embeds workflow (no option to skip)
 *
 * Both implementations:
 * 1. Save canvas state -> update view to fit all nodes -> draw canvas
 * 2. Convert canvas to PNG blob
 * 3. Parse PNG binary, insert tEXt chunk with workflow JSON
 * 4. Download the modified blob
 * 5. Restore canvas state and redraw
 *
 * PNG tEXt chunk binary format:
 *   [4-byte data length][tEXt][keyword\0value][4-byte CRC]
 *   The 4-byte data length field stores the length of only the data portion
 *   (excluding the 4-byte chunk type).
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

/**
 * Calculate CRC32 checksum for PNG chunk validation.
 * The CRC is computed over the chunk type and chunk data.
 */
function crc32(data: Uint8Array): number {
  const table = getCrcTable();
  let crc = 0 ^ -1;
  for (let i = 0; i < data.byteLength; i++) {
    crc = (crc >>> 8) ^ table[(crc ^ data[i]!) & 0xff]!;
  }
  return (crc ^ -1) >>> 0;
}

/**
 * Convert a 32-bit number to a big-endian Uint8Array (4 bytes)
 */
function n2b(n: number): Uint8Array {
  return new Uint8Array([(n >> 24) & 0xff, (n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]);
}

/**
 * Concatenate multiple Uint8Arrays into a single Uint8Array
 */
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

/**
 * Embed workflow JSON into a PNG blob using a tEXt chunk.
 *
 * PNG file structure:
 *   [8-byte Signature][IHDR chunk][...other chunks...][IEND chunk]
 *
 * Each chunk structure:
 *   [4-byte data length][4-byte chunk type][data bytes][4-byte CRC]
 *
 * We insert a tEXt chunk right after the IHDR chunk:
 *   tEXt chunk: [data length][tEXt]["workflow\0" + JSON][CRC]
 *
 * @param pngBlob - The original PNG blob from canvas
 * @param workflow - The workflow JSON string to embed
 * @returns A new PNG Blob with the embedded tEXt chunk
 */
async function embedWorkflowInPng(pngBlob: Blob, workflow: string): Promise<Blob> {
  const buffer = await pngBlob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);

  // Build tEXt chunk data: chunk type + keyword + null byte + value
  // The keyword is "workflow", followed by a null byte, then the JSON value
  const chunkData = new TextEncoder().encode(`tEXtworkflow\0${workflow}`);

  // PNG chunk length field = size of data portion only (excluding chunk type)
  const chunkDataLen = chunkData.byteLength - 4; // subtract 4 bytes for "tEXt"

  // Compute CRC over the entire chunk data (type + data)
  const chunkCrc = crc32(chunkData);

  // Assemble the complete tEXt chunk:
  //   [4-byte data length][chunk type + data][4-byte CRC]
  const chunk = concat(n2b(chunkDataLen), chunkData, n2b(chunkCrc));

  // Find insertion point: right after the IHDR chunk
  // PNG signature is 8 bytes, IHDR starts at offset 8
  // IHDR data length is at offset 8 (first 4 bytes of the chunk after signature)
  const ihdrDataLen = view.getUint32(8);
  // Position after IHDR = 8 (signature) + 4 (length) + 4 (type) + ihdrDataLen + 4 (CRC)
  const insertPos = 8 + 4 + 4 + ihdrDataLen + 4;

  // Insert tEXt chunk after IHDR
  const result = concat(bytes.subarray(0, insertPos), chunk, bytes.subarray(insertPos));

  return new Blob([result.buffer as ArrayBuffer], { type: 'image/png' });
}

/**
 * Export the current workflow as a PNG image.
 *
 * @param includeWorkflow - Whether to embed the workflow JSON in the PNG tEXt chunk
 */
export async function exportPng(includeWorkflow: boolean): Promise<void> {
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
    //    (increased from 500ms because complex nodes with remote models need more time)
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
    // We must manually draw their text content onto the canvas.
    drawWidgetTextOnCanvas(bounds);

    // Get PNG blob from canvas
    let blob = await canvasToPngBlob();

    // Optionally embed workflow data
    if (includeWorkflow) {
      const workflow = getWorkflowJson();
      blob = await embedWorkflowInPng(blob, workflow);
    }

    // Download
    downloadBlob(blob, 'workflow.png');
  } finally {
    // Always restore canvas state
    if (savedState) {
      restoreCanvasState(savedState);
      drawCanvas(true, true);
    }
  }
}
