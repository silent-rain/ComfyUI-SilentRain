const crcTable: number[] = [];

function getCrcTable(): number[] {
  if (crcTable.length === 0) {
    for (let i = 0; i < 256; i++) {
      let c = i;
      for (let k = 0; k < 8; k++) {
        c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      }
      crcTable.push(c);
    }
  }
  return crcTable;
}

function crc32(bytes: Uint8Array): number {
  const table = getCrcTable();
  let crc = -1;
  for (let i = 0; i < bytes.length; i++) {
    const byte = bytes[i]!;
    const tableEntry = table[(crc ^ byte) & 0xff]!;
    crc = tableEntry ^ (crc >>> 8);
  }
  return crc ^ -1;
}

function n2b(num: number): Uint8Array {
  const b = new Uint8Array(4);
  new DataView(b.buffer).setInt32(0, num);
  return b;
}

function concat(chunks: Uint8Array[]): Uint8Array {
  let total = 0;
  for (const c of chunks) total += c.length;
  const result = new Uint8Array(total);
  let i = 0;
  for (const c of chunks) {
    result.set(c, i);
    i += c.length;
  }
  return result;
}

/**
 * Embeds workflow JSON string into PNG tEXt chunks and returns a new PNG Uint8Array.
 * @param pngBuffer Original PNG file data as Uint8Array.
 * @param workflowStr Workflow JSON string.
 * @returns New PNG data with embedded workflow.
 */
export function embedWorkflowInPng(pngBuffer: Uint8Array, workflowStr: string): Uint8Array {
  const encoder = new TextEncoder();
  const keyword = encoder.encode('workflow\0');
  const value = encoder.encode(workflowStr);
  const textChunkData = concat([keyword, value]);
  const textChunkType = encoder.encode('tEXt');
  const textChunkCrc = crc32(concat([textChunkType, textChunkData]));
  const textChunk = concat([
    n2b(textChunkData.length),
    textChunkType,
    textChunkData,
    n2b(textChunkCrc),
  ]);

  const ihdrIndex = pngBuffer.findIndex(
    (_, i) =>
      pngBuffer[i] === 0x49 &&
      pngBuffer[i + 1] === 0x48 &&
      pngBuffer[i + 2] === 0x44 &&
      pngBuffer[i + 3] === 0x52,
  );
  if (ihdrIndex === -1) {
    throw new Error('[WorkflowImage] tEXt not found in PNG');
  }

  const ihdrLen = new DataView(pngBuffer.buffer, ihdrIndex - 4, 4).getInt32(0);
  const insertPos = ihdrIndex + 4 + ihdrLen + 4;
  const pre = pngBuffer.subarray(0, insertPos);
  const post = pngBuffer.subarray(insertPos);

  return concat([pre, textChunk, post]);
}
