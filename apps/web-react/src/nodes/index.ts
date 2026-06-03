/**
 * 节点扩展总表 —— 新增节点时只需：
 *   1) 创建 src/nodes/<your-node>/index.tsx 并导出 ComfyExtension
 *   2) 在此处 import 并加入 NODE_EXTENSIONS 数组
 */

import type { ComfyExtension } from '../core';
import { ParamHubExtension } from './param-hub';
import { ParamPortExtension } from './param-port';

export const NODE_EXTENSIONS: ComfyExtension[] = [
  ParamHubExtension,
  ParamPortExtension,
];
