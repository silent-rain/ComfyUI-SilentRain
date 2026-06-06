/**
 * 节点扩展总表
 */

import type { ComfyExtension } from '@comfyorg/comfyui-frontend-types';
// import { ParamHubExtension } from './param-hub';
// import { ParamPortExtension } from './param-port';
import StringDynList2 from './string-dyn-list2';

export const NODE_EXTENSIONS: ComfyExtension[] = [
  // ParamHubExtension,
  // ParamPortExtension,
  StringDynList2(),
];
