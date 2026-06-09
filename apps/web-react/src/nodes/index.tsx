/**
 * 节点扩展总表
 */

import type { ComfyExtension } from '@comfyorg/comfyui-frontend-types';
import StringDynList2 from './string-dyn-list2';
import ParamHub from './param-hub';
import ParamPort from './param-port';

export const NODE_EXTENSIONS: ComfyExtension[] = [StringDynList2(), ParamHub(), ParamPort()];
