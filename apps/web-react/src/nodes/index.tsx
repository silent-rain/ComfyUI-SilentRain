/**
 * 节点扩展总表
 */

import type { ComfyExtension } from '@comfyorg/comfyui-frontend-types';
import AnyDynList from './any-dyn-list';
import StringDynList2 from './string-dyn-list2';
import ParamHub from './param-hub';
import ParamPort from './param-port';
import MenuItems from './menu-item';

export const NODE_EXTENSIONS: ComfyExtension[] = [
  AnyDynList(),
  StringDynList2(),
  ParamHub(),
  ParamPort(),
  MenuItems(),
];
