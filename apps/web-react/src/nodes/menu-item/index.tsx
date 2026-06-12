/* menu-item Node - Workflow Image Export/Import Menu Integration
 *
 * 参考实现:
 * https://github.com/pythongosssss/ComfyUI-Custom-Scripts/blob/main/web/js/workflowImage.js
 * https://github.com/BobRandomNumber/ComfyUI-QoL-Pack/blob/main/web/js/QoL_WorkflowImage.js
 */

import type { ComfyExtension } from '@comfyorg/comfyui-frontend-types';

import { exportPng } from './components/ExportPng';
import { exportSvg } from './components/ExportSvg';
import { exportJson } from './components/ExportJson';
import { exportImageAndWorkflow } from './components/ExportImageAndWorkflow';
import { importWorkflow } from './components/ImportWorkflow';

const NODE_NAME = 'MenuItems';

// ── Context Menu Types ────────────────────────────────────────────────
// Adapted from @comfyorg/comfyui-frontend-types (global declarations)
// These types are not explicitly exported from the package, so we define
// them locally based on the interface structure in the .d.ts file.

interface IContextMenuOptions<TValue = unknown> {
  title?: string;
  callback?: (
    this: HTMLDivElement,
    value?: TValue,
    options?: unknown,
    event?: MouseEvent,
    previous_menu?: unknown,
    extra?: unknown,
  ) => void | boolean | Promise<void | boolean>;
  ignore_item_callbacks?: boolean;
  event?: MouseEvent;
  className?: string;
}

interface IContextMenuSubmenu<TValue = unknown> extends IContextMenuOptions<TValue> {
  options: readonly (string | IContextMenuValue<TValue> | null)[];
}

interface IContextMenuValue<TValue = unknown, TExtra = unknown, TCallbackValue = unknown> {
  value?: TValue;
  content: string | undefined;
  has_submenu?: boolean;
  disabled?: boolean;
  submenu?: IContextMenuSubmenu<TValue>;
  property?: string;
  type?: string;
  slot?: any;
  callback?(
    this: HTMLDivElement,
    value?: TCallbackValue,
    options?: unknown,
    event?: MouseEvent,
    previous_menu?: unknown,
    extra?: TExtra,
  ): void | boolean | Promise<void | boolean>;
}

// ── Menu Builder ──────────────────────────────────────────────────────

/**
 * Build the "Workflow Image" canvas context menu items.
 *
 * Menu structure (matching the reference screenshot):
 *
 *   Workflow Image
 *     Import
 *     Export
 *       png (with embedded workflow)
 *       png (no embedded workflow)
 *       svg (with embedded workflow)
 *       svg (no embedded workflow)
 *       json
 */
function buildWorkflowImageMenu(): (IContextMenuValue | null)[] {
  const pngWithWorkflow: IContextMenuValue = {
    content: 'png (with embedded workflow)',
    callback: () => {
      exportPng(true);
    },
  };

  const pngWithoutWorkflow: IContextMenuValue = {
    content: 'png (no embedded workflow)',
    callback: () => {
      exportPng(false);
    },
  };

  const svgWithWorkflow: IContextMenuValue = {
    content: 'svg (with embedded workflow)',
    callback: () => {
      exportSvg(true);
    },
  };

  const svgWithoutWorkflow: IContextMenuValue = {
    content: 'svg (no embedded workflow)',
    callback: () => {
      exportSvg(false);
    },
  };

  const jsonItem: IContextMenuValue = {
    content: 'json',
    callback: () => {
      exportJson();
    },
  };

  const exportMenu: IContextMenuValue = {
    content: 'Export',
    has_submenu: true,
    submenu: {
      options: [pngWithWorkflow, pngWithoutWorkflow, svgWithWorkflow, svgWithoutWorkflow, jsonItem],
    },
  };

  const exportImageAndWorkflowItem: IContextMenuValue = {
    content: 'Export Image + Workflow',
    callback: () => {
      exportImageAndWorkflow();
    },
  };

  const importItem: IContextMenuValue = {
    content: 'Import',
    callback: () => {
      importWorkflow();
    },
  };

  const workflowImageMenu: IContextMenuValue = {
    content: 'Sr Workflow Image',
    has_submenu: true,
    submenu: {
      options: [importItem, exportMenu, null, exportImageAndWorkflowItem],
    },
  };

  // null represents a menu separator
  return [null, workflowImageMenu];
}

export const MenuItems = (): ComfyExtension => {
  return {
    name: `SilentRain.${NODE_NAME}`,

    init: async _app => {
      // No initialization needed - widgets are handled by DOM overlay system
    },

    setup: async _app => {
      // Node setup
    },

    nodeCreated: (_node, _app) => {
      // Node created callback
    },

    loadedGraphNode: (_node, _app) => {
      // Graph node loaded callback
    },

    getCanvasMenuItems: _canvas => {
      return buildWorkflowImageMenu();
    },

    beforeRegisterNodeDef: async (_nodeType, _nodeData, _app) => {},
  };
};

export default MenuItems;
