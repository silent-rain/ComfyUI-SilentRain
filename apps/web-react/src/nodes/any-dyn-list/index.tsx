/* AnyDynList Node */

import { ISlotType } from '../../enums/comfy';
import type { ComfyExtension } from '@comfyorg/comfyui-frontend-types';

const NODE_NAME = 'AnyDynList';

export const AnyDynList = (): ComfyExtension => {
  return {
    name: `SilentRain.${NODE_NAME}`,
    init: async _app => {
      // Node initialization
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
      return [];
    },
    beforeRegisterNodeDef: async (nodeType, nodeData, _app) => {
      // Only handle specific node
      if (nodeData.name !== NODE_NAME) return;

      nodeType.prototype.onConnectionsChange = function (type, index, isConnected, link_info) {
        if (!link_info) return;

        if (type !== ISlotType.Input) return;

        if (isConnected) {
          // Check if there is an empty "any_" slot
          const anyInputs = this.inputs.filter(slot => slot.name.startsWith('any_'));
          const hasEmptySlot = anyInputs.some(slot => !slot.link);

          if (this.inputs.length >= 2 && !hasEmptySlot) {
            // Add a new empty slot
            const firstAnyInput = this.inputs[1];
            if (!firstAnyInput) return;

            const newIndex = this.inputs.length + 1;
            this.addInput(`any_${newIndex}`, firstAnyInput.type);
          }
        } else {
          setTimeout(() => {
            // If the slot still has a connection, don't remove it
            if (this.inputs[index]?.link) return;

            // If there are only 2 or fewer inputs, don't remove
            if (this.inputs.length <= 2) return;

            this.removeInput(index);

            // Rename all "any_" slots
            let nameCount = 0;
            for (const item of this.inputs.filter(slot => slot.name.startsWith('any_'))) {
              nameCount += 1;
              const label = `any_${nameCount}`;
              item.name = label;
              item.label = label;
            }
          }, 100);
        }
      };
    },
  };
};

export default AnyDynList;
