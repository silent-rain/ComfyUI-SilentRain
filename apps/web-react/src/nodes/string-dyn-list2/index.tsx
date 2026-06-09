/* StringDynList2 Node */

import { ISlotType } from '../../enums/comfy';
import type { ComfyExtension } from '@comfyorg/comfyui-frontend-types';

const NODE_NAME = 'StringDynList2';

export const StringDynList2 = (): ComfyExtension => {
  return {
    name: `SilentRain.${NODE_NAME}`,
    init: async _app => {
      // Node initialization
    },
    setup: async _app => {
      // Node setup
    },
    getCanvasMenuItems: _canvas => {
      return [];
    },
    nodeCreated: (_node, _app) => {
      // Node created callback
    },
    loadedGraphNode: (_node, _app) => {
      // Graph node loaded callback
    },
    beforeRegisterNodeDef: async (nodeType, nodeData, _app) => {
      // Only handle specific node
      if (nodeData.name !== NODE_NAME) return;

      nodeType.prototype.onConnectionsChange = function (type, _index, isConnected, link_info) {
        if (!link_info) return;

        if (type !== ISlotType.Input) return;

        if (isConnected) {
          const inputTotal = this.inputs.length;
          const linkCount = this.inputs
            .filter(slot => slot.name !== 'delimiter')
            .filter(slot => !slot.link).length;
          console.log(`inputTotal: ${inputTotal}, linkCount: ${linkCount}`);

          if (linkCount === 0) {
            const firstInput = this.inputs[0];
            if (!firstInput) return;

            // 添加一个空闲slot
            const newIndex = inputTotal + 1;
            this.addInput(`string_${newIndex}`, firstInput.type);
          }
        } else {
          // Only remove if this slot has no other connections
          // const hasConnection = this.inputs[index]?.link !== null;
          // if (!hasConnection) {
          //     this.removeInput(index);
          // }

          // 移除所有的空闲slot
          const inputTotal = this.inputs.length;
          let linkCount = this.inputs
            .filter(slot => slot.name !== 'delimiter')
            .filter(slot => !slot.link).length;
          console.log(`inputTotal: ${inputTotal}, linkCount: ${linkCount}`);
          for (let i = 0; i < inputTotal - 1; i++) {
            const input = this.inputs[i];
            if (!input || input.name === 'delimiter') continue;

            // 保留至少一个空闲slot
            if (linkCount <= 1) {
              break;
            }

            if (!input.link) {
              this.removeInput(i);
              linkCount -= 1;
            }
          }
        }

        let nameCount = 0;
        for (const item of this.inputs) {
          if (item.name === 'delimiter') continue;
          nameCount += 1;
          const name = `string_${nameCount}`;
          item.name = name;
          item.label = name;
        }
      };
    },
  };
};

export default StringDynList2;
