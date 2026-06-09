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
          // 添加一个空闲 slot
          const strInputTotal = this.inputs.filter(slot => slot.name !== 'delimiter').length;
          const strLinkCount = this.inputs
            .filter(slot => slot.name !== 'delimiter')
            .filter(slot => !slot.link).length;
          if (strLinkCount === 0) {
            const firstInput = this.inputs[0];
            if (!firstInput) return;

            const newIndex = strInputTotal + 1;
            this.addInput(`string_${newIndex}`, firstInput.type);
          }
        } else {
          setTimeout(() => {
            // 如果slot有连接，则不删除
            if (this.inputs[index]?.link) return;

            // 如果只有一个 string slot，则不删除
            if (this.inputs.filter(slot => slot.name !== 'delimiter').length === 1) return;

            this.removeInput(index);

            // 重命名所有slot
            let nameCount = 0;
            for (const item of this.inputs.filter(slot => slot.name !== 'delimiter')) {
              nameCount += 1;
              const label = `string_${nameCount}`;
              item.name = label;
              item.label = label;
            }
          }, 500);
        }
      };
    },
  };
};

export default StringDynList2;
