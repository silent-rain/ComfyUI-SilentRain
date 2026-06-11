/*便捷工具 */

import { HUB_SLOTS_PROPERTY } from '@/constant/param-hub';
import { getParamHubStoreState } from '@/store';
import type { NodeId } from '@/types/comfy';

// 更新节点 Properties
export const updateProperties = (nodeId: NodeId) => {
  const store = getParamHubStoreState();
  const slots = store.getHubSlots(nodeId);

  try {
    const app = window.app!;
    if (app?.rootGraph) {
      const node = app.rootGraph.getNodeById(nodeId);
      if (node) {
        // 直接存储 Slot[] 数组
        node.properties = node.properties ?? {};
        node.properties[HUB_SLOTS_PROPERTY] = JSON.stringify(slots);
      }
    }
  } catch (error) {
    console.error('Failed to update input label:', error);
  }
};
