import { create } from 'zustand';
import type { NodeId, Slot } from '../types/comfy';
import { HUB_SLOTS_PROPERTY } from '@/nodes/param-hub';

/** 扩展 Slot，增加来源 hub 节点 ID */
export interface HubSlot extends Slot {
  /** 来源 ParamHub 节点 ID */
  hubNodeId: NodeId;
}

interface ParamHubStore {
  /**
   * 所有 ParamHub 节点的 slot 映射
   */
  hubs: Map<NodeId, Slot[]>;

  /**
   * 节点是否已完成首次加载（loadedGraphNode 中用到）
   */
  loadedFlags: Map<NodeId, boolean>;

  /** 设置或替换整个 hub 的 slots */
  setHub: (nodeId: NodeId, slots: Slot[]) => void;

  /** 移除整个 hub */
  removeHub: (nodeId: NodeId) => void;

  /** 获取一个 hub 的所有 slots（不存在时返回空数组） */
  getHubSlots: (nodeId: NodeId) => Slot[];

  /**
   * 获取所有 hub 的 slots，供 ParamPort 使用
   */
  getAllHubSlots: () => HubSlot[];

  /** 更新指定 slot 的标签（根据 name） */
  updateSlotLabel: (nodeId: NodeId, name: string, label: string) => void;

  /** 检查节点是否已加载过 */
  isNodeLoaded: (nodeId: NodeId) => boolean;

  /** 标记节点为已加载 */
  markNodeLoaded: (nodeId: NodeId) => void;
}

export const useParamHubStore = create<ParamHubStore>((set, get) => ({
  hubs: new Map(),
  loadedFlags: new Map(),

  setHub: (nodeId, slots) => {
    set(state => {
      const next = new Map(state.hubs);
      next.set(nodeId, [...slots]); // 浅拷贝数组
      return { hubs: next };
    });
  },

  removeHub: nodeId => {
    set(state => {
      const nextHubs = new Map(state.hubs);
      const nextFlags = new Map(state.loadedFlags);
      nextHubs.delete(nodeId);
      nextFlags.delete(nodeId);
      return { hubs: nextHubs, loadedFlags: nextFlags };
    });
  },

  getHubSlots: nodeId => {
    return get().hubs.get(nodeId) ?? [];
  },

  getAllHubSlots: () => {
    const { hubs } = get();
    const allSlots: HubSlot[] = [];
    for (const [nodeId, slots] of hubs.entries()) {
      for (const slot of slots) {
        allSlots.push({ ...slot, hubNodeId: nodeId });
      }
    }
    return allSlots;
  },

  isNodeLoaded: nodeId => {
    return get().loadedFlags.get(nodeId) ?? false;
  },

  markNodeLoaded: nodeId => {
    set(state => {
      const next = new Map(state.loadedFlags);
      next.set(nodeId, true);
      return { loadedFlags: next };
    });
  },

  /** 更新指定 slot 的 label（根据 name） */
  updateSlotLabel: (nodeId, name, label) => {
    set(state => {
      const slots = state.hubs.get(nodeId);
      if (!slots) return state;

      const slotIndex = slots.findIndex(s => s.name === name);
      if (slotIndex < 0) return state;

      const next = new Map(state.hubs);
      const nextSlots = [...slots];
      nextSlots[slotIndex] = Object.assign({}, nextSlots[slotIndex], { label });
      next.set(nodeId, nextSlots);

      // 同步更新 LiteGraph 节点的 input label
      try {
        const app = window.app!;
        if (app?.rootGraph) {
          const node = app.rootGraph.getNodeById(nodeId);
          if (node?.inputs) {
            const input = node.inputs.find((inp: any) => inp.name === name);
            if (input) {
              input.label = label;
            }

            // 直接存储 Slot[] 数组
            node.properties = node.properties ?? {};
            node.properties[HUB_SLOTS_PROPERTY] = JSON.stringify(slots);
          }
        }
      } catch (error) {
        console.error('Failed to update input label:', error);
      }

      return { hubs: next };
    });
  },
}));

/**
 * Zustand 选择器辅助函数：在外部（非 React 组件）直接读取当前状态。
 * 适用于 ComfyExtension 的回调函数中读取 Store。
 */
export function getParamHubStoreState(): ParamHubStore {
  return useParamHubStore.getState();
}
