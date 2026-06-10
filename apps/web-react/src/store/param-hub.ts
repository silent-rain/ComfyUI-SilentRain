import { create } from 'zustand';
import type { NodeId, Slot } from '../types/comfy';

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

  /** 设置或替换整个 hub 的 slots */
  setHub: (nodeId: NodeId, slots: Slot[]) => void;

  /** 移除整个 hub */
  removeHub: (nodeId: NodeId) => void;

  /**
   * 设置/更新单个 slot
   * 根据 linkId 查找，如果找到则更新，否则追加到末尾
   */
  setHubSlot: (nodeId: NodeId, slot: Slot) => void;

  /** 移除单个 slot（根据 linkId） */
  removeHubSlot: (nodeId: NodeId, linkId: number) => void;

  /** 根据 index 移除单个 slot */
  removeHubSlotByIndex: (nodeId: NodeId, index: number) => void;

  /** 获取一个 hub 的所有 slots（不存在时返回空数组） */
  getHubSlots: (nodeId: NodeId) => Slot[];

  /**
   * 获取所有 hub 的 slots，供 ParamPort 使用
   */
  getAllHubSlots: () => HubSlot[];

  /** 更新指定 slot 的标签（根据 linkId） */
  updateSlotLabel: (nodeId: NodeId, linkId: number, label: string) => void;
}

export const useParamHubStore = create<ParamHubStore>((set, get) => ({
  hubs: new Map(),

  setHub: (nodeId, slots) => {
    set(state => {
      const next = new Map(state.hubs);
      next.set(nodeId, [...slots]); // 浅拷贝数组
      return { hubs: next };
    });
  },

  removeHub: nodeId => {
    set(state => {
      const next = new Map(state.hubs);
      next.delete(nodeId);
      return { hubs: next };
    });
  },

  setHubSlot: (nodeId, slot) => {
    set(state => {
      const next = new Map(state.hubs);
      const slots = next.get(nodeId) ?? [];

      // 根据 linkId 查找是否已存在
      const existingIndex = slots.findIndex(s => s.linkId === slot.linkId);

      const nextSlots = [...slots];
      if (existingIndex >= 0) {
        // 更新已存在的 slot
        nextSlots[existingIndex] = slot;
      } else {
        // 追加新 slot
        nextSlots.push(slot);
      }

      next.set(nodeId, nextSlots);
      return { hubs: next };
    });
  },

  removeHubSlot: (nodeId, linkId) => {
    set(state => {
      const slots = state.hubs.get(nodeId);
      if (!slots) return state;

      const next = new Map(state.hubs);
      const nextSlots = slots.filter(s => s.linkId !== linkId);
      next.set(nodeId, nextSlots);
      return { hubs: next };
    });
  },

  removeHubSlotByIndex: (nodeId, index) => {
    set(state => {
      const slots = state.hubs.get(nodeId);
      if (!slots || index < 0 || index >= slots.length) return state;

      const next = new Map(state.hubs);
      const nextSlots = [...slots];
      nextSlots.splice(index, 1);
      next.set(nodeId, nextSlots);
      return { hubs: next };
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

  /** 更新指定 slot 的 label（根据 linkId） */
  updateSlotLabel: (nodeId, linkId, label) => {
    set(state => {
      const slots = state.hubs.get(nodeId);
      if (!slots) return state;

      const slotIndex = slots.findIndex(s => s.linkId === linkId);
      if (slotIndex < 0) return state;

      const next = new Map(state.hubs);
      const nextSlots = [...slots];
      nextSlots[slotIndex] = Object.assign({}, nextSlots[slotIndex], { label });
      next.set(nodeId, nextSlots);

      // 同步更新 LiteGraph 节点的 input label
      try {
        const app = (window as any).app;
        if (app?.graph) {
          const node = app.graph.getNodeById(nodeId);
          if (node?.inputs) {
            const input = node.inputs.find((inp: any) => inp.link === linkId);
            if (input) {
              input.label = label;
            }
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
