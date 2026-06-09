import { create } from 'zustand';
import type { Slot } from '../types/comfy';

interface ParamHubStore {
  /** 所有 ParamHub 节点的 slot 映射 */
  hubs: Map<number, Map<number, Slot>>;

  /** 设置或替换整个 hub */
  setHub: (hubNodeId: number, hub: Map<number, Slot>) => void;

  /** 移除整个 hub */
  removeHub: (hubNodeId: number) => void;

  /** 获取一个 hub（不存在时返回空 Map） */
  getHub: (hubNodeId: number) => Map<number, Slot>;

  /** 设置/更新单个 slot */
  setHubSlot: (hubNodeId: number, slot: Slot) => void;

  /** 移除单个 slot */
  removeHubSlot: (hubNodeId: number, linkId: number) => void;

  /** 获取单个 slot */
  getHubSlot: (hubNodeId: number, linkId: number) => Slot | null;
}

export const useParamHubStore = create<ParamHubStore>((set, get) => ({
  hubs: new Map(),

  setHub: (hubNodeId, hub) =>
    set((state) => {
      const next = new Map(state.hubs);
      next.set(hubNodeId, hub);
      return { hubs: next };
    }),

  removeHub: (hubNodeId) =>
    set((state) => {
      const next = new Map(state.hubs);
      next.delete(hubNodeId);
      return { hubs: next };
    }),

  getHub: (hubNodeId) => {
    return get().hubs.get(hubNodeId) ?? new Map();
  },

  setHubSlot: (hubNodeId, slot) =>
    set((state) => {
      const next = new Map(state.hubs);
      const slots = new Map(next.get(hubNodeId) ?? []);
      slots.set(slot.linkId, slot);
      next.set(hubNodeId, slots);
      return { hubs: next };
    }),

  removeHubSlot: (hubNodeId, linkId) =>
    set((state) => {
      const slots = state.hubs.get(hubNodeId);
      if (!slots) return state;
      const next = new Map(state.hubs);
      const nextSlots = new Map(slots);
      nextSlots.delete(linkId);
      next.set(hubNodeId, nextSlots);
      return { hubs: next };
    }),

  getHubSlot: (hubNodeId, linkId) => {
    return get().hubs.get(hubNodeId)?.get(linkId) ?? null;
  },
}));

/**
 * Zustand 选择器辅助函数：在外部（非 React 组件）直接读取当前状态。
 * 适用于 ComfyExtension 的回调函数中读取 Store。
 */
export function getParamHubStoreState() {
  return useParamHubStore.getState();
}