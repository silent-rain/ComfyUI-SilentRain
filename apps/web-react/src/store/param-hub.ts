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
   * key: hubNodeId, value: Map<linkId, Slot>
   */
  hubs: Map<NodeId, Map<number, Slot>>;

  /** 设置或替换整个 hub */
  setHub: (nodeId: NodeId, slots: Map<number, Slot>) => void;

  /** 移除整个 hub */
  removeHub: (nodeId: NodeId) => void;

  /** 设置/更新单个 slot */
  setHubSlot: (nodeId: NodeId, slot: Slot) => void;

  /** 移除单个 slot */
  removeHubSlot: (nodeId: NodeId, linkId: number) => void;

  /** 获取一个 hub 的所有 slots（不存在时返回空 Map） */
  getHubSlots: (nodeId: NodeId) => Map<number, Slot>;

  /**
   * 获取所有 hub 的 slots，供 ParamPort 使用
   * 返回扁平化的 slot 数组，每个 slot 包含 hubNodeId
   */
  getAllHubSlots: () => HubSlot[];

  /** 更新指定 slot 的标签 */
  updateSlotLabel: (nodeId: NodeId, linkId: number, label: string) => void;
}

export const useParamHubStore = create<ParamHubStore>((set, get) => ({
  hubs: new Map(),

  setHub: (hubNodeId, slots) =>
    set(state => {
      const next = new Map(state.hubs);
      next.set(hubNodeId, slots);
      return { hubs: next };
    }),

  removeHub: hubNodeId =>
    set(state => {
      const next = new Map(state.hubs);
      next.delete(hubNodeId);
      return { hubs: next };
    }),

  setHubSlot: (hubNodeId, slot) =>
    set(state => {
      const next = new Map(state.hubs);
      const slots = new Map(next.get(hubNodeId) ?? []);
      slots.set(slot.linkId, slot);
      next.set(hubNodeId, slots);
      return { hubs: next };
    }),

  removeHubSlot: (hubNodeId, linkId) =>
    set(state => {
      const slots = state.hubs.get(hubNodeId);
      if (!slots) return state;
      const next = new Map(state.hubs);
      const nextSlots = new Map(slots);
      nextSlots.delete(linkId);
      next.set(hubNodeId, nextSlots);
      return { hubs: next };
    }),

  getHubSlots: hubNodeId => {
    return get().hubs.get(hubNodeId) ?? new Map();
  },

  getAllHubSlots: () => {
    const { hubs } = get();
    const allSlots: HubSlot[] = [];
    for (const [hubNodeId, slots] of hubs.entries()) {
      for (const [, slot] of slots.entries()) {
        allSlots.push({ ...slot, hubNodeId });
      }
    }
    return allSlots;
  },

  /** 更新指定 slot 的标签 */
  updateSlotLabel: (nodeId: NodeId, linkId: number, label: string) =>
    set(state => {
      const slots = state.hubs.get(nodeId);
      if (!slots) return state;
      const slot = slots.get(linkId);
      if (!slot) return state;
      const next = new Map(state.hubs);
      const nextSlots = new Map(slots);
      nextSlots.set(linkId, { ...slot, label });

      // 同步更新 LiteGraph 节点的 input label
      try {
        const app = (window as any).app;
        if (app?.graph) {
          const node = app.graph.getNodeById(nodeId);
          if (node?.inputs) {
            const input = node.inputs.find((inp: any) => inp.link === linkId);
            if (input) {
              input.name = label;
              input.label = label;
            }
          }
          // 强制刷新画布
          node?.setDirtyCanvas(true, true);
        }
      } catch (error) {
        console.error('Failed to update input label:', error);
      }

      return { hubs: next };
    }),
}));

/**
 * Zustand 选择器辅助函数：在外部（非 React 组件）直接读取当前状态。
 * 适用于 ComfyExtension 的回调函数中读取 Store。
 */
export function getParamHubStoreState(): ParamHubStore {
  return useParamHubStore.getState();
}
