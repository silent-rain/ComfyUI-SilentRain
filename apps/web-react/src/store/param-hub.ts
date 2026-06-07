import { create } from 'zustand';
import type { Slot } from '../types/param-hub';

interface ParamHubStore {
  hubs: Map<number, Map<number, Slot>>;

  setHub: (hubNodeId: number, hub: Map<number, Slot>) => void;

  removeHub: (hubNodeId: number) => void;

  getHub: (hubNodeId: number) => Map<number, Slot>;


  setHubSolt: (hubNodeId: number, slot: Slot) => void;

  removeHubSolt: (hubNodeId: number, linkId: number) => void;

  getHubSolt: (hubNodeId: number, linkId: number) => Slot | null;
}


export const useParamHubStore = create<ParamHubStore>((set, get) => ({
  hubs: new Map(),

  setHub: (hubNodeId, hub) =>
    set((state) => ({
      hubs: state.hubs.set(hubNodeId, hub)
    })),

  removeHub: (hubNodeId) =>
    set((state) => {
      state.hubs.delete(hubNodeId);
      return state;
    }),


  getHub: (hubNodeId) => {
    const state = get();
    const hub = state.hubs.get(hubNodeId);
    return hub ?? new Map();
  },

  setHubSolt: (hubNodeId: number, slot: Slot) => set((state) => {
    const solts = state.hubs.get(hubNodeId);
    if (!solts) {
      return state;
    }

    solts.set(slot.linkId, slot)

    state.setHub(hubNodeId, solts);
    return state;
  }),

  removeHubSolt: (hubNodeId, linkId) =>
    set((state) => {
      const solts = state.hubs.get(hubNodeId);
      if (!solts) {
        return state;
      }

      // 移除指定 solt
      solts.delete(linkId)

      state.setHub(hubNodeId, solts)
      return state;
    }),

  getHubSolt: (hubNodeId: number, linkId: number) => {
    const state = get();
    const solts = state.hubs.get(hubNodeId);
    if (!solts) {
      return null;
    }
    return solts.get(linkId) || null;
  }
}));