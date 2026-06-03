import { create } from 'zustand';
import type { ParamHubMeta } from '../nodes/param-hub/param-protocol';

interface ParamHubStore {
  /** key = hub LGraphNode.id, value = 该 hub 的最新 meta */
  hubs: Record<number, ParamHubMeta>;

  /** Hub 端调用：写入/更新某个 hub 的 meta */
  setHubMeta: (hubNodeId: number, meta: ParamHubMeta) => void;

  /** Hub 端调用：节点被移除时清理 */
  removeHub: (hubNodeId: number) => void;
}

export const useParamHubStore = create<ParamHubStore>((set) => ({
  hubs: {},

  setHubMeta: (hubNodeId, meta) =>
    set((state) => ({
      hubs: { ...state.hubs, [hubNodeId]: meta },
    })),

  removeHub: (hubNodeId) =>
    set((state) => {
      const { [hubNodeId]: _, ...rest } = state.hubs;
      return { hubs: rest };
    }),
}));

/**
 * Port 端使用：订阅特定 hub 的 meta 变化
 *
 * Zustand v5 的 subscribe 只接受 (state, prevState) => void listener，
 * 因此我们在 listener 内部手动做 selector + equality 检查。
 *
 * @param hubNodeId 要监听的 hub 节点 id
 * @param callback  meta 变化时的回调（参数为最新 meta，null 表示该 hub 不存在）
 * @returns unsubscribe 函数
 */
export function subscribeHubMeta(
  hubNodeId: number,
  callback: (meta: ParamHubMeta | null) => void,
): () => void {
  let lastSerialized = '';

  return useParamHubStore.subscribe((state, _prevState) => {
    const meta = state.hubs[hubNodeId] ?? null;
    const serialized = JSON.stringify(meta);

    // 只有该 hub 的 meta 实际变化时才触发 callback
    if (serialized !== lastSerialized) {
      lastSerialized = serialized;
      callback(meta);
    }
  });
}
