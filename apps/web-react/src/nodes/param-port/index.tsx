/**
 * ParamPort 节点扩展
 *
 * ParamPort 是 ParamHub 的成对节点，用于转发输出端点。
 * 从 ParamHub store 读取数据，动态构建输出端点。
 *
 * 核心机制：
 * - ParamHub 的 slots 数据存储在 Zustand store 中
 * - ParamPort 通过 store 订阅实时感知 ParamHub slots 的变化
 * - 连接建立时从 store 获取 slots 并动态构建 this.outputs
 * - 利用 Zustand subscribe 在非 React 上下文中监听 store 变化
 * - 当 ParamHub 的 slots 发生变化时自动重建 outputs
 */
import type { ComfyExtension } from '@comfyorg/comfyui-frontend-types';
import { ISlotType } from '../../enums/comfy';
import { getParamHubStoreState, useParamHubStore } from '@/store';
import type { NodeId, Slot } from '../../types/comfy';

const NODE_NAME = 'ParamPort';

// ── 重建 outputs（核心函数）──────────────────────────────────

/**
 * 根据给定的 slots 数据重建节点的 this.outputs。
 * 保留现有 outputs 的连接信息，避免断开已有连线。
 */
function rebuildOutputs(node: any, slots: Slot[]): void {
  const oldOutputs = [...(node.outputs || [])];

  // 清空并重建 outputs
  node.outputs = [];
  for (const slot of slots) {
    node.addOutput(slot.label || slot.name, slot.type);
  }

  // 尝试恢复旧 outputs 中的连接
  for (let i = 0; i < node.outputs.length; i++) {
    if (i < oldOutputs.length && oldOutputs[i].links) {
      node.outputs[i].links = oldOutputs[i].links;
    }
  }

  // 触发节点重绘
  if (node.graph) {
    node.graph.setDirtyCanvas(true, true);
  }
}

// ── Zustand Store 订阅 ──────────────────────────────────────────

/**
 * 初始化 Store 订阅。
 * 仅在 onConnectionsChange 首次连接时调用（此时已有 originNodeId）。
 * 用 __sr_store_subscribed 状态标记避免重复初始化。
 */
function initStoreSubscription(node: any, originNodeId: NodeId): void {
  const nodeAny = node as any;
  if (nodeAny.__sr_store_subscribed) return;

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  let prevSlots: Slot[] | undefined;

  const unsubscribe = useParamHubStore.subscribe(state => {
    const slots = state.hubs.get(originNodeId);
    // 仅在 slots 引用变化时重建 outputs
    if (slots !== prevSlots) {
      prevSlots = slots;
      rebuildOutputs(node, slots ?? []);
    }
  });

  // 标记已订阅，并保存 unsubscribe 引用
  nodeAny.__sr_store_subscribed = true;
  nodeAny.__sr_store_unsubscribe = unsubscribe;
}

// ── 清理函数 ──────────────────────────────────────────────

/** 取消 ParamPort 节点上的 store 订阅 */
function cleanupStoreSubscription(node: any): void {
  const nodeAny = node as any;

  // 取消 Zustand store 订阅
  if (nodeAny.__sr_store_unsubscribe) {
    nodeAny.__sr_store_unsubscribe();
    nodeAny.__sr_store_unsubscribe = null;
  }
}

// ── ParamPort factory ─────────────────────────────────────

const ParamPort = (): ComfyExtension => {
  return {
    name: `SilentRain.${NODE_NAME}`,

    init: async _app => {},

    setup: async _app => {},

    loadedGraphNode: (node, _app) => {
      if (node.comfyClass !== NODE_NAME && node.type !== NODE_NAME) return;
    },

    nodeCreated: (node, _app) => {
      if (node.comfyClass !== NODE_NAME && node.type !== NODE_NAME) return;
      console.log(`[ParamPort] nodeId: ${node.id} nodeCreated`);
    },

    getCanvasMenuItems: _canvas => {
      return [];
    },

    beforeRegisterNodeDef: async (nodeType, nodeData, _app) => {
      if (nodeData.name !== NODE_NAME) return;

      nodeType.prototype.onConnectionsChange = function (
        type,
        _index,
        isConnected,
        link_info,
        _inputOrOutput,
      ) {
        if (!link_info) return;

        if (type !== ISlotType.Input) return;

        if (isConnected) {
          // 首次连接时初始化 Store 订阅（此时已有 originNodeId）
          const originNodeId = link_info.origin_id;
          initStoreSubscription(this, originNodeId);

          // 直接用 ParamHub 的 slots 覆盖 outputs
          const store = getParamHubStoreState();
          const slots = store.getHubSlots(originNodeId);
          rebuildOutputs(this, slots);
        } else {
          // 连接断开：直接清空所有 outputs
          rebuildOutputs(this, []);
        }
      };

      // 节点移除时的清理
      nodeType.prototype.onRemoved = function () {
        cleanupStoreSubscription(this);
      };
    },
  };
};

export default ParamPort;
