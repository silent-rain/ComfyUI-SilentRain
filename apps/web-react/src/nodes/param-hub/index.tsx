/**
 * ParamHub 节点扩展
 *
 * 保留原始 ComfyExtension 的所有回调结构，便于参考与扩展。
 * React UI（HubPanel）在 loadedGraphNode / nodeCreated 中直接挂载。
 *
 * 持久化策略：
 * - 每个 ParamHub 节点把自己的 slots 序列化到 node.properties[HUB_SLOTS_PROPERTY]
 * - 刷新页面 / 加载 workflow 时，从 properties 恢复 slots 到 store 和 input labels
 */
import React from 'react';
import type { ComfyExtension } from '@comfyorg/comfyui-frontend-types';
import { ISlotType } from '../../enums/comfy';
import { mountReactWidget } from '../../core';
import { getParamHubStoreState } from '../../store';
import type { Slot, SlotType } from '../../types/comfy';
import { HubPanel } from './components/HubPanel';

const NODE_NAME = 'ParamHub';

const HUB_SLOTS_PROPERTY = 'sr_hub_slots';
const HUB_PANEL_NAME = 'hub_panel';

// ── Properties 持久化辅助函数 ──────────────────────────────

/** 将当前节点的 slots 从 store 序列化到 node.properties[HUB_SLOTS_PROPERTY] */
function saveSlotsToProperties(node: any): void {
  if (!node) return;
  const store = getParamHubStoreState();
  const slots = store.getHubSlots(node.id);

  // 将 Map<linkId, Slot> 转为普通对象以便 JSON 序列化
  const obj: Record<string, Slot> = {};
  for (const [linkId, slot] of slots.entries()) {
    obj[String(linkId)] = slot;
  }

  node.properties = node.properties ?? {};
  node.properties[HUB_SLOTS_PROPERTY] = JSON.stringify(obj);
}

/** 从 node.properties[HUB_SLOTS_PROPERTY] 恢复 slots 到 store 和 input labels */
function loadSlotsFromProperties(node: any): void {
  if (!node?.properties?.[HUB_SLOTS_PROPERTY]) return;

  try {
    const raw = JSON.parse(node.properties[HUB_SLOTS_PROPERTY]);
    const slotsMap = new Map<number, Slot>();

    for (const [linkIdStr, slotRaw] of Object.entries(raw as Record<string, any>)) {
      const linkId = Number(linkIdStr);
      // 兼容旧数据：确保 linkId 字段存在，且 label 不为空
      const slot: Slot = {
        linkId,
        name: slotRaw?.name ?? '',
        label: slotRaw?.label ?? '',
        type: slotRaw?.type ?? '*',
        value: slotRaw?.value,
      };
      slotsMap.set(linkId, slot);
    }

    // 恢复到 store
    const store = getParamHubStoreState();
    store.setHub(node.id, slotsMap);

    // 恢复 input label（匹配 linkId）
    if (node.inputs) {
      for (const input of node.inputs) {
        if (input.link != null) {
          const savedSlot = slotsMap.get(input.link);
          if (savedSlot) {
            if (savedSlot.label) {
              input.label = savedSlot.label;
            }
          }
        }
      }
    }
  } catch (e) {
    console.error('[ParamHub] Failed to load slots from properties:', e);
  }
}

// ── 导出函数：供 React 组件调用 ────────────────────────────

/** 保存指定节点的 slots 到 properties（可从 React 组件调用） */
export function saveNodeSlotsToProperties(nodeId: number): void {
  const app = (window as any).app;
  if (!app?.graph) return;
  const node = app.graph.getNodeById(nodeId);
  if (node) {
    saveSlotsToProperties(node);
  }
}

// ── React UI 绑定 ──────────────────────────────────────────

/** 将 React UI 挂载到指定节点（仅在首次调用时执行一次） */
function bindReactUI(node: any): void {
  if ((node as any).__sr_ui_bound) return;
  (node as any).__sr_ui_bound = true;

  mountReactWidget(node, HUB_PANEL_NAME, <HubPanel nodeId={node.id} />, {
    minHeight: 30,
  });
}

// ──  ParamHub factory  ─────────────────────

const ParamHub = (): ComfyExtension => {
  return {
    // 扩展名的名称
    name: `SilentRain.${NODE_NAME}`,

    // 允许任何初始化，例如加载资源。在画布创建后但在添加节点之前调用
    init: async _app => {
      // Node initialization
    },

    // 允许在应用程序完全设置并运行后调用任何其他设置
    setup: async _app => {
      // Node setup
    },

    // 允许扩展将上下文菜单项添加到画布右键菜单
    getCanvasMenuItems: _canvas => {
      return [];
    },

    // 允许扩展在节点构造函数之后运行代码
    nodeCreated: (node, _app) => {
      if (node.comfyClass !== NODE_NAME && node.type !== NODE_NAME) return;
      bindReactUI(node);
    },

    // 允许扩展修改已重新加载到图形上的节点。
    // 如果你破坏了后端的某些东西，并想修补前端的工作流
    loadedGraphNode: (node, _app) => {
      if (node.comfyClass !== NODE_NAME && node.type !== NODE_NAME) return;
      // 从 properties 恢复 slots
      loadSlotsFromProperties(node);
      bindReactUI(node);
    },

    // 允许扩展在向 LGraph 注册节点之前向其添加额外的处理
    beforeRegisterNodeDef: async (nodeType, nodeData, _app) => {
      // Only handle specific node
      if (nodeData.name !== NODE_NAME) return;

      nodeType.prototype.onConnectionsChange = function (
        type,
        index,
        isConnected,
        link_info,
        _inputOrOutput,
      ) {
        if (!link_info) return;

        if (type !== ISlotType.Input) return;

        const store = getParamHubStoreState();
        const nodeId = this.id;

        if (isConnected) {
          // 更新当前的 slot 信息为 link_info
          if (this.inputs[index]) {
            // this.inputs[index].name = String(link_info!.type);
            this.inputs[index].label = String(link_info!.type);
            this.inputs[index].type = link_info.type;
          }

          // 同步到 store：添加/更新 slot
          const slot: Slot = {
            linkId: link_info.id as number,
            name: this.inputs[index]?.name ?? '',
            label: this.inputs[index]?.label ?? '',
            type: link_info.type as SlotType,
          };
          store.setHubSlot(nodeId, slot);

          // 同步到 properties
          saveSlotsToProperties(this);

          // 添加一个空闲 slot
          const inputTotal = this.inputs.length;
          const linkCount = this.inputs.filter(slot => !slot.link).length;
          if (linkCount === 0) {
            const firstInput = this.inputs[0];
            if (!firstInput) return;

            const newIndex = inputTotal + 1;
            this.addInput(`param_${newIndex}`, '*');
          }
        } else {
          const self = this;
          // 在断开时立即获取 linkId， delayed 后 inputs[index].link 已被清除
          const disconnectedLinkId = link_info.id as number;

          setTimeout(() => {
            // 如果slot已重新连接，则不删除
            if (self.inputs[index]?.link) return;

            // 从 store 移除 slot（使用断开时就保存的 linkId）
            store.removeHubSlot(self.id, disconnectedLinkId);

            // 如果只有一个 string slot，则不删除
            if (self.inputs.length === 1) {
              // 保存 properties（即使没删除也要同步）
              saveSlotsToProperties(self);
              return;
            }

            self.removeInput(index);

            // 重命名所有slot
            let nameCount = 0;
            for (const item of self.inputs) {
              nameCount += 1;
              const label = `param_${nameCount}`;
              item.label = label;
            }

            // 同步到 properties
            saveSlotsToProperties(self);
          }, 500);
        }
      };
    },
  };
};

export default ParamHub;
