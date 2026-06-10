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
import type { Slot } from '../../types/comfy';
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

  // 直接存储 Slot[] 数组
  node.properties = node.properties ?? {};
  node.properties[HUB_SLOTS_PROPERTY] = JSON.stringify(slots);
}

/** 从 node.properties[HUB_SLOTS_PROPERTY] 恢复 slots 到 store 和 input labels */
function loadSlotsFromProperties(node: any): void {
  if (!node?.properties?.[HUB_SLOTS_PROPERTY]) return;

  try {
    const slots = JSON.parse(node.properties[HUB_SLOTS_PROPERTY]) as Slot[];
    // 恢复到 store
    const store = getParamHubStoreState();
    store.setHub(node.id, slots);

    // 恢复 input label（匹配 linkId）
    // if (node.inputs) {
    //   for (const input of node.inputs) {
    //     if (input.link != null) {
    //       const savedSlot = slots.find(s => s.linkId === input.link);
    //       if (savedSlot) {
    //         if (savedSlot.label) {
    //           input.label = savedSlot.label;
    //         }
    //       }
    //     }
    //   }
    // }
  } catch (e) {
    console.error('[ParamHub] Failed to load slots from properties:', e);
  }
}

// ── React UI 绑定 ──────────────────────────────────────────

/**
 * 将 React UI 挂载到指定节点
 * - 首次调用时创建 UI 并保存 handle
 * - 后续调用时（如 loadedGraphNode）使用 rerender 更新 props
 */
function bindReactUI(node: any): void {
  const nodeAny = node as any;

  // 如果已经创建过 widget，使用 rerender 更新
  if (nodeAny.__sr_widget_handle) {
    // 使用 rerender 传入正确的 nodeId
    nodeAny.__sr_widget_handle.rerender(<HubPanel nodeId={node.id} />);
    return;
  }

  // 首次创建 UI
  const handle = mountReactWidget(node, HUB_PANEL_NAME, <HubPanel nodeId={node.id} />, {
    minHeight: 30,
  });

  // 保存 handle 引用，供后续 rerender 使用
  nodeAny.__sr_widget_handle = handle;
  nodeAny.__sr_ui_bound = true;
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
          // 先尝试从 store 中获取已保存的 slot 数据（避免刷新页面后数据重置）
          const slots = store.getHubSlots(nodeId);
          const existingSlot = slots.find(s => s.linkId === link_info.id);

          // 重置名称和类型
          if (this.inputs[index]) {
            if (existingSlot) {
              // 如果 store 中存在，使用 store 中的 label（保留用户自定义的）
              this.inputs[index].label = existingSlot.label || String(link_info.type);
              this.inputs[index].type = existingSlot.type;
            } else {
              // 如果 store 中不存在，使用 link_info 创建新的
              this.inputs[index].label = String(link_info.type);
              this.inputs[index].type = link_info.type;
            }
          }

          // 同步到 store：添加/更新 slot
          const slot: Slot = {
            linkId: link_info.id as number,
            name: this.inputs[index]?.name ?? '',
            label: this.inputs[index]?.label ?? '',
            type: this.inputs[index]?.type ?? '*',
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
          // 在断开时立即获取 linkId，delayed 后 inputs[index].link 已被清除
          const disconnectedLinkId = link_info.id as number;
          const nodeId = this.id;

          setTimeout(() => {
            // 如果slot已重新连接，则不删除
            if (this.inputs[index]?.link) return;

            // 从 store 移除 slot（使用断开时就保存的 linkId）
            store.removeHubSlot(nodeId, disconnectedLinkId);

            // 如果只有一个 string slot，则不删除
            if (this.inputs.length === 1) {
              // 保存 properties
              saveSlotsToProperties(this);
              return;
            }

            this.removeInput(index);

            // 保存 properties
            saveSlotsToProperties(this);
          }, 500);
        }

        // 重命名所有slot
        let nameCount = 0;
        for (const item of this.inputs) {
          nameCount += 1;
          const name = `param_${nameCount}`;
          item.name = name;
          if (item.label?.startsWith('param_')) {
            item.label = name;
          }
        }
      };
    },
  };
};

export default ParamHub;
