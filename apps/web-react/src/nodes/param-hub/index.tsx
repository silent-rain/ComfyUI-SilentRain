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
import type { NodeId, Slot } from '../../types/comfy';
import { HubPanel } from './components/HubPanel';

const NODE_NAME = 'ParamHub';

export const HUB_SLOTS_PROPERTY = 'sr_hub_slots';
const HUB_PANEL_NAME = 'hub_panel';

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

// ── 批量同步 slots（以 this.inputs 为唯一事实来源） ─────────

/**
 * 从节点的 this.inputs 批量重建 slots 并写入 store，同时合并 store 中已有的自定义 label。
 * 这是保证 store 数据和 LiteGraph inputs 完全一致的唯一可靠方式。
 */
function syncHubSlotsFromInputs(node: any): void {
  const store = getParamHubStoreState();
  const nodeId = node.id as NodeId;

  // 获取已保存在 store 中的 label 映射（linkId -> label），用于保留用户自定义 label
  const existingSlots = store.getHubSlots(nodeId);
  const labelMap = new Map<number, Slot>();
  for (const slot of existingSlots) {
    if (slot.label) {
      labelMap.set(slot.linkId, slot);
    }
  }

  console.log('[ParamHub] Syncing slots from inputs', labelMap);

  // 从 this.inputs 扫描，批量构建当前 slots
  const slots: Slot[] = [];
  for (const input of node.inputs) {
    // 只保留已连接的 input（link 存在）
    if (input.link != null) {
      const linkId = input.link as number;
      const customSlot = labelMap.get(linkId);
      slots.push({
        linkId,
        name: input.name,
        label: customSlot?.label ?? input.type,
        type: customSlot?.label ?? input.type,
      });
    }
  }

  // 一次性写入 store（替换整个数组，保证顺序和最新数据）
  store.setHub(nodeId, slots);

  // 同步回 properties 持久化（代替 saveSlotsToProperties 的独立调用）
  node.properties = node.properties ?? {};
  node.properties[HUB_SLOTS_PROPERTY] = JSON.stringify(slots);
}

/**
 * 更新（新增 / 覆盖）单个 slot 到 store 与 Properties 持久化。
 * 用于连接建立时把当前 link 对应的 slot 写入 store 并同步 properties。
 */
function updateSingleSlot(node: any, index: number, link_info: any): void {
  const store = getParamHubStoreState();
  const nodeId = node.id as NodeId;

  // 首次加载：从 properties 恢复 slots
  if (!store.isNodeLoaded(nodeId)) {
    loadSlotsFromProperties(node);
    store.markNodeLoaded(nodeId);
  }

  const input = node.inputs[index];

  let newSlot: Slot = {
    linkId: link_info.id,
    name: input.name,
    label: input.label,
    type: input.type,
  };
  console.log('[ParamHub] Syncing slots from input', newSlot);

  // 重置名称和类型
  // 从 store 获取当前 slots
  const slots = store.getHubSlots(nodeId);

  // 重置名称和类型
  const existingSlot = slots.find(s => s.name === input.name);
  console.log('[ParamHub] Syncing slots from existingSlot', existingSlot);
  if (existingSlot) {
    // 如果 store 中存在，使用 store 中的 label（保留用户自定义的）
    node.inputs[index].label = existingSlot.label || String(link_info.type);
    node.inputs[index].type = existingSlot.type;
  } else {
    // 如果 store 中不存在，使用 link_info 创建新的
    node.inputs[index].label = String(link_info.type);
    node.inputs[index].type = link_info.type;

    newSlot.label = String(link_info.type);
    newSlot.type = link_info.type;

    console.log('[ParamHub] Syncing slots from existingSlot2', newSlot);
  }
  console.log('[ParamHub] Syncing slots from newSlot3', newSlot);

  if (index >= 0) {
    slots[index] = newSlot;
  } else {
    slots.push(newSlot);
  }

  // 一次性写入 store
  store.setHub(nodeId, slots);

  // 同步回 properties 持久化
  node.properties = node.properties ?? {};
  node.properties[HUB_SLOTS_PROPERTY] = JSON.stringify(slots);
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
          console.log(
            '[ParamHub] onConnectionsChange called',
            type,
            index,
            isConnected,
            link_info,
            _inputOrOutput,
          );
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

          // 同时更新单个 slot 到 store 与 Properties
          updateSingleSlot(this, index, link_info);

          // 添加一个空闲 slot（如果没有空位了）
          const inputTotal = this.inputs.length;
          const linkCount = this.inputs.filter(slot => !slot.link).length;
          if (linkCount === 0) {
            const newIndex = inputTotal + 1;
            this.addInput(`param_${newIndex}`, '*');
          }
        } else {
          setTimeout(() => {
            // 如果slot已重新连接，则不删除
            if (this.inputs[index]?.link) {
              syncHubSlotsFromInputs(this);
              return;
            }
            // 断开连接：如果只有一个 slot，则不删除 input，只清空
            if (this.inputs.length === 1) {
              // 保持至少一个空位
            } else {
              // 延迟删除 input，避免与重新连接冲突
              // 但同步逻辑交给 syncHubSlotsFromInputs，无需 setTimeout
              this.removeInput(index);
            }

            // 重命名所有 slot 的 name，保证 name 顺序正确
            let nameCount = 0;
            for (const item of this.inputs) {
              nameCount += 1;
              const name = `param_${nameCount}`;
              item.name = name;
              // 如果 label 是自动生成的（以 param_ 开头），同步更新
              if (item.label?.startsWith('param_')) {
                item.label = name;
              }
            }

            // 从 this.inputs 批量重建 slots 并写入 store（核心：以 inputs 为唯一事实来源）
            syncHubSlotsFromInputs(this);
          }, 200);
        }
      };
    },
  };
};

export default ParamHub;
