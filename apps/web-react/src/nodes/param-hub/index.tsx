/**
 * ParamHub 节点扩展
 *
 * 保留原始 ComfyExtension 的所有回调结构，便于参考与扩展。
 * React UI（HubPanel）在 loadedGraphNode / nodeCreated 中直接挂载。
 */
import React from 'react';
import type { ComfyExtension } from '@comfyorg/comfyui-frontend-types';
import { ISlotType } from '../../enums/comfy';
import { mountReactWidget } from '../../core';
import { getParamHubStoreState } from '../../store';
import type { Slot, SlotType } from '../../types/comfy';
import { HubPanel } from './components/HubPanel';

const NODE_NAME = 'ParamHub';

/** 将 React UI 挂载到指定节点（仅在首次调用时执行一次） */
function bindReactUI(node: any): void {
  if ((node as any).__sr_ui_bound) return;
  (node as any).__sr_ui_bound = true;

  mountReactWidget(node, 'hub_panel', <HubPanel nodeId={node.id} />, {
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
      // bindReactUI(node);
    },

    // 允许扩展修改已重新加载到图形上的节点。
    // 如果你破坏了后端的某些东西，并想修补前端的工作流
    loadedGraphNode: (node, _app) => {
      if (node.comfyClass !== NODE_NAME && node.type !== NODE_NAME) return;
      console.log('loaded Graph Node widgets_values:', (node as any).widgets_values);
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
          setTimeout(() => {
            // 如果slot有连接，则不删除
            if (this.inputs[index]?.link) return;

            // 从 store 移除 slot
            const linkId = this.inputs[index]?.link;
            if (linkId) {
              store.removeHubSlot(nodeId, linkId);
            }

            // 如果只有一个 string slot，则不删除
            if (this.inputs.length === 1) return;

            this.removeInput(index);

            // 重命名所有slot
            let nameCount = 0;
            for (const item of this.inputs) {
              nameCount += 1;
              const label = `param_${nameCount}`;
              item.name = label;
              // item.label = label;
            }
          }, 500);
        }
      };
    },
  };
};

export default ParamHub;
