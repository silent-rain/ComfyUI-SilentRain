/**
 * Sr Param Hub 节点：动态多输入聚合
 *
 * 节点 inputs 是真相源；hubCore 负责所有命令式同步，本文件只负责：
 *   - 钩子注册（onCreate / onConfigure / onConnectionsChange）
 *   - React 渲染（订阅 Zustand Store 显示最新状态）
 */

import React, { type ReactElement } from 'react';
import { defineReactNode, findWidget, hideWidget, type LGraphNode } from '../../core';
import { useParamHubStore } from '../../store/paramHubStore';
import { emptyHubMeta, safeParseHubMeta, SR_PARAMS_TYPE } from './param-protocol';
import {
  META_WIDGET,
  applyConnectionEvent,
  buildMetaFromNode,
  isInStructuralOp,
  removeSlot,
  renameSlot,
  syncFromInputs,
  syncFromMeta,
} from './hubCore';
import { HubDialogWithTrigger } from './HubPanel';

/* -------------------------------------------------------------------------- */
/* React 容器                                                                  */
/* -------------------------------------------------------------------------- */

interface HubRootProps {
  node: LGraphNode;
}

function HubRoot({ node }: HubRootProps) {
  const meta = useParamHubStore((state) => state.hubs[node.id] ?? emptyHubMeta());

  return (
    <HubDialogWithTrigger
      meta={meta}
      node={node}
      onRename={(slotName: string, label: string) => renameSlot(node, slotName, label)}
      onRemove={(slotName: string) => removeSlot(node, slotName)}
    />
  );
}

/* -------------------------------------------------------------------------- */
/* 扩展定义                                                                    */
/* -------------------------------------------------------------------------- */

export const ParamHubExtension = defineReactNode({
  comfyClass: 'ParamHub',
  extensionName: 'SilentRain.ParamHub',
  widgetName: 'sr_param_hub_ui',
  minHeight: 0,

  // 关键：阻止 ComfyUI 在创建节点时按 INPUT_TYPES 预先塞入 32 个 param_* 槽。
  // 后端 INPUT_TYPES 仍保留这 32 个 optional 声明（用于 server 端 prompt 校验），
  // 前端运行期通过 addInput 按需动态创建实际用到的槽。
  stripOptionalInputs: [/^param_\d+$/],

  onCreate({ node }) {
    const w = findWidget(node, META_WIDGET);
    if (!w) {
      console.warn('[ParamHub] params_meta_json widget not found');
      return;
    }
    hideWidget(w);

    // 输出端口类型矫正
    if (node.outputs?.[0]) {
      node.outputs[0].type = SR_PARAMS_TYPE;
    }

    // backfill 判别：node 已经被加入到 graph 时，说明 ComfyUI 已经走完
    //   createNode → nodeCreated → graph.add → configure(...) 全流程并完成
    //   link 反序列化。此时若像 onConfigure 那样 clear-and-rebuild，removeInput
    //   会真的销毁用户连线，必须改走"就地修补"路线。
    //
    // 注意：不能用 "inputs 里有 link" 来判别——某些节点的 link 信息可能尚未关联
    // 到 inputs 数组（取决于反序列化阶段），且即使确实没有外部连线时这条规则
    // 也成立。最稳妥的信号是节点本身已经被注册进 graph._nodes。
    const inGraph = !!(
      node.graph &&
      Array.isArray((node.graph as { _nodes?: LGraphNode[] })._nodes) &&
      (node.graph as { _nodes: LGraphNode[] })._nodes.includes(node)
    );

    syncFromMeta(node, { preserveLinks: inGraph });
    // 将当前 meta 同步到 Zustand store，确保 Port 节点能读取到完整 label
    useParamHubStore.getState().setHubMeta(node.id, buildMetaFromNode(node));
    // 节点尺寸收敛由 main.ts 在 React widget 完成首次布局后统一处理
  },
  onConfigure({ node }) {
    // 加载工作流后 widgets_values 已经填好；按 meta 重建 inputs
    // preserveLinks: true — 此时 inputs 的 link 信息已由 ComfyUI 反序列化恢复，
    // 必须走"就地修补"路线而非"清空重建"，否则 removeInput 会销毁上游连线，
    // 导致 Port 端 pickLiveSlots 找不到 linked slots 而清空输出端点名称。
    syncFromMeta(node, { preserveLinks: true });
    // 同时将持久化的 meta 写入 Zustand store，确保 Port 节点在刷新后
    // 能从 store 中读取到完整的 label 信息（而不是从可能丢失 label 的 inputs）
    const w = findWidget(node, META_WIDGET);
    if (w) {
      const meta = safeParseHubMeta(w.value);
      useParamHubStore.getState().setHubMeta(node.id, meta);
    }
  },

  render({ node }): ReactElement {
    return <HubRoot node={node} />;
  },

  onNodeContextMenu({ node }) {
    return [
      {
        content: 'Edit params',
        callback: () => {
          // 通过 dispatch 触发节点内部 open 状态
          // 由于 React 组件与这里的上下文隔离，我们用自定义事件来通信
          const ev = new CustomEvent('sr-hub-open-dialog', {
            detail: { nodeId: node.id },
          });
          window.dispatchEvent(ev);
        },
      },
    ];
  },

  onConnectionsChange({ node, type, index, connected, link }) {
    // 仅处理 input 端（type === 1 in LiteGraph 为 INPUT）
    if (type !== 1) return;
    // 结构变更期间（removeSlot / rebuildInputsFromMeta 等）禁用同步重入，
    // 否则会出现「删除 VAE2 后 VAE3 被错误清空」之类的下标漂移问题。
    if (isInStructuralOp()) return;
    applyConnectionEvent(node, index, connected, link);
    syncFromInputs(node);
  },
});
