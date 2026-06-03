/**
 * ParamHub 的核心同步层（与 React 解耦的纯命令式逻辑）
 *
 * 真相源 = 节点上的真实 inputs 数组（仅 param_* 槽参与统计）。
 * 派生：
 *   - widget `params_meta_json` 是它的序列化镜像（用于持久化）
 *   - bus 事件是它的实时推送
 *   - React state 是它的展示镜像
 *
 * 所有外部触发（onCreate / onConfigure / onConnectionsChange / 用户编辑）
 * 都最终汇聚到 `syncFromInputs(node)` 一个出口，由它统一：
 *   1. 算出最新 meta
 *   2. 维护"末尾恰好一个空闲槽"不变量（必要时 add/remove input）
 *   3. 写回 widget
 *   4. emit bus
 */

import { findWidget, fitNodeHeight, type LGraphNode, type INodeSlot, type LLink } from '../../core';
import { useParamHubStore } from '../../store/paramHubStore';
import {
  emptyHubMeta,
  MAX_PARAM_SLOTS,
  nextAvailableSlotIndex,
  safeParseHubMeta,
  slotInputName,
  type ParamHubMeta,
  type SlotMeta,
} from './param-protocol';

export const META_WIDGET = 'params_meta_json';

/* -------------------------------------------------------------------------- */
/* 节点 inputs 操作的小工具                                                   */
/* -------------------------------------------------------------------------- */

interface AddInputCapable {
  addInput?: (name: string, type: string, options?: unknown) => INodeSlot;
  removeInput?: (slot: number) => void;
  disconnectInput?: (slot: number) => void;
}

/**
 * 结构变更重入保护标记
 *
 * removeInput / disconnectInput 都会同步触发 onConnectionsChange 回调，
 * 该回调又会调用 syncFromInputs / ensureExactlyOneEmptySlot 去改 inputs 数组，
 * 导致调用方所持的下标在下一句语变得指向错误的 slot。
 *
 * 例：删除 VAE2 (idx=3) 时 disconnectInput(3) 陆续触发：
 *   1. applyConnectionEvent 把 VAE2 重置成 空闲槽
 *   2. syncFromInputs 调 ensureExactlyOneEmptySlot 发现有两个空闲槽（VAE2 + param_6）
 *   3. 把 VAE2 提前删了
 *   4. 回到 removeSlot 继续跳 cap.removeInput(3) → 误删 VAE3
 *
 * 进入这个父作业期间，onConnectionsChange 不应该走同步同步逻辑，而是交由父作业结束
 * 后统一 syncFromInputs。
 */
let structuralOpDepth = 0;

function runStructural<T>(fn: () => T): T {
  structuralOpDepth++;
  try {
    return fn();
  } finally {
    structuralOpDepth--;
  }
}

export function isInStructuralOp(): boolean {
  return structuralOpDepth > 0;
}

function nodeInputs(node: LGraphNode): INodeSlot[] {
  return node.inputs ?? [];
}

function findInputIndexByName(node: LGraphNode, name: string): number {
  return nodeInputs(node).findIndex((s) => s.name === name);
}

function isParamSlot(slot: INodeSlot): boolean {
  return !!slot.name && slot.name.startsWith('param_');
}

function getUpstreamType(node: LGraphNode, link: LLink | null): string {
  if (!link || !node.graph) return '*';
  const up = node.graph.getNodeById?.(link.origin_id);
  const out = up?.outputs?.[link.origin_slot];
  return out?.type || link.type || '*';
}

/* -------------------------------------------------------------------------- */
/* meta 计算                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * 直接读取节点上的真实 param_* input，构造 meta。
 *
 * 注意：本函数不会修改任何状态，纯派生。
 */
export function buildMetaFromNode(node: LGraphNode): ParamHubMeta {
  const slots: SlotMeta[] = [];
  for (const inp of nodeInputs(node)) {
    if (!isParamSlot(inp)) continue;
    const labeled = inp as INodeSlot & { label?: string };
    slots.push({
      name: inp.name,
      label: labeled.label || inp.name,
      type: typeof inp.type === 'string' ? inp.type : '*',
    });
  }
  return { version: 1, slots };
}

/* -------------------------------------------------------------------------- */
/* 节点结构变更（核心入口）                                                    */
/* -------------------------------------------------------------------------- */

/**
 * 删除节点上所有 param_* input（用于 onCreate / onConfigure 重建前清场）
 */
export function clearAllParamInputs(node: LGraphNode): void {
  const cap = node as unknown as AddInputCapable;
  if (typeof cap.removeInput !== 'function') return;

  runStructural(() => {
    // 倒序删避免索引漂移
    const inputs = nodeInputs(node);
    for (let i = inputs.length - 1; i >= 0; i--) {
      const inp = inputs[i];
      if (!inp || !isParamSlot(inp)) continue;
      // 不手动 disconnectInput：removeInput 内部会负责断开 link，
      // 额外调用反而会多触发一轮 onConnectionsChange。
      cap.removeInput!.call(node, i);
    }
  });
}

/**
 * 按 meta 重建节点 param_* inputs（先清空再添加）
 */
export function rebuildInputsFromMeta(
  node: LGraphNode,
  meta: ParamHubMeta,
): void {
  const cap = node as unknown as AddInputCapable;
  if (typeof cap.addInput !== 'function') return;

  runStructural(() => {
    clearAllParamInputs(node);
    for (const slot of meta.slots) {
      const added = cap.addInput!.call(node, slot.name, slot.type || '*');
      if (added) {
        (added as INodeSlot & { label?: string }).label =
          slot.label || slot.name;
      }
    }
  });
}

/**
 * 按 meta 对齐节点 param_* inputs（保留已有连线）
 *
 * 与 rebuildInputsFromMeta 不同，本函数走"就地修补"路线：
 *   - 不删除任何带 link 的 input，避免误断用户连线
 *   - 按 meta.slots 顺序对位修改前 N 个 param_* input 的 name/type/label
 *   - meta 比当前 input 少：尾部多余且无 link 的 param_* input 删掉
 *   - meta 比当前 input 多：末尾 addInput 补足
 *
 * 主要用于 backfill 场景（React bundle 晚于 workflow 加载，节点已被 ComfyUI
 * 创建并完成 link 反序列化）。此时若像 onConfigure 那样 clear-and-rebuild，
 * removeInput 会真的把已经建立的 link 一并销毁。
 */
export function reconcileInputsToMeta(
  node: LGraphNode,
  meta: ParamHubMeta,
): void {
  const cap = node as unknown as AddInputCapable;
  if (typeof cap.addInput !== 'function') return;

  runStructural(() => {
    // 1) 收集当前 param_* input 的下标
    const inputs = nodeInputs(node);
    const paramIdx: number[] = [];
    for (let i = 0; i < inputs.length; i++) {
      const inp = inputs[i];
      if (inp && isParamSlot(inp)) paramIdx.push(i);
    }

    const targetSlots = meta.slots;

    // 2) 前 N 个对位修补属性
    const overlap = Math.min(paramIdx.length, targetSlots.length);
    for (let k = 0; k < overlap; k++) {
      const inp = node.inputs![paramIdx[k]!] as INodeSlot & { label?: string };
      const target = targetSlots[k]!;
      inp.name = target.name;
      inp.type = target.type || '*';
      inp.label = target.label || target.name;
    }

    // 3) 当前比 meta 多：尾部多余 input 倒序删除（仅删无 link 的）
    if (paramIdx.length > targetSlots.length && typeof cap.removeInput === 'function') {
      for (let k = paramIdx.length - 1; k >= targetSlots.length; k--) {
        const i = paramIdx[k]!;
        const slot = node.inputs![i];
        if (slot && slot.link == null) {
          cap.removeInput.call(node, i);
        }
      }
    }

    // 4) meta 比当前多：末尾 addInput 补足
    if (targetSlots.length > paramIdx.length) {
      for (let k = paramIdx.length; k < targetSlots.length; k++) {
        const target = targetSlots[k]!;
        const added = cap.addInput!.call(node, target.name, target.type || '*');
        if (added) {
          (added as INodeSlot & { label?: string }).label =
            target.label || target.name;
        }
      }
    }
  });
}

/**
 * 维护"末尾恰好一个空闲槽"不变量（直接改节点 inputs）
 *
 * 清理策略：
 *   - 保留所有已连线（link != null）的槽
 *   - 末尾保留恰好 1 个空闲槽作为占位
 *   - 中间断开连接的槽全部删除（避免界面上出现冗余的断开 VAE）
 *
 * 返回：是否发生了 input 数组结构变化
 */
function ensureExactlyOneEmptySlot(node: LGraphNode): boolean {
  const cap = node as unknown as AddInputCapable;
  let changed = false;

  // 1. 收集当前 param_* 槽下标
  const inputs = nodeInputs(node);
  const paramIdx: number[] = [];
  for (let i = 0; i < inputs.length; i++) {
    const inp = inputs[i];
    if (inp && isParamSlot(inp)) paramIdx.push(i);
  }

  // 2. 从尾部往前累计「连续空闲槽」
  const isEmpty = (i: number) => node.inputs![i]?.link == null;
  const trailingEmpty: number[] = [];
  for (let k = paramIdx.length - 1; k >= 0; k--) {
    const i = paramIdx[k]!;
    if (isEmpty(i)) trailingEmpty.unshift(i);
    else break;
  }

  // 3. 删除中间所有断开连接的槽（不在 trailingEmpty 中的空槽）
  if (typeof cap.removeInput === 'function') {
    const middleEmpty = paramIdx.filter(i => isEmpty(i) && !trailingEmpty.includes(i));
    // 倒序删避免下标漂移
    for (const i of middleEmpty.sort((a, b) => b - a)) {
      cap.removeInput.call(node, i);
      changed = true;
    }
  }

  // 4. 末尾保留恰好 1 个空闲槽，多余的尾部空闲槽倒序删
  if (trailingEmpty.length > 1) {
    const keep = trailingEmpty[trailingEmpty.length - 1];
    const toRemove = trailingEmpty
      .filter((i) => i !== keep)
      .sort((a, b) => b - a);
    if (typeof cap.removeInput === 'function') {
      for (const i of toRemove) {
        cap.removeInput.call(node, i);
        changed = true;
      }
    }
  }

  // 5. 末尾没有空闲槽时追加 param_<next>
  if (trailingEmpty.length === 0) {
    const meta = buildMetaFromNode(node);
    if (meta.slots.length < MAX_PARAM_SLOTS && typeof cap.addInput === 'function') {
      const idx = nextAvailableSlotIndex(meta);
      if (idx > 0) {
        const name = slotInputName(idx);
        const added = cap.addInput.call(node, name, '*');
        if (added) {
          (added as INodeSlot & { label?: string }).label = name;
        }
        changed = true;
      }
    }
  }

  return changed;
}

/**
 * 把节点 inputs 状态持久化到 widget + 写入 Zustand Store
 */
function publishMeta(node: LGraphNode, meta: ParamHubMeta): void {
  const w = findWidget(node, META_WIDGET);
  if (w) {
    const serialized = JSON.stringify(meta);
    if (w.value !== serialized) {
      w.value = serialized;
    }
  }
  useParamHubStore.getState().setHubMeta(node.id, meta);
  (
    node.graph as unknown as {
      setDirtyCanvas?: (a: boolean, b: boolean) => void;
    }
  )?.setDirtyCanvas?.(true, true);
}

/**
 * 统一同步入口：节点 inputs → meta + widget + bus
 *
 * 调用时机：
 *   - 用户连/断线（onConnectionsChange）
 *   - 用户重命名 / 删除槽
 *   - 用户手动添加空闲槽
 *
 * 内部会先维护"恰好一个空闲槽"，然后才计算/广播 meta。
 */
export function syncFromInputs(node: LGraphNode): ParamHubMeta {
  ensureExactlyOneEmptySlot(node);
  const meta = buildMetaFromNode(node);
  publishMeta(node, meta);
  return meta;
}

/**
 * onCreate / onConfigure 入口：
 *   1. 解析 widget 中的 meta
 *   2. 清空预留槽并按 meta 重建（解决 ComfyUI 把 32 个 optional 全铺出来的问题）
 *   3. 调用 syncFromInputs 维护不变量并广播
 *
 * @param opts.preserveLinks 走"就地对齐"路线，不删除带 link 的 input。
 *   适用于 backfill 场景：节点已被 ComfyUI 创建且 link 已反序列化完成，
 *   再像 onConfigure 那样 clear-and-rebuild 会真的销毁用户连线。
 */
export function syncFromMeta(
  node: LGraphNode,
  opts: { preserveLinks?: boolean } = {},
): ParamHubMeta {
  const w = findWidget(node, META_WIDGET);
  const meta = w ? safeParseHubMeta(w.value) : emptyHubMeta();
  if (opts.preserveLinks) {
    reconcileInputsToMeta(node, meta);
  } else {
    rebuildInputsFromMeta(node, meta);
  }
  const result = syncFromInputs(node);
  // input 数量变化必然影响节点高度，主动收敛一次（widget 容器尺寸不变时
  // ResizeObserver 不会触发，所以这里必须显式调用）
  fitNodeHeight(node);
  return result;
}

/* -------------------------------------------------------------------------- */
/* 连线变化的细节处理                                                          */
/* -------------------------------------------------------------------------- */

/**
 * 在调 syncFromInputs 之前，先按连线事件更新 inputs[i] 的 type/label
 */
export function applyConnectionEvent(
  node: LGraphNode,
  index: number,
  connected: boolean,
  link: LLink | null,
): void {
  const inputs = nodeInputs(node);
  const slot = inputs[index];
  if (!slot || !isParamSlot(slot)) return;
  const labeled = slot as INodeSlot & { label?: string };
  if (connected) {
    const upType = getUpstreamType(node, link);
    slot.type = upType;
    // 如果 label 还是默认 name，则用上游类型作为初始 label
    if (!labeled.label || labeled.label === slot.name) {
      labeled.label = upType !== '*' ? upType : slot.name;
    }
  } else {
    // 断开连线：保留用户的自定义命名与类型期望，不做无脑重置。
    //
    //   - 用户改过 label（label !== slot.name）：保留，方便后续重连同名/同义源
    //   - 用户没改过 label：回归底层 name 作为占位
    //   - type 保留为断开前的类型：方便用户重新接同类源；
    //     若要换其它类型，connect 分支会以上游类型覆盖
    if (!labeled.label) {
      labeled.label = slot.name;
    }
  }
}

/**
 * 用户在 React 面板里改名
 */
export function renameSlot(
  node: LGraphNode,
  slotName: string,
  newLabel: string,
): void {
  const idx = findInputIndexByName(node, slotName);
  if (idx < 0) return;
  const inp = node.inputs![idx] as INodeSlot & { label?: string };
  inp.label = (newLabel || slotName).trim() || slotName;
  syncFromInputs(node);
}

/**
 * 用户删除一个槽
 *
 * 重要：不手动调用 disconnectInput，removeInput 内部会自动断开 link。
 * 另外整个过程要包裹在 runStructural 里，避免
 * onConnectionsChange 重入同步调用 syncFromInputs 从而误删后续槽。
 */
export function removeSlot(node: LGraphNode, slotName: string): void {
  const cap = node as unknown as AddInputCapable;
  const idx = findInputIndexByName(node, slotName);
  if (idx < 0) return;

  runStructural(() => {
    if (typeof cap.removeInput === 'function') {
      cap.removeInput.call(node, idx);
    }
  });
  syncFromInputs(node);
}

/**
 * 用户主动追加一个空闲槽（一般无需手动调用）
 */
export function addEmptySlot(node: LGraphNode): void {
  syncFromInputs(node);
}
