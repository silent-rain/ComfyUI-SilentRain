/**
 * Sr Param Port 节点：从上游 Sr Param Hub 中按上游连线槽，
 * 在本节点上动态生成对应的"输出端口"（每个输出对应 hub 的一个 label）。
 *
 * - 反向追溯 `params` input 的 origin 节点（必须是 ParamHub），读取其 meta
 * - 按 meta.slots 中"已连线"的项，依次维护本节点 outputs：
 *     outputs[0] = label_1 (type_1)
 *     outputs[1] = label_2 (type_2)
 *     ...
 *   未启用的位由前端 removeOutput 清掉（后端预留 32 个 any 输出做兜底）
 * - 把端口顺序写回 hidden widget `out_labels_json`，供后端 execute 按位取值
 * - 通过 Zustand Store 订阅特定 hub 的 meta 变化实现实时联动
 */

import {
  defineReactNode,
  findWidget,
  fitNodeHeight,
  hideWidget,
  type INodeSlot,
  type LGraphNode,
  type LLink,
} from '../../core';
import { subscribeHubMeta, useParamHubStore } from '../../store/paramHubStore';
import {
  buildMetaFromNode,
  type ParamHubMeta,
  type SlotMeta,
} from '../param-hub/param-protocol';

const OUT_LABELS_WIDGET = 'out_labels_json';
const INPUT_PORT_NAME = 'params';

/** 旧版本工作流可能残留的 widget 名（含已废弃的 param_name），统一隐藏 */
const LEGACY_HIDDEN_WIDGETS = new Set<string>(['param_name']);

/**
 * 隐藏所有非"白名单"中的 widget，避免旧工作流中遗留的 widget 显示出来。
 * 白名单：仅保留 out_labels_json（仍需序列化但不显示）。
 */
function hideStaleWidgets(node: LGraphNode): void {
  const widgets = node.widgets ?? [];
  for (const w of widgets) {
    if (w.name === OUT_LABELS_WIDGET || LEGACY_HIDDEN_WIDGETS.has(w.name)) {
      hideWidget(w);
    }
  }
}

interface OutputCapableNode {
  addOutput?: (name: string, type: string, options?: unknown) => INodeSlot;
  removeOutput?: (slot: number) => void;
  disconnectOutput?: (slot: number) => void;
}

/* -------------------------------------------------------------------------- */
/* 上游 meta 读取                                                              */
/* -------------------------------------------------------------------------- */

function readUpstream(node: LGraphNode): {
  linked: boolean;
  meta: ParamHubMeta | null;
  hubNodeId: number | null;
} {
  const graph = node.graph;
  if (!graph) return { linked: false, meta: null, hubNodeId: null };
  const inputs = node.inputs ?? [];
  const idx = inputs.findIndex(s => s.name === INPUT_PORT_NAME);
  if (idx < 0) return { linked: false, meta: null, hubNodeId: null };
  const linkId = inputs[idx]?.link;
  if (linkId == null) return { linked: false, meta: null, hubNodeId: null };
  const link: LLink | undefined = graph.links?.[linkId];
  if (!link) return { linked: false, meta: null, hubNodeId: null };
  const upstream = graph.getNodeById?.(link.origin_id) ?? null;
  if (!upstream) return { linked: true, meta: null, hubNodeId: link.origin_id };

  // 优先从 Zustand store 读取 meta（包含持久化的 label）
  // 因为 upstream.inputs 在 ComfyUI 反序列化后可能丢失 label 属性
  const storeMeta = useParamHubStore.getState().hubs[upstream.id];
  const meta = storeMeta ?? buildMetaFromNode(upstream);

  return {
    linked: true,
    meta,
    hubNodeId: link.origin_id,
  };
}

/** 仅取 hub 上"已连线"的槽 —— 这些才是有意义的可分发参数 */
function pickLiveSlots(meta: ParamHubMeta, hubNode: LGraphNode | null): SlotMeta[] {
  if (!hubNode) return meta.slots.filter(s => s.label && s.label.trim());
  const inputs = hubNode.inputs ?? [];
  const linkedNames = new Set(
    inputs.filter(i => i.link != null && i.name?.startsWith('param_')).map(i => i.name),
  );
  return meta.slots.filter(s => linkedNames.has(s.name) && s.label && s.label.trim());
}

function getHubNode(node: LGraphNode): LGraphNode | null {
  const graph = node.graph;
  if (!graph) return null;
  const inputs = node.inputs ?? [];
  const idx = inputs.findIndex(s => s.name === INPUT_PORT_NAME);
  if (idx < 0) return null;
  const linkId = inputs[idx]?.link;
  if (linkId == null) return null;
  const link: LLink | undefined = graph.links?.[linkId];
  if (!link) return null;
  return graph.getNodeById?.(link.origin_id) ?? null;
}

/* -------------------------------------------------------------------------- */
/* outputs 同步                                                                */
/* -------------------------------------------------------------------------- */

/**
 * 用目标 slots 列表去校准 node.outputs：
 * - 已存在且 label/type 一致：保留
 * - 否则：**就地修补** name/type，避免 removeOutput 把已建立的下游连线一并销毁
 * - 多余的尾部 output：仅删除"无 link"的；带 link 的保留以维持用户工作流
 * - 不足：末尾 addOutput 补足
 */
function syncOutputs(node: LGraphNode, target: SlotMeta[]): boolean {
  const cap = node as unknown as OutputCapableNode;
  if (typeof cap.addOutput !== 'function') return false;

  const current = node.outputs ?? [];

  // 快速比较：完全一致就跳过
  const same =
    current.length === target.length &&
    current.every(
      (o, i) => o.name === target[i]!.label && o.type === (target[i]!.type || '*'),
    );
  if (same) return false;

  let changed = false;

  // 1) 前 N 个对位修补属性（保留 links）
  const overlap = Math.min(current.length, target.length);
  for (let i = 0; i < overlap; i++) {
    const out = current[i]!;
    const t = target[i]!;
    const desiredType = t.type || '*';
    if (out.name !== t.label) {
      out.name = t.label;
      changed = true;
    }
    if (out.type !== desiredType) {
      out.type = desiredType;
      changed = true;
    }
  }

  // 2) 多余尾部：仅删无 link 的输出
  if (current.length > target.length && typeof cap.removeOutput === 'function') {
    for (let i = current.length - 1; i >= target.length; i--) {
      const out = current[i]!;
      const linkIds = out.links ?? [];
      if (linkIds.length === 0) {
        cap.removeOutput.call(node, i);
        changed = true;
      }
    }
  }

  // 3) 不足：末尾追加
  if (target.length > current.length) {
    for (let i = current.length; i < target.length; i++) {
      const t = target[i]!;
      cap.addOutput.call(node, t.label, t.type || '*');
      changed = true;
    }
  }

  return changed;
}

function writeOutLabels(node: LGraphNode, labels: string[]): void {
  const w = findWidget(node, OUT_LABELS_WIDGET);
  if (!w) return;
  const serialized = JSON.stringify(labels);
  if (w.value !== serialized) {
    w.value = serialized;
  }
}

/* 新增：从 Port 自身 widget 读取持久化的 labels */
function readSelfLabels(node: LGraphNode): SlotMeta[] {
  const w = findWidget(node, OUT_LABELS_WIDGET);
  if (!w) return [];
  try {
    const arr = JSON.parse(w.value);
    if (!Array.isArray(arr)) return [];
    return arr
      .filter((s: unknown) => typeof s === 'string' && s.trim())
      .map((label: string, i: number) => ({
        name: `out_${i + 1}`,
        label,
        type: node.outputs?.[i]?.type || '*',
      }));
  } catch {
    return [];
  }
}

/**
 * 端口同步主入口：从上游 hub 读 meta -> 重建 outputs -> 写 widget
 *
 * 刷新恢复时的时序问题：
 *   1. Hub 的 onCreate 同步执行 → Store 中写入空/不完整 meta
 *   2. Port 的 onCreate 同步执行 → 从 Store 读不到正确 meta
 *   3. Hub 的 onConfigure（microtask）→ Store 更新为正确 meta
 *   4. Port 的 subscribeHubMeta 回调 → 再次 refreshPort
 *
 * 因此 refreshPort 需要双重保险：
 *   - 优先使用 Store meta + pickLiveSlots（正常连线场景）
 *   - 如果 Store meta 为空或 liveSlots 不完整，回退到 Port 自身持久化的 labels
 */
function refreshPort(node: LGraphNode): {
  linked: boolean;
  liveSlots: SlotMeta[];
} {
  const r = readUpstream(node);
  const hubNode = getHubNode(node);
  let liveSlots = r.linked && r.meta ? pickLiveSlots(r.meta, hubNode) : [];

  // 回退机制：如果从上游读不到足够有效的 slots（刷新时序问题），
  // 使用 Port 自身 widget 中持久化的 labels 来恢复。
  // 条件 1：上游返回的 liveSlots 数量少于 Port 已持久化的 labels 数量。
  const selfLabels = readSelfLabels(node);
  let useSelfLabels = liveSlots.length < selfLabels.length;

  // 条件 2：store 返回的 label 看起来全是默认值（如 param_1, param_2…），
  // 而 widget 中保存着用户自定义的 label。这种情况发生在 Hub 的 onConfigure
  // 晚于 Port 的 onConfigure 执行，导致 Port 首次刷新时读到的 store 数据
  // 还是 Hub onCreate 写入的不完整默认值。此时优先使用 widget 持久化数据。
  if (!useSelfLabels && liveSlots.length > 0 && selfLabels.length > 0) {
    const defaultLike = liveSlots.filter(
      (s) => s.label === s.name || s.label.startsWith('param_'),
    ).length;
    const selfLike = selfLabels.filter(
      (s) => s.label === s.name || s.label.startsWith('param_'),
    ).length;
    // 当 store 的默认值比例显著高于 widget 时，认为 store 数据不可靠
    if (defaultLike > selfLike) {
      useSelfLabels = true;
    }
  }

  if (useSelfLabels) {
    liveSlots = selfLabels;
  }

  const changed = syncOutputs(node, liveSlots);
  writeOutLabels(
    node,
    liveSlots.map((s) => s.label),
  );
  if (changed) {
    (
      node.graph as unknown as {
        setDirtyCanvas?: (a: boolean, b: boolean) => void;
      }
    )?.setDirtyCanvas?.(true, true);
  }
  // outputs 数量变化必然影响节点高度，主动收敛一次（widget 容器尺寸不变时
  // ResizeObserver 不会触发，所以这里必须显式调用）
  fitNodeHeight(node);
  return { linked: r.linked, liveSlots };
}

/* -------------------------------------------------------------------------- */
/* 扩展定义                                                                    */
/*                                                                            */
/* 设计：本节点没有任何节点内 UI，所以不提供 render（参考 cg-use-everywhere      */
/* 「全局输入」节点：不挂任何 DOM widget，节点高度由 outputs 自然决定）。       */
/* 与上游 hub 的联动通过 onConnectionsChange + bus 订阅完成。                   */
/* -------------------------------------------------------------------------- */

export const ParamPortExtension = defineReactNode({
  comfyClass: 'ParamPort',
  extensionName: 'SilentRain.ParamPort',

  onCreate({ node }) {
    let currentUnsub: (() => void) | null = null;

    /** 绑定到当前上游 hub，订阅其 meta 变化 */
    function bindToHub() {
      currentUnsub?.();
      currentUnsub = null;

      const r = readUpstream(node);
      if (!r.linked || !r.hubNodeId) return;

      currentUnsub = subscribeHubMeta(r.hubNodeId, () => {
        // 再次确认当前上游仍是该 hub（防止连线已切换但回调延迟到达）
        const now = readUpstream(node);
        if (now.hubNodeId !== r.hubNodeId) return;
        refreshPort(node);
      });
    }

    bindToHub();
    (node as unknown as { __sr_port_bind__?: () => void }).__sr_port_bind__ = bindToHub;

    // 隐藏后端预留的 hidden widget + 旧工作流可能残留的 widget（如 param_name）
    hideStaleWidgets(node);

    // backfill 判别：若节点已在 graph._nodes 中，说明 ComfyUI 已经反序列化
    //   完成、outputs 上的 link 表已建立。此时绝不能 disconnect/remove 已有
    //   output，否则会真的销毁用户下游连线。
    const inGraph = !!(
      node.graph &&
      Array.isArray((node.graph as { _nodes?: LGraphNode[] })._nodes) &&
      (node.graph as { _nodes: LGraphNode[] })._nodes.includes(node)
    );

    if (!inGraph) {
      // 新建节点：后端预留了 MAX_PARAM_SLOTS 个 any 输出，全部清掉，
      // 之后由 refreshPort 按上游 hub 的实际 slots 动态 add
      const cap = node as unknown as OutputCapableNode;
      if (typeof cap.removeOutput === 'function') {
        for (let i = (node.outputs ?? []).length - 1; i >= 0; i--) {
          const outs = node.outputs ?? [];
          const out = outs[i]!;
          const linkIds = out.links ?? [];
          if (linkIds.length && typeof cap.disconnectOutput === 'function') {
            cap.disconnectOutput.call(node, i);
          }
          cap.removeOutput.call(node, i);
        }
      }
    }

    refreshPort(node);
  },

  onRemoved({ node }) {
    const ref = node as unknown as { __sr_port_bind__?: () => void };
    ref.__sr_port_bind__?.();
    delete ref.__sr_port_bind__;
  },

  onConnectionsChange({ node, type }) {
    // type === 1 是 INPUT，连/断 hub 时重新绑定订阅并刷新端口
    if (type !== 1) return;
    (node as unknown as { __sr_port_bind__?: () => void }).__sr_port_bind__?.();
    refreshPort(node);
  },

  onConfigure({ node }) {
    // 工作流加载时，根据持久化的 widget + 当前 graph 状态恢复输出端口
    hideStaleWidgets(node);
    // 页面刷新后重新绑定 store 订阅（上游 hub 可能在更晚的 microtask 才完成 configure，
    // 因此必须在此处重新尝试 bindToHub，否则 hub 后续更新 store 时 port 无法感知）
    (node as unknown as { __sr_port_bind__?: () => void }).__sr_port_bind__?.();
    refreshPort(node);
  },
});
