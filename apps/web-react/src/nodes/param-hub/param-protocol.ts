// 与后端 Rust ParamHub / ParamPort 共享的协议类型
// 必须与 apps/server/src/utils/param_hub.rs 的 serde 结构体保持一致

/** 后端预留的最大动态 input 槽数 (与 Rust 端 MAX_PARAM_SLOTS 保持一致) */
export const MAX_PARAM_SLOTS = 32;

/** 自定义连线类型代号 (与 Rust 端 SR_PARAMS_TYPE 保持一致) */
export const SR_PARAMS_TYPE = "SR_PARAMS";

/** 单个槽的元数据 */
export interface SlotMeta {
  /** 后端 input 名 (param_1 ... param_N)，与 Rust INPUT_TYPES 中的 key 一致 */
  name: string;
  /** 用户重命名后的显示名 (例如 "VAE2")，同时也是 dict key */
  label: string;
  /** 当前连线类型，用于前端展示徽标；未连线时为 "*" */
  type: string;
}

/** ParamHub 持久化数据 (写入 hidden widget params_meta_json) */
export interface ParamHubMeta {
  version: number;
  slots: SlotMeta[];
}

export const PARAM_PROTOCOL_VERSION = 1;

export function emptyHubMeta(): ParamHubMeta {
  return { version: PARAM_PROTOCOL_VERSION, slots: [] };
}

/** 从节点 inputs 中读取 param_* 槽，构造 meta（纯派生，不改节点） */
export function buildMetaFromNode(node: { inputs?: Array<{ name: string; type: string | number; label?: string }> }): ParamHubMeta {
  const slots: SlotMeta[] = [];
  for (const inp of node.inputs ?? []) {
    if (!inp.name?.startsWith('param_')) continue;
    slots.push({
      name: inp.name,
      label: inp.label || inp.name,
      type: typeof inp.type === 'string' ? inp.type : '*',
    });
  }
  return { version: PARAM_PROTOCOL_VERSION, slots };
}

export function safeParseHubMeta(raw: unknown): ParamHubMeta {
  if (typeof raw !== "string" || !raw.trim()) return emptyHubMeta();
  try {
    const obj = JSON.parse(raw);
    if (!obj || typeof obj !== "object") return emptyHubMeta();
    const slots: SlotMeta[] = Array.isArray(obj.slots)
      ? obj.slots
          .filter((s: unknown) => s && typeof s === "object")
          .map((s: Record<string, unknown>) => ({
            name: String(s.name ?? ""),
            label: String(s.label ?? s.name ?? ""),
            type: String(s.type ?? "*"),
          }))
          .filter((s: SlotMeta) => s.name)
      : [];
    return {
      version: typeof obj.version === "number" ? obj.version : PARAM_PROTOCOL_VERSION,
      slots,
    };
  } catch {
    return emptyHubMeta();
  }
}

/** 生成第 i 个槽对应的后端 input name (1-based) */
export function slotInputName(index1Based: number): string {
  return `param_${index1Based}`;
}

/** 找出最小的可用槽 index (1-based)；返回 -1 表示已用满 */
export function nextAvailableSlotIndex(meta: ParamHubMeta): number {
  const used = new Set(meta.slots.map((s) => s.name));
  for (let i = 1; i <= MAX_PARAM_SLOTS; i++) {
    if (!used.has(slotInputName(i))) return i;
  }
  return -1;
}
