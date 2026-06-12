export declare interface Slot {
  linkId: number;
  /** 后端 input 名 (param_1 ... param_N)，作为 slot 的唯一标识 */
  name: string;
  /** 显示的标签名称，在 EditSlot 窗口中编辑 */
  label: string;
  type: SlotType; // 输入的类型
}

export declare type NodeId = number | string;
export declare type SlotType = number | string;
