export declare interface Slot {
    linkId: number;
    /** 后端 input 名 (param_1 ... param_N)，与 Rust INPUT_TYPES 中的 key 一致 */
    name?: string;
    type: string | number; // 输入的类型
    value?: any; // 输入的值
}

export declare type NodeId = number | string;