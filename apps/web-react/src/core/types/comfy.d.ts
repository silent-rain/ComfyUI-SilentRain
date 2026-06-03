// ComfyUI / LiteGraph 的最小化类型声明（仅声明本项目使用的部分）
// 真正的类型来自全局 window.app / window.LiteGraph，本文件仅供 TS 静态检查。

export {};

declare global {
  interface Window {
    app?: ComfyApp;
    LiteGraph?: any;
  }
}

export interface ComfyApp {
  registerExtension: (ext: ComfyExtension) => void;
  graph: LGraph;
  canvas: any;
  ui?: any;
}

export interface ComfyExtension {
  name: string;
  init?: () => void | Promise<void>;
  setup?: () => void | Promise<void>;
  beforeRegisterNodeDef?: (
    nodeType: any,
    nodeData: any,
    app: ComfyApp
  ) => void | Promise<void>;
  nodeCreated?: (node: LGraphNode, app: ComfyApp) => void | Promise<void>;
  loadedGraphNode?: (node: LGraphNode, app: ComfyApp) => void;
}

export interface LGraph {
  _nodes: LGraphNode[];
  links: Record<number, LLink>;
  getNodeById?: (id: number) => LGraphNode | null;
}

export interface LLink {
  id: number;
  origin_id: number;
  origin_slot: number;
  target_id: number;
  target_slot: number;
  type: string;
}

export interface LGraphNode {
  id: number;
  type: string;
  comfyClass?: string;
  title: string;
  size: [number, number];
  pos: [number, number];
  widgets?: IWidget[];
  inputs?: INodeSlot[];
  outputs?: INodeSlot[];
  graph?: LGraph;

  addWidget(
    type: string,
    name: string,
    value: any,
    callback?: ((v: any) => void) | null,
    options?: any
  ): IWidget;

  addDOMWidget?(
    name: string,
    type: string,
    element: HTMLElement,
    options?: any
  ): IWidget;

  removeWidget?(widget: IWidget | number): void;

  setSize?(size: [number, number]): void;
  computeSize?(): [number, number];
  onNodeCreated?: () => void;
  onConfigure?: (info: any) => void;
  onConnectionsChange?: (
    type: number,
    index: number,
    connected: boolean,
    link: LLink | null,
    slot: INodeSlot
  ) => void;
  onSerialize?: (info: any) => void;
  onRemoved?: () => void;
}

export interface IWidget {
  name: string;
  type: string;
  value: any;
  options?: any;
  element?: HTMLElement;
  hidden?: boolean;
  computedHeight?: number;
  callback?: (v: any) => void;
  computeSize?: (width: number) => [number, number];
  serializeValue?: (node: LGraphNode, idx: number) => any;
  draw?: (...args: any[]) => void;
  onMouseDown?: (...args: any[]) => boolean;
}

export interface INodeSlot {
  name: string;
  type: string;
  link?: number | null;
  links?: number[] | null;
  widget?: { name: string };
}
