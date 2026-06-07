/* ParamPort Node */

import { ISlotType } from "../../enums/comfy";
import type { ComfyExtension } from "@comfyorg/comfyui-frontend-types";



const NODE_NAME = "ParamPort";


const ParamPort = (): ComfyExtension => {
    return {
        // 扩展名的名称
        name: NODE_NAME,
        // 允许任何初始化，例如加载资源。在画布创建后但在添加节点之前调用
        init: async (_app) => {
            // Node initialization
        },
        // 允许在应用程序完全设置并运行后调用任何其他设置
        setup: async (_app) => {
            // Node setup
        },
        // *允许扩展修改已重新加载到图形上的节点。
        // *如果你破坏了后端的某些东西，并想修补前端的工作流
        loadedGraphNode: (node, _app) => {
            if (node.type !== NODE_NAME) return;
            // Graph node loaded callback
            console.log("loaded Graph Node:", node);
            console.log("loaded Graph Node widgets_values:", (node as any).widgets_values);

            // for (let i = 10; i < node.outputs.length - 1; i++) {
            //     const output = node.outputs[i];
            //     console.log("output:", output);
            //     node.removeOutput(i);
            //     // node.addOutput("output", ISlotType.Output)
            // }

            // 自动计算并调整节点大小
            const newSize = node.computeSize();
            // 可以给一个最小宽度，防止节点太窄
            newSize[0] = Math.max(newSize[0], 150);
            node.setSize(newSize);

            // 强制刷新画布
            // node.setDirtyCanvas(true, true);


            node.addOutput("output1", ISlotType.Output)
            node.addOutput("output2", ISlotType.Output)
        },
        // 允许扩展在节点构造函数之后运行代码
        nodeCreated: (_node, _app) => {
            // Node created callback
        },
        // 允许扩展将上下文菜单项添加到画布右键菜单
        getCanvasMenuItems: (_canvas) => {
            return [];
        },
        // *允许扩展在向**LGraph注册节点之前向其添加额外的处理**
        beforeRegisterNodeDef: async (nodeType, nodeData, _app) => {
            // Only handle specific node
            if (nodeData.name !== NODE_NAME) return;

            nodeType.prototype.onConnectionsChange = function (type, _index, isConnected, link_info, inputOrOutput) {
                if (!link_info) return;

                if (type !== ISlotType.Input) return;

                console.log("onConnectionsChange called", inputOrOutput);
                console.log(type, _index, isConnected, link_info);


                // out_labels_json

                const outLabelsJsons = this.inputs.filter((slot) => slot.name === 'out_labels_json');
                if (outLabelsJsons.length === 0) return;

                const outLabelsJson = outLabelsJsons[0];
                console.log("===========:", outLabelsJson);

                if (isConnected) {
                } else {
                }
            };
        },
    };
};



export default ParamPort;