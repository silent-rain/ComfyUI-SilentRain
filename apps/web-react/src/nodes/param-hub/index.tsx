/* ParamHub Node */

// import { useParamHubStore } from "../../store";
import { ISlotType } from "../../enums/comfy";
import type { ComfyExtension } from "@comfyorg/comfyui-frontend-types";



const NODE_NAME = "ParamHub";


const ParamHub = (): ComfyExtension => {
    return {
        name: NODE_NAME,
        init: async (_app) => {
            // Node initialization
        },
        setup: async (_app) => {
            // Node setup
        },
        loadedGraphNode: (node, _app) => {
            if (node.type !== NODE_NAME) return;
            // Graph node loaded callback
            console.log("loaded Graph Node:", node);
            console.log("loaded _app Node:", _app);
            console.log("loaded Graph Node widgets_values:", (node as any).widgets_values);
        },
        nodeCreated: (_node, _app) => {
            // Node created callback
        },
        getCanvasMenuItems: (_canvas) => {
            return [];
        },
        beforeRegisterNodeDef: async (nodeType, nodeData, _app) => {
            // Only handle specific node
            if (nodeData.name !== NODE_NAME) return;



            nodeType.prototype.onConnectionsChange = function (type, index, isConnected, link_info) {
                if (!link_info) return;

                // const _paramHubStore = useParamHubStore();
                // console.log("===================: ", _paramHubStore.getHub(1))

                const id = _app.rootGraph.id;
                console.log("===================: ", _app)
                console.log("===================: ", _app.rootGraph)
                console.log("===================: ", id)

                console.log("0", type, index, isConnected, link_info)
                if (type !== ISlotType.Input) return;

                if (isConnected) {
                    console.log("11", type, index, isConnected, link_info)

                    // 更新输入slot的类型
                    if (this.inputs[index]) {
                        // this.inputs[index].name = String(link_info!.type);
                        // this.inputs[index].label = String(link_info!.type);
                        this.inputs[index].type = link_info!.type;
                    }

                    // if (linkCount === 0) {
                    //     // 添加一个空闲slot
                    //     const newIndex = inputTotal + 1;
                    //     this.addInput(`param_${newIndex}`, '*');
                    // }

                } else {
                    console.log("22", type, index, isConnected, link_info)

                    // 延迟移除
                    setTimeout(() => {
                        if (this.inputs[index]?.link) {
                            // 如果有连接，则不移除
                            return
                        }
                        this.removeInput(index);
                        console.log(`removeInput: ${index}`);
                    }, 1500);

                    // 移除所有的空闲slot
                    // const inputTotal = this.inputs.length;
                    // const inputTotal = this.inputs.length;
                    // let linkCount = this.inputs.filter((slot) => !slot.link).length;
                    // console.log(`inputTotal: ${inputTotal}, linkCount: ${linkCount}`);
                    // for (let i = 0; i < inputTotal - 1; i++) {
                    //     const input = this.inputs[i];
                    //     if (!input) continue;

                    //     // 保留至少一个空闲slot
                    //     if (linkCount <= 1) {
                    //         break;
                    //     }

                    //     if (!input.link) {
                    //         this.removeInput(i);
                    //         linkCount -= 1;
                    //     }
                    // }
                }


                {
                    // 添加一个空闲slot
                    const inputTotal = this.inputs.length;
                    const linkCount = this.inputs.filter((slot) => !slot.link).length;
                    console.log(`inputTotal: ${inputTotal}, linkCount: ${linkCount}`);
                    if (linkCount === 0) {
                        const newIndex = inputTotal + 1;
                        this.addInput(`param_${newIndex}`, '*');
                    }
                }

                // this.inputs
                console.log(this.inputs)






                // let nameCount = 0;
                // for (const item of this.inputs) {
                //     nameCount += 1;
                //     const name = `string_${nameCount}`;
                //     item.name = name;
                //     item.label = name;
                // }
            };
        },
    };
};



export default ParamHub;