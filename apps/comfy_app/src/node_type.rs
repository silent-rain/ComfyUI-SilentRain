//! 节点类型的对象

use std::collections::HashMap;

use js_sys::{Object, Reflect};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

use crate::LGraphNode;

/// 连接信息封装
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LinkInfo {
    id: f32,
    #[serde(rename = "type")]
    link_type: String,
    origin_id: u32,
    origin_slot: u32,
    target_id: u32,
    target_slot: u32,
    #[serde(rename = "_data")]
    data: Value,
    #[serde(rename = "_pos")]
    pos: HashMap<Value, Value>,
}

impl LinkInfo {
    /// 从 JsValue 反序列化
    pub fn from_js(js_value: JsValue) -> Result<LinkInfo, JsValue> {
        let result: LinkInfo = serde_wasm_bindgen::from_value(js_value)?;
        Ok(result)
    }

    /// 序列化为 JsValue
    pub fn to_js(&self) -> Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&self)?)
    }
}

impl TryFrom<LinkInfo> for JsValue {
    type Error = JsValue;

    fn try_from(value: LinkInfo) -> Result<Self, Self::Error> {
        value.to_js()
    }
}

/// 节点类型封装
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct NodeType {
    inner: Object,
}

#[wasm_bindgen]
impl NodeType {
    #[wasm_bindgen(constructor)]
    pub fn new(inner: Object) -> Self {
        Self { inner }
    }

    /// 获取inner对象
    pub fn get_inner(&self) -> Object {
        self.inner.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn prototype(&self) -> Result<LGraphNode, JsValue> {
        let prototype: JsValue = Reflect::get(&self.inner, &"prototype".into())?;
        Ok(LGraphNode::new(prototype))
    }

    // /// 获取输入槽位列表
    // #[wasm_bindgen(getter)]
    // pub fn inputs(&self) -> Result<Array, JsValue> {
    //     let prototype = Reflect::get(&self.inner, &"prototype".into())?;
    //     Reflect::get(&prototype, &"inputs".into())?.dyn_into::<Array>()
    // }

    // /// 设置输入槽位列表
    // #[wasm_bindgen(setter)]
    // pub fn set_inputs(&self, inputs: &Array) -> Result<bool, JsValue> {
    //     let prototype = Reflect::get(&self.inner, &"prototype".into())?;
    //     Reflect::set(&prototype, &"inputs".into(), inputs)
    // }

    // /// 获取输出槽位列表
    // #[wasm_bindgen(getter)]
    // pub fn outputs(&self) -> Result<Array, JsValue> {
    //     let prototype = Reflect::get(&self.inner, &"prototype".into())?;
    //     Reflect::get(&prototype, &"outputs".into())?.dyn_into::<Array>()
    // }

    // /// 设置输出槽位列表
    // #[wasm_bindgen(setter)]
    // pub fn set_outputs(&self, outputs: &Array) -> Result<bool, JsValue> {
    //     let prototype = Reflect::get(&self.inner, &"prototype".into())?;
    //     Reflect::set(&prototype, &"outputs".into(), outputs)
    // }

    // /// 获取小部件列表
    // #[wasm_bindgen(getter)]
    // pub fn widgets(&self) -> Result<Array, JsValue> {
    //     // 检查节点类型本身
    //     console::log_1(&"Checking nodeType itself".into());
    //     if let Ok(widgets) = Reflect::get(&self.inner, &"widgets".into()) {
    //         console::log_1(&format!("nodeType.widgets1: {:#?}", widgets).into());
    //     }

    //     // 检查原型
    //     console::log_1(&"Checking prototype".into());
    //     let prototype = Reflect::get(&self.inner, &"prototype".into())?;
    //     console::log_1(&"Checking prototype2".into());
    //     if let Ok(widgets) = Reflect::get(&prototype, &"widgets".into()) {
    //         console::log_1(&format!("prototype.widgets2: {:#?}", widgets).into());
    //     }

    //     // 打印 nodeType 的所有属性
    //     let keys = Object::keys(&self.inner);
    //     console::log_1(&"nodeType keys:".into());
    //     console::log_1(&keys);

    //     // 1. 获取 nodeData
    //     let node_data = Reflect::get(&self.inner, &"nodeData".into())?;
    //     console::log_1(&"nodeData:".into());
    //     console::log_1(&node_data);

    //     // 2. 从 nodeData 获取 widgets
    //     let widgets = Reflect::get(&node_data, &"widgets".into())?;
    //     console::log_1(&"widgets from nodeData:".into());
    //     console::log_1(&widgets);

    //     let prototype = Reflect::get(&self.inner, &"prototype".into())?;
    //     Reflect::get(&prototype, &"widgets".into())?.dyn_into::<Array>()
    // }

    // /// 设置小部件列表
    // #[wasm_bindgen(setter)]
    // pub fn set_widgets(&self, widgets: &Array) -> Result<bool, JsValue> {
    //     let prototype = Reflect::get(&self.inner, &"prototype".into())?;
    //     Reflect::set(&prototype, &"widgets".into(), widgets)
    // }

    // /// 添加输入槽位
    // pub fn add_input(&self, name: &str, slot_type: &str) -> Result<(), JsValue> {
    //     let prototype = Reflect::get(&self.inner, &"prototype".into())?;
    //     let add_input_fn = Reflect::get(&prototype, &"addInput".into())?.dyn_into::<Function>()?;

    //     add_input_fn.call2(
    //         &prototype,
    //         &JsValue::from_str(name),
    //         &JsValue::from_str(slot_type),
    //     )?;

    //     Ok(())
    // }

    // /// 移除输入槽位
    // pub fn remove_input(&self, index: i32) -> Result<(), JsValue> {
    //     let prototype = Reflect::get(&self.inner, &"prototype".into())?;
    //     let remove_input_fn =
    //         Reflect::get(&prototype, &"removeInput".into())?.dyn_into::<Function>()?;

    //     remove_input_fn.call1(&prototype, &JsValue::from_f64(index as f64))?;

    //     Ok(())
    // }
}

impl NodeType {
    // /// 断开输入连接
    // ///
    // /// binding: disconnectInput
    // pub fn disconnect_input(&self, slot: i32) -> Result<(), JsValue> {
    //     let prototype = Reflect::get(&self.inner, &"prototype".into())?;
    //     let handler =
    //         Reflect::get(&prototype, &"disconnectInput".into())?.dyn_into::<Function>()?;

    //     handler.call1(&prototype, &JsValue::from_f64(slot as f64))?;

    //     Ok(())
    // }

    // /// 获取输入槽位
    // pub fn get_input(&self, index: usize) -> Result<SlotInfo, JsValue> {
    //     let inputs = self.inputs()?;
    //     let input = inputs.get(index as u32);
    //     SlotInfo::from_js(input)
    // }

    // /// 获取输出槽位
    // pub fn get_output(&self, index: usize) -> Result<SlotInfo, JsValue> {
    //     let outputs = self.outputs()?;
    //     let output = outputs.get(index as u32);
    //     SlotInfo::from_js(output)
    // }

    // /// 获取小部件
    // pub fn get_widget(&self, index: usize) -> Result<Widget, JsValue> {
    //     let widgets = self.widgets()?;
    //     let widget = widgets.get(index as u32);
    //     Widget::from_js(widget)
    // }

    // /// 设置小部件值
    // pub fn set_widget_value(&self, index: usize, value: f64) -> Result<(), JsValue> {
    //     let widgets = self.widgets()?;
    //     let widget = widgets.get(index as u32);

    //     if !widget.is_undefined() {
    //         Reflect::set(&widget, &"value".into(), &JsValue::from_f64(value))?;
    //     }

    //     Ok(())
    // }

    // /// 设置小部件选项最大值
    // pub fn set_widget_option_max(&self, index: usize, max: f64) -> Result<(), JsValue> {
    //     let widgets = self.widgets()?;
    //     let widget = widgets.get(index as u32);

    //     if !widget.is_undefined() {
    //         let options = Reflect::get(&widget, &"options".into())?;
    //         Reflect::set(&options, &"max".into(), &JsValue::from_f64(max))?;
    //     }

    //     Ok(())
    // }

    // /// 设置输出槽位类型
    // pub fn set_output_type(&self, index: usize, slot_type: &str) -> Result<(), JsValue> {
    //     let outputs = self.outputs()?;
    //     let output = outputs.get(index as u32);

    //     if !output.is_undefined() {
    //         Reflect::set(&output, &"type".into(), &JsValue::from_str(slot_type))?;
    //     }

    //     Ok(())
    // }

    // /// 设置输出槽位标签
    // pub fn set_output_label(&self, index: usize, label: &str) -> Result<(), JsValue> {
    //     let outputs = self.outputs()?;
    //     let output = outputs.get(index as u32);

    //     if !output.is_undefined() {
    //         Reflect::set(&output, &"label".into(), &JsValue::from_str(label))?;
    //     }

    //     Ok(())
    // }

    // /// 设置输出槽位名称
    // pub fn set_output_name(&self, index: usize, name: &str) -> Result<(), JsValue> {
    //     let outputs = self.outputs()?;
    //     let output = outputs.get(index as u32);

    //     if !output.is_undefined() {
    //         Reflect::set(&output, &"name".into(), &JsValue::from_str(name))?;
    //     }

    //     Ok(())
    // }

    // /// 设置输入槽位类型
    // pub fn set_input_type(&self, index: usize, slot_type: &str) -> Result<(), JsValue> {
    //     let inputs = self.inputs()?;
    //     let input = inputs.get(index as u32);

    //     if !input.is_undefined() {
    //         Reflect::set(&input, &"type".into(), &JsValue::from_str(slot_type))?;
    //     }

    //     Ok(())
    // }

    // /// 设置输入槽位名称
    // pub fn set_input_name(&self, index: usize, name: &str) -> Result<(), JsValue> {
    //     let inputs = self.inputs()?;
    //     let input = inputs.get(index as u32);

    //     if !input.is_undefined() {
    //         Reflect::set(&input, &"name".into(), &JsValue::from_str(name))?;
    //     }

    //     Ok(())
    // }
}

impl From<NodeType> for Object {
    fn from(value: NodeType) -> Self {
        value.get_inner()
    }
}
