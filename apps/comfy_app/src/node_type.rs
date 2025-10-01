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
}

impl From<NodeType> for Object {
    fn from(node_type: NodeType) -> Self {
        node_type.get_inner()
    }
}
