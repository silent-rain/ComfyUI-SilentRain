//! 节点类型的对象

use std::collections::HashMap;

use js_sys::{Object, Reflect};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use wasm_bindgen::{
    JsValue,
    prelude::{Closure, wasm_bindgen},
};
use web_sys::console;

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
    fn get_inner(&self) -> Object {
        self.inner.clone()
    }
}

impl NodeType {
    /// 设置原型方法 `onConnectionsChange`
    pub fn on_connections_change<F>(&self, handler: F) -> Result<(), JsValue>
    where
        F: Fn(i32, i32, i32, Option<LinkInfo>) -> Result<(), JsValue> + 'static,
    {
        // 将 Rust 闭包转换为 JavaScript 函数
        let handler = Closure::wrap(Box::new(
            move |r#type: i32, index: i32, connected: i32, link_info: JsValue| {
                if link_info.is_null() {
                    return handler(r#type, index, connected, None);
                }

                console::log_1(&format!("link_info: {:#?}", link_info).into());
                let link_info = LinkInfo::from_js(link_info.clone())?;

                handler(r#type, index, connected, Some(link_info))
            },
        )
            as Box<dyn Fn(i32, i32, i32, JsValue) -> Result<(), JsValue>>);

        // 获取原型对象
        let prototype = Reflect::get(&self.inner, &"prototype".into())?;

        // 将 `onConnectionsChange` 方法设置到原型上
        Reflect::set(
            &prototype,
            &"onConnectionsChange".into(),
            &handler.as_ref().clone(),
        )?;

        // 保持闭包生命周期
        handler.forget();

        Ok(())
    }
}

impl From<NodeType> for Object {
    fn from(value: NodeType) -> Self {
        value.get_inner()
    }
}
