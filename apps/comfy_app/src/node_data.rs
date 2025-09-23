//! 包含节点信息的对象
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};

/// 节点数据封装
#[wasm_bindgen]
pub struct NodeData {
    inner: Object,
}

#[wasm_bindgen]
impl NodeData {
    #[wasm_bindgen(constructor)]
    pub fn new(inner: Object) -> Self {
        Self { inner }
    }

    /// 获取节点名称
    pub fn name(&self) -> Result<String, JsValue> {
        Reflect::get(&self.inner, &"name".into()).and_then(|v| v.as_string().ok_or(JsValue::NULL))
    }

    /// 获取输入列表
    pub fn inputs(&self) -> Result<Array, JsValue> {
        Reflect::get(&self.inner, &"inputs".into()).and_then(|v| v.dyn_into::<Array>())
    }

    /// 获取输出列表
    pub fn outputs(&self) -> Result<Array, JsValue> {
        Reflect::get(&self.inner, &"outputs".into()).and_then(|v| v.dyn_into::<Array>())
    }
    /// 获取inner对象
    fn get_inner(&self) -> Object {
        self.inner.clone()
    }
}

impl From<NodeData> for Object {
    fn from(value: NodeData) -> Self {
        value.get_inner()
    }
}
