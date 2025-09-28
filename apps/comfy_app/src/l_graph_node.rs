use js_sys::{Boolean, Reflect};
use wasm_bindgen::{
    JsCast, JsValue,
    prelude::{Closure, wasm_bindgen},
};

use crate::node_type::LinkInfo;

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct LGraphNode {
    inner: JsValue,
}

#[wasm_bindgen]
impl LGraphNode {
    #[wasm_bindgen(constructor)]
    pub fn new(inner: JsValue) -> Self {
        Self { inner }
    }

    /// 获取inner对象
    pub fn get_inner(&self) -> JsValue {
        self.inner.clone()
    }
}

#[wasm_bindgen]
impl LGraphNode {
    #[wasm_bindgen(getter)]
    pub fn comfy_class(&self) -> Result<String, JsValue> {
        Reflect::get(&self.inner, &"comfyClass".into())
            .and_then(|v| v.as_string().ok_or(JsValue::NULL))
    }

    #[wasm_bindgen(getter)]
    pub fn cm_menu_added(&self) -> Result<Boolean, JsValue> {
        Reflect::get(&self.inner, &"cm_menu_added".into())?.dyn_into()
    }
}

impl LGraphNode {
    /// 设置onConnectionsChange钩子
    ///
    /// binding: onConnectionsChange
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

                let link_info = LinkInfo::from_js(link_info.clone())?;

                handler(r#type, index, connected, Some(link_info))
            },
        )
            as Box<dyn Fn(i32, i32, i32, JsValue) -> Result<(), JsValue>>);

        // 将 `onConnectionsChange` 方法设置到原型上
        Reflect::set(
            &self.inner,
            &"onConnectionsChange".into(),
            &handler.as_ref().clone(),
        )?;

        // 保持闭包生命周期
        handler.forget();

        Ok(())
    }
}
