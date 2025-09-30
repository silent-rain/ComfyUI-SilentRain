use js_sys::{Boolean, Function, Reflect};
use serde_repr::{Deserialize_repr, Serialize_repr};
use wasm_bindgen::{
    JsCast, JsValue,
    prelude::{Closure, wasm_bindgen},
};

use crate::{NodeType, node_type::LinkInfo};

/// 连接类型
#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
pub enum ConnectionType {
    /// 未知
    Unknown = 0,
    /// 输入卡槽
    Input = 1,
    /// 输出卡槽
    Output = 2,
}

impl From<i32> for ConnectionType {
    fn from(value: i32) -> Self {
        match value {
            1 => Self::Input,
            2 => Self::Output,
            _ => Self::Unknown,
        }
    }
}

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
    ///
    /// Args:
    /// * this: 内置NodeType对象
    /// * type: 连接类型, 1: input, 2: output
    /// * index: 连接索引
    /// * connected: 是否连接, 1: 连接, 0: 断开
    /// * link_info: 连接信息
    pub fn on_connections_change<F>(&self, handler: F) -> Result<(), JsValue>
    where
        F: Fn(NodeType, ConnectionType, usize, bool, Option<LinkInfo>) -> Result<(), JsValue>
            + 'static,
    {
        // 创建 Rust 闭包，接受 this 作为第一个参数
        let rust_handler = Closure::wrap(Box::new(
            move |this: JsValue, r#type: i32, index: usize, connected: i32, link_info: JsValue| {
                let node_type_this = NodeType::new(this.into());
                let connection_type: ConnectionType = r#type.into();
                let connected = connected == 1;

                if link_info.is_null() {
                    return handler(
                        node_type_this.clone(),
                        connection_type,
                        index,
                        connected,
                        None,
                    );
                }

                let link_info = LinkInfo::from_js(link_info.clone())?;

                // 调用用户提供的处理函数
                handler(
                    node_type_this.clone(),
                    connection_type,
                    index,
                    connected,
                    Some(link_info),
                )
            },
        )
            as Box<dyn Fn(JsValue, i32, usize, i32, JsValue) -> Result<(), JsValue>>);

        // 创建 JavaScript 包装函数, 用于透传 this 对象到rust的闭包
        // 在包装函数中添加错误捕获
        let wrapper_js = r#"
            return function(type, index, connected, linkInfo) {
                try {
                    rustHandler(this, type, index, connected, linkInfo);
                } catch (e) {
                    console.error("Error in onConnectionsChange:", e);
                    throw e;
                }
            };
        "#;

        let create_wrapper = Function::new_with_args("rustHandler", wrapper_js);
        let wrapper = create_wrapper
            .call1(&JsValue::NULL, &rust_handler.as_ref().clone())?
            .dyn_into::<Function>()?;

        Reflect::set(&self.inner, &"onConnectionsChange".into(), &wrapper)?;

        // 保持闭包生命周期
        rust_handler.forget();

        Ok(())
    }
}
