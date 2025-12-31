//! 扩展入口
//!
//! 网页加载时：
//! init → addCustomNodeDefs → getCustomWidgets → beforeRegisterNodeDef → registerCustomNodes → beforeConfigureGraph → nodeCreated → loadedGraphNode → afterConfigureGraph → setup
//!
//! 加载工作流时：
//! beforeConfigureGraph → nodeCreated → loadedGraphNode → afterConfigureGraph
//!
//! 添加新节点时：
//! nodeCreated
//!

use std::fmt;

use js_sys::{Function, Object, Reflect};
use wasm_bindgen::{
    JsCast, JsValue,
    prelude::{Closure, wasm_bindgen},
};
use web_sys::console;

use crate::{ComfyApp, Node, NodeData, NodeType};

/// 扩展
#[wasm_bindgen]
#[derive(Clone)]
pub struct Extension {
    /// 扩展名称, 需要唯一
    #[wasm_bindgen(skip)]
    pub name: String,
    // 扩展对象
    inner: JsValue,
}

/// Debug 展示
impl fmt::Debug for Extension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Extension {{ extension: JsValue, name: {} }}", self.name)
    }
}

impl Extension {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            inner: Object::new().into(),
        }
    }

    /// 获取扩展对象
    pub fn get_inner(&self) -> JsValue {
        self.inner.clone()
    }

    /// 注册扩展
    pub fn _register(self, app: Object) -> Result<(), JsValue> {
        Reflect::set(&self.inner, &"name".into(), &self.name.into())?;

        let register_func =
            Reflect::get(&app, &"registerExtension".into())?.dyn_into::<Function>()?;

        register_func.call1(&app, &self.inner)?;

        Ok(())
    }
}

#[wasm_bindgen]
impl Extension {
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> Result<String, JsValue> {
        Reflect::get(&self.inner, &"name".into()).and_then(|v| v.as_string().ok_or(JsValue::NULL))
    }
}

/// Comfy 钩子（Hooks）
impl Extension {
    /// 设置init钩子
    ///
    /// async init()
    pub fn init<F>(&mut self, handler: F) -> Result<(), JsValue>
    where
        F: FnOnce() -> Result<(), JsValue> + 'static,
    {
        console::log_1(&JsValue::from_str("init"));

        let handler = Closure::once_into_js(move || {
            if let Err(e) = handler() {
                console::error_1(&e);
            }
            JsValue::undefined()
        });

        // 设置init方法
        Reflect::set(&self.inner, &"init ...".into(), &handler)?;

        Ok(())
    }

    /// 设置addCustomNodeDefs钩子
    ///
    /// async addCustomNodeDefs(defs, app)
    pub fn add_custom_node_defs<F>(&mut self, handler: F) -> Result<(), JsValue>
    where
        F: Fn(JsValue, JsValue) -> Result<(), JsValue> + 'static,
    {
        console::log_1(&JsValue::from_str("add_custom_node_defs ..."));

        let handler = Closure::wrap(
            Box::new(handler) as Box<dyn Fn(JsValue, JsValue) -> Result<(), JsValue>>
        );

        // 设置addCustomNodeDefs方法
        Reflect::set(
            &self.inner,
            &"addCustomNodeDefs".into(),
            &handler.as_ref().clone(),
        )?;

        Ok(())
    }

    /// 设置getCustomWidgets钩子
    ///
    /// async getCustomWidgets(app)
    pub fn get_custom_widgets<F>(&mut self, handler: F) -> Result<(), JsValue>
    where
        F: Fn(ComfyApp) -> Result<(), JsValue> + 'static,
    {
        console::log_1(&JsValue::from_str("get_custom_widgets ..."));

        let handler = Closure::wrap(Box::new(move |app: Object| {
            let app = ComfyApp::from_app(app);
            handler(app)
        }) as Box<dyn Fn(Object) -> Result<(), JsValue>>);

        // 设置getCustomWidgets方法
        Reflect::set(
            &self.inner,
            &"getCustomWidgets".into(),
            &handler.as_ref().clone(),
        )?;

        Ok(())
    }

    /// 设置beforeRegisterNodeDef钩子
    ///
    /// async beforeRegisterNodeDef(nodeType, nodeData, app)
    pub fn before_register_node_def<F>(&mut self, handler: F) -> Result<(), JsValue>
    where
        F: Fn(NodeType, NodeData, ComfyApp) -> Result<JsValue, JsValue> + 'static,
    {
        console::log_1(&JsValue::from_str("before_register_node_def ..."));

        let handler = Closure::wrap(Box::new(
            move |node_type: Object, node_data: Object, app: Object| {
                let node_type = NodeType::new(node_type);
                let node_data = NodeData::new(node_data);
                let app = ComfyApp::from_app(app);
                handler(node_type, node_data, app)
            },
        )
            as Box<dyn Fn(Object, Object, Object) -> Result<JsValue, JsValue>>);

        Reflect::set(
            &self.inner,
            &"beforeRegisterNodeDef".into(),
            &handler.as_ref().clone(),
        )?;

        // 保持闭包生命周期
        handler.forget();

        Ok(())
    }

    /// 设置registerCustomNodes钩子
    ///
    /// async registerCustomNodes(app)
    pub fn register_custom_nodes<F>(&mut self, handler: F) -> Result<(), JsValue>
    where
        F: Fn(JsValue) -> Result<JsValue, JsValue> + 'static,
    {
        console::log_1(&JsValue::from_str("register_custom_nodes ..."));

        let handler =
            Closure::wrap(Box::new(handler) as Box<dyn Fn(JsValue) -> Result<JsValue, JsValue>>);

        // 设置registerCustomNodes方法
        Reflect::set(
            &self.inner,
            &"registerCustomNodes".into(),
            &handler.as_ref().clone(),
        )?;

        // 保持闭包生命周期
        handler.forget();

        Ok(())
    }

    /// 设置nodeCreated钩子
    ///
    /// async nodeCreated(node)
    pub fn node_created<F>(&mut self, handler: F) -> Result<(), JsValue>
    where
        F: Fn(Node, ComfyApp) -> Result<JsValue, JsValue> + 'static,
    {
        console::log_1(&JsValue::from_str("node_created ..."));

        let handler = Closure::wrap(Box::new(move |node: Object, app: Object| {
            let node = Node::new(node);
            let app = ComfyApp::from_app(app);
            handler(node, app)
        })
            as Box<dyn Fn(Object, Object) -> Result<JsValue, JsValue>>);

        // 设置nodeCreated方法
        Reflect::set(
            &self.inner,
            &"nodeCreated".into(),
            &handler.as_ref().clone(),
        )?;

        // 保持闭包生命周期
        handler.forget();

        Ok(())
    }

    /// 设置loadedGraphNode钩子
    ///
    /// loadedGraphNode(node, app)
    pub fn loaded_graph_node<F>(&mut self, handler: F) -> Result<(), JsValue>
    where
        F: Fn(Node, ComfyApp) -> Result<JsValue, JsValue> + 'static,
    {
        console::log_1(&JsValue::from_str("loaded_graph_node ..."));

        let handler = Closure::wrap(Box::new(move |node: Object, app: Object| {
            let node = Node::new(node);
            let app = ComfyApp::from_app(app);
            handler(node, app)
        })
            as Box<dyn Fn(Object, Object) -> Result<JsValue, JsValue>>);

        // 设置loadedGraphNode方法
        Reflect::set(
            &self.inner,
            &"loadedGraphNode".into(),
            &handler.as_ref().clone(),
        )?;

        // 保持闭包生命周期
        handler.forget();

        Ok(())
    }

    /// 设置afterConfigureGraph钩子
    ///
    /// afterConfigureGraph: (missingNodeTypes)
    pub fn after_configure_graph<F>(&mut self, handler: F) -> Result<(), JsValue>
    where
        F: Fn(JsValue) -> Result<(), JsValue> + 'static,
    {
        console::log_1(&JsValue::from_str("after_configure_graph ..."));

        let handler =
            Closure::wrap(Box::new(handler) as Box<dyn Fn(JsValue) -> Result<(), JsValue>>);

        // 设置afterConfigureGraph方法
        Reflect::set(
            &self.inner,
            &"afterConfigureGraph".into(),
            &handler.as_ref().clone(),
        )?;

        Ok(())
    }

    /// 设置setup钩子
    ///
    /// async setup()
    ///
    /// 注意：根据文档，setup在启动流程结束时调用，只调用一次
    pub fn setup<F>(&mut self, handler: F) -> Result<(), JsValue>
    where
        F: FnOnce() -> Result<(), JsValue> + 'static,
    {
        console::log_1(&JsValue::from_str("setup ..."));

        // 使用once_into_js处理FnOnce闭包
        let js_func = Closure::once_into_js(move || {
            if let Err(e) = handler() {
                console::error_1(&e);
            }
            JsValue::undefined()
        });

        // 设置setup方法
        Reflect::set(&self.inner, &"setup".into(), &js_func.as_ref().clone())?;

        Ok(())
    }

    /// 设置getCanvasMenuItems钩子
    ///
    /// getCanvasMenuItems(canvas)
    pub fn get_canvas_menu_items<F>(&mut self, handler: F) -> Result<(), JsValue>
    where
        F: Fn(Object) -> Result<JsValue, JsValue> + 'static,
    {
        console::log_1(&JsValue::from_str("get_canvas_menu_items ..."));

        let handler = Closure::wrap(
            Box::new(handler) as Box<dyn Fn(Object) -> Result<JsValue, JsValue>>
        );

        // 设置getCanvasMenuItems方法
        Reflect::set(
            &self.inner,
            &"getCanvasMenuItems".into(),
            &handler.as_ref().clone(),
        )?;

        // 保持闭包生命周期
        handler.forget();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple() -> Result<(), JsValue> {
        Ok(())
    }
}
