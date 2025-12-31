//! Comfy Js App main
//!

use js_sys::{Function, Object, Reflect};
use wasm_bindgen::{JsValue, prelude::*};

use crate::extension::Extension;

#[wasm_bindgen(raw_module = "/scripts/app.js")]
extern "C" {
    // import { app } from "../../scripts/app.js";
    #[wasm_bindgen(thread_local_v2, js_name = app)]
    static APP: Object;
}

#[derive(Debug, Clone)]
#[wasm_bindgen]
pub struct ComfyApp {
    app: Object,
}

#[wasm_bindgen]
impl ComfyApp {
    pub fn new() -> Result<ComfyApp, JsValue> {
        let app_obj = APP.with(|app| app.clone());
        Ok(ComfyApp { app: app_obj })
    }

    pub fn from_app(app: Object) -> ComfyApp {
        ComfyApp { app }
    }

    /// 获取 app 对象
    pub fn app(&self) -> Object {
        self.app.clone()
    }

    /// 注册扩展
    ///
    /// registerExtension
    pub fn register_extension(&self, extension: &Extension) -> Result<(), JsValue> {
        let extension_obj = extension.get_inner();

        // 设置name属性
        Reflect::set(
            &extension_obj,
            &"name".into(),
            &extension.name.clone().into(),
        )?;

        // 获取 registerExtension 方法
        let register_func =
            Reflect::get(&self.app, &"registerExtension".into())?.dyn_into::<Function>()?;

        // 调用 registerExtension
        register_func.call1(&self.app, &extension_obj)?;

        Ok(())
    }

    /// 获取图形对象
    pub fn graph(&self) -> Result<JsValue, JsValue> {
        Reflect::get(&self.app, &"graph".into())
    }

    /// 根据节点 ID 获取节点
    pub fn get_node_by_id(&self, id: &str) -> Result<JsValue, JsValue> {
        let graph = self.graph()?;

        let get_node_func = Reflect::get(&graph, &"getNodeById".into())?.dyn_into::<Function>()?;

        get_node_func.call1(&graph, &id.into())
    }
}

impl From<ComfyApp> for Object {
    fn from(value: ComfyApp) -> Self {
        value.app()
    }
}
