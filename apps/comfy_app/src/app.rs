//! Comfy Js App main
//!

use js_sys::{Function, Reflect};
use wasm_bindgen::{JsValue, prelude::*};

use crate::extension::Extension;

#[wasm_bindgen(raw_module = "/scripts/app.js")]
extern "C" {
    // import { app } from "../../scripts/app.js";
    #[wasm_bindgen(thread_local_v2, js_name = app)]
    static APP: JsValue;
}

#[wasm_bindgen]
pub struct ComfyApp {
    app: JsValue,
}

#[wasm_bindgen]
impl ComfyApp {
    pub fn new() -> Result<ComfyApp, JsValue> {
        let app_obj = APP.with(|app| app.clone());
        Ok(ComfyApp { app: app_obj })
    }

    /// 获取 app 对象
    pub fn app(&self) -> JsValue {
        self.app.clone()
    }

    pub fn register_extension(&self, extension: &Extension) -> Result<(), JsValue> {
        let extension_obj = extension.as_js_value();

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
}
