//! 封装 ComfyUI 的 app 对象
//!
//! 原文:
//! https://docs.comfy.org/zh-CN/custom-nodes/js/javascript_overview
//! https://docs.comfy.org/zh-CN/custom-nodes/js/javascript_hooks#beforeregisternodedef
//!
use js_sys::{Array, Function, Object, Promise, Reflect};
use std::collections::HashMap;
use std::rc::Rc;
use wasm_bindgen::{JsValue, prelude::*};
use wasm_bindgen_futures::JsFuture;
use web_sys::console;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = import)]
    fn js_import(module_path: &str) -> Promise;
}

pub type SetupFn = Box<dyn FnOnce(&ComfyApp) -> Result<(), JsValue>>;
pub type InitFn = Box<dyn FnOnce(&ComfyApp) -> Result<(), JsValue>>;
pub type BeforeRegisterFn = Box<dyn FnMut(&JsValue, &str) -> Result<(), JsValue>>;

/// 扩展配置
#[derive(Default)]
pub struct ExtensionConfig {
    pub name: String,
    pub setup_fn: Option<SetupFn>,
    pub init_fn: Option<InitFn>,
    pub before_register_node_def_fn: Option<BeforeRegisterFn>,
}

/// 节点定义
pub struct NodeDef {
    pub name: String,
    pub display_name: Option<String>,
    pub category: String,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub output_types: Vec<String>,
    pub properties: HashMap<String, JsValue>,
}

/// ComfyApp 结构体封装了 ComfyUI 的 app 对象
#[derive(Clone)]
pub struct ComfyApp {
    app: JsValue,
}

#[allow(unused)]
impl ComfyApp {
    /// 从 JavaScript 的 app 对象创建 ComfyApp
    ///
    /// import { app } from "../../scripts/app.js";
    pub fn new(app: JsValue) -> Self {
        Self { app }
    }

    /// 从指定路径加载 app 对象
    pub async fn load(script_path: &str) -> Result<Self, JsValue> {
        // 动态导入 JavaScript 模块
        console::log_1(&JsValue::from_str(&format!(
            "正在加载 ComfyUI app: {}",
            script_path
        )));

        let module = JsFuture::from(js_import(script_path)).await?;
        let app = Reflect::get(&module, &JsValue::from_str("app"))?;

        Ok(Self { app })
    }

    /// 注册扩展
    pub fn register_extension(self, config: ExtensionConfig) -> Result<(), JsValue> {
        // 将 self 转换为 Rc<Self>，这样可以在闭包中安全地使用
        let app = Rc::new(self);
        let extension = Object::new();

        // 设置扩展名称
        Reflect::set(
            &extension,
            &JsValue::from_str("name"),
            &JsValue::from_str(&config.name),
        )?;

        // 设置 setup 函数
        if let Some(setup_fn) = config.setup_fn {
            let app_clone = Rc::clone(&app);
            let setup_closure = Closure::once(move || {
                if let Err(e) = setup_fn(&app_clone) {
                    console::error_1(&e);
                }
            });

            Reflect::set(
                &extension,
                &JsValue::from_str("setup"),
                setup_closure.as_ref(),
            )?;
            setup_closure.forget(); // 防止被回收
        }

        // 设置 init 函数
        if let Some(init_fn) = config.init_fn {
            let app_clone = Rc::clone(&app);
            let init_closure = Closure::once(move || {
                if let Err(e) = init_fn(&app_clone) {
                    console::error_1(&e);
                }
            });

            Reflect::set(
                &extension,
                &JsValue::from_str("init"),
                init_closure.as_ref(),
            )?;
            init_closure.forget(); // 防止被回收
        }

        // 设置 beforeRegisterNodeDef 函数
        if let Some(mut before_register_fn) = config.before_register_node_def_fn {
            let before_register_closure =
                Closure::wrap(Box::new(move |node_def: JsValue, node_name: String| {
                    if let Err(e) = before_register_fn(&node_def, &node_name) {
                        console::error_1(&e);
                    }
                    Ok(JsValue::undefined())
                })
                    as Box<dyn FnMut(JsValue, String) -> Result<JsValue, JsValue>>);

            Reflect::set(
                &extension,
                &JsValue::from_str("beforeRegisterNodeDef"),
                before_register_closure.as_ref(),
            )?;
            before_register_closure.forget(); // 防止被回收
        }

        // 调用 app.registerExtension 方法
        let register_extension = Reflect::get(&app.app, &JsValue::from_str("registerExtension"))?;
        let register_fn = Function::from(register_extension);
        register_fn.call1(&app.app, &extension)?;

        Ok(())
    }

    /// 注册节点
    pub fn register_node(&self, node_def: NodeDef) -> Result<(), JsValue> {
        let js_node_def = Object::new();

        // 设置节点名称
        Reflect::set(
            &js_node_def,
            &JsValue::from_str("name"),
            &JsValue::from_str(&node_def.name),
        )?;

        // 设置显示名称（如果有）
        if let Some(display_name) = node_def.display_name {
            Reflect::set(
                &js_node_def,
                &JsValue::from_str("display_name"),
                &JsValue::from_str(&display_name),
            )?;
        }

        // 设置类别
        Reflect::set(
            &js_node_def,
            &JsValue::from_str("category"),
            &JsValue::from_str(&node_def.category),
        )?;

        // 设置输入名称
        let input_array = Array::new();
        for (i, name) in node_def.input_names.iter().enumerate() {
            input_array.set(i as u32, JsValue::from_str(name));
        }
        Reflect::set(&js_node_def, &JsValue::from_str("input"), &input_array)?;

        // 设置输出名称
        let output_array = Array::new();
        for (i, name) in node_def.output_names.iter().enumerate() {
            output_array.set(i as u32, JsValue::from_str(name));
        }
        Reflect::set(&js_node_def, &JsValue::from_str("output"), &output_array)?;

        // 设置输出类型
        let output_type_array = Array::new();
        for (i, type_name) in node_def.output_types.iter().enumerate() {
            output_type_array.set(i as u32, JsValue::from_str(type_name));
        }
        Reflect::set(
            &js_node_def,
            &JsValue::from_str("output_type"),
            &output_type_array,
        )?;

        // 设置其他属性
        for (key, value) in node_def.properties {
            Reflect::set(&js_node_def, &JsValue::from_str(&key), &value)?;
        }

        // 调用 app.registerNodeDef 方法
        let register_node_def = Reflect::get(&self.app, &JsValue::from_str("registerNodeDef"))?;
        let register_fn = Function::from(register_node_def);
        register_fn.call2(&self.app, &JsValue::from_str(&node_def.name), &js_node_def)?;

        Ok(())
    }

    /// 获取图表数据
    pub fn get_graph(&self) -> Result<JsValue, JsValue> {
        Reflect::get(&self.app, &JsValue::from_str("graph"))
    }

    /// 获取画布数据
    pub fn get_canvas(&self) -> Result<JsValue, JsValue> {
        Reflect::get(&self.app, &JsValue::from_str("canvas"))
    }

    /// 获取 LiteGraph 对象
    pub fn get_litegraph(&self) -> Result<JsValue, JsValue> {
        Reflect::get(&self.app, &JsValue::from_str("LiteGraph"))
    }
}
