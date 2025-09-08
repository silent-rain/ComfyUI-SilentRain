use js_sys::eval;
use std::collections::HashMap;
use wasm_bindgen::{JsValue, prelude::*};
use web_sys::{console, window};

mod comfy_app;
use comfy_app::{ComfyApp, ExtensionConfig, NodeDef};

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

#[wasm_bindgen]
pub async fn register_extension() -> Result<(), JsValue> {
    console::log_1(&JsValue::from_str("正在加载Rust脚本"));

    // 加载 ComfyUI app
    let app = ComfyApp::load("/scripts/app.js").await?;

    // 创建扩展配置
    let config = ExtensionConfig {
        name: "a.unique.name.for.a.useless.extension".to_string(),
        setup_fn: Some(Box::new(|_app| {
            console::log_1(&JsValue::from_str("Setup complete 0!"));

            let _ = eval("alert('Setup complete 1!');");

            if let Some(window) = window() {
                let _ = window.alert_with_message("Setup complete 2!");
            }

            Ok(())
        })),
        init_fn: None,
        before_register_node_def_fn: None,
    };

    // 注册扩展
    app.register_extension(config)?;

    Ok(())
}

#[wasm_bindgen]
pub async fn register_custom_node() -> Result<(), JsValue> {
    console::log_1(&JsValue::from_str("注册自定义节点"));

    // 加载 ComfyUI app
    let app = ComfyApp::load("/scripts/app.js").await?;

    // 创建自定义节点定义
    let mut properties = HashMap::new();

    // 添加节点执行函数
    let execute_fn = js_sys::Function::new_with_args(
        "inputs",
        r#"
        return {
            output: "Hello from Rust WASM Node: " + (inputs.text || "")
        };
        "#,
    );
    properties.insert("execute".to_string(), execute_fn.into());

    // 创建节点定义
    let node_def = NodeDef {
        name: "RustWasmTextNode".to_string(),
        display_name: Some("Rust WASM Text Node".to_string()),
        category: "utils".to_string(),
        input_names: vec!["text".to_string()],
        output_names: vec!["output".to_string()],
        output_types: vec!["STRING".to_string()],
        properties,
    };

    // 注册节点
    app.register_node(node_def)?;

    Ok(())
}
