use wasm_bindgen::{JsValue, prelude::*};
use web_sys::console;

use comfy_app::{ComfyApp, Extension};

#[wasm_bindgen(start)]
fn run() -> Result<(), JsValue> {
    let mut extension = Extension::new("test.hook.types");

    extension.init(|| Ok(()))?;

    // extension.get_custom_widgets(|_app| Ok(()))?;

    extension.before_register_node_def(|node_type, node_data, app| {
        console::log_1(&"📋 JS beforeRegisterNodeDef".into());

        if node_data.name()? == "ImpactSwitch" {
            console::log_1(&node_type.clone().into());
            console::log_1(&node_data.into());
            console::log_1(&app.into());

            node_type.on_connections_change(|r#type, index, connected, link_info| {
                console::log_1(&format!("🔗 连接类型: {}", r#type).into());
                console::log_1(&format!("🔗 连接类型: {}", index).into());
                console::log_1(&format!("🔗 连接类型: {}", connected).into());
                console::log_1(&format!("🔗 连接类型: {:#?}", link_info).into());

                Ok(())
            })?;
        }

        Ok(JsValue::undefined())
    })?;

    extension.setup(|| {
        console::log_1(&"⚙️  JS setup called!".into());

        // use js_sys::eval;
        // let _ = eval("alert('Setup complete 1!');");

        // if let Some(window) = window() {
        //     let _ = window.alert_with_message("Setup complete 2!");
        // }

        Ok(())
    })?;

    let app = ComfyApp::new()?;

    app.register_extension(&extension)?;

    console::log_1(&"🚀 Rust init called!".into());

    Ok(())
}
