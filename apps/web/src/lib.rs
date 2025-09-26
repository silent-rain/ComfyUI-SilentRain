use wasm_bindgen::{JsValue, prelude::*};
use web_sys::console;

use comfy_app::{ComfyApp, Extension};

#[wasm_bindgen(start)]
fn run() -> Result<(), JsValue> {
    let mut extension = Extension::new("ComfyUI-SilentRain");

    extension.init(|| Ok(()))?;

    // extension.get_custom_widgets(|_app| Ok(()))?;

    extension.before_register_node_def(|node_type, node_data, app| {
        console::log_1(&"📋 JS beforeRegisterNodeDef".into());

        if node_data.name()? == "ImpactSwitch" {
            console::log_1(&node_type.clone().into());
            console::log_1(&node_data.into());
            console::log_1(&app.into());

            let node_type_c = node_type.clone();
            node_type.on_connections_change(move |r#type, index, connected, link_info| {
                let link_info = match link_info {
                    Some(link_info) => link_info,
                    None => return Ok(()),
                };
                console::log_1(&format!("🔗 连接类型: {}", r#type).into());
                console::log_1(&format!("🔗 连接索引: {}", index).into());
                console::log_1(&format!("🔗 是否连接: {}", connected).into());
                console::log_1(&format!("🔗 连接参数: {:#?}", link_info).into());

                console::log_1(&format!("🔗 node_type_c: {:#?}", node_type_c.get_input(0)).into());
                console::log_1(&format!("🔗 node_type_c: {:#?}", node_type_c.get_widget(0)).into());

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
