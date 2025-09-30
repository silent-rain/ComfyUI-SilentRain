use wasm_bindgen::{JsValue, prelude::*};
use web_sys::console;

use comfy_app::{ComfyApp, Extension};

mod string_dyn_list2;
use string_dyn_list2::StringDynList2;

mod string_dyn_list;
use string_dyn_list::StringDynList;

#[wasm_bindgen(start)]
fn run() -> Result<(), JsValue> {
    let mut extension = Extension::new("ComfyUI-SilentRain");

    extension.init(|| Ok(()))?;

    // extension.get_custom_widgets(|_app| Ok(()))?;

    extension.before_register_node_def(|node_type, node_data, _app| {
        // if node_data.name()? == "StringDynList" {
        //     StringDynList::on_connections_change(&node_type)?;
        // }

        if node_data.name()? == "StringDynList2" {
            let _ = StringDynList2::on_connections_change(&node_type).map_err(|e| {
                console::log_1(&format!("🔗 更新 StringDynList2 UI 失败， err: {:#?}", e).into());
                e
            });
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
