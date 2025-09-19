use js_sys::eval;
use wasm_bindgen::{JsValue, prelude::*};
use web_sys::{console, window};

use comfy_app::{ComfyApp, Extension};

#[wasm_bindgen(start)]
fn run() -> Result<(), JsValue> {
    let mut extension = Extension::new("test.hook.types");

    extension.init(|| {
        console::log_1(&"🚀 JS init called!".into());
        Ok(())
    })?;

    extension.get_custom_widgets(|app| {
        console::log_1(&"🎨 JS getCustomWidgets called!".into());
        console::log_1(&app);
        Ok(())
    })?;

    extension.before_register_node_def(|node_type, node_data, app| {
        console::log_1(&"📋 JS beforeRegisterNodeDef".into());
        console::log_1(&node_type);
        console::log_1(&node_data);
        console::log_1(&app);

        Ok(JsValue::undefined())
    })?;

    extension.setup(|| {
        console::log_1(&"⚙️  JS setup called!".into());

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
