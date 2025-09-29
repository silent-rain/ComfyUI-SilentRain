use wasm_bindgen::{JsValue, prelude::*};
use web_sys::console;

use comfy_app::{ComfyApp, Extension};

#[wasm_bindgen(start)]
fn run() -> Result<(), JsValue> {
    let mut extension = Extension::new("ComfyUI-SilentRain");

    extension.init(|| Ok(()))?;

    // extension.get_custom_widgets(|_app| Ok(()))?;

    extension.before_register_node_def(|node_type, node_data, _app| {
        if node_data.name()? == "StringList" {
            node_type.prototype()?.on_connections_change(
                move |this, r#type, index, connected, link_info| {
                    let link_info = match link_info {
                        Some(link_info) => link_info,
                        None => return Ok(()),
                    };
                    console::log_1(&format!("🔗 连接类型: {}", r#type).into());
                    console::log_1(&format!("🔗 连接索引: {}", index).into());
                    console::log_1(&format!("🔗 是否连接: {}", connected).into());
                    console::log_1(&format!("🔗 连接参数: {:#?}", link_info).into());

                    console::log_1(
                        &"***********************************************************************:"
                            .into(),
                    );

                    let widgets = this.widgets()?;
                    console::log_1(&format!("widgets: {:#?}", widgets).into());

                    let inputs = this.inputs()?;
                    console::log_1(&format!("inputs: {:#?}", inputs).into());

                    let outputs = this.outputs()?;
                    console::log_1(&format!("outputs: {:#?}", outputs).into());

                    Ok(())
                },
            )?;
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
