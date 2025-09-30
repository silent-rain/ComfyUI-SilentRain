use js_sys::JSON;
use serde_json::json;
use wasm_bindgen::{JsValue, prelude::*};
use web_sys::console;

use comfy_app::{ComfyApp, ConnectionType, Extension};

#[wasm_bindgen(start)]
fn run() -> Result<(), JsValue> {
    let mut extension = Extension::new("ComfyUI-SilentRain");

    extension.init(|| Ok(()))?;

    // extension.get_custom_widgets(|_app| Ok(()))?;

    extension.before_register_node_def(|node_type, node_data, _app| {
        if node_data.name()? == "StringDynList" {
            node_type.prototype()?.on_connections_change(
                move |this, r#type, index, connected, link_info| {
                    let link_info = match link_info {
                        Some(link_info) => link_info,
                        None => return Ok(()),
                    };
                    console::log_1(&format!("🔗 连接类型: {:?}", r#type).into());
                    console::log_1(&format!("🔗 连接索引: {}", index).into());
                    console::log_1(&format!("🔗 是否连接: {}", connected).into());
                    console::log_1(&format!("🔗 连接参数: {:#?}", link_info).into());

                    let widgets = this.widgets()?;
                    let inputs = this.inputs()?;
                    let outputs = this.outputs()?;

                    console::log_1(&format!("widgets: {:#?}", widgets).into());
                    console::log_1(&format!("inputs: {:#?}", inputs).into());
                    console::log_1(&format!("outputs: {:#?}", outputs).into());

                    if r#type == ConnectionType::Input {
                        // connect input
                        // 移除空卡槽
                        // if connected == 0 && inputs.length() > 2 {
                        //     this.remove_input(index)?;
                        // }

                        // 添加新的卡槽
                        let widgets_length = widgets.length();
                        let inputs_length = inputs.length();
                        console::log_1(&format!("🔗 卡槽数量: {}", widgets_length).into());
                        console::log_1(&format!("🔗 输入数量: {}", inputs_length).into());

                        // 移除"string_"的空卡槽
                        // for (i, _input) in inputs.iter().enumerate() {
                        //     let input = this.get_input(i as u32)?;
                        //     if !input.name.starts_with("string_") {
                        //         continue;
                        //     }

                        //     // 暂时暂定 shape 为空时为空卡槽
                        //     if input.link.is_none() && input.shape.is_none() {
                        //         console::log_1(
                        //             &format!("🔗 移除空卡槽: {} input: {:#?}", i, input).into(),
                        //         );
                        //         this.remove_input(i as i32)?;
                        //     }
                        // }

                        // 添加空卡槽
                        // let extra_info = json!({
                        //     "widget": {"name":format!("string_{}", inputs_length + 1)},
                        //     "shape": 7,
                        // });
                        // let extra_info = serde_wasm_bindgen::to_value(&extra_info)?;
                        // let first_string_input = this.get_input(1)?;
                        // this.add_input(
                        //     &format!("string_{}", inputs_length + 1),
                        //     &first_string_input.r#type,
                        //     extra_info,
                        // )
                        // .map_err(|e| {
                        //     console::log_1(&format!("🔗 添加空卡槽失败: {:#?}", e).into());
                        //     e
                        // })?;
                        // console::log_1(&format!("🔗 添加空卡槽: {}", inputs_length + 1).into());

                        // 添加空组件
                        /*
                        Object({"name":"string_2","localized_name":"string_2","label":"string_2","type":"STRING","shape":7,
                        "widget":{"name":"string_2"},"boundingRect":{"0":3780,"1":4454,"2":15,"3":20},"pos":[10,124],"link":null}), 
                         */

                        // // 重命名"string_"卡槽
                        // for (i, _input) in widgets.iter().enumerate() {
                        //     let input = this.get_input(i as u32)?;
                        //     if !input.name.starts_with("string_") {
                        //         continue;
                        //     }
                        //     // this.set_widget(i as i32, &format!("string_{}", i + 1))?;
                        // }
                    } else {
                        // connect output
                    }

                    // console::log_1(
                    //     &"***********************************************************************:"
                    //         .into(),
                    // );

                    // let widgets = this.widgets()?;
                    // console::log_1(&format!("widgets: {:#?}", widgets).into());

                    // let widget = this.get_widget(0)?;
                    // console::log_1(&format!("widget: {:#?}", widget).into());

                    // console::log_1(&"---------------------".into());

                    // let inputs = this.inputs()?;
                    // console::log_1(&format!("inputs: {:#?}", inputs).into());

                    // let input = this.get_input(0)?;
                    // console::log_1(&format!("input: {:#?}", input).into());

                    // console::log_1(&"---------------------".into());

                    // let outputs = this.outputs()?;
                    // console::log_1(&format!("outputs: {:#?}", outputs).into());

                    // let output = this.get_output(0)?;
                    // console::log_1(&format!("output: {:#?}", output).into());

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
