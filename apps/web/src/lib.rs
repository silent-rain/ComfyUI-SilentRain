use js_sys::Object;
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};
use web_sys::console;

use comfy_app::{ComfyApp, Extension};

mod string_dyn_list2;
use string_dyn_list2::StringDynList2;

mod string_dyn_list;
use string_dyn_list::StringDynList;

mod any_dyn_list;
use any_dyn_list::AnyDynList;

mod workflow_image;
use workflow_image::register_workflow_image_export;

mod workflow_image_export;
use workflow_image_export::register_workflow_image_export as register_workflow_image_export_full;

#[wasm_bindgen(start)]
fn run() -> Result<(), JsValue> {
    let mut extension = Extension::new("ComfyUI-SilentRain");

    extension.init(|| Ok(()))?;

    // extension.get_custom_widgets(|_app| Ok(()))?;

    extension.node_created(|_node, _app| {
        console::log_1(&"🚀 node_created called!".into());

        Ok(JsValue::undefined())
    })?;

    extension.before_register_node_def(|node_type, node_data, _app| {
        if node_data.name()? == "StringDynList2" {
            let _ = StringDynList2::on_connections_change(&node_type).map_err(|e| {
                console::log_1(&format!("🔗 更新 StringDynList2 UI 失败， err: {:#?}", e).into());
                e
            });
        }

        if node_data.name()? == "StringDynList" {
            let _ = StringDynList::on_node_created(&node_type).map_err(|e| {
                console::log_1(&format!("🔗 更新 StringDynList UI 失败， err: {:#?}", e).into());
                e
            });
        }

        if node_data.name()? == "AnyDynList" {
            let _ = AnyDynList::on_connections_change(&node_type).map_err(|e| {
                console::log_1(&format!("🔗 更新 AnyDynList UI 失败， err: {:#?}", e).into());
                e
            });
        }

        Ok(JsValue::undefined())
    })?;

    extension.loaded_graph_node(|node, _app| {
        console::log_1(&"🚀 loaded_graph_nodes called!".into());

        if node.title()? == "Sr String Dyn List" {
            let _ = StringDynList::init_widget(&node).map_err(|e| {
                console::log_1(&format!("🔗 更新 StringDynList UI 失败， err: {:#?}", e).into());
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

    // 注册画布菜单项
    extension.get_canvas_menu_items(|_canvas: Object| {
        let menu_items = js_sys::Array::new();

        // 使用新的工作流导出功能
        let full_menu = register_workflow_image_export_full()?;
        for i in 0..full_menu.length() {
            menu_items.push(&full_menu.get(i));
        }

        // 添加分隔符
        menu_items.push(&JsValue::NULL);

        // 注册工作流图片导出功能注册菜单
        menu_items.push(&register_workflow_image_export()?);

        Ok(menu_items.into())
    })?;

    let app = ComfyApp::new()?;

    app.register_extension(&extension)?;

    console::log_1(&"🚀 Rust init called!".into());

    Ok(())
}
