use wasm_bindgen::JsValue;
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::console;

use comfy_app::{ComfyApp, Extension};

#[test]
#[wasm_bindgen_test]
fn test_extension_hook() -> Result<(), JsValue> {
    let mut extension = Extension::new("test.hook.types");

    extension.init(|| {
        console::log_1(&"🚀 JS init called!".into());
        Ok(())
    })?;

    // 添加自定义节点定义
    extension.add_custom_node_defs(|defs, app| {
        console::log_1(&"🔧 JS addCustomNodeDefs called!".into());
        console::log_1(&defs);
        console::log_1(&app);
        Ok(())
    })?;

    // 返回自定义小部件类型
    extension.get_custom_widgets(|app| {
        console::log_1(&"🎨 JS getCustomWidgets called!".into());
        console::log_1(&app.into());
        Ok(())
    })?;

    // 修改节点行为
    extension.before_register_node_def(|node_type, node_data, app| {
        console::log_1(&"📋 JS beforeRegisterNodeDef".into());
        console::log_1(&node_type.into());
        console::log_1(&node_data.into());
        console::log_1(&app.into());

        Ok(JsValue::undefined())
    })?;

    extension.register_custom_nodes(|app| {
        console::log_1(&"🏗️  JS registerCustomNodes called!".into());
        console::log_1(&app);

        Ok(JsValue::undefined())
    })?;

    extension.node_created(|app| {
        console::log_1(&"📦 JS nodeCreated".into());
        console::log_1(&app);

        Ok(JsValue::undefined())
    })?;

    // 处理节点加载
    extension.loaded_graph_node(|node, app| {
        console::log_1(&"📥 JS loadedGraphNode".into());
        console::log_1(&node);
        console::log_1(&app);

        Ok(JsValue::undefined())
    })?;

    // 处理节点创建
    extension.after_configure_graph(|missing_node_types| {
        console::log_1(&"📊 JS afterConfigureGraph called!".into());
        console::log_1(&missing_node_types);

        Ok(())
    })?;

    extension.setup(|| {
        console::log_1(&"⚙️  JS setup called!".into());
        Ok(())
    })?;

    // Comfy App 实例
    let app = ComfyApp::new()?;

    // 注册扩展
    app.register_extension(&extension)?;
    Ok(())
}
