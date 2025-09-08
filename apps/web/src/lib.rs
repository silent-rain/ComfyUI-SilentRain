use js_sys::{Function, Object, Promise, Reflect};
use wasm_bindgen::{JsValue, prelude::*};
use wasm_bindgen_futures::JsFuture;
use web_sys::console;

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = import)]
    fn js_import(module_path: &str) -> Promise;
}

#[wasm_bindgen]
pub async fn register_extension() -> Result<(), JsValue> {
    console::log_1(&JsValue::from_str("正在加载Rust脚本"));

    // 动态导入JavaScript模块
    let module = JsFuture::from(js_import("/scripts/app.js")).await?;
    let app = Reflect::get(&module, &JsValue::from_str("app"))?;

    // 创建扩展对象
    let extension = Object::new();
    Reflect::set(
        &extension,
        &JsValue::from_str("name"),
        &JsValue::from_str("a.unique.name.for.a.useless.extension"),
    )?;

    // 创建setup函数
    let setup = Function::new_no_args(
        "
        alert('Setup complete!');
    ",
    );

    Reflect::set(&extension, &JsValue::from_str("setup"), &setup)?;

    // 调用app.registerExtension方法
    let register_extension = Reflect::get(&app, &JsValue::from_str("registerExtension"))?;
    let register_fn = Function::from(register_extension);
    register_fn.call1(&app, &extension)?;

    Ok(())
}
