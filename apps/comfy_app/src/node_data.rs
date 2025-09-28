//! 包含节点信息的对象
use js_sys::{Array, Boolean, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue, prelude::wasm_bindgen};

/// 节点数据封装
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct NodeData {
    inner: Object,
}

#[wasm_bindgen]
impl NodeData {
    #[wasm_bindgen(constructor)]
    pub fn new(inner: Object) -> Self {
        Self { inner }
    }

    /// 获取inner对象
    fn get_inner(&self) -> Object {
        self.inner.clone()
    }
}

#[wasm_bindgen]
impl NodeData {
    #[wasm_bindgen(getter)]
    pub fn input(&self) -> Result<JsValue, JsValue> {
        Reflect::get(&self.inner, &"input".into())
    }

    #[wasm_bindgen(getter)]
    pub fn input_order(&self) -> Result<JsValue, JsValue> {
        Reflect::get(&self.inner, &"input_order".into())
    }

    #[wasm_bindgen(getter)]
    pub fn output(&self) -> Result<Vec<String>, JsValue> {
        let js_value = Reflect::get(&self.inner, &"output".into())?;
        let result: Vec<String> = serde_wasm_bindgen::from_value(js_value)?;
        Ok(result)
    }

    #[wasm_bindgen(getter)]
    pub fn output_is_list(&self) -> Result<Vec<Boolean>, JsValue> {
        let arr = Reflect::get(&self.inner, &"output_is_list".into())?.dyn_into::<Array>()?;
        let results = arr
            .iter()
            .map(|v| v.dyn_into::<Boolean>())
            .collect::<Result<Vec<Boolean>, JsValue>>()?;
        Ok(results)
    }

    #[wasm_bindgen(getter)]
    pub fn output_name(&self) -> Result<Vec<String>, JsValue> {
        let js_value = Reflect::get(&self.inner, &"output_name".into())?;
        let result: Vec<String> = serde_wasm_bindgen::from_value(js_value)?;
        Ok(result)
    }

    #[wasm_bindgen(getter)]
    pub fn name(&self) -> Result<String, JsValue> {
        Reflect::get(&self.inner, &"name".into()).and_then(|v| v.as_string().ok_or(JsValue::NULL))
    }

    #[wasm_bindgen(getter)]
    pub fn display_name(&self) -> Result<String, JsValue> {
        Reflect::get(&self.inner, &"display_name".into())
            .and_then(|v| v.as_string().ok_or(JsValue::NULL))
    }

    #[wasm_bindgen(getter)]
    pub fn description(&self) -> Result<String, JsValue> {
        Reflect::get(&self.inner, &"description".into())
            .and_then(|v| v.as_string().ok_or(JsValue::NULL))
    }

    #[wasm_bindgen(getter)]
    pub fn python_module(&self) -> Result<String, JsValue> {
        Reflect::get(&self.inner, &"python_module".into())
            .and_then(|v| v.as_string().ok_or(JsValue::NULL))
    }

    #[wasm_bindgen(getter)]
    pub fn category(&self) -> Result<String, JsValue> {
        Reflect::get(&self.inner, &"category".into())
            .and_then(|v| v.as_string().ok_or(JsValue::NULL))
    }

    #[wasm_bindgen(getter)]
    pub fn output_node(&self) -> Result<bool, JsValue> {
        Reflect::get(&self.inner, &"output_node".into())
            .and_then(|v| v.as_bool().ok_or(JsValue::NULL))
    }

    #[wasm_bindgen(getter)]
    pub fn output_tooltips(&self) -> Result<Vec<String>, JsValue> {
        let js_value = Reflect::get(&self.inner, &"output_tooltips".into())?;
        let result: Vec<String> = serde_wasm_bindgen::from_value(js_value)?;
        Ok(result)
    }
}

impl From<NodeData> for Object {
    fn from(value: NodeData) -> Self {
        value.get_inner()
    }
}
