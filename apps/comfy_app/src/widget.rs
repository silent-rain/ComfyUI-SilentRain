//! 部件节点

use serde::{Deserialize, Serialize};
use wasm_bindgen::JsValue;

/// 小部件信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    pub name: String,
    pub value: f64,
    pub options: WidgetOptions,
}

impl Widget {
    /// 创建新的小部件
    pub fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value,
            options: WidgetOptions::default(),
        }
    }

    /// 从 JsValue 反序列化
    pub fn from_js(js_value: JsValue) -> Result<Widget, JsValue> {
        let result: Widget = serde_wasm_bindgen::from_value(js_value)?;
        Ok(result)
    }

    /// 序列化为 JsValue
    pub fn to_js(&self) -> Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&self)?)
    }
}

/// 输入/输出槽位信息
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SlotInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub slot_type: String,
    pub label: Option<String>,
}

impl SlotInfo {
    /// 从 JsValue 反序列化
    pub fn from_js(js_value: JsValue) -> Result<SlotInfo, JsValue> {
        let result: SlotInfo = serde_wasm_bindgen::from_value(js_value)?;
        Ok(result)
    }

    /// 序列化为 JsValue
    pub fn to_js(&self) -> Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&self)?)
    }
}

/// 小部件选项
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct WidgetOptions {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub values: Option<Vec<String>>,
}

impl WidgetOptions {
    /// 从 JsValue 反序列化
    pub fn from_js(js_value: JsValue) -> Result<WidgetOptions, JsValue> {
        let result: WidgetOptions = serde_wasm_bindgen::from_value(js_value)?;
        Ok(result)
    }

    /// 序列化为 JsValue
    pub fn to_js(&self) -> Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&self)?)
    }
}
