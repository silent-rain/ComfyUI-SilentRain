//! 部件节点

use serde::{Deserialize, Serialize};
use wasm_bindgen::JsValue;

/// 小部件信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    pub name: String,
    pub label: String,
    #[serde(rename = "type")]
    pub widget_type: String,
    pub value: f64,
    pub y: u32,
    pub options: WidgetOptions,
}

impl Widget {
    /// 创建新的小部件
    pub fn new(name: &str, value: f64) -> Self {
        Self {
            name: name.to_string(),
            value,
            options: WidgetOptions::default(),
            label: name.to_string(),
            widget_type: name.to_string(),
            y: 0,
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

/// 表示单个 Input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Input {
    pub name: String,
    pub localized_name: String,
    pub label: String,
    #[serde(rename = "type")]
    pub input_type: String,
    #[serde(default)]
    pub shape: Option<i32>,
    #[serde(rename = "boundingRect")]
    pub bounding_rect: BoundingRect,
    #[serde(default)]
    pub widget: Option<WidgetRef>,
    #[serde(default)]
    pub link: Option<i32>,
}

impl Input {
    /// 从 JsValue 反序列化
    pub fn from_js(js_value: JsValue) -> Result<Input, JsValue> {
        let result: Input = serde_wasm_bindgen::from_value(js_value)?;
        Ok(result)
    }

    /// 序列化为 JsValue
    pub fn to_js(&self) -> Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&self)?)
    }
}

/// 表示单个 Output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Output {
    pub name: String,
    pub localized_name: String,
    pub label: String,
    #[serde(rename = "type")]
    pub output_type: String,
    #[serde(default)]
    pub shape: Option<i32>,
    #[serde(default, rename = "boundingRect")]
    pub bounding_rect: BoundingRect,
    #[serde(default)]
    pub links: Option<Vec<i32>>,
}

impl Output {
    /// 从 JsValue 反序列化
    pub fn from_js(js_value: JsValue) -> Result<Output, JsValue> {
        let result: Output = serde_wasm_bindgen::from_value(js_value)?;
        Ok(result)
    }

    /// 序列化为 JsValue
    pub fn to_js(&self) -> Result<JsValue, JsValue> {
        Ok(serde_wasm_bindgen::to_value(&self)?)
    }
}

/// 表示边界矩形
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundingRect {
    #[serde(rename = "0")]
    pub x1: i32,
    #[serde(rename = "1")]
    pub y1: i32,
    #[serde(rename = "2")]
    pub x2: i32,
    #[serde(rename = "3")]
    pub y2: i32,
}

/// 表示 Widget 引用
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WidgetRef {
    pub name: String,
}
