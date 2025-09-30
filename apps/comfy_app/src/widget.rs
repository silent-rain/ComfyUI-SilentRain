//! 部件节点

use serde::{Deserialize, Serialize};
use wasm_bindgen::JsValue;

/// 小部件信息
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Widget {
    pub name: String,
    pub label: String,
    #[serde(rename = "type")]
    pub r#type: String,
    pub options: WidgetOptions,
    pub y: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_y: Option<f64>,
    #[serde(rename = "computedDisabled", skip_serializing_if = "Option::is_none")]
    pub computed_disabled: Option<bool>,
    #[serde(rename = "computedHeight", skip_serializing_if = "Option::is_none")]
    pub computed_height: Option<f64>,
}

impl Widget {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
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

/*
obj: Object {
        obj: JsValue([
        Object({"name":"delimiter","localized_name":"delimiter","label":"delimiter","type":"STRING","widget":{"name":"delimiter"},
        "boundingRect":{"0":3780,"1":4406,"2":15,"3":20},"pos":[10,76],"link":null}),
        Object({"name":"string_1","localized_name":"string_1","label":"string_1","type":"STRING","shape":7,"widget":{"name":"string_1"},
        "boundingRect":{"0":3780,"1":4430,"2":15,"3":20},"pos":[10,100],"link":81}),
        Object({"name":"string_5","label":"string_5","type":"STRING","boundingRect":{"0":3780,"1":4344,"2":20,"3":20},"link":83}),
        Object({"name":"string_5","label":"string_5","type":"STRING","boundingRect":{"0":3780,"1":4364,"2":20,"3":20},"link":null}),
        Object({"name":"string_6","label":"string_6","type":"STRING","boundingRect":{"0":3780,"1":4384,"2":20,"3":20},"link":85})]),
*/

/// 表示单个 Input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Input {
    pub name: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,

    #[serde(rename = "type")]
    pub r#type: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub localized_name: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<i32>,

    #[serde(rename = "boundingRect", skip_serializing_if = "Option::is_none")]
    pub bounding_rect: Option<BoundingRect>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub widget: Option<WidgetRef>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub pos: Option<Vec<i32>>,

    #[serde(skip_serializing_if = "Option::is_none")]
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
    pub r#type: String,
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
pub struct WidgetRef {
    pub name: String,
}
