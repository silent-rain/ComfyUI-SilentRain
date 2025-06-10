//! 工作流输入参数 extra_pnginfo 解析

use pyo3::{
    pyclass,
    types::{PyDict, PyDictMethods},
    Bound,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Error;

#[pyclass]
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct ExtraPnginfo {
    pub workflow: Workflow,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Workflow {
    pub id: String,
    pub revision: u32,
    #[serde(rename = "last_node_id")]
    pub last_node_id: u32,
    #[serde(rename = "last_link_id")]
    pub last_link_id: u32,
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    pub groups: Vec<Group>,
    pub config: Value, // 空对象用Value表示
    pub extra: Extra,
    pub version: f32,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Node {
    pub id: u32,
    #[serde(rename = "type")]
    pub node_type: String,
    pub pos: (f32, f32),
    pub size: (f32, f32),
    pub flags: Value, // 空对象用Value表示
    pub order: u32,
    pub mode: u32,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub properties: Properties,
    #[serde(rename = "widgets_values")]
    pub widgets_values: Option<Value>,
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub bgcolor: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Input {
    pub label: String,
    pub name: String,
    pub shape: Option<u32>,
    #[serde(rename = "type")]
    pub input_type: String,
    pub link: Option<u32>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Output {
    pub name: String,
    pub shape: Option<u32>,
    #[serde(rename = "type")]
    pub output_type: String,
    pub links: Option<Vec<u32>>, // 可能为null
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Properties {
    pub cnr_id: Option<String>,
    pub ver: Option<String>,
    #[serde(rename = "Node name for S&R")]
    pub node_name_for_sr: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Link(pub u32, pub u32, pub u32, pub u32, pub u32, pub String);

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Group {
    pub id: u32,
    pub title: String,
    pub bounding: (f32, f32, f32, f32),
    pub color: String,
    pub font_size: u32,
    pub flags: Value, // 空对象用Value表示
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Extra {
    pub ds: Ds,
    #[serde(rename = "frontendVersion")]
    pub frontend_version: String,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Ds {
    pub scale: f32,
    pub offset: (f32, f32),
}

impl ExtraPnginfo {
    /// 从 python kwargs 解析 comfyui 输入参数 extra_pnginfo
    pub fn parse(kwargs: &Option<Bound<'_, PyDict>>) -> Result<Self, Error> {
        // 默认不支持解析到自定义结构体
        // let extra_pnginfo = kwargs
        //     .clone()
        //     .ok_or(Error::PyMissingKwargs(
        //         "the py kwargs parameter does not exist".to_string(),
        //     ))?
        //     .get_item("extra_pnginfo")?
        //     .ok_or(Error::PyMissingKwargs(
        //         "the extra_pnginfo parameter does not exist".to_string(),
        //     ))?
        //     .extract::<ExtraPnginfo>()
        //     .map_err(|e| {
        //         error!("extract extra_pnginfo failed: {:#?}", e);
        //         e
        //     })?;

        // Option<Result<T, E>> 转换为 Result<Option<T>, E>
        // let extra_pnginfo: Result<Option<ExtraPnginfo>, _> = kwargs
        //     .get_item("extra_pnginfo")?
        //     .map(|unique_id_any| unique_id_any.extract::<ExtraPnginfo>())
        //     .transpose();

        let extra_pnginfo_obj = kwargs
            .clone()
            .ok_or(Error::PyMissingKwargs(
                "the py kwargs parameter does not exist".to_string(),
            ))?
            .get_item("extra_pnginfo")?
            .ok_or(Error::PyMissingKwargs(
                "the extra_pnginfo parameter does not exist".to_string(),
            ))?;

        let extra_pnginfo: ExtraPnginfo = pythonize::depythonize(&extra_pnginfo_obj)?;

        Ok(extra_pnginfo)
    }
}

pub struct InputKwargs<'py> {
    kwargs: Bound<'py, PyDict>,
}

impl<'py> InputKwargs<'py> {
    pub fn new(kwargs: &Bound<'py, PyDict>) -> Self {
        Self {
            kwargs: kwargs.clone(),
        }
    }

    /// 从 python kwargs 解析输入参数
    pub fn parse<T>(&self, key: &str) -> Result<T, Error>
    where
        T: serde::de::DeserializeOwned,
    {
        let value_obj = self
            .kwargs
            .get_item(key)?
            .ok_or(Error::PyMissingKwargs(format!(
                "the {key} parameter does not exist"
            )))?;

        let value: T = pythonize::depythonize(&value_obj)?;

        Ok(value)
    }

    pub fn extra_pnginfo(&self) -> Result<ExtraPnginfo, Error> {
        self.parse("extra_pnginfo")
    }

    pub fn unique_id(&self) -> Result<String, Error> {
        self.parse("unique_id")
    }

    pub fn prompt(&self) -> Result<Value, Error> {
        self.parse("prompt")
    }

    pub fn dynprompt(&self) -> Result<Value, Error> {
        self.parse("dynprompt")
    }
}
