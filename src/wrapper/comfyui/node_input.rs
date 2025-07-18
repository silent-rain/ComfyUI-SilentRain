//! 工作流输入参数 extra_pnginfo 解析

use pyo3::{
    pyclass,
    types::{PyDict, PyDictMethods},
    Bound,
};
use rust_decimal::Decimal;
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
    #[serde(serialize_with = "serialize_decimal_two_digits")]
    pub version: Decimal,
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
    #[serde(rename = "widgets_values", skip_serializing_if = "Option::is_none")]
    pub widgets_values: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bgcolor: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Input {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<u32>,
    #[serde(rename = "type")]
    pub input_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub widget: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<u32>,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Output {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape: Option<u32>,
    #[serde(rename = "type")]
    pub output_type: String,
    #[serde(default)]
    pub links: Option<Vec<u32>>, // 可能为null
}

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Properties {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cnr_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
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

fn serialize_decimal_two_digits<S>(value: &Decimal, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let rounded = value.round_dp(2);
    let float_value: f64 = rounded.try_into().unwrap_or(0.0); // 尝试转换，失败时返回默认值
    serializer.serialize_f64(float_value)
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

    /// 获取额外附加信息
    pub fn extra_pnginfo(&self) -> Result<ExtraPnginfo, Error> {
        self.parse("extra_pnginfo")
    }

    /// 获取工作流信息
    pub fn workflow(&self) -> Result<Workflow, Error> {
        Ok(self.parse::<ExtraPnginfo>("extra_pnginfo")?.workflow)
    }

    /// 获取当前节点ID
    pub fn unique_id(&self) -> Result<u32, Error> {
        Ok(self.parse::<String>("unique_id")?.parse::<u32>()?)
    }

    /// 获取提示词
    pub fn prompt(&self) -> Result<Value, Error> {
        self.parse("prompt")
    }

    /// 获取动态提示词
    pub fn dynprompt(&self) -> Result<Value, Error> {
        self.parse("dynprompt")
    }

    /// 获取指定节点
    pub fn node(&self, node_id: u32) -> Result<Option<Node>, Error> {
        let workflow = self.workflow()?;
        for node in workflow.nodes {
            if node.id == node_id {
                return Ok(Some(node));
            }
        }
        Ok(None)
    }

    /// 获取前一个节点的信息
    pub fn pre_node(&self) -> Result<Option<Node>, Error> {
        // 获取当前节点id
        let cur_node_id = self.unique_id()?;

        // 直接获取到前一个节点的 id
        // let pre_node_id: Option<u32> = self
        //     .node(cur_node_id)?
        //     .and_then(|cur_node| cur_node.inputs.iter().find_map(|input| input.link));

        // 获取当前节点
        let cur_node = self.node(cur_node_id)?.ok_or_else(|| {
            Error::OptionNone(format!(
                "node_id: {cur_node_id}, the current node does not exist"
            ))
        })?;

        // 获取前一个节点的id, 默认第一个输入的id
        let pre_input_id = cur_node
            .inputs
            .iter()
            .find_map(|input| input.link)
            .ok_or_else(|| {
                Error::OptionNone(format!(
                    "node_id: {cur_node_id}, the current node does not have a previous node"
                ))
            })?;

        // 获取前一个节点的信息
        let mut pre_node = None;
        let workflow = self.workflow()?;
        for node in workflow.nodes {
            for output in node.outputs.clone() {
                if let Some(links) = output.links {
                    let link = links.iter().find(|link| **link == pre_input_id);
                    if link.is_some() {
                        pre_node = Some(node);
                        break;
                    }
                }
            }
        }

        Ok(pre_node)
    }
}

/// 提示词入参
#[derive(Debug)]
pub struct InputPrompt {
    prompt: Value,
}

impl InputPrompt {
    pub fn new<'py>(prompt: Bound<'py, PyDict>) -> Result<Self, Error> {
        let prompt: Value = pythonize::depythonize(&prompt)?;

        Ok(Self { prompt })
    }

    /// 将python的prompt转换为json字符串
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        self.prompt.to_string()
    }
}

/// 附加信息
pub struct InputExtraPnginfo {
    extra_pnginfo: ExtraPnginfo,
}

impl InputExtraPnginfo {
    pub fn new<'py>(extra_pnginfo: Bound<'py, PyDict>) -> Result<Self, Error> {
        let extra_pnginfo: ExtraPnginfo = pythonize::depythonize(&extra_pnginfo)?;

        Ok(Self { extra_pnginfo })
    }

    /// 将python的extra_pnginfo转换为json字符串
    pub fn to_string(&self) -> Result<String, Error> {
        let extra_pnginfo = serde_json::to_string(&self.extra_pnginfo)?;

        Ok(extra_pnginfo)
    }
}
