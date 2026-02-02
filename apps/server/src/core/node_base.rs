//! ComfyUI 节点输入输出构建器
//!
//! 提供类型安全、易用的构建器 API，用于生成 ComfyUI 节点的 INPUT_TYPES
//!
//! # 使用示例
//!
//! ```rust
//! Python::attach(|py| {
//!     InputSpec::new()
//!         .with_required("text", InputType::string().default("hello").multiline(true))
//!         .with_required("count", InputType::int().default(10).min(0).max(100).step(1))
//!         .with_optional("mode", InputType::enum_list(vec!["A".to_string(), "B".to_string()]).default("A"))
//!         .with_hidden("unique_id", InputType::none())
//!         .build(py)
//! })
//! ```

use indexmap::IndexMap;
use pyo3::{
    Bound, IntoPyObject, Py, PyAny, PyResult, Python, pyclass,
    types::{PyDict, PyDictMethods},
};

use crate::wrapper::comfyui::types::{
    NODE_BOOLEAN, NODE_CLIP, NODE_CONDITIONING, NODE_FLOAT, NODE_IMAGE, NODE_INT, NODE_JSON,
    NODE_LATENT, NODE_LIST, NODE_MASK, NODE_METADATA_RAW, NODE_MODEL, NODE_NONE, NODE_STRING,
    NODE_VAE,
};

/// 输入规范构建器
///
/// 用于构建 ComfyUI 节点的 INPUT_TYPES 字典结构
#[pyclass]
pub struct InputSpec {
    required: Vec<(String, InputType)>,
    optional: Vec<(String, InputType)>,
    hidden: Vec<(String, InputType)>,
}

impl InputSpec {
    /// 创建一个新的输入规范构建器
    pub fn new() -> Self {
        Self {
            required: Vec::new(),
            optional: Vec::new(),
            hidden: Vec::new(),
        }
    }

    /// 添加必需输入
    pub fn with_required(mut self, name: impl Into<String>, input: InputType) -> Self {
        self.required.push((name.into(), input));
        self
    }

    /// 添加可选输入
    pub fn with_optional(mut self, name: impl Into<String>, input: InputType) -> Self {
        self.optional.push((name.into(), input));
        self
    }

    /// 添加隐藏输入
    pub fn with_hidden(mut self, name: impl Into<String>, input: InputType) -> Self {
        self.hidden.push((name.into(), input));
        self
    }

    /// 构建 PyDict
    ///
    /// 生成符合 ComfyUI INPUT_TYPES 格式的字典结构
    pub fn build(self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        // Required 字段
        let required = PyDict::new(py);
        for (name, input_type) in self.required {
            required.set_item(name, input_type.to_py_tuple(py)?)?;
        }
        dict.set_item("required", required)?;

        // Optional 字段
        let optional = PyDict::new(py);
        for (name, input_type) in self.optional {
            optional.set_item(name, input_type.to_py_tuple(py)?)?;
        }
        dict.set_item("optional", optional)?;

        // Hidden 字段
        let hidden = PyDict::new(py);
        for (name, input_type) in self.hidden {
            hidden.set_item(name, input_type.to_py_tuple(py)?)?;
        }
        dict.set_item("hidden", hidden)?;

        Ok(dict.into())
    }
}

impl Default for InputSpec {
    fn default() -> Self {
        Self::new()
    }
}

/// 输入类型（包含类型和参数）
///
/// 统一了类型定义和参数设置，简化了 API
pub struct InputType {
    kind: InputKind,
    params: IndexMap<String, ParamValue>,
    list_options: Vec<String>,
}

/// 输入类型枚举（仅表示类型，不包含参数）
#[derive(Debug, Clone, PartialEq)]
enum InputKind {
    None,
    String,
    Int,
    Float,
    Bool,
    List,
    Image,
    Json,
    MetadataRaw,
    Mask,
    Model,
    Clip,
    Conditioning,
    Vae,
    Latent,
}

/// 参数值
#[derive(Debug, Clone)]
enum ParamValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl InputType {
    /// 创建新的输入类型
    fn new(kind: InputKind) -> Self {
        Self {
            kind,
            params: IndexMap::new(),
            list_options: Vec::new(),
        }
    }

    /// None 类型
    pub fn none() -> Self {
        Self::new(InputKind::None)
    }

    /// String 类型
    pub fn string() -> Self {
        Self::new(InputKind::String)
    }

    /// Int 类型
    pub fn int() -> Self {
        Self::new(InputKind::Int)
    }

    /// Float 类型
    pub fn float() -> Self {
        Self::new(InputKind::Float)
    }

    /// Bool 类型
    pub fn bool() -> Self {
        Self::new(InputKind::Bool)
    }

    /// Image 类型
    pub fn image() -> Self {
        Self::new(InputKind::Image)
    }

    /// 动态选项列表
    pub fn list(options: Vec<String>) -> Self {
        Self {
            kind: InputKind::List,
            params: IndexMap::new(),
            list_options: options,
        }
    }

    /// Json 类型
    pub fn json() -> Self {
        Self::new(InputKind::Json)
    }

    /// MetadataRaw 类型
    pub fn metadata_raw() -> Self {
        Self::new(InputKind::MetadataRaw)
    }

    /// Mask 类型
    pub fn mask() -> Self {
        Self::new(InputKind::Mask)
    }

    /// Model 类型
    pub fn model() -> Self {
        Self::new(InputKind::Model)
    }

    /// Clip 类型
    pub fn clip() -> Self {
        Self::new(InputKind::Clip)
    }

    /// Conditioning 类型
    pub fn conditioning() -> Self {
        Self::new(InputKind::Conditioning)
    }

    /// Vae 类型
    pub fn vae() -> Self {
        Self::new(InputKind::Vae)
    }

    /// Latent 类型
    pub fn latent() -> Self {
        Self::new(InputKind::Latent)
    }
}

impl InputType {
    // ============ 通用参数 ============

    /// 设置默认值
    pub fn default<V: Into<ParamValue>>(mut self, value: V) -> Self {
        self.params.insert("default".to_string(), value.into());
        self
    }

    /// 设置提示文本
    pub fn tooltip<V: Into<ParamValue>>(mut self, value: V) -> Self {
        self.params.insert("tooltip".to_string(), value.into());
        self
    }

    /// 强制输入
    pub fn force_input(mut self, force: bool) -> Self {
        self.params
            .insert("forceInput".to_string(), ParamValue::Bool(force));
        self
    }

    // ============ 数值类型参数 ============

    /// 设置最小值
    pub fn min<V: Into<ParamValue>>(mut self, value: V) -> Self {
        self.params.insert("min".to_string(), value.into());
        self
    }

    /// 设置最大值
    pub fn max<V: Into<ParamValue>>(mut self, value: V) -> Self {
        self.params.insert("max".to_string(), value.into());
        self
    }

    /// 设置步长
    pub fn step<V: Into<ParamValue>>(mut self, value: V) -> Self {
        self.params.insert("step".to_string(), value.into());
        self
    }

    /// 设置整数数最小值
    pub fn min_int(mut self, value: i64) -> Self {
        self.params
            .insert("min".to_string(), ParamValue::Int(value));
        self
    }

    /// 设置整数最大值
    pub fn max_int(mut self, value: i64) -> Self {
        self.params
            .insert("max".to_string(), ParamValue::Int(value));
        self
    }

    /// 设置整数步长
    pub fn step_int(mut self, value: i64) -> Self {
        self.params
            .insert("step".to_string(), ParamValue::Int(value));
        self
    }

    /// 设置浮点数最小值
    pub fn min_float(mut self, value: f64) -> Self {
        self.params
            .insert("min".to_string(), ParamValue::Float(value));
        self
    }

    /// 设置浮点数最大值
    pub fn max_float(mut self, value: f64) -> Self {
        self.params
            .insert("max".to_string(), ParamValue::Float(value));
        self
    }

    /// 设置浮点数步长
    pub fn step_float(mut self, value: f64) -> Self {
        self.params
            .insert("step".to_string(), ParamValue::Float(value));
        self
    }

    // ============ 字符串类型参数 ============

    /// 设置多行文本
    pub fn multiline(mut self, multiline: bool) -> Self {
        self.params
            .insert("multiline".to_string(), ParamValue::Bool(multiline));
        self
    }

    // ============ 布尔类型参数 ============

    /// 设置开标签
    pub fn label_on(mut self, label: impl Into<String>) -> Self {
        self.params
            .insert("label_on".to_string(), ParamValue::String(label.into()));
        self
    }

    /// 设置关标签
    pub fn label_off(mut self, label: impl Into<String>) -> Self {
        self.params
            .insert("label_off".to_string(), ParamValue::String(label.into()));
        self
    }

    /// 转换为 Python 元组
    ///
    /// 列表类型：(options_list, params_dict)
    /// 其他类型：(type_name, params_dict)
    fn to_py_tuple<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.kind == InputKind::List {
            // 列表类型：返回 (options_list, params_dict)
            let params_dict = self.params.to_py_dict(py)?;
            return Ok((self.list_options, params_dict)
                .into_pyobject(py)?
                .into_any());
        }

        let type_str = match &self.kind {
            InputKind::None => NODE_NONE,
            InputKind::String => NODE_STRING,
            InputKind::Int => NODE_INT,
            InputKind::Float => NODE_FLOAT,
            InputKind::Bool => NODE_BOOLEAN,
            InputKind::Image => NODE_IMAGE,
            InputKind::List => NODE_LIST,
            InputKind::Json => NODE_JSON,
            InputKind::MetadataRaw => NODE_METADATA_RAW,
            InputKind::Mask => NODE_MASK,
            InputKind::Model => NODE_MODEL,
            InputKind::Clip => NODE_CLIP,
            InputKind::Conditioning => NODE_CONDITIONING,
            InputKind::Vae => NODE_VAE,
            InputKind::Latent => NODE_LATENT,
        };
        let params_dict = self.params.to_py_dict(py)?;
        Ok((type_str, params_dict).into_pyobject(py)?.into_any())
    }
}

/// 将 IndexMap<String, ParamValue> 转换为 PyDict
trait ToPyDict {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>>;
}

impl ToPyDict for IndexMap<String, ParamValue> {
    fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in self {
            match value {
                ParamValue::String(s) => dict.set_item(key, s.as_str())?,
                ParamValue::Int(i) => dict.set_item(key, *i)?,
                ParamValue::Float(f) => dict.set_item(key, *f)?,
                ParamValue::Bool(b) => dict.set_item(key, *b)?,
            }
        }
        Ok(dict.into())
    }
}

// ============ ParamValue 的 From 实现 ============

impl From<String> for ParamValue {
    fn from(s: String) -> Self {
        ParamValue::String(s)
    }
}

impl From<&str> for ParamValue {
    fn from(s: &str) -> Self {
        ParamValue::String(s.to_string())
    }
}

impl From<i64> for ParamValue {
    fn from(i: i64) -> Self {
        ParamValue::Int(i)
    }
}

impl From<i32> for ParamValue {
    fn from(i: i32) -> Self {
        ParamValue::Int(i as i64)
    }
}

impl From<usize> for ParamValue {
    fn from(i: usize) -> Self {
        ParamValue::Int(i as i64)
    }
}

impl From<f64> for ParamValue {
    fn from(f: f64) -> Self {
        ParamValue::Float(f)
    }
}

impl From<f32> for ParamValue {
    fn from(f: f32) -> Self {
        ParamValue::Float(f as f64)
    }
}

impl From<bool> for ParamValue {
    fn from(b: bool) -> Self {
        ParamValue::Bool(b)
    }
}

/// 输出规范
#[derive(Debug, Clone)]
pub struct OutputSpec {
    pub types: Vec<&'static str>,
    pub names: Vec<&'static str>,
}

impl OutputSpec {
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            names: Vec::new(),
        }
    }

    pub fn with_output(mut self, type_name: &'static str, name: &'static str) -> Self {
        self.types.push(type_name);
        self.names.push(name);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试 InputSpec 构建器
    #[test]
    fn test_input_spec_builder() -> anyhow::Result<()> {
        let spec = InputSpec::new()
            .with_required("text", InputType::string().default("hello"))
            .with_required("count", InputType::int().default(10).min(0).max(100))
            .with_optional("enabled", InputType::bool().default(false));

        assert_eq!(spec.required.len(), 2);
        assert_eq!(spec.optional.len(), 1);
        assert_eq!(spec.required[0].0, "text");
        Ok(())
    }

    /// 测试枚举类型
    #[test]
    fn test_enum_input_type() -> anyhow::Result<()> {
        let options = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let spec =
            InputSpec::new().with_required("mode", InputType::list(options.clone()).default("A"));

        assert_eq!(spec.required.len(), 1);
        assert!(spec.required[0].1.list_options.is_empty());
        Ok(())
    }

    /// 测试多行文本
    #[test]
    fn test_multiline_string() -> anyhow::Result<()> {
        let spec = InputSpec::new().with_required("prompt", InputType::string().multiline(true));

        assert_eq!(spec.required.len(), 1);
        Ok(())
    }

    /// 测试浮点数范围
    #[test]
    fn test_float_with_range() -> anyhow::Result<()> {
        let spec = InputSpec::new().with_required(
            "temperature",
            InputType::float()
                .default(0.7)
                .min_float(0.0)
                .max_float(2.0),
        );

        assert_eq!(spec.required.len(), 1);
        Ok(())
    }

    /// 测试隐藏字段
    #[test]
    fn test_hidden_fields() -> anyhow::Result<()> {
        let spec = InputSpec::new()
            .with_hidden("unique_id", InputType::none())
            .with_hidden("prompt", InputType::none());

        assert_eq!(spec.hidden.len(), 2);
        Ok(())
    }

    /// 测试复杂示例
    #[test]
    fn test_complex_example() -> anyhow::Result<()> {
        let model_list = vec!["model1".to_string(), "model2".to_string()];
        let spec = InputSpec::new()
            .with_required(
                "model_path",
                InputType::list(model_list.clone())
                    .default("model1")
                    .tooltip("模型路径"),
            )
            .with_required("prompt", InputType::string().multiline(true))
            .with_required("steps", InputType::int().default(20).min(1).max(150))
            .with_optional(
                "temperature",
                InputType::float()
                    .default(0.7)
                    .min_float(0.0)
                    .max_float(2.0),
            )
            .with_optional("seed", InputType::int().default(-1))
            .with_hidden("unique_id", InputType::none());

        assert_eq!(spec.required.len(), 3);
        assert_eq!(spec.optional.len(), 2);
        assert_eq!(spec.hidden.len(), 1);
        Ok(())
    }
}
