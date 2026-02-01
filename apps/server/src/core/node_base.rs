//! ComfyUI 节点输入输出

use pyo3::{
    Bound, IntoPyObject, Py, PyAny, PyResult, Python,
    types::{PyAnyMethods, PyDict},
};

use crate::wrapper::comfyui::types::{
    NODE_BOOLEAN, NODE_FLOAT, NODE_IMAGE, NODE_INT, NODE_NONE, NODE_STRING,
};

/*



 Python::attach(|py| {
            let spec = InputSpec::new()
                // 模型选择
                .with_required("model_path", {
                    let models = Self::get_model_list();
                    InputType::Enum(models)
                })
                // 提示词
                .with_required(
                    "system_prompt",
                    InputType::String {
                        default: String::new(),
                        multiline: true,
                    },
                )
                .with_required(
                    "user_prompt",
                    InputType::String {
                        default: String::new(),
                        multiline: true,
                    },
                )
                // 生成参数
                .with_required(
                    "n_ctx",
                    InputType::Int {
                        default: 4096,
                        min: Some(256),
                        max: Some(i64::MAX),
                        step: 1,
                    },
                )
                .with_required(
                    "n_predict",
                    InputType::Int {
                        default: 2048,
                        min: Some(-1),
                        max: Some(i64::MAX),
                        step: 1,
                    },
                )
                .with_required(
                    "seed",
                    InputType::Int {
                        default: -1,
                        min: Some(-1),
                        max: Some(i64::MAX),
                        step: 1,
                    },
                )
                // GPU 参数
                .with_required(
                    "main_gpu",
                    InputType::Int {
                        default: 0,
                        min: Some(0),
                        max: None,
                        step: 1,
                    },
                )
                .with_required(
                    "n_gpu_layers",
                    InputType::Int {
                        default: 0,
                        min: Some(0),
                        max: Some(10000),
                        step: 1,
                    },
                )
                // 缓存选项
                .with_required(
                    "keep_context",
                    InputType::Bool { default: false },
                )
                .with_required(
                    "cache_model",
                    InputType::Bool { default: false },
                );

            let defaults = NodeOptions::default();
            spec.build(py, &defaults)
        })
*/

/// 输入规范
pub struct InputSpec<'py> {
    pub required: Vec<(String, InputTypeAttr<'py>)>,
    pub optional: Vec<(String, InputTypeAttr<'py>)>,
    pub hidden: Vec<(String, InputTypeAttr<'py>)>,
}

impl<'py> InputSpec<'py> {
    pub fn new() -> Self {
        Self {
            required: Vec::new(),
            optional: Vec::new(),
            hidden: Vec::new(),
        }
    }

    pub fn with_required(mut self, name: impl Into<String>, input_type: InputTypeAttr) -> Self {
        self.required.push((name.into(), input_type));
        self
    }

    pub fn with_optional(mut self, name: impl Into<String>, input_type: InputTypeAttr) -> Self {
        self.optional.push((name.into(), input_type));
        self
    }

    pub fn with_hidden(mut self, name: impl Into<String>, input_type: InputTypeAttr) -> Self {
        self.hidden.push((name.into(), input_type));
        self
    }

    /// 构建 PyDict
    pub fn build<'py>(&self, py: Python<'py>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);

        // Required 字段
        let required = PyDict::new(py);
        for (name, input_type) in &self.required {
            let (type_name, params) = input_type.to_py_dict(py)?;
            required.set_item(name, (type_name, params))?;
        }
        dict.set_item("required", required)?;

        // Optional 字段
        let optional = PyDict::new(py);
        for (name, input_type) in &self.optional {
            let (type_name, params) = input_type.to_py_dict(py)?;
            optional.set_item(name, (type_name, params))?;
        }
        dict.set_item("optional", optional)?;

        // Hidden 字段
        let hidden = PyDict::new(py);
        for (name, input_type) in &self.hidden {
            let (type_name, params) = input_type.to_py_dict(py)?;
            hidden.set_item(name, (type_name, params))?;
        }
        dict.set_item("hidden", hidden)?;

        Ok(dict.into())
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

/// 输入类型属性
pub struct InputTypeAttr<'py> {
    input_type: InputType,
    params: Bound<'py, PyDict>,
}

unsafe impl<'py> Send for InputTypeAttr<'py> {}
unsafe impl<'py> Sync for InputTypeAttr<'py> {}

impl<'py> InputTypeAttr<'py> {
    pub fn new(py: Python<'py>) -> InputTypeAttr {
        Self {
            input_type: InputType::None,
            params: PyDict::new(py),
        }
    }

    pub fn input_type(mut self, input_type: InputType) -> Self {
        self.input_type = input_type;
        self
    }

    pub fn default_value<V: IntoPyObject<'py>>(self, defaault: V) -> PyResult<Self> {
        self.params.set_item("defaault", defaault)?;
        Ok(self)
    }

    pub fn multiline<V: IntoPyObject<'py>>(self, multiline: V) -> PyResult<Self> {
        self.params.set_item("multiline", multiline)?;
        Ok(self)
    }

    pub fn tooltip<V: IntoPyObject<'py>>(self, tooltip: V) -> PyResult<Self> {
        self.params.set_item("tooltip", tooltip)?;
        Ok(self)
    }

    pub fn min<V: IntoPyObject<'py>>(self, min: V) -> PyResult<Self> {
        self.params.set_item("min", min)?;
        Ok(self)
    }

    pub fn max<V: IntoPyObject<'py>>(self, max: V) -> PyResult<Self> {
        self.params.set_item("max", max)?;
        Ok(self)
    }

    pub fn step<V: IntoPyObject<'py>>(self, step: V) -> PyResult<Self> {
        self.params.set_item("step", step)?;
        Ok(self)
    }

    pub fn label_on<V: IntoPyObject<'py>>(self, label_on: V) -> PyResult<Self> {
        self.params.set_item("label_on", label_on)?;
        Ok(self)
    }

    pub fn label_off<V: IntoPyObject<'py>>(self, label_off: V) -> PyResult<Self> {
        self.params.set_item("label_off", label_off)?;
        Ok(self)
    }

    pub fn force_input<V: IntoPyObject<'py>>(self, force_input: V) -> PyResult<Self> {
        self.params.set_item("force_input", force_input)?;
        Ok(self)
    }

    pub fn to_py_dict(self) -> PyResult<(&'static str, Py<PyDict>)> {
        Ok((self.input_type.into(), self.params.into()))
    }
}

/// 输入类型
#[derive(Debug, Default, Clone)]
pub enum InputType {
    #[default]
    None,
    String,
    Int,
    Float,
    Bool,
    Image,
}

impl From<InputType> for &str {
    fn from(input_type: InputType) -> Self {
        match input_type {
            InputType::None => NODE_NONE,
            InputType::String => NODE_STRING,
            InputType::Int => NODE_INT,
            InputType::Float => NODE_FLOAT,
            InputType::Bool => NODE_BOOLEAN,
            InputType::Image => NODE_IMAGE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple() -> anyhow::Result<()> {
        let input_spec = InputSpec::new()
            .with_required(
                "input_string",
                InputType::String {
                    default: "default".to_string(),
                    multiline: false,
                    tooltip: "tooltip".to_string(),
                },
            )
            .with_required(
                "input_string",
                InputType::String {
                    default: "default".to_string(),
                    multiline: todo!(),
                    tooltip: todo!(),
                },
            );
        Ok(())
    }
}
