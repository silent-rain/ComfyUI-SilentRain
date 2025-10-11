//! 控制台打印 Conditioning 内容

use std::{collections::HashMap, fs};

use candle_core::Device;
use log::{error, info};
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyList, PyType},
};

use crate::{
    core::category::CATEGORY_CONDITIONING,
    error::Error,
    wrapper::{
        comfy::node_helpers::{Conditioning, ConditioningShape, conditionings_py2rs},
        comfyui::{
            PromptServer,
            types::{NODE_BOOLEAN, NODE_CONDITIONING, NODE_STRING},
        },
    },
};

/// 控制台打印 Conditioning 内容
#[pyclass(subclass)]
pub struct ConditioningConsoleDebug {
    device: Device,
}

impl PromptServer for ConditioningConsoleDebug {}

#[pymethods]
impl ConditioningConsoleDebug {
    #[new]
    fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_NODE")]
    fn output_node() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() {}

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() {}

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() {}

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_CONDITIONING;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Console printing of Condition Shape content"
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);

               required.set_item(
                    "conditioning",
                    (NODE_CONDITIONING, {
                        
                        PyDict::new(py)
                    }),
                )?;

                required.set_item(
                    "shape",
                    (NODE_BOOLEAN, {
                        let conditioning_shape = PyDict::new(py);
                        conditioning_shape.set_item("default", false)?;
                        conditioning_shape.set_item("tooltip", "print conditioning shape")?;
                        conditioning_shape
                    }),
                )?;

                required.set_item(
                    "detail",
                    (NODE_BOOLEAN, {
                        let conditioning_detail = PyDict::new(py);
                        conditioning_detail.set_item("default", false)?;
                        conditioning_detail.set_item("tooltip", "print conditioning details")?;
                        conditioning_detail
                    }),
                )?;

                required.set_item(
                    "filepath",
                    (NODE_STRING, {
                        let filepath = PyDict::new(py);
                        filepath.set_item("default", "")?;
                        filepath.set_item(
                            "tooltip",
                            "If the file path is configured, debugging information will be stored in a file.",
                        )?;
                        filepath
                    }),
                )?;

                required
            })?;

            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        conditioning: Bound<'py, PyList>,
        shape: bool,
        detail: bool,
        filepath: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let results = self.console_debug(conditioning, shape, detail, filepath);

        match results {
            Ok(_v) => self.node_result(py),
            Err(e) => {
                error!("ConditioningShapeConsoleDebug error, {e}");
                if let Err(e) = self.send_error(
                    py,
                    "ConditioningShapeConsoleDebug".to_string(),
                    e.to_string(),
                ) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}
impl ConditioningConsoleDebug {
    /// 组合为前端需要的数据结构
    fn node_result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("result", HashMap::<String, String>::new())?;
        Ok(dict)
    }
}

impl ConditioningConsoleDebug {
    /// print conditionings shape
    pub fn to_conditionings_shape(
        conditionings: &[Conditioning],
    ) -> Result<Vec<ConditioningShape>, Error> {
        let mut shapes = Vec::new();
        for conditioning in conditionings.iter() {
            shapes.push(conditioning.shape());
        }

        Ok(shapes)
    }

    /// 保存图像
    fn console_debug<'py>(
        &self,
        conditioning: Bound<'py, PyList>,
        shape: bool,
        detail: bool,
        filepath: &str,
    ) -> Result<(), Error> {
        // py params to rust
        let conditionings = conditionings_py2rs(conditioning, &self.device)?;

        let mut contents = String::new();

        if shape {
            let shapes = Self::to_conditionings_shape(&conditionings)?;

            let results = serde_json::to_string_pretty(&shapes)?;
            info!("conditionings shape: \n{results:?}");

            contents.push_str(&results);
        }

        if detail {
            let results = serde_json::to_string_pretty(&conditionings)?;
            info!("conditionings detail: \n{results:?}");

            contents.push_str(&results);
        }

        if !filepath.is_empty() {
            fs::write(filepath, contents.as_bytes())?;
        }
        Ok(())
    }
}
