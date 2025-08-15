//! 控制台调试信息打印

use std::collections::HashMap;

use log::{error, info};
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_UTILS,
    wrapper::comfyui::{types::any_type, PromptServer},
};

/// 控制台调试
#[pyclass(subclass)]
pub struct ConsoleDebug {}

impl PromptServer for ConsoleDebug {}

#[pymethods]
impl ConsoleDebug {
    #[new]
    fn new() -> Self {
        Self {}
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
    const CATEGORY: &'static str = CATEGORY_UTILS;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Output node information on the console."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);
                required.set_item(
                    "any",
                    (any_type(py)?, {
                        let any = PyDict::new(py);
                        any.set_item("tooltip", "any input")?;
                        any
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
        any: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let results = self.console_debug(&any);

        match results {
            Ok(_v) => self.node_result(py),
            Err(e) => {
                error!("ConsoleDebug error, {e}");
                if let Err(e) = self.send_error(py, "ConsoleDebug".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}
impl ConsoleDebug {
    /// 组合为前端需要的数据结构
    fn node_result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("result", HashMap::<String, String>::new())?;
        Ok(dict)
    }
}

impl ConsoleDebug {
    /// 保存图像
    fn console_debug<'py>(&self, any: &Bound<'py, PyAny>) -> Result<(), String> {
        info!("console_debug: \n{any:#?}");
        Ok(())
    }
}
