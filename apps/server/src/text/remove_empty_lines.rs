//! 去除空行

use log::error;
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};

use crate::{
    core::category::CATEGORY_TEXT,
    error::Error,
    wrapper::comfyui::{PromptServer, types::NODE_STRING},
};

/// 去除空行
#[pyclass(subclass)]
pub struct RemoveEmptyLines {}

impl PromptServer for RemoveEmptyLines {}

#[pymethods]
impl RemoveEmptyLines {
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
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> &'static str {
        NODE_STRING
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> &'static str {
        "text"
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_TEXT;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Remove empty lines from text."
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
                required.set_item("text", (NODE_STRING, { PyDict::new(py) }))?;
                required.set_item(
                    "keep_whitespace_lines",
                    ("BOOLEAN", {
                        let keep_whitespace_lines = PyDict::new(py);
                        keep_whitespace_lines.set_item("default", false)?;
                        keep_whitespace_lines.set_item(
                            "tooltip",
                            "Keep lines that contain only whitespace characters",
                        )?;
                        keep_whitespace_lines
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
        text: &str,
        keep_whitespace_lines: bool,
    ) -> PyResult<String> {
        let result = self.remove_empty_lines(text, keep_whitespace_lines);

        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("RemoveEmptyLines error, {e}");
                if let Err(e) = self.send_error(py, "RemoveEmptyLines".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl RemoveEmptyLines {
    /// 去除空行
    fn remove_empty_lines(&self, text: &str, keep_whitespace_lines: bool) -> Result<String, Error> {
        let lines: Vec<&str> = text.lines().collect();
        let filtered_lines: Vec<&str> = if keep_whitespace_lines {
            lines
                .iter()
                .filter(|&line| !line.is_empty())
                .copied()
                .collect()
        } else {
            lines
                .iter()
                .filter(|&line| !line.trim().is_empty())
                .copied()
                .collect()
        };

        Ok(filtered_lines.join("\n"))
    }
}
