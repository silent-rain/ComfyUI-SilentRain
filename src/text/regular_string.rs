//! 正则字符串

use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyErr, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_TEXT,
    error::Error,
    wrapper::comfyui::{
        types::{NODE_INT, NODE_STRING},
        PromptServer,
    },
};

/// 正则字符串
#[pyclass(subclass)]
pub struct RegularString {}

impl PromptServer for RegularString {}

#[pymethods]
impl RegularString {
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
    fn return_types() -> (&'static str, &'static str, &'static str) {
        (NODE_STRING, NODE_STRING, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str, &'static str) {
        ("results", "nth_result", "count")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool, bool) {
        (true, false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_TEXT;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Regular Text Node."
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
                    "string",
                    (NODE_STRING, {
                        let string = PyDict::new(py);
                        string
                    }),
                )?;
                required.set_item(
                    "pattern",
                    (NODE_STRING, {
                        let pattern = PyDict::new(py);
                        pattern.set_item("default", r#"(\d{4})-(\d{2})-(\d{2})"#)?;
                        pattern.set_item("tooltip", "regular expression")?;
                        pattern
                    }),
                )?;

                required.set_item(
                    "index",
                    (NODE_INT, {
                        let start_index = PyDict::new(py);
                        start_index.set_item("default", 0)?;
                        start_index.set_item("tooltip", "The Nth matching result")?;
                        start_index
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
        string: &str,
        pattern: &str,
        index: usize,
    ) -> PyResult<(Vec<String>, String, usize)> {
        let results = self.regular_string(string, pattern, index);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("RegularString error, {e}");
                if let Err(e) = self.send_error(py, "RegularString".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl RegularString {
    /// 正则字符串
    fn regular_string(
        &self,
        string: &str,
        pattern: &str,
        index: usize,
    ) -> Result<(Vec<String>, String, usize), Error> {
        let re = regex::Regex::new(pattern)?;
        let matches: Vec<String> = re
            .find_iter(string)
            .map(|m| m.as_str().to_string())
            .collect();
        let index = if index > matches.len() {
            matches.len()
        } else {
            index
        };

        let nth_result = matches[index].to_string();
        let count = matches.len();

        Ok((matches, nth_result, count))
    }
}
