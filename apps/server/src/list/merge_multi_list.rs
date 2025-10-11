//! 合并列表

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};

use crate::{
    core::category::CATEGORY_LIST,
    error::Error,
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_INT, any_type},
    },
};

/// 合并列表
#[pyclass(subclass)]
pub struct MergeMultiList {}

impl PromptServer for MergeMultiList {}

#[pymethods]
impl MergeMultiList {
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
    fn return_types(py: Python<'_>) -> (Bound<'_, PyAny>, &'static str) {
        let any_type = any_type(py).unwrap();
        (any_type, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("list", "count")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (true, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LIST;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Merge multiple lists into one list."
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
                    "list_1",
                    (any_type(py)?, {
                        let batch = PyDict::new(py);
                        batch.set_item("tooltip", "Input any list")?;
                        batch
                    }),
                )?;
                required.set_item(
                    "list_2",
                    (any_type(py)?, {
                        let batch = PyDict::new(py);
                        batch.set_item("tooltip", "Input any list")?;
                        batch
                    }),
                )?;
                required
            })?;
            Ok(dict.into())
        })
    }

    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        list_1: Vec<Bound<'py, PyAny>>,
        list_2: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<(Vec<Bound<'py, PyAny>>, usize)> {
        let results = self.merge_multi_list(list_1, list_2);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("MergeMultiList error, {e}");
                if let Err(e) = self.send_error(py, "MergeMultiList".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl MergeMultiList {
    pub fn merge_multi_list<'py>(
        &self,
        list_1: Vec<Bound<'py, PyAny>>,
        list_2: Vec<Bound<'py, PyAny>>,
    ) -> Result<(Vec<Bound<'py, PyAny>>, usize), Error> {
        let mut results = list_1;
        results.extend(list_2);

        let total = results.len();

        Ok((results, total))
    }
}
