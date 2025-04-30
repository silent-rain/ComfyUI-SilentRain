//! 任意节点桥接

use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyAny, PyResult, Python,
};

use crate::core::{category::CATEGORY_UTILS, types::any_type, PromptServer};

/// 任意节点桥接
#[pyclass(subclass)]
pub struct BridgeAnything {}

impl PromptServer for BridgeAnything {}

#[pymethods]
impl BridgeAnything {
    #[new]
    fn new() -> Self {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_level(true)
            .with_file(true)
            .with_line_number(true)
            .try_init();
        Self {}
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types(py: Python) -> (Bound<'_, PyAny>,) {
        let any_type = any_type(py).unwrap();
        (any_type,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("out",)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_UTILS;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "When any output is passed through this node, it collects and organizes data before forwarding, ensuring that the sub workflow of the previous stage has been completed."
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
    fn execute<'py>(&mut self, any: Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>,)> {
        Ok((any,))
    }
}
