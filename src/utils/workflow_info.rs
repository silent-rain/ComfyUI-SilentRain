//! 获取工作流信息

use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};
use pythonize::pythonize;

use crate::{
    core::category::CATEGORY_UTILS,
    error::Error,
    wrapper::comfyui::{
        node_input::InputKwargs,
        types::{any_type, NODE_STRING},
        PromptServer,
    },
};

/// 工作流信息
#[pyclass(subclass)]
pub struct WorkflowInfo {}

impl PromptServer for WorkflowInfo {}

#[pymethods]
impl WorkflowInfo {
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
    fn return_types(py: Python<'_>) -> (&'static str, Bound<'_, PyAny>, Bound<'_, PyAny>) {
        let any_type = any_type(py).unwrap();
        (NODE_STRING, any_type.clone(), any_type)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str, &'static str) {
        ("workflow_id", "workflow", "pre_node")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool, bool) {
        (false, false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_UTILS;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Retrieve workflow data and retrieve data from the previous node."
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

            dict.set_item("hidden", {
                let hidden = PyDict::new(py);
                hidden.set_item("prompt", "PROMPT")?;
                hidden.set_item("dynprompt", "DYNPROMPT")?;
                hidden.set_item("extra_pnginfo", "EXTRA_PNGINFO")?;
                // UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side.
                // It is commonly used in client-server communications (see messages).
                hidden.set_item("unique_id", "UNIQUE_ID")?;

                hidden
            })?;

            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute", signature = (any, **kwargs))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        any: Bound<'py, PyAny>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(String, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let result = self.get_info(py, any, kwargs);

        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("WorkflowInfo error, {e}");
                if let Err(e) = self.send_error(py, "WorkflowInfo".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl WorkflowInfo {
    /// 获取工作流信息及节点信息
    fn get_info<'py>(
        &self,
        py: Python<'py>,
        _any: Bound<'py, PyAny>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<(String, Bound<'py, PyAny>, Bound<'py, PyAny>), Error> {
        let kwargs = kwargs.clone().ok_or_else(|| {
            Error::PyMissingKwargs("the py kwargs parameter does not exist".to_string())
        })?;
        let kwargs = InputKwargs::new(&kwargs);
        let workflow = kwargs.workflow()?;
        let workflow_id = workflow.id.clone();
        let pre_node = kwargs.pre_node()?;

        // Rust -> Python
        let workflow_obj = pythonize(py, &workflow)?;
        let pre_node_obj = pythonize(py, &pre_node)?;

        Ok((workflow_id, workflow_obj, pre_node_obj))
    }
}
