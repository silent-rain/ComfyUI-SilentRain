//! 字符串列表转字符串

use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_TEXT,
    wrapper::comfyui::{types::NODE_STRING, PromptServer},
};

/// 文本转列表
#[pyclass(subclass)]
pub struct StringListToSting {}

impl PromptServer for StringListToSting {}

#[pymethods]
impl StringListToSting {
    #[new]
    fn new() -> Self {
        Self {}
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_STRING,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("string",)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_TEXT;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Convert a string list to a string."
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
                    "string_list",
                    (NODE_STRING, {
                        let string_list = PyDict::new(py);
                        string_list.set_item("forceInput", true)?;
                        string_list
                    }),
                )?;
                required.set_item(
                    "join_with",
                    (NODE_STRING, {
                        let join_with = PyDict::new(py);
                        join_with.set_item("default", "\\n")?;
                        join_with.set_item("tooltip", "String list linking symbol.")?;
                        join_with
                    }),
                )?;
                required
            })?;
            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute")]
    fn execute(&mut self, string_list: Vec<String>, join_with: Vec<String>) -> PyResult<(String,)> {
        if string_list.is_empty() {
            return Ok(("".to_string(),));
        }
        let join_with = join_with[0].clone();
        let result = self.string_list_to_sting(string_list, join_with);

        Ok((result,))
    }
}

impl StringListToSting {
    fn string_list_to_sting(&self, string_list: Vec<String>, join_with: String) -> String {
        string_list.join(&join_with)
    }
}
