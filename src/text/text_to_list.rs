//! 文本转列表

use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_TEXT,
    wrapper::comfyui::{
        types::{NODE_INT, NODE_STRING},
        PromptServer,
    },
};

/// 文本转列表
#[pyclass(subclass)]
pub struct TextToList {}

impl PromptServer for TextToList {}

#[pymethods]
impl TextToList {
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
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str) {
        (NODE_STRING, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("string", "total")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (true, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_TEXT;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Convert text into a string list based on line breaks."
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
                    "text",
                    (NODE_STRING, {
                        let text = PyDict::new(py);
                        text.set_item("default", "")?;
                        text.set_item("multiline", true)?;
                        text
                    }),
                )?;
                required.set_item(
                    "start_index",
                    (NODE_INT, {
                        let start_index = PyDict::new(py);
                        start_index.set_item("default", 0)?;
                        start_index.set_item("min", 0)?;
                        start_index.set_item("max", 1000000)?;
                        start_index.set_item("step", 1)?;
                        start_index
                    }),
                )?;
                required.set_item(
                    "limit",
                    (NODE_INT, {
                        let start_index = PyDict::new(py);
                        start_index.set_item("default", -1)?;
                        start_index.set_item("min", -1)?;
                        start_index.set_item("max", 1000000)?;
                        start_index.set_item("step", 1)?;
                        start_index
                    }),
                )?;
                required
            })?;
            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute")]
    fn execute(
        &mut self,
        text: &str,
        start_index: i32,
        limit: i32,
    ) -> PyResult<(Vec<String>, usize)> {
        let result = self.string_to_list(text, start_index, limit);

        Ok(result)
    }
}

impl TextToList {
    fn string_to_list(&self, text: &str, start_index: i32, limit: i32) -> (Vec<String>, usize) {
        let lines = text
            .split("\n")
            .map(|v| v.to_string())
            .collect::<Vec<String>>();

        let len = lines.len() as i32;
        let end_index = if limit == -1 || limit > len {
            len
        } else {
            limit
        };

        let results = lines[start_index as usize..end_index as usize].to_vec();
        let total = results.len();
        (results, total)
    }
}
