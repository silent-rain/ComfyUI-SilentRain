//! 文本框

use std::io::{self, BufRead};

use log::error;
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyErr, PyResult, Python,
};

use crate::{
    core::{
        category::CATEGORY_TEXT,
        types::{NODE_BOOLEAN, NODE_STRING},
        PromptServer,
    },
    error::Error,
};

/// 文本框
#[pyclass(subclass)]
pub struct TextBox {}

impl PromptServer for TextBox {}

#[pymethods]
impl TextBox {
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
                    "strip_newlines",
                    (NODE_BOOLEAN, {
                        let strip_newlines = PyDict::new(py);
                        strip_newlines.set_item("default", true)?;
                        strip_newlines
                    }),
                )?;
                required
            })?;
            Ok(dict.into())
        })
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_STRING,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("text",)
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
        "This is a multi line text box node."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[pyo3(name = "execute")]
    fn execute(&mut self, py: Python, text: &str, strip_newlines: bool) -> PyResult<(String,)> {
        let result = self.stringify(text, strip_newlines);

        match result {
            Ok(v) => Ok((v,)),
            Err(e) => {
                error!("scan files failed, {e}");
                if let Err(e) = self.send_error(py, "SCAN_FILES_ERROR".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        e.to_string(),
                    ));
                };
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    e.to_string(),
                ))
            }
        }
    }
}

impl TextBox {
    fn stringify(&self, text: &str, strip_newlines: bool) -> Result<String, Error> {
        let mut new_string: Vec<String> = Vec::new();
        let reader = io::Cursor::new(text);
        for line in reader.lines() {
            let line = line?;
            if !line.starts_with('\n') && strip_newlines {
                new_string.push(line.replace("\\n", "\n"));
            } else if line.starts_with('\n') && strip_newlines {
                new_string.push(line.replace("\\n", ""));
            } else {
                new_string.push(line);
            }
        }
        Ok(new_string.join("\n"))
    }
}
