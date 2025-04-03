//! 任意索引

use log::error;
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyList, PyListMethods, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};

use crate::{error::Error, prompt_server::PromptServer, utils::any_type};

#[pyclass(subclass)]
pub struct IndexAny {}

impl PromptServer for IndexAny {}

#[pymethods]
impl IndexAny {
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
                    "any",
                    (any_type(py)?, {
                        let any = PyDict::new(py);
                        any.set_item("tooltip", "Input any list")?;
                        any
                    }),
                )?;
                required.set_item(
                    "index",
                    ("INT", {
                        let index = PyDict::new(py);
                        index.set_item("default", 0)?;
                        index.set_item("min", 0)?;
                        index.set_item("max", 1000000)?;
                        index.set_item("step", 1)?;
                        index
                    }),
                )?;
                required
            })?;
            Ok(dict.into())
        })
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
    fn output_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = "SilentRain/Logic";

    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python,
        any: Bound<'py, PyAny>,
        index: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        println!("=========================: {:#?}", any);
        let results = self.get_list_index(any, index);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("get list index error, {e}");
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

impl IndexAny {
    /// 获取列表指定索引的数据
    fn get_list_index<'py>(
        &self,
        any: Bound<'py, PyAny>,
        index: usize,
    ) -> Result<Bound<'py, PyAny>, Error> {
        match any.downcast::<PyList>() {
            Ok(value) => {
                if value.is_empty() {
                    return Err(Error::InputListEmpty);
                }

                if index >= value.len() {
                    return Err(Error::IndexOutOfRange(format!(
                        "The length of the list is {}",
                        value.len()
                    )));
                }
                value
                    .get_item(index)
                    .map_err(|e| Error::GetItemAtIndex(e.to_string()))
            }
            Err(e) => Err(Error::DowncastFailed(e.to_string())),
        }
    }
}
