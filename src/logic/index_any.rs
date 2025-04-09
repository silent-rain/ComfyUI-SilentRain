//! 任意索引

use log::error;
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};

use crate::{
    core::{
        types::{any_type, NODE_INT},
        PromptServer,
    },
    error::Error,
};

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
                    (NODE_INT, {
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
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types(py: Python) -> (Bound<'_, PyAny>, &'static str) {
        let any_type = any_type(py).unwrap();
        (any_type, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("out", "total")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (false, false)
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
        any: Vec<Bound<'py, PyAny>>,
        index: Vec<usize>,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let total = any.len();
        let result = self.get_list_index(any, index[0]);

        match result {
            Ok(v) => Ok((v, total)),
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
        anys: Vec<Bound<'py, PyAny>>,
        index: usize,
    ) -> Result<Bound<'py, PyAny>, Error> {
        if anys.is_empty() {
            return Err(Error::InputListEmpty);
        }

        if index >= anys.len() {
            return Err(Error::IndexOutOfRange(format!(
                "The length of the list is {}",
                anys.len()
            )));
        }
        let result = anys.get(index).ok_or(Error::GetListIndex)?.clone();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ffi::CString;

    use pyo3::types::PyStringMethods;
    use pyo3::{types::PyString, Python};

    #[test]
    fn test_get_valid_index() -> anyhow::Result<()> {
        Python::with_gil(|py| {
            let index_any = IndexAny::new();

            // 创建包含Python对象的列表
            let elem1 = PyString::new(py, "1").into_any();
            let elem2 = PyString::new(py, "2").into_any();
            let anys = vec![elem1, elem2];

            // 测试第一个元素
            let result = index_any.get_list_index(anys.clone(), 0)?;
            assert_eq!(result.downcast::<PyString>().unwrap().to_str()?, "1");

            // 测试第二个元素
            let result = index_any.get_list_index(anys, 1)?;
            assert_eq!(result.downcast::<PyString>().unwrap().to_str()?, "2");

            Ok(())
        })
    }

    #[test]
    fn test_index_out_of_range() {
        Python::with_gil(|py| {
            let index_any = IndexAny::new();
            let elem = PyString::new(py, "test").into_any();
            let anys = vec![elem];

            let err = index_any
                .get_list_index(anys, 1)
                .expect_err("Should return index out of range error");

            if let Error::IndexOutOfRange(msg) = err {
                assert!(msg.contains("The length of the list is 1"));
            } else {
                panic!("Unexpected error type: {:?}", err);
            }
        })
    }

    #[test]
    fn test_empty_list() {
        let index_any = IndexAny::new();
        let err = index_any
            .get_list_index(vec![], 0)
            .expect_err("Should return empty list error");

        assert!(matches!(err, Error::InputListEmpty));
    }

    #[test]
    fn test_mixed_types() -> anyhow::Result<()> {
        Python::with_gil(|py| {
            let index_any = IndexAny::new();

            // 创建包含不同类型元素的列表
            let elements = vec![
                PyString::new(py, "text").into_any(),
                py.eval(&CString::new("42").unwrap(), None, None)?
                    .into_any(),
                py.eval(&CString::new("3.12").unwrap(), None, None)?
                    .into_any(),
            ];

            // 测试字符串类型
            let result = index_any.get_list_index(elements.clone(), 0)?;
            assert!(result.downcast::<PyString>().is_ok());

            // 测试整数类型
            let result = index_any.get_list_index(elements.clone(), 1)?;
            assert_eq!(result.extract::<i32>()?, 42);

            // 测试浮点数类型
            let result = index_any.get_list_index(elements, 2)?;
            assert_eq!(result.extract::<f64>()?, 3.12);

            Ok(())
        })
    }
}
