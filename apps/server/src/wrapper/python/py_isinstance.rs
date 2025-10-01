//! python 原生对象或函数封装

use std::ffi::CString;

use pyo3::{
    pyfunction,
    types::{PyAnyMethods, PyModule},
    Bound, PyAny, PyResult, Python,
};

/// Instance judgment of torch
/// py_type: torch.Tensor
#[pyfunction]
pub fn isinstance_by_torch<'py>(
    py: Python<'py>,
    py_any: &Bound<'py, PyAny>,
    py_type: &str,
) -> PyResult<bool> {
    let py_type = py_type.replace("torch.", "");
    let torch_module = py.import("torch")?;
    let tensor = torch_module.getattr(py_type)?;
    isinstance_py(py, py_any, &tensor)
}

/// Python `isinstance` function wrapper with proper implementation
#[pyfunction]
pub fn isinstance<'py>(
    py: Python<'py>,
    py_any: &Bound<'py, PyAny>,
    py_type: &str,
) -> PyResult<bool> {
    let code = CString::new(format!("isinstance({py_any:?}, {py_type})")).unwrap();
    let res = py.eval(&code, None, None)?;
    res.extract()
}

/// Python `isinstance` function wrapper with proper implementation
#[pyfunction]
pub fn isinstance_py<'py>(
    py: Python<'py>,
    py_any: &Bound<'py, PyAny>,
    py_type: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    // Import the built-in isinstance function
    let builtins = PyModule::import(py, "builtins")?;
    let isinstance_fn = builtins.getattr("isinstance")?;

    // Call isinstance(obj, type)
    let result = isinstance_fn.call1((py_any, py_type))?;

    // Extract the boolean result
    result.extract()
}

#[cfg(test)]
mod tests {
    use pyo3::types::{PyInt, PyString};

    use super::*;

    #[test]
    #[ignore]
    fn test_isinstance() -> anyhow::Result<()> {
        Python::attach(|py| {
            let binding = PyString::new(py, "this is a str.");
            let py_any = binding.as_any();
            let result = isinstance(py, py_any, "str").unwrap();
            assert!(result);

            let result = isinstance(py, py_any, "int").unwrap();
            assert!(result);
        });

        Ok(())
    }

    #[test]
    #[ignore]
    fn test_isinstance2() -> anyhow::Result<()> {
        Python::attach(|py| {
            let binding = PyString::new(py, "this is a str.");
            let py_any = binding.as_any();

            let binding = py_any.get_type();
            let py_type = binding.as_any();

            let result = isinstance_py(py, py_any, py_type).unwrap();
            assert!(result);

            {
                let binding = PyInt::new(py, 1);
                let py_any = binding.as_any();

                let binding = py_any.get_type();
                let py_type = binding.as_any();

                let result = isinstance_py(py, py_any, py_type).unwrap();
                assert!(!result);
            }
        });

        Ok(())
    }
}
