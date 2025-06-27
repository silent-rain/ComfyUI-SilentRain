//! python 原生对象或函数封装

use std::ffi::CString;

use pyo3::{
    pyfunction,
    types::{PyAnyMethods, PyModule},
    Bound, PyAny, PyResult, Python,
};

/// Python `isinstance` function wrapper with proper implementation
#[pyfunction]
pub fn isinstance<'py>(
    py: Python<'py>,
    py_any: &Bound<'py, PyAny>,
    py_type: &str,
) -> PyResult<bool> {
    let code = CString::new(format!("isinstance({py_any:?}, {py_type})")).unwrap();
    let res = if py_type.starts_with("torch") {
        //         let code = c_str!(
        //             "
        // import torch

        // tensor = torch.Tensor
        //             "
        //         );
        //         py.run(code, None, None)?;

        //         let code = CString::new(format!("isinstance({:?}, {})", py_any, "tensor")).unwrap();
        //         py.eval(&code, None, None)?

        let torch_module = py.import("torch")?;
        let tensor = torch_module.getattr("Tensor")?;
        return isinstance2(py, py_any, &tensor);
    } else {
        py.eval(&code, None, None)?
    };

    res.extract()
}

/// Python `isinstance` function wrapper with proper implementation
#[pyfunction]
pub fn isinstance2<'py>(
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
        Python::with_gil(|py| {
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
        Python::with_gil(|py| {
            let binding = PyString::new(py, "this is a str.");
            let py_any = binding.as_any();

            let binding = py_any.get_type();
            let py_type = binding.as_any();

            let result = isinstance2(py, py_any, py_type).unwrap();
            assert!(result);

            {
                let binding = PyInt::new(py, 1);
                let py_any = binding.as_any();

                let binding = py_any.get_type();
                let py_type = binding.as_any();

                let result = isinstance2(py, py_any, py_type).unwrap();
                assert!(!result);
            }
        });

        Ok(())
    }
}
