//! python 包装

mod py_isinstance;
pub use py_isinstance::isinstance;
pub use py_isinstance::isinstance_by_torch;

use pyo3::{
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult, Python,
};

/// 核心模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "python")?;
    submodule.add_function(wrap_pyfunction!(isinstance_by_torch, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(isinstance, &submodule)?)?;
    Ok(submodule)
}
