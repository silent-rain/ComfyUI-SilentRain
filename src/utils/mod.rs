//! 工具

pub mod always_equal_proxy;

use pyo3::{
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult, Python,
};

pub use always_equal_proxy::{any_type, AlwaysEqualProxy};

/// 工具模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "utils")?;
    submodule.add_class::<AlwaysEqualProxy>()?;
    submodule.add_function(wrap_pyfunction!(any_type, &submodule)?)?;

    Ok(submodule)
}
