//! 工具

pub mod always_equal_proxy;

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

pub use always_equal_proxy::AlwaysEqualProxy;

/// 工具模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "utils")?;
    submodule.add_class::<AlwaysEqualProxy>()?;
    Ok(submodule)
}
