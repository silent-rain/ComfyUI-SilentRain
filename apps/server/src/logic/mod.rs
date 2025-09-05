//! 逻辑

mod batch_float;
pub use batch_float::BatchFloat;

use pyo3::{types::PyModule, Bound, PyResult, Python};

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "logic")?;
    // submodule.add_class::<IndexAnything>()?;
    Ok(submodule)
}
