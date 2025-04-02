//! 逻辑

pub mod index_any;

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

pub use index_any::IndexAny;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "logic")?;
    submodule.add_class::<IndexAny>()?;
    Ok(submodule)
}
