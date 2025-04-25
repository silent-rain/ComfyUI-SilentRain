//! 逻辑

pub mod index_anything;
pub mod list_count;

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

pub use index_anything::IndexAnything;
pub use list_count::ListCount;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "logic")?;
    submodule.add_class::<IndexAnything>()?;
    Ok(submodule)
}
