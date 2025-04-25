//! 逻辑

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

mod index_anything;
pub use index_anything::IndexAnything;

mod list_count;
pub use list_count::ListCount;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "logic")?;
    submodule.add_class::<IndexAnything>()?;
    Ok(submodule)
}
