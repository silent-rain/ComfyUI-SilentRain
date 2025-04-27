//! 工具

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

mod file_scanner;
pub use file_scanner::FileScanner;

mod index_anything;
pub use index_anything::IndexAnything;

mod list_count;
pub use list_count::ListCount;

mod shuffle_any_list;
pub use shuffle_any_list::ShuffleAnyList;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "utils")?;
    submodule.add_class::<IndexAnything>()?;
    Ok(submodule)
}
