//! 文本相关的节点
pub mod file_scanner;

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

pub use file_scanner::FileScanner;

/// 文本模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "text")?;
    submodule.add_class::<FileScanner>()?;
    Ok(submodule)
}
