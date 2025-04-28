//! 工具

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

mod file_scanner;
pub use file_scanner::FileScanner;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "utils")?;
    submodule.add_class::<FileScanner>()?;
    Ok(submodule)
}
