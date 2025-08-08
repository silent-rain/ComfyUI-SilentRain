//! 模型

use pyo3::{types::PyModule, Bound, PyResult, Python};

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "model")?;
    // submodule.add_class::<JoyCaptionExtraOptions>()?;
    Ok(submodule)
}
