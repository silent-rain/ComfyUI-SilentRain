// python 包装
pub mod comfyui;
pub mod python;
pub mod torch;

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

/// 核心模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "wrapper")?;
    submodule.add_submodule(&python::submodule(py)?)?;
    Ok(submodule)
}
