// python 包装
pub mod comfy;
pub mod comfyui;
pub mod python;
pub mod torch;

use pyo3::{
    Bound, PyResult, Python,
    types::{PyModule, PyModuleMethods},
};

/// 核心模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "wrapper")?;
    submodule.add_submodule(&python::submodule(py)?)?;
    Ok(submodule)
}
