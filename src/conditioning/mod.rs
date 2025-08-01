//! Conditioning

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

mod flux_kontext_inpainting_conditioning;
pub use flux_kontext_inpainting_conditioning::FluxKontextInpaintingConditioning;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "conditioning")?;
    submodule.add_class::<FluxKontextInpaintingConditioning>()?;
    Ok(submodule)
}
