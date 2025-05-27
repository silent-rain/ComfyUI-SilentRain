//! 图片

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

mod image_simple_resolution;
pub use image_simple_resolution::ImageSimpleResolution;

mod image_resolution;
pub use image_resolution::ImageResolution;

mod image_resolution2;
pub use image_resolution2::ImageResolution2;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "utils")?;
    submodule.add_class::<ImageSimpleResolution>()?;
    Ok(submodule)
}
