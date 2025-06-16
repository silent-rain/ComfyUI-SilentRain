//! 图片

mod image_simple_resolution;
pub use image_simple_resolution::ImageSimpleResolution;

mod image_resolution;
pub use image_resolution::ImageResolution;

mod image_resolution2;
pub use image_resolution2::ImageResolution2;

mod load_images_from_folder;
pub use load_images_from_folder::LoadImagesFromFolder;

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "utils")?;
    submodule.add_class::<ImageSimpleResolution>()?;
    Ok(submodule)
}
