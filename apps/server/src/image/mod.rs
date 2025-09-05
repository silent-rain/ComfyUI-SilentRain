//! 图片

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

use crate::core::node::NodeRegister;

mod image_simple_resolution;
pub use image_simple_resolution::ImageSimpleResolution;

mod image_resolution;
pub use image_resolution::ImageResolution;

mod image_resolution2;
pub use image_resolution2::ImageResolution2;

mod load_images_from_folder;
pub use load_images_from_folder::LoadImagesFromFolder;

mod save_images;
pub use save_images::SaveImages;

mod save_image_text;
pub use save_image_text::SaveImageText;

mod image_split_grid;
pub use image_split_grid::ImageSplitGrid;

mod image_grid_composite;
pub use image_grid_composite::ImageGridComposite;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "image")?;
    submodule.add_class::<ImageSimpleResolution>()?;
    submodule.add_class::<ImageResolution>()?;
    submodule.add_class::<ImageResolution2>()?;
    submodule.add_class::<LoadImagesFromFolder>()?;
    submodule.add_class::<SaveImages>()?;
    submodule.add_class::<SaveImageText>()?;
    submodule.add_class::<ImageSplitGrid>()?;
    submodule.add_class::<ImageGridComposite>()?;
    Ok(submodule)
}

/// Image node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        NodeRegister(
            "ImageSimpleResolution",
            py.get_type::<ImageSimpleResolution>(),
            "Sr Image Simple Resolution",
        ),
        NodeRegister(
            "ImageResolution",
            py.get_type::<ImageResolution>(),
            "Sr Image Resolution",
        ),
        NodeRegister(
            "ImageResolution2",
            py.get_type::<ImageResolution2>(),
            "Sr Image Resolution2",
        ),
        NodeRegister(
            "LoadImagesFromFolder",
            py.get_type::<LoadImagesFromFolder>(),
            "Sr Load Images From Folder",
        ),
        NodeRegister("SaveImages", py.get_type::<SaveImages>(), "Sr Save Images"),
        NodeRegister(
            "SaveImageText",
            py.get_type::<SaveImageText>(),
            "Sr Save Image Text",
        ),
        NodeRegister(
            "ImageSplitGrid",
            py.get_type::<ImageSplitGrid>(),
            "Sr Image Split Grid",
        ),
        NodeRegister(
            "ImageGridComposite",
            py.get_type::<ImageGridComposite>(),
            "Sr Image Grid Composite",
        ),
    ];
    Ok(nodes)
}
