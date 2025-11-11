//! 模型

use pyo3::{
    Bound, PyResult, Python,
    types::{PyModule, PyModuleMethods},
};

use crate::core::node::NodeRegister;

mod nunchaku_sdxl_unet_loader;
pub use nunchaku_sdxl_unet_loader::NunchakuSdxlUnetLoader;

mod qwen_image_block_swap_path;
pub use qwen_image_block_swap_path::QwenImageBlockSwapPatch;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "model")?;
    submodule.add_class::<NunchakuSdxlUnetLoader>()?;
    submodule.add_class::<QwenImageBlockSwapPatch>()?;
    Ok(submodule)
}

/// model node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        // NodeRegister(
        //     "NunchakuSdxlUnetLoader",
        //     py.get_type::<NunchakuSdxlUnetLoader>(),
        //     "Nunchaku SDXL UNet Loader",
        // ),
        NodeRegister(
            "QwenImageBlockSwapPatch",
            py.get_type::<QwenImageBlockSwapPatch>(),
            "Qwen Image Block Swap Patch",
        ),
    ];
    Ok(nodes)
}
