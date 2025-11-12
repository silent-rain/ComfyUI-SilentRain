//! 模型

use pyo3::{
    Bound, PyResult, Python,
    types::{PyModule, PyModuleMethods},
};

use crate::core::node::NodeRegister;

mod nunchaku_sdxl_unet_loader_v2;
pub use nunchaku_sdxl_unet_loader_v2::NunchakuSdxlUnetLoaderV2;

mod nunchaku_sdxl_lora_loader;
pub use nunchaku_sdxl_lora_loader::NunchakuSdxlLoraLoader;

mod qwen_image_block_swap_path;
pub use qwen_image_block_swap_path::QwenImageBlockSwapPatch;

mod flux_block_swap_path;
pub use flux_block_swap_path::FluxBlockSwapPatch;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "model")?;
    submodule.add_class::<NunchakuSdxlUnetLoaderV2>()?;
    submodule.add_class::<NunchakuSdxlLoraLoader>()?;
    submodule.add_class::<QwenImageBlockSwapPatch>()?;
    submodule.add_class::<FluxBlockSwapPatch>()?;
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
            "NunchakuSdxlUnetLoaderV2",
            py.get_type::<NunchakuSdxlUnetLoaderV2>(),
            "Nunchaku SDXL UNet Loader V2",
        ),
        NodeRegister(
            "NunchakuSdxlLoraLoader",
            py.get_type::<NunchakuSdxlLoraLoader>(),
            "Nunchaku SDXL LoRA Loader",
        ),
        NodeRegister(
            "QwenImageBlockSwapPatch",
            py.get_type::<QwenImageBlockSwapPatch>(),
            "Qwen Image Block Swap Patch",
        ),
        NodeRegister(
            "FluxBlockSwapPatch",
            py.get_type::<FluxBlockSwapPatch>(),
            "Flux Block Swap Patch",
        ),
    ];
    Ok(nodes)
}
