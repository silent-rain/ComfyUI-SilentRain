//! llama.cpp
//!
//! git: https://github.com/utilityai/llama-cpp-rs
//!

use pyo3::{
    Bound, PyResult, Python,
    types::{PyModule, PyModuleMethods},
};

use crate::core::node::NodeRegister;

mod llama_cpp_options;
pub use llama_cpp_options::{LlamaCppOptions, PromptMessageRole};

mod llama_cpp_base_context;
pub use llama_cpp_base_context::LlamaCppBaseContext;

mod llama_cpp_mtmd_context;
pub use llama_cpp_mtmd_context::LlamaCppMtmdContext;

mod llama_cpp_pipeline;
pub use llama_cpp_pipeline::LlamaCppPipeline;

mod llama_cpp_vision;
pub use llama_cpp_vision::LlamaCppVision;

mod llama_cpp_chat;
pub use llama_cpp_chat::LlamaCppChat;

mod model_manager;
pub use model_manager::ModelManager;

mod llama_cpp_purge_vram;
pub use llama_cpp_purge_vram::LlamaCppPurgeVram;

// v2 版本节点
mod llama_cpp_options_v2;
pub use llama_cpp_options_v2::LlamaCppOptionsv2;
mod llama_cpp_model_v2;
pub use llama_cpp_model_v2::LlamaCppModelv2;
mod llama_cpp_prompt_helper_v2;
pub use llama_cpp_prompt_helper_v2::LlamaCppPromptHelperv2;
mod llama_cpp_image_caption_v2;
pub use llama_cpp_image_caption_v2::LlamaCppImageCaptionv2;
mod llama_cpp_purge_vram_v2;
pub use llama_cpp_purge_vram_v2::LlamaCppPurgeVramv2;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "math")?;
    // v1 节点
    submodule.add_class::<LlamaCppOptions>()?;
    submodule.add_class::<LlamaCppVision>()?;
    submodule.add_class::<LlamaCppChat>()?;
    submodule.add_class::<LlamaCppPurgeVram>()?;
    // v2 节点
    submodule.add_class::<LlamaCppModelv2>()?;
    submodule.add_class::<LlamaCppOptionsv2>()?;
    submodule.add_class::<LlamaCppPromptHelperv2>()?;
    submodule.add_class::<LlamaCppImageCaptionv2>()?;
    submodule.add_class::<LlamaCppPurgeVramv2>()?;
    Ok(submodule)
}

/// llama.cpp node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        // v1 节点
        NodeRegister(
            "LlamaCppOptions",
            py.get_type::<LlamaCppOptions>(),
            "Sr Llama Cpp Options",
        ),
        NodeRegister(
            "LlamaCppVision",
            py.get_type::<LlamaCppVision>(),
            "Sr Llama Cpp Vision",
        ),
        NodeRegister(
            "LlamaCppChat",
            py.get_type::<LlamaCppChat>(),
            "Sr Llama Cpp Chat",
        ),
        NodeRegister(
            "LlamaCppPurgeVram",
            py.get_type::<LlamaCppPurgeVram>(),
            "Sr Llama Cpp Purge Vram",
        ),
        // v2 节点（实验性）
        NodeRegister(
            "LlamaCppModelv2",
            py.get_type::<LlamaCppModelv2>(),
            "Sr Llama Cpp Model v2",
        ),
        NodeRegister(
            "LlamaCppOptionsv2",
            py.get_type::<LlamaCppOptionsv2>(),
            "Sr Llama Cpp Options v2",
        ),
        NodeRegister(
            "LlamaCppPromptHelperv2",
            py.get_type::<LlamaCppPromptHelperv2>(),
            "Sr Llama Cpp Prompt Helper v2",
        ),
        NodeRegister(
            "LlamaCppImageCaptionv2",
            py.get_type::<LlamaCppImageCaptionv2>(),
            "Sr Llama Cpp Image Caption v2",
        ),
        NodeRegister(
            "LlamaCppPurgeVramv2",
            py.get_type::<LlamaCppPurgeVramv2>(),
            "Sr Llama Cpp Purge Vram v2",
        ),
    ];
    Ok(nodes)
}
