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
mod llama_cpp_model_v2;
pub use llama_cpp_model_v2::{LlamaCppModelv2, ModelHandleV2};

mod llama_cpp_chat_v2;
pub use llama_cpp_chat_v2::LlamaCppChatv2;

mod llama_cpp_vision_v2;
pub use llama_cpp_vision_v2::LlamaCppVisionv2;

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
    submodule.add_class::<LlamaCppChatv2>()?;
    submodule.add_class::<LlamaCppVisionv2>()?;
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
            "LlamaCppChatv2",
            py.get_type::<LlamaCppChatv2>(),
            "Sr Llama Cpp Chat v2",
        ),
        NodeRegister(
            "LlamaCppVisionv2",
            py.get_type::<LlamaCppVisionv2>(),
            "Sr Llama Cpp Vision v2",
        ),
        NodeRegister(
            "LlamaCppPurgeVramv2",
            py.get_type::<LlamaCppPurgeVramv2>(),
            "Sr Llama Cpp Purge Vram v2",
        ),
    ];
    Ok(nodes)
}
