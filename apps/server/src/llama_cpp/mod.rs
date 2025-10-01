//! llama.cpp
//!
//! git: https://github.com/utilityai/llama-cpp-rs
//!

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
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

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "math")?;
    submodule.add_class::<LlamaCppOptions>()?;
    submodule.add_class::<LlamaCppVision>()?;
    submodule.add_class::<LlamaCppChat>()?;
    Ok(submodule)
}

/// llama.cpp node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
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
    ];
    Ok(nodes)
}
