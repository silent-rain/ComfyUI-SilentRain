//! 公共库
//!
//! 相关节点定义: ComfyUI/comfy/comfy_types/node_typing.py

mod prompt_server;
pub use prompt_server::PromptServer;
use pyo3::{
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult, Python,
};

mod always_equal_proxy;
mod py_wrapper;
pub mod tensor_wrapper;
pub use py_wrapper::isinstance;
pub use py_wrapper::isinstance2;

pub mod types;

pub mod category;
pub mod node;

/// 核心模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "core")?;
    submodule.add_function(wrap_pyfunction!(isinstance, &submodule)?)?;
    submodule.add_function(wrap_pyfunction!(isinstance2, &submodule)?)?;
    Ok(submodule)
}
