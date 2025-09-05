//! 公共库
//!
//! 相关节点定义: ComfyUI/comfy/comfy_types/node_typing.py

mod always_equal_proxy;
use always_equal_proxy::AlwaysEqualProxy;

pub mod category;
pub mod node;
pub mod utils;

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "core")?;
    submodule.add_class::<AlwaysEqualProxy>()?;
    Ok(submodule)
}
