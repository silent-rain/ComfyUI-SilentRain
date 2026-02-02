//! 公共库
//!
//! 相关节点定义: ComfyUI/comfy/comfy_types/node_typing.py

use pyo3::{
    Bound, PyResult, Python,
    types::{PyModule, PyModuleMethods},
};

mod always_equal_proxy;
use always_equal_proxy::AlwaysEqualProxy;

mod always_equal_proxy2;
use always_equal_proxy2::AlwaysEqualProxy2;

pub mod category;
pub mod node;
pub mod node_base;
pub mod utils;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "core")?;
    submodule.add_class::<AlwaysEqualProxy>()?;
    submodule.add_class::<AlwaysEqualProxy2>()?;
    Ok(submodule)
}
