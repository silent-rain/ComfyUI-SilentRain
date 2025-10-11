//! 模型

use pyo3::{Bound, PyResult, Python, types::PyModule};

use crate::core::node::NodeRegister;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "model")?;
    // submodule.add_class::<JoyCaptionExtraOptions>()?;
    Ok(submodule)
}

/// model node register
pub fn node_register(_py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![];
    Ok(nodes)
}
