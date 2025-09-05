//! 遮罩

use pyo3::{types::PyModule, Bound, PyResult, Python};

use crate::core::node::NodeRegister;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "math")?;
    // submodule.add_class::<JoyCaptionExtraOptions>()?;
    Ok(submodule)
}

/// Logic node register
pub fn node_register(_py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![];
    Ok(nodes)
}
