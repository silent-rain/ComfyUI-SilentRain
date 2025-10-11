//! 逻辑

use pyo3::{
    Bound, PyResult, Python,
    types::{PyModule, PyModuleMethods},
};

use crate::core::node::NodeRegister;

mod batch_float;
pub use batch_float::BatchFloat;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "logic")?;
    submodule.add_class::<BatchFloat>()?;
    Ok(submodule)
}

/// Logic node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![NodeRegister(
        "BatchFloat",
        py.get_type::<BatchFloat>(),
        "Sr Batch Float",
    )];
    Ok(nodes)
}
