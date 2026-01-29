//! 逻辑

use pyo3::{
    Bound, PyResult, Python,
    types::{PyModule, PyModuleMethods},
};

use crate::core::node::NodeRegister;

mod batch_float;
pub use batch_float::BatchFloat;

mod is_empty;
pub use is_empty::IsEmpty;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "logic")?;
    submodule.add_class::<BatchFloat>()?;
    submodule.add_class::<IsEmpty>()?;
    Ok(submodule)
}

/// Logic node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        NodeRegister("BatchFloat", py.get_type::<BatchFloat>(), "Sr Batch Float"),
        NodeRegister("IsEmpty", py.get_type::<IsEmpty>(), "Sr Is Empty"),
    ];
    Ok(nodes)
}
