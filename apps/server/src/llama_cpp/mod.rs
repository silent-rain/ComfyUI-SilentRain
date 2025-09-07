//! 数学运算

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

use crate::core::node::NodeRegister;

mod llama_cpp_options;
pub use llama_cpp_options::LlamaCppOptions;

mod llama_cpp_vision;
pub use llama_cpp_vision::LlamaCppVision;

mod lllama_cpp_mtmd_context;
pub use lllama_cpp_mtmd_context::LlamaCppMtmdContext;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "math")?;
    // submodule.add_class::<IntConstant>()?;
    Ok(submodule)
}

/// Logic node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        // NodeRegister(
        //     "IntConstant",
        //     py.get_type::<IntConstant>(),
        //     "Sr Int Constant",
        // ),
    ];
    Ok(nodes)
}
