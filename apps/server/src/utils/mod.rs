//! 工具
use pyo3::{
    Bound, PyResult, Python,
    types::{PyModule, PyModuleMethods},
};

use crate::core::node::NodeRegister;

mod template;

mod file_scanner;
pub use file_scanner::FileScanner;

mod bridge_anything;
pub use bridge_anything::BridgeAnything;

mod workflow_info;
pub use workflow_info::WorkflowInfo;

mod batch_rename;
pub use batch_rename::BatchRename;

mod console_debug;
pub use console_debug::ConsoleDebug;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "utils")?;
    submodule.add_class::<FileScanner>()?;
    submodule.add_class::<BridgeAnything>()?;
    submodule.add_class::<WorkflowInfo>()?;
    submodule.add_class::<BatchRename>()?;
    submodule.add_class::<ConsoleDebug>()?;
    Ok(submodule)
}

/// Utils node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        NodeRegister(
            "FileScanner",
            py.get_type::<FileScanner>(),
            "Sr File Scanner",
        ),
        NodeRegister(
            "BridgeAnything",
            py.get_type::<BridgeAnything>(),
            "Sr Bridge Anything",
        ),
        NodeRegister(
            "WorkflowInfo",
            py.get_type::<WorkflowInfo>(),
            "Sr Workflow Info",
        ),
        NodeRegister(
            "BatchRename",
            py.get_type::<BatchRename>(),
            "Sr Batch Rename",
        ),
        NodeRegister(
            "ConsoleDebug",
            py.get_type::<ConsoleDebug>(),
            "Sr Console Debug",
        ),
    ];
    Ok(nodes)
}
