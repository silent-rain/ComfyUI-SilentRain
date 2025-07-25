//! 工具

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

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "utils")?;
    submodule.add_class::<FileScanner>()?;
    Ok(submodule)
}
