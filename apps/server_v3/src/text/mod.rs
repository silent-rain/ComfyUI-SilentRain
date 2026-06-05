/// Text processing nodes for SilentRain v3.
use pyo3::{prelude::*, types::PyType};

mod text_echo;
pub use text_echo::TextEcho;

/// 顶层模块导出
#[pymodule]
pub mod text {
    pub use super::*;

    #[pymodule_export]
    pub use TextEcho;
}

/// Collect all text nodes as Python types.
pub fn node_register(py: Python<'_>) -> PyResult<Vec<Py<PyType>>> {
    Ok(vec![py.get_type::<TextEcho>().into()])
}
