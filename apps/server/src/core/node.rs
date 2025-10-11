//! 节点相关

use pyo3::{Bound, types::PyType};

/// 节点注册
/// (节点class名称, 节点对象, 节点显示名称)
pub struct NodeRegister<'py>(pub &'static str, pub Bound<'py, PyType>, pub &'static str);
