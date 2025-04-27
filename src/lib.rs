pub mod core;
pub mod error;
pub mod register;

pub mod list;
pub mod logic;
pub mod text;
pub mod utils;

use pyo3::{
    pyfunction, pymodule,
    types::{PyDict, PyDictMethods, PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult, Python,
};

use crate::register::node_register;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "comfyui_silentrain")] // 需要与包名保持一致
fn py_init(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 添加函数demo
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    // 添加子模块
    m.add_submodule(&text::submodule(py)?)?;
    m.add_submodule(&logic::submodule(py)?)?;
    m.add_submodule(&utils::submodule(py)?)?;

    // 注册 ComfyUI NODE_CLASS_MAPPINGS/NODE_DISPLAY_NAME_MAPPINGS
    let node_mapping = PyDict::new(py);
    let name_mapping = PyDict::new(py);
    // 注册单个节点
    // node_mapping.set_item("FileScanner", py.get_type::<FileScanner>())?;
    // name_mapping.set_item("FileScanner", "Sr File Scanner")?;

    // 批量注册节点, 简化注册流程
    let nodes = node_register(py)?;
    for node in nodes {
        node_mapping.set_item(node.0, node.1)?;
        name_mapping.set_item(node.0, node.2)?;
    }

    m.add("NODE_CLASS_MAPPINGS", node_mapping)?;
    m.add("NODE_DISPLAY_NAME_MAPPINGS", name_mapping)?;
    Ok(())
}
