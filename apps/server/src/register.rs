//! 注册
use pyo3::{prelude::*, types::PyDict};

use crate::{
    conditioning,
    core::{self, node::NodeRegister},
    image, joycaption, list, llama_cpp, logic, mask, math, model, text, utils, wrapper,
};

/// ComfyUI entrypoint function.
#[pyfunction]
pub fn entrypoint_submodules<'py>(py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
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

/// 节点注册
fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let mut nodes: Vec<NodeRegister> = Vec::new();
    nodes.extend(utils::node_register(py)?);
    nodes.extend(text::node_register(py)?);
    nodes.extend(list::node_register(py)?);
    nodes.extend(logic::node_register(py)?);
    nodes.extend(image::node_register(py)?);
    nodes.extend(mask::node_register(py)?);
    nodes.extend(conditioning::node_register(py)?);
    nodes.extend(joycaption::node_register(py)?);
    nodes.extend(model::node_register(py)?);
    nodes.extend(math::node_register(py)?);
    nodes.extend(llama_cpp::node_register(py)?);
    Ok(nodes)
}

/// Register sub-modules into the given Python module.
///
/// # Errors
///
/// Returns `Err` if any submodule fails to initialize.
pub fn register_submodules<'py>(py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_submodule(&core::submodule(py)?)?;
    m.add_submodule(&wrapper::submodule(py)?)?;
    m.add_submodule(&text::submodule(py)?)?;
    m.add_submodule(&list::submodule(py)?)?;
    m.add_submodule(&logic::submodule(py)?)?;
    m.add_submodule(&math::submodule(py)?)?;
    m.add_submodule(&utils::submodule(py)?)?;
    m.add_submodule(&image::submodule(py)?)?;
    m.add_submodule(&mask::submodule(py)?)?;
    m.add_submodule(&model::submodule(py)?)?;
    m.add_submodule(&conditioning::submodule(py)?)?;
    m.add_submodule(&joycaption::submodule(py)?)?;
    m.add_submodule(&llama_cpp::submodule(py)?)?;

    // 注册 ComfyUI V1 entrypoint
    entrypoint_submodules(py, m)?;
    Ok(())
}
