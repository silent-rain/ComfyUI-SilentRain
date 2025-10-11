pub mod asset;
pub mod core;
pub mod error;
pub mod wrapper;

pub mod conditioning;
pub mod image;
pub mod joycaption;
pub mod list;
pub mod llama_cpp;
pub mod logic;
pub mod mask;
pub mod math;
pub mod model;
pub mod text;
pub mod utils;

use pyo3::{
    Bound, PyResult, Python, pyfunction, pymodule,
    types::{PyDict, PyDictMethods, PyModule, PyModuleMethods},
    wrap_pyfunction,
};

use crate::{core::node::NodeRegister, wrapper::comfy::init_folder_paths::apply_custom_paths};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "comfyui_silentrain")] // 需要与包名保持一致
fn py_init(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 初始化日志
    // 每个扩展模块都有自己的全局变量，因此所使用的记录器也与其他 Rust 原生扩展无关。
    // 因此，每个扩展模块可以根据自己的需要自行设置记录器。
    let _ = tracing_subscriber::fmt()
        .with_ansi(true)
        .with_max_level(tracing::Level::DEBUG)
        .with_level(true)
        .with_file(true)
        .with_line_number(true)
        .with_target(false)
        .try_init();

    // 添加函数demo
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    // 添加子模块
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

    const WEB_DIRECTORY: &str = "./web";

    m.add("NODE_CLASS_MAPPINGS", node_mapping)?;
    m.add("NODE_DISPLAY_NAME_MAPPINGS", name_mapping)?;
    m.add("WEB_DIRECTORY", WEB_DIRECTORY)?;

    // 添加自定义路径
    apply_custom_paths();
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
