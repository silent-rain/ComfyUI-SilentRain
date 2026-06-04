/// ComfyUI v3 Extension — SilentRain
///
/// This crate provides ComfyUI nodes implemented in Rust using the `comfyui_v3` SDK.
use pyo3::prelude::*;

pub mod core;
pub mod image;
pub mod text;
pub mod utils;

use comfyui_v3::node::ExtensionBuilder;

/// Build the SilentRain v3 extension with all registered nodes.
///
/// Each sub-module collects its own nodes via `node_register`;
/// this function merges all lists and passes them to the builder.
#[pyfunction]
pub fn build_extension<'py>(py: Python<'py>) -> PyResult<Py<PyAny>> {
    let nodes = node_collect(py)?;

    let mut builder = ExtensionBuilder::new();
    for node in nodes {
        builder = builder.add_node(node);
    }
    builder.build(py)
}

/// Merge node collections from every sub-module.
fn node_collect(py: Python<'_>) -> PyResult<Vec<Py<pyo3::types::PyType>>> {
    let mut nodes: Vec<Py<pyo3::types::PyType>> = Vec::new();
    nodes.extend(image::node_register(py)?);
    nodes.extend(text::node_register(py)?);
    Ok(nodes)
}

/// ComfyUI entrypoint function.
///
/// ```python
/// async def comfy_entrypoint():
///     return SilentRainV3Extension()
/// ```
#[pyfunction]
pub fn comfy_entrypoint(py: Python<'_>) -> PyResult<Py<PyAny>> {
    build_extension(py)
}

/// Python module for SilentRain v3.
///
/// When compiled as a `cdylib`, this module can be imported from Python:
///
/// ```python
/// import comfyui_silentrain_v3
/// ext = comfyui_silentrain_v3.build_extension()
/// ```
#[pymodule]
#[pyo3(name = "comfyui_silentrain_v3")]
fn init_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt()
        .with_ansi(true)
        .with_max_level(tracing::Level::DEBUG)
        .with_level(true)
        .with_file(true)
        .with_line_number(true)
        .with_target(false)
        .try_init();

    m.add_function(pyo3::wrap_pyfunction!(build_extension, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(comfy_entrypoint, m)?)?;

    // 添加子模块
    m.add_submodule(&image::submodule(py)?)?;
    m.add_submodule(&text::submodule(py)?)?;

    Ok(())
}
