//! 注册
use pyo3::prelude::*;

use comfyui_v3::node::ExtensionBuilder;

use crate::example;
use crate::image;
use crate::text;

/// ComfyUI entrypoint function.
///
/// ```python
/// def comfy_entrypoint():
///     return SilentRainV3Extension()
/// ```
#[pyfunction]
pub fn comfy_entrypoint<'py>(py: Python<'py>) -> PyResult<Py<PyAny>> {
    build_extension(py)
}

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

/// * Merge node collections from every sub-module.
fn node_collect(py: Python<'_>) -> PyResult<Vec<Py<pyo3::types::PyType>>> {
    let mut nodes: Vec<Py<pyo3::types::PyType>> = Vec::new();
    nodes.extend(image::node_register(py)?);
    nodes.extend(text::node_register(py)?);
    nodes.extend(example::node_register(py)?);
    Ok(nodes)
}

/// 顶层模块导出
#[pymodule]
pub mod v3 {
    use crate::*;

    // 顶层子模块
    #[pymodule_export]
    use image::image;

    #[pymodule_export]
    use text::text;

    #[pymodule_export]
    use example::example;
}
