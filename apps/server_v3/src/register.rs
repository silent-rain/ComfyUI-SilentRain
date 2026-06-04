//! 注册
use pyo3::prelude::*;

use comfyui_v3::node::ExtensionBuilder;

use crate::image;
use crate::text;

/// ComfyUI entrypoint function.
///
/// ```python
/// async def comfy_entrypoint():
///     return SilentRainV3Extension()
/// ```
#[pyfunction]
#[pyo3()]
pub async fn comfy_entrypoint() -> PyResult<Py<PyAny>> {
    println!("==============================1");

    Python::attach(|py| -> PyResult<Py<PyAny>> { build_extension(py) })
}

/// Build the SilentRain v3 extension with all registered nodes.
///
/// Each sub-module collects its own nodes via `node_register`;
/// this function merges all lists and passes them to the builder.
#[pyfunction]
pub fn build_extension<'py>(py: Python<'py>) -> PyResult<Py<PyAny>> {
    println!("==============================2");
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
    Ok(nodes)
}

/// Register sub-modules into the given Python module.
///
/// # Errors
///
/// Returns `Err` if any submodule fails to initialize.
pub fn register_submodules<'py>(py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_submodule(&image::submodule(py)?)?;
    m.add_submodule(&text::submodule(py)?)?;
    Ok(())
}
