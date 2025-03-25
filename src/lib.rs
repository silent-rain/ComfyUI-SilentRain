use pyo3::{
    pyfunction, pymodule,
    types::{PyDict, PyDictMethods, PyModule, PyModuleMethods},
    wrap_pyfunction, Bound, PyResult, Python,
};
use text::FileScanner;

mod error;
pub mod text;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "ComfyUI_SilentRain")] // 需要与包名保持一致
fn py_init(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

    m.add_submodule(&text::text_module(py)?)?;

    // 注册 ComfyUI NODE_CLASS_MAPPINGS/NODE_DISPLAY_NAME_MAPPINGS
    let node_mapping = PyDict::new(py);
    node_mapping.set_item("FileScanner", py.get_type::<FileScanner>())?;

    let name_mapping = PyDict::new(py);
    name_mapping.set_item("FileScanner", "文本文件扫描器")?;

    m.add("NODE_CLASS_MAPPINGS", node_mapping)?;
    m.add("NODE_DISPLAY_NAME_MAPPINGS", name_mapping)?;
    Ok(())
}
