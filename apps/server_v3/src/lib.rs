/// ComfyUI v3 Extension — SilentRain
///
/// This crate provides ComfyUI nodes implemented in Rust using the `comfyui_v3` SDK.
use pyo3::prelude::*;

pub mod register;

pub mod core;
pub mod image;
pub mod text;
pub mod utils;

/// Python module for SilentRain v3.
///
/// When compiled as a `cdylib`, this module can be imported from Python:
///
/// ```python
/// import comfyui_silentrain_v3
/// ext = comfyui_silentrain_v3.comfy_entrypoint()
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

    // ComfyUI V3 核心入口
    m.add_function(pyo3::wrap_pyfunction!(register::comfy_entrypoint, m)?)?;

    // 添加子模块
    register::register_submodules(py, m)?;

    Ok(())
}
