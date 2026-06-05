use pyo3::{PyResult, pyfunction, pymodule};

pub mod asset;
pub mod core;
pub mod error;
pub mod register;
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

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Python module for SilentRain.
///
/// When compiled as a `cdylib`, this module can be imported from Python:
#[pymodule]
#[pyo3(name = "comfyui_silentrain")] // 需要与包名保持一致
mod extension_module {
    use pyo3::{
        Bound, PyResult,
        types::{PyModule, PyModuleMethods},
        wrap_pyfunction,
    };

    use comfyui_silentrain_v3::{core::init_log, register};

    use crate::{
        register::register_submodules, sum_as_string,
        wrapper::comfy::init_folder_paths::apply_custom_paths,
    };

    // ==================== ComfyUI V3 核心入口 ====================
    #[pymodule_export]
    use register::build_extension;
    #[pymodule_export]
    use register::comfy_entrypoint;
    // ==================== ComfyUI V3 子模块 ====================
    #[pymodule_export]
    use register::v3;

    // 模块初始化时运行代码
    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Initialize tracing
        init_log();

        // Arbitrary code to run at the module initialization
        // 添加函数demo
        m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;

        // ==================== ComfyUI V1 ====================
        register_submodules(m.py(), m)?;

        // ==================== ComfyUI web ====================
        const WEB_DIRECTORY: &str = "./web";
        m.add("WEB_DIRECTORY", WEB_DIRECTORY)?;

        // 添加自定义路径
        apply_custom_paths();
        Ok(())
    }
}
