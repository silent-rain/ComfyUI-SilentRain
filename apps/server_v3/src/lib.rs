/// ComfyUI v3 Extension — SilentRain
///
/// This crate provides ComfyUI nodes implemented in Rust using the `comfyui_v3` SDK.
use pyo3::{pyfunction, pymodule};

pub mod constant;
pub mod core;
pub mod register;

pub mod example;
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
mod extension_module {
    use pyo3::{
        Bound, PyResult,
        types::{PyAnyMethods, PyModule, PyModuleMethods},
    };

    use crate::core::init_log;

    #[pymodule_export]
    use super::double;

    #[pymodule_export]
    use super::register::comfy_entrypoint;

    #[pymodule_export]
    use crate::register::v3;

    // 模块初始化时运行代码
    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Initialize tracing
        init_log();

        // Arbitrary code to run at the module initialization
        m.add("double2", m.getattr("double")?)?;
        m.add_class::<crate::text::TextEcho>()?;

        // 子模块注册
        {
            let sub = PyModule::new(m.py(), "v3")?;
            sub.add_class::<crate::text::TextEcho>()?;
            m.add_submodule(&sub)?;
        }

        Ok(())
    }
}

#[pyfunction]
fn double(x: usize) -> usize {
    x * 2
}
