//! 类型定义
//! 相关节点定义: ComfyUI/comfy/comfy_types/node_typing.py

use pyo3::{ffi::c_str, pyfunction, Bound, PyResult, Python};

pub const NODE_INT: &str = "INT";
pub const NODE_FLOAT: &str = "FLOAT";
pub const NODE_STRING: &str = "STRING";
pub const NODE_BOOLEAN: &str = "BOOLEAN";
pub const NODE_LIST: &str = "LIST";
pub const NODE_JSON: &str = "JSON";
pub const NODE_METADATA_RAW: &str = "METADATA_RAW";
pub const NODE_IMAGE: &str = "IMAGE";
pub const NODE_MASK: &str = "MASK";
pub const NODE_MODEL: &str = "MODEL";
pub const NODE_CLIP: &str = "CLIP";
pub const NODE_CONDITIONING: &str = "CONDITIONING";
pub const NODE_VAE: &str = "VAE";
pub const NODE_LATENT: &str = "LATENT";

pub const NODE_LLAMA_CPP_OPTIONS: &str = "LlamaCppExtraOptions";

pub const NODE_INT_MAX: u64 = 0xffffffffffffffffu64;
pub const NODE_SEED_MAX: u64 = 10000000;

/// 任意类型
#[pyfunction]
pub fn any_type(py: Python<'_>) -> PyResult<Bound<'_, pyo3::PyAny>> {
    let code = c_str!(
        "class AlwaysEqualProxy(str):
            def __eq__(self, _):
                return True
            def __ne__(self, _):
                return False

        # any_type = AlwaysEqualProxy('*')
        "
    );

    py.run(code, None, None)?;
    py.eval(c_str!("AlwaysEqualProxy('*')"), None, None)
}
