//! llama.cpp Model Loader v2

use llama_cpp_core::{CacheManager, model::ModelConfig};
use log::error;
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyDict, PyType},
};
use std::{collections::HashMap, sync::Arc};

use crate::{
    core::{
        category::CATEGORY_LLAMA_CPP,
        node_base::{InputSpec, InputType},
    },
    error::Error,
    wrapper::{
        comfy::folder_paths::FolderPaths,
        comfyui::{PromptServer, types::NODE_LLAMA_CPP_MODEL_V2},
    },
};

/// LlamaCpp Model Loader v2
#[pyclass(subclass)]
pub struct LlamaCppModelv2 {
    cache: Arc<CacheManager>,
}

impl PromptServer for LlamaCppModelv2 {}

#[pymethods]
impl LlamaCppModelv2 {
    #[new]
    fn new() -> Self {
        Self {
            cache: CacheManager::global(),
        }
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_LLAMA_CPP_MODEL_V2,) // 返回模型缓存 key
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("model",)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LLAMA_CPP;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "llama.cpp model loader v2"
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classattr]
    #[pyo3(name = "EXPERIMENTAL")]
    fn experimental() -> bool {
        true
    }

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        let model_list = Self::get_model_list();
        let config = ModelConfig::default();

        InputSpec::new()
            // 模型路径
            .with_required(
                "model_path",
                InputType::list(model_list.clone()).tooltip("Path to the GGUF model file"),
            )
            .with_required(
                "mmproj_path",
                InputType::list(model_list).tooltip("Path to the GGUF model file"),
            )
            // // GPU 配置
            // .with_required(
            //     "main_gpu",
            //     InputType::int()
            //         .default(config.main_gpu)
            //         .min(0)
            //         .step_int(1)
            //         .tooltip("Main GPU index"),
            // )
            // .with_required(
            //     "n_gpu_layers",
            //     InputType::int()
            //         .default(config.n_gpu_layers as i64)
            //         .min(0)
            //         .max_int(10000)
            //         .step_int(1)
            //         .tooltip("Number of layers to offload to GPU (0 = CPU only)"),
            // )
            // // 多 GPU 设备
            // .with_optional(
            //     "devices",
            //     InputType::string().default(config.devices_str()).tooltip(
            //         "GPU devices to use (comma-separated, e.g., 0,1,2). Overrides main_gpu",
            //     ),
            // )
            // .with_required(
            //     "media_marker",
            //     InputType::string()
            //         .default(config.media_marker.unwrap_or_default())
            //         .tooltip("Media marker. If not provided, the default marker will be used."),
            // )
            // .with_required(
            //     "n_threads",
            //     InputType::string().default(config.n_threads).tooltip("Number of threads to use during generation. Set to a specific value to limit CPU usage."),
            // )
            // // MoE 配置
            // .with_optional(
            //     "cmoe",
            //     InputType::bool()
            //         .default(config.cmoe)
            //         .tooltip("Keep MoE layers on CPU"),
            // )
            // // 内存锁定
            // .with_optional(
            //     "use_mlock",
            //     InputType::bool()
            //         .default(config.use_mlock)
            //         .tooltip("Force system to keep model in RAM (use mlock)"),
            // )
            .with_required(
                "cache_model",
                InputType::bool()
                    .default(config.cache_model)
                    .tooltip("Cache model in memory for reuse"),
            )
            // 详细信息
            // .with_optional(
            //     "verbose",
            //     InputType::bool()
            //         .default(config.verbose)
            //         .tooltip("Print detailed information about model loading"),
            // )
            .build()
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        model_path: String,
        mmproj_path: String,
        cache_model: bool,
    ) -> PyResult<(HashMap<String, String>,)> {
        let result = self.load_model(model_path, mmproj_path, cache_model);

        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("LlamaCppModelv2 error: {e}");
                if let Err(e) = self.send_error(py, "LlamaCppModelv2".to_string(), e.to_string()) {
                    error!("send error failed: {e}");
                }
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LlamaCppModelv2 {
    /// 加载模型并返回缓存 key
    fn load_model(
        &self,
        model_path: String,
        mmproj_path: String,
        cache_model: bool,
    ) -> Result<(HashMap<String, String>,), Error> {
        // 解析完整路径
        let base_models_dir = FolderPaths::default().model_path();
        let model_path = base_models_dir.join("LLM").join(&model_path);
        let mmproj_path = base_models_dir.join("LLM").join(&mmproj_path);

        if !model_path.exists() {
            return Err(Error::InvalidPath(format!(
                "Model not found: {}",
                model_path.display()
            )));
        }

        let cache_model_key = "comfyui:model";
        let cache_mmproj_key = "comfyui:mmproj";

        Ok((HashMap::from([
            (
                "model_path".to_string(),
                model_path.to_string_lossy().to_string(),
            ),
            (
                "mmproj_path".to_string(),
                mmproj_path.to_string_lossy().to_string(),
            ),
            ("cache_model".to_string(), cache_model.to_string()),
            ("cache_model_key".to_string(), cache_model_key.to_string()),
            ("cache_mmproj_key".to_string(), cache_mmproj_key.to_string()),
        ]),))
    }

    /// 获取模型列表
    fn get_model_list() -> Vec<String> {
        let llm_list = FolderPaths::default().get_filename_list("LLM");
        let text_encoders_list = FolderPaths::default().get_filename_list("text_encoders");

        [llm_list, text_encoders_list]
            .concat()
            .into_iter()
            .filter(|v| v.ends_with(".gguf"))
            .collect()
    }
}
