//! llama.cpp Model Loader v2
//!
//! v2 版本改进：
//! - 独立的模型加载节点，支持模型复用
//! - 通过缓存 key 与推理节点联动
//! - 减少重复加载，提升性能

use log::{error, info};
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use std::sync::Arc;

use llama_cpp_core::{
    CacheManager, ModelConfig,
    model::ModelLoadRequest,
};

use crate::{
    core::category::CATEGORY_LLAMA_CPP,
    error::Error,
    wrapper::{
        comfy::folder_paths::FolderPaths,
        comfyui::{
            PromptServer,
            types::{NODE_BOOLEAN, NODE_INT, NODE_STRING},
        },
    },
};

/// 模型信息结构
#[derive(Clone)]
pub struct ModelHandleV2 {
    pub cache_key: String,
    pub model_path: String,
    pub config: ModelConfig,
}

/// LlamaCpp Model Loader v2
/// 
//! ### 版本差异
//! - **v1**: 每个推理节点独立加载模型，重复开销大
//! - **v2**: 独立模型节点，通过缓存 key 传递，支持模型复用
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
        (NODE_STRING,)  // 返回模型缓存 key
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("model_handle",)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LLAMA_CPP;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "llama.cpp model loader v2 - Load and cache model for reuse"
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
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);

                // 模型路径
                let model_list = Self::get_model_list();
                required.set_item(
                    "model_path",
                    (model_list, {
                        let params = PyDict::new(py);
                        params.set_item("tooltip", "Path to the GGUF model file")?;
                        params
                    }),
                )?;

                // GPU 配置
                required.set_item(
                    "main_gpu",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 0)?;
                        params.set_item("min", 0)?;
                        params.set_item("step", 1)?;
                        params.set_item("tooltip", "Main GPU index")?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_gpu_layers",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 0)?;
                        params.set_item("min", 0)?;
                        params.set_item("step", 1)?;
                        params.set_item("max", 10000)?;
                        params.set_item("tooltip", "Number of layers to offload to GPU (0 = CPU only)")?;
                        params
                    }),
                )?;

                required.set_item(
                    "disable_gpu",
                    (NODE_BOOLEAN, {
                        let params = PyDict::new(py);
                        params.set_item("default", false)?;
                        params.set_item("tooltip", "Disable GPU acceleration")?;
                        params
                    }),
                )?;

                required.set_item(
                    "cache_model",
                    (NODE_BOOLEAN, {
                        let params = PyDict::new(py);
                        params.set_item("default", true)?;
                        params.set_item("tooltip", "Cache model in memory for reuse")?;
                        params
                    }),
                )?;

                required
            })?;

            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute", signature = (model_path, main_gpu, n_gpu_layers, disable_gpu, cache_model))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        model_path: String,
        main_gpu: i32,
        n_gpu_layers: u32,
        disable_gpu: bool,
        cache_model: bool,
    ) -> PyResult<(String,)> {
        let result = self.load_model(
            model_path,
            main_gpu,
            n_gpu_layers,
            disable_gpu,
            cache_model,
        );

        match result {
            Ok(handle) => Ok((handle.cache_key,)),
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
        main_gpu: i32,
        n_gpu_layers: u32,
        disable_gpu: bool,
        cache_model: bool,
    ) -> Result<ModelHandleV2, Error> {
        // 解析完整路径
        let base_models_dir = FolderPaths::default().model_path();
        let full_path = base_models_dir.join("LLM").join(&model_path);

        if !full_path.exists() {
            return Err(Error::InvalidPath(format!(
                "Model not found: {}",
                full_path.display()
            )));
        }

        // 生成缓存 key
        let cache_key = format!(
            "model_v2:{}:gpu{}:layers{}",
            model_path,
            main_gpu,
            if disable_gpu { 0 } else { n_gpu_layers }
        );

        let config = ModelConfig {
            main_gpu: if disable_gpu { -1 } else { main_gpu },
            n_gpu_layers: if disable_gpu { 0 } else { n_gpu_layers },
        };

        info!("Loading model: {} with config: {:?}", model_path, config);

        // 检查缓存
        if cache_model {
            if let Ok(Some(_)) = self.cache.get_data::<()>(&cache_key) {
                info!("Model cache hit: {}", cache_key);
                return Ok(ModelHandleV2 {
                    cache_key,
                    model_path,
                    config,
                });
            }
        }

        // 创建加载请求（实际加载由推理节点触发，这里只存储配置）
        let _request = ModelLoadRequest::new(&full_path)
            .with_main_gpu(config.main_gpu)
            .with_gpu_layers(config.n_gpu_layers);

        // 将配置存入缓存
        if cache_model {
            let params = vec![
                model_path.clone(),
                config.main_gpu.to_string(),
                config.n_gpu_layers.to_string(),
            ];
            self.cache.force_update(
                &cache_key,
                &params,
                Arc::new((model_path.clone(), config)),
            ).map_err(|e| Error::LockError(e.to_string()))?;
        }

        info!("Model loaded successfully: {}", cache_key);

        Ok(ModelHandleV2 {
            cache_key,
            model_path,
            config,
        })
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
