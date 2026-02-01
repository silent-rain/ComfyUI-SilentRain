//! llama.cpp Chat v2
//!
//! v2 版本改进：
//! - 支持从 LlamaCppModelv2 接收模型缓存 key
//! - 上下文历史自动保存和恢复
//! - 使用 llama_cpp_core 的 Pipeline 进行推理

use log::{error, info};
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use std::{path::PathBuf, sync::Arc};

use llama_cpp_core::{
    CacheManager, Pipeline, InferenceRequest, PipelineConfig,
    config::SamplerConfig,
    history::HistoryMessage,
};

use crate::{
    core::category::CATEGORY_LLAMA_CPP,
    error::Error,
    wrapper::{
        comfy::folder_paths::FolderPaths,
        comfyui::{
            PromptServer,
            types::{NODE_BOOLEAN, NODE_INT, NODE_INT_MAX, NODE_SEED_MAX, NODE_STRING},
        },
    },
};

/// LlamaCpp Chat v2
///
//! ### 版本差异
//! - **v1**: 独立加载模型，无上下文保持
//! - **v2**: 
//!   - 复用 LlamaCppModelv2 加载的模型
//!   - 自动保存/恢复对话上下文
//!   - 使用 llama_cpp_core 的高性能 Pipeline
#[pyclass(subclass)]
pub struct LlamaCppChatv2 {
    cache: Arc<CacheManager>,
    pipeline: Arc<Pipeline>,
}

impl PromptServer for LlamaCppChatv2 {}

#[pymethods]
impl LlamaCppChatv2 {
    #[new]
    fn new() -> PyResult<Self> {
        let cache = CacheManager::global();
        let pipeline = Pipeline::with_cache(cache.clone())
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("Failed to create pipeline: {}", e)))?;
        
        Ok(Self {
            cache,
            pipeline: Arc::new(pipeline),
        })
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str) {
        (NODE_STRING, NODE_STRING)  // (response, full_context)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("response", "context_key")
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LLAMA_CPP;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "llama.cpp chat v2 - With context management and model reuse"
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

                // 模型句柄（来自 LlamaCppModelv2）
                required.set_item(
                    "model_handle",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("tooltip", "Model cache key from LlamaCppModelv2")?;
                        params
                    }),
                )?;

                required.set_item(
                    "system_prompt",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", "You are a helpful assistant.")?;
                        params.set_item("multiline", true)?;
                        params.set_item("tooltip", "System prompt that guides the model's behavior")?;
                        params
                    }),
                )?;

                required.set_item(
                    "user_prompt",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", "")?;
                        params.set_item("multiline", true)?;
                        params.set_item("tooltip", "User input to the model")?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_ctx",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 4096)?;
                        params.set_item("min", 256)?;
                        params.set_item("max", NODE_INT_MAX)?;
                        params.set_item("step", 1)?;
                        params.set_item("tooltip", "Context window size")?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_predict",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 2048)?;
                        params.set_item("min", -1)?;
                        params.set_item("max", NODE_INT_MAX)?;
                        params.set_item("step", 1)?;
                        params.set_item("tooltip", "Max tokens to predict (-1 = unlimited)")?;
                        params
                    }),
                )?;

                required.set_item(
                    "seed",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", -1)?;
                        params.set_item("min", -1)?;
                        params.set_item("max", NODE_SEED_MAX)?;
                        params.set_item("step", 1)?;
                        params.set_item("tooltip", "Random seed (-1 = random)")?;
                        params
                    }),
                )?;

                required.set_item(
                    "temperature",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", "0.6")?;
                        params.set_item("tooltip", "Sampling temperature")?;
                        params
                    }),
                )?;

                required.set_item(
                    "keep_context",
                    (NODE_BOOLEAN, {
                        let params = PyDict::new(py);
                        params.set_item("default", true)?;
                        params.set_item("tooltip", "Keep conversation context between calls")?;
                        params
                    }),
                )?;

                required
            })?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);
                
                // 上下文 key（用于恢复历史对话）
                optional.set_item(
                    "context_key",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", "")?;
                        params.set_item("tooltip", "Previous context key to restore conversation history")?;
                        params
                    }),
                )?;

                optional
            })?;

            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute", signature = (model_handle, system_prompt, user_prompt, n_ctx, n_predict, seed, temperature, keep_context, context_key=None))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        model_handle: String,
        system_prompt: String,
        user_prompt: String,
        n_ctx: u32,
        n_predict: i32,
        seed: i32,
        temperature: String,
        keep_context: bool,
        context_key: Option<String>,
    ) -> PyResult<(String, String)> {
        let result = self.generate(
            model_handle,
            system_prompt,
            user_prompt,
            n_ctx,
            n_predict,
            seed,
            temperature,
            keep_context,
            context_key,
        );

        match result {
            Ok((response, new_context_key)) => Ok((response, new_context_key)),
            Err(e) => {
                error!("LlamaCppChatv2 error: {e}");
                if let Err(e) = self.send_error(py, "LlamaCppChatv2".to_string(), e.to_string()) {
                    error!("send error failed: {e}");
                }
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LlamaCppChatv2 {
    /// 生成聊天响应
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        model_handle: String,
        system_prompt: String,
        user_prompt: String,
        n_ctx: u32,
        n_predict: i32,
        seed: i32,
        temperature: String,
        keep_context: bool,
        context_key: Option<String>,
    ) -> Result<(String, String), Error> {
        // 从模型句柄解析信息
        let (model_path, model_config) = self.resolve_model_handle(&model_handle)?;

        // 恢复或创建历史上下文
        let mut history = if let Some(ref key) = context_key {
            self.load_history(key)?
        } else {
            HistoryMessage::new()
        };

        // 添加系统提示（仅在首次对话时）
        if history.messages().is_empty() && !system_prompt.is_empty() {
            history.add_system(system_prompt.clone())?;
        }

        // 添加用户消息
        history.add_user(user_prompt.clone())?;

        // 构建采样配置
        let sampler_config = SamplerConfig::default()
            .temperature(temperature.parse::<f32>().unwrap_or(0.6))
            .seed(seed);

        // 构建推理请求
        let request = InferenceRequest::new(&model_path)
            .with_system_prompt(&system_prompt)
            .with_user_prompt(&user_prompt)
            .with_sampler(sampler_config)
            .with_model_config(model_config)
            .with_cache_model(true)
            .with_keep_context(keep_context);

        info!("Chat inference with model: {}", model_handle);

        // 执行推理（需要在 tokio runtime 中）
        let response = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.pipeline.infer(request).await
            })
        }).map_err(|e| Error::InvalidInput(format!("Generation failed: {}", e)))?;

        // 添加助手回复到历史
        history.add_assistant(response.text.clone())?;

        // 生成新的上下文 key
        let new_context_key = if keep_context {
            let key = format!("chat_ctx:{}", uuid::Uuid::new_v4());
            self.save_history(&key, &history)?;
            key
        } else {
            String::new()
        };

        info!("Generated {} tokens", response.tokens_generated);

        Ok((response.text, new_context_key))
    }

    /// 解析模型句柄，返回模型路径和配置
    fn resolve_model_handle(&self, handle: &str) -> Result<(PathBuf, llama_cpp_core::config::ModelConfig), Error> {
        // 从缓存获取模型信息
        if let Some((model_path, config)) = self.cache.get_data::<(String, llama_cpp_core::config::ModelConfig)>(handle)? {
            let base_models_dir = FolderPaths::default().model_path();
            let full_path = base_models_dir.join("LLM").join(&model_path);
            return Ok((full_path, *config));
        }

        // 如果缓存中没有，尝试作为普通路径处理
        let base_models_dir = FolderPaths::default().model_path();
        let full_path = base_models_dir.join("LLM").join(handle);
        
        if full_path.exists() {
            Ok((full_path, llama_cpp_core::config::ModelConfig::default()))
        } else {
            Err(Error::FileNotFound(handle.to_string()))
        }
    }

    /// 加载历史上下文
    fn load_history(&self, key: &str) -> Result<HistoryMessage, Error> {
        if let Some(messages) = self.cache.get_data::<Vec<llama_cpp_2::model::LlamaChatMessage>>(key)? {
            Ok(HistoryMessage::from_vec(messages.to_vec()))
        } else {
            Ok(HistoryMessage::new())
        }
    }

    /// 保存历史上下文
    fn save_history(&self, key: &str, history: &HistoryMessage) -> Result<(), Error> {
        let params = vec![key.to_string()];
        self.cache.force_update(
            key,
            &params,
            Arc::new(history.messages()),
        ).map_err(|e| Error::LockError(e.to_string()))?;
        Ok(())
    }
}
