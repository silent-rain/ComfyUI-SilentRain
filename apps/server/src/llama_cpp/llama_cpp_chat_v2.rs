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
    types::{PyAnyMethods, PyType},
};
use std::{path::PathBuf, sync::Arc};

use llama_cpp_core::{
    CacheManager, Pipeline, PipelineConfig,
    config::{ModelConfig, SamplerConfig},
    history::HistoryMessage,
    types::PromptMessageRole,
};

use crate::{
    core::{
        category::CATEGORY_LLAMA_CPP,
        node_base::{InputSpec, InputType},
    },
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
#[pyclass(subclass)]
pub struct LlamaCppChatv2 {
    cache: Arc<CacheManager>,
}

impl PromptServer for LlamaCppChatv2 {}

#[pymethods]
impl LlamaCppChatv2 {
    #[new]
    fn new() -> PyResult<Self> {
        let cache = CacheManager::global();

        Ok(Self { cache })
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str) {
        (NODE_STRING, NODE_STRING) // (response, full_context)
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
            let spec = InputSpec::new()
                .with_required(
                    "model_handle",
                    InputType::string().tooltip("Model cache key from LlamaCppModelv2"),
                )
                .with_required(
                    "system_prompt",
                    InputType::string()
                        .default("You are a helpful assistant.")
                        .multiline(true)
                        .tooltip("System prompt that guides the model's behavior"),
                )
                .with_required(
                    "user_prompt",
                    InputType::string()
                        .default("")
                        .multiline(true)
                        .tooltip("User input to the model"),
                )
                .with_required(
                    "n_ctx",
                    InputType::int()
                        .default(4096)
                        .min(256)
                        .max(NODE_INT_MAX as i64)
                        .step_int(1)
                        .tooltip("Context window size"),
                )
                .with_required(
                    "n_predict",
                    InputType::int()
                        .default(2048)
                        .min(-1)
                        .max(NODE_INT_MAX as i64)
                        .step_int(1)
                        .tooltip("Max tokens to predict (-1 = unlimited)"),
                )
                .with_required(
                    "seed",
                    InputType::int()
                        .default(-1)
                        .min(-1)
                        .max(NODE_SEED_MAX as i64)
                        .step_int(1)
                        .tooltip("Random seed (-1 = random)"),
                )
                .with_required(
                    "temperature",
                    InputType::string()
                        .default("0.6")
                        .tooltip("Sampling temperature"),
                )
                .with_required(
                    "keep_context",
                    InputType::bool()
                        .default(true)
                        .tooltip("Keep conversation context between calls"),
                )
                .with_optional(
                    "context_key",
                    InputType::string()
                        .default("")
                        .tooltip("Previous context key to restore conversation history"),
                );

            spec.build(py)
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

        // 构建 Pipeline 配置
        let pipeline_config = PipelineConfig::new(model_path.to_string_lossy().to_string(), None)
            .with_system_prompt(system_prompt)
            .with_user_prompt(user_prompt)
            .with_n_ctx(n_ctx)
            .with_n_predict(n_predict)
            .with_disable_gpu(model_config.disable_gpu)
            .with_n_gpu_layers(model_config.n_gpu_layers)
            .with_main_gpu(model_config.main_gpu)
            .with_keep_context(keep_context)
            .with_cache_model(true);

        // 创建 Pipeline
        let mut pipeline = Pipeline::try_new(pipeline_config).with_history_message(history);

        info!("Chat inference with model: {}", model_handle);

        // 执行推理（需要在 tokio runtime 中）
        let response = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async { pipeline.infer().await })
        })
        .map_err(|e| Error::InvalidInput(format!("Generation failed: {}", e)))?;

        // 生成新的上下文 key
        let new_context_key = if keep_context {
            let key = format!("chat_ctx:{}", uuid::Uuid::new_v4());
            self.save_history(&key, &pipeline.history_message())?;
            key
        } else {
            String::new()
        };

        info!("Generated {} tokens", response.tokens_generated);

        Ok((response.text, new_context_key))
    }

    /// 解析模型句柄，返回模型路径和配置
    fn resolve_model_handle(&self, handle: &str) -> Result<(PathBuf, ModelConfig), Error> {
        // 从缓存获取模型信息
        if let Some((model_path, config)) = self.cache.get_data::<(String, ModelConfig)>(handle)? {
            let base_models_dir = FolderPaths::default().model_path();
            let full_path = base_models_dir.join("LLM").join(&model_path);
            return Ok((full_path, config));
        }

        // 如果缓存中没有，尝试作为普通路径处理
        let base_models_dir = FolderPaths::default().model_path();
        let full_path = base_models_dir.join("LLM").join(handle);

        if full_path.exists() {
            Ok((full_path, ModelConfig::default()))
        } else {
            Err(Error::FileNotFound(handle.to_string()))
        }
    }

    /// 加载历史上下文
    fn load_history(&self, key: &str) -> Result<HistoryMessage, Error> {
        HistoryMessage::from_cache(key.to_string())
    }

    /// 保存历史上下文
    fn save_history(&self, key: &str, history: &HistoryMessage) -> Result<(), Error> {
        let mut history_copy = history.clone();
        history_copy.force_update_cache(key.to_string())?;
        Ok(())
    }
}
