//! llama.cpp options
//!
//!
//! ## media_marker
//!     - chat_template.json
//!         - https://hf-mirror.com/mlabonne/gemma-3-4b-it-abliterated/blob/main/chat_template.json
//!         - image: <start_of_image>
//!     - gguf model
//!         - tokenizer.chat_template

use std::path::PathBuf;

use log::error;
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use pythonize::{depythonize, pythonize};
use rand::TryRngCore;
use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

use crate::{
    core::category::CATEGORY_LLAMA_CPP,
    error::Error,
    wrapper::{
        comfy::folder_paths::FolderPaths,
        comfyui::{
            PromptServer,
            types::{NODE_BOOLEAN, NODE_FLOAT, NODE_INT, NODE_LLAMA_CPP_OPTIONS, NODE_STRING},
        },
    },
};

const SUBFOLDER: &str = "LLM";

/// 对话消息角色枚举
#[derive(Debug, Clone, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum PromptMessageRole {
    /// 系统角色（用于初始化或系统级指令）
    #[strum(to_string = "System")]
    System,
    /// 用户角色（人类用户输入）
    #[strum(to_string = "User")]
    User,
    /// AI 助手角色（模型生成的回复）
    #[strum(to_string = "Assistant")]
    Assistant,
    /// 可选：自定义角色（如多AI代理场景）
    #[strum(transparent)]
    Custom(String),
}

impl PromptMessageRole {
    pub fn custom(role: &str) -> Self {
        Self::Custom(role.to_string())
    }
}

/// 处理模式枚举
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum PoolingTypeMode {
    /// 无模式
    #[strum(to_string = "None")]
    None,

    /// 均值模式
    #[strum(to_string = "Mean")]
    Mean,

    /// 分类模式
    #[strum(to_string = "Cls")]
    Cls,

    /// 最后模式
    #[strum(to_string = "Last")]
    Last,

    /// 排序模式
    #[strum(to_string = "Rank")]
    Rank,

    /// 未指定模式
    #[strum(to_string = "Unspecified")]
    Unspecified,
}

/// Options for the llama_cpp library
#[pyclass(subclass)]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LlamaCppOptions {
    /// Path to the model file (e.g., "ggml-model.bin")
    #[serde(default)]
    pub model_path: String,

    /// Path to the multimodal projection file (e.g., "mmproj-model.bin")
    /// Required for models with multimodal capabilities (e.g., vision or audio).
    #[serde(default)]
    pub mmproj_path: String,

    /// The system prompt (or instruction) that guides the model's behavior.
    /// This is typically a high-level directive (e.g., "You are a helpful assistant.").
    /// It is often static and set once per session.
    #[serde(default)]
    pub system_prompt: String,

    /// The user-provided input or query to the model.
    /// This is the dynamic part of the prompt that changes with each interaction.
    /// May include media markers - else they will be added automatically.
    #[serde(default)]
    pub user_prompt: String,

    /// Controls diversity via top-k sampling.
    /// Higher values mean more diverse outputs.
    #[serde(default)]
    pub top_k: i32,

    /// Controls diversity via nucleus sampling.
    /// Lower values mean more focused outputs.
    #[serde(default)]
    pub top_p: f32,

    /// Controls randomness.
    /// Higher values mean more random outputs.
    #[serde(default)]
    pub temperature: f32,

    /// Seed for random number generation.
    /// Set to a fixed value for reproducible outputs.
    #[serde(default)]
    pub seed: i32,

    /// Number of threads to use during generation.
    /// Set to a specific value to limit CPU usage.
    #[serde(default)]
    pub n_threads: i32,

    /// Number of threads to use during batch and prompt processing.
    /// Useful for optimizing multi-threaded workloads.
    #[serde(default)]
    pub n_threads_batch: i32,

    /// Batch size for prompt processing.
    /// Larger values may improve throughput but increase memory usage.
    #[serde(default)]
    pub n_batch: u32,

    /// Size of the prompt context window.
    /// Defines the maximum context length the model can handle.
    #[serde(default)]
    pub n_ctx: u32,

    /// Number of tokens to predict (-1 for unlimited)
    #[serde(default)]
    pub n_predict: i32,

    /// Index of the main GPU to use.
    /// Relevant for multi-GPU systems.
    #[serde(default)]
    pub main_gpu: i32,

    /// Number of GPU layers to offload.
    /// Higher values offload more work to the GPU.
    #[serde(default)]
    pub n_gpu_layers: u32,

    /// If set to `true`, disables GPU offloading for the multimodal projection (mmproj) .
    /// This forces mmproj computations to run on CPU, even if the main model runs on GPU.
    // #[serde(default)]
    // pub no_mmproj_offload: bool,

    // TODO 尚未实现
    /// Enables flash attention for faster inference.
    /// Requires compatible hardware and model support.
    #[serde(default)]
    pub flash_attention: bool,

    /// Whether to keep context between requests
    #[serde(default)]
    pub keep_context: bool,

    /// Whether to cache model between requests
    #[serde(default)]
    pub cache_model: bool,

    /// Size of the sliding window for repeat penalty
    /// Specifies how many most recent tokens to consider for repeat penalty
    #[serde(default)]
    pub penalty_last_n: i32,

    /// Repeat penalty coefficient
    /// Penalizes repeated tokens - higher values enforce more diversity
    #[serde(default)]
    pub penalty_repeat: f32,

    /// Frequency penalty coefficient
    /// Penalizes tokens based on their frequency in the text
    #[serde(default)]
    pub penalty_freq: f32,

    /// Presence penalty coefficient
    /// Penalizes tokens already present in the context
    #[serde(default)]
    pub penalty_present: f32,

    /// Pooling type for embeddings.
    /// Options: "None", "Mean", "Cls", "Last", "Rank", "Unspecified".
    #[serde(default)]
    pub pooling_type: String,

    /// Media marker. If not provided, the default marker will be used.
    #[serde(default)]
    pub media_marker: Option<String>,

    /// Chat template to use, default template if not provided
    // #[arg(long = "chat-template", value_name = "TEMPLATE")]
    #[serde(default)]
    pub chat_template: Option<String>,

    /// Path to image file(s)
    #[serde(default)]
    pub images: Vec<Vec<u8>>,

    /// Path to audio file(s)
    #[serde(default)]
    pub audio: Vec<Vec<u8>>,

    /// Enables verbose logging from llama.cpp.
    /// Useful for debugging and performance analysis.
    #[serde(default)]
    pub verbose: bool,

    // *************************
    /// Whether to normalise the produced embeddings
    #[serde(default)]
    pub normalise: bool,

    /// The documents to embed and compare against
    #[serde(default)]
    pub documents: Vec<String>,

    /// override some parameters of the model
    // #[arg(short = 'o', value_parser = parse_key_val)]
    // key_value_overrides: Vec<(String, ParamOverrideValue)>,

    /// set the length of the prompt + output in tokens
    // #[arg(long, default_value_t = 32)]
    #[serde(default)]
    pub n_len: i32,

    #[serde(default)]
    pub model_cache_key: String,
    #[serde(default)]
    pub model_mtmd_context_cache_key: String,
    #[serde(default)]
    pub context_history_cache_key: String,
}

impl PromptServer for LlamaCppOptions {}

#[pymethods]
impl LlamaCppOptions {
    #[new]
    fn new() -> Self {
        Default::default()
    }

    // #[classattr]
    // #[pyo3(name = "EXPERIMENTAL")]
    // fn experimental() -> bool {
    //     true
    // }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_LLAMA_CPP_OPTIONS,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("extra_options",)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() {}

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LLAMA_CPP;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "llama.cpp extra options"
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);

                let options = LlamaCppOptions::default();

                required.set_item(
                    "top_k",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.top_k)?;
                        params.set_item("step", 1)?;
                        params.set_item(
                            "tooltip",
                            "Controls diversity via top-k sampling.  Higher values mean more diverse outputs.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "top_p",
                    (NODE_FLOAT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.top_p)?;
                        params.set_item("step", 0.01)?;
                        params.set_item(
                            "tooltip",
                            "Controls diversity via nucleus sampling.  Lower values mean more focused outputs.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "temperature",
                    (NODE_FLOAT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.temperature)?;
                        params.set_item("step", 0.05)?;
                        params.set_item(
                            "tooltip",
                            "Controls randomness. Higher values mean more random outputs.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_threads",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.n_threads)?;
                        params.set_item("min", 0)?;
                        params.set_item("step", 1)?;
                        params.set_item(
                            "tooltip",
                            "Number of threads to use during generation. Set to a specific value to limit CPU usage.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_threads_batch",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.n_threads_batch)?;
                        params.set_item("min", 0)?;
                        params.set_item("step", 1)?;
                        params.set_item(
                            "tooltip",
                            "Number of threads to use during batch and prompt processing. Useful for optimizing multi-threaded workloads.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_batch",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.n_batch)?;
                        params.set_item("min", 0)?;
                        params.set_item("step", 1)?;
                        params.set_item(
                            "tooltip",
                            "Batch size for prompt processing. Larger values may improve throughput but increase memory usage.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "flash_attention",
                    (NODE_BOOLEAN, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.flash_attention)?;
                        params.set_item("tooltip", "Enables flash attention for faster inference. Requires compatible hardware and model support.")?;
                        params
                    }),
                )?;

                required.set_item(
                    "penalty_last_n",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.penalty_last_n)?;
                        params.set_item("step", 1)?;
                        params.set_item(
                            "tooltip",
                            "Size of the sliding window for repeat penalty Specifies how many most recent tokens to consider for repeat penalty.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "penalty_repeat",
                    (NODE_FLOAT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.penalty_repeat)?;
                        params.set_item("step", 0.01)?;
                        params.set_item(
                            "tooltip",
                            "Penalizes repeated tokens - higher values enforce more diversity.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "penalty_freq",
                    (NODE_FLOAT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.penalty_freq)?;
                        params.set_item("step", 0.01)?;
                        params.set_item(
                            "tooltip",
                            "Penalizes tokens based on their frequency in the text - higher values enforce more diversity.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "penalty_present",
                    (NODE_FLOAT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.penalty_present)?;
                        params.set_item("step", 0.01)?;
                        params.set_item(
                            "tooltip",
                            "Penalizes tokens already present in the context - higher values enforce more diversity.",
                        )?;
                        params
                    }),
                )?;


                required.set_item(
                    "pooling_type",
                    (
                        vec![
                            PoolingTypeMode::None.to_string(),
                            PoolingTypeMode::Mean.to_string(),
                            PoolingTypeMode::Cls.to_string(),
                            PoolingTypeMode::Last.to_string(),
                            PoolingTypeMode::Rank.to_string(),
                            PoolingTypeMode::Unspecified.to_string(),
                        ],
                        {
                            let params = PyDict::new(py);
                            params.set_item("default", options.pooling_type)?;
                            params.set_item("tooltip", r#"Pooling type for embeddings. Options: "None", "Mean", "Cls", "Last", "Rank", "Unspecified"."#)?;
                            params
                        },
                    ),
                )?;

                required.set_item(
                    "media_marker",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.media_marker)?;
                        params.set_item(
                            "tooltip",
                            "Media marker. If not provided, the default marker will be used.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "chat_template",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.chat_template)?;
                        params.set_item("tooltip", "Chat template to use, default template if not provided.")?;
                        params
                    }),
                )?;

                required.set_item(
                    "verbose",
                    (NODE_BOOLEAN, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.verbose)?;
                        params.set_item("tooltip", "Enables verbose logging from llama.cpp. Useful for debugging and performance analysis..")?;
                        params
                    }),
                )?;

                required
            })?;

            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute", signature = (**kwargs))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyDict>,)> {
        let results = self.options_parser(py, kwargs);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("LlamaCppOptions error, {e}");
                if let Err(e) = self.send_error(py, "LlamaCppOptions".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LlamaCppOptions {
    fn options_parser<'py>(
        &self,
        py: Python<'py>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<(Bound<'py, PyDict>,), Error> {
        let kwargs =
            kwargs.ok_or_else(|| Error::InvalidParameter("parameters is required".to_string()))?;
        let options: LlamaCppOptions = depythonize(&kwargs)?;

        let py_options = pythonize(py, &options)?.extract::<Bound<'py, PyDict>>()?;
        Ok((py_options,))
    }
}

impl LlamaCppOptions {
    /// get model path
    pub fn get_model_path(&self) -> Result<PathBuf, Error> {
        let base_models_dir = FolderPaths::default().model_path();

        let model_path = base_models_dir
            .join(SUBFOLDER)
            .join(self.model_path.clone());

        // Validate required parameters
        if !model_path.exists() {
            return Err(Error::InvalidPath(format!(
                "Model file not found: {}",
                model_path.to_string_lossy()
            )));
        }

        Ok(model_path)
    }

    /// get mmproj path
    pub fn get_mmproj_path(&self) -> Result<String, Error> {
        let base_models_dir = FolderPaths::default().model_path();

        let mmproj_path = base_models_dir
            .join(SUBFOLDER)
            .join(self.mmproj_path.clone());

        // Validate required parameters
        if !mmproj_path.exists() {
            return Err(Error::InvalidPath(format!(
                "Multimodal projection file not found: {}",
                mmproj_path.to_string_lossy()
            )));
        }

        Ok(mmproj_path.to_string_lossy().to_string())
    }

    /// max predict
    pub fn max_predict(&self) -> i32 {
        if self.n_predict < 0 {
            i32::MAX
        } else {
            self.n_predict
        }
    }

    /// get seed
    pub fn get_seed(&self) -> u32 {
        // 随机值
        if self.seed == -1 {
            // 随机值
            rand::rng().try_next_u32().unwrap_or(0)
        } else {
            self.seed as u32
        }
    }
}

impl Default for LlamaCppOptions {
    fn default() -> Self {
        Self {
            // 模型文件路径（需用户指定，默认留空）
            model_path: String::new(),
            // 多模态投影文件路径（需用户指定，默认留空）
            mmproj_path: String::new(),

            // 文本生成参数
            system_prompt: String::new(), // 描述模型行为的系统级指令（例如“你是一个有用的助手”）。
            user_prompt: String::new(),   // 用户提供的输入或查询

            // 采样参数
            top_k: 40,        // 默认 top-k 采样值
            top_p: 0.95,      // 默认 top-p 采样值
            temperature: 0.6, // 默认温度值
            seed: -1,         // 默认随机种子（-1 表示随机）

            // 线程和批处理参数
            n_threads: 0,       // 0 表示自动使用所有可用线程
            n_threads_batch: 0, // 0 表示自动使用所有可用线程
            n_batch: 512,       // 默认批处理大小
            n_ctx: 4096,        // 默认上下文窗口大小
            n_predict: 2048,    // 要预测的Token数量， -1 表示无限生成

            // GPU 相关参数
            main_gpu: 0,     // 默认主 GPU 索引
            n_gpu_layers: 0, // 默认不启用 GPU 卸载
            // no_mmproj_offload: false, // 默认启用 mmproj 的 GPU 卸载
            flash_attention: false, // 默认禁用 Flash Attention

            keep_context: false, // 保持上下文（默认禁用）
            cache_model: false,  // 缓存模型（默认禁用）

            // Penalizes tokens for being present in the context.
            penalty_last_n: 64,   // 重复惩罚的窗口大小
            penalty_repeat: 1.2,  // 重复惩罚系数
            penalty_freq: 1.1,    // 重复频率惩罚系数
            penalty_present: 0.0, // 存在惩罚系数

            // 池化类型（默认未指定）
            pooling_type: PoolingTypeMode::Unspecified.to_string(),

            // 多模态输入（默认留空）
            media_marker: Some("<__media__>".to_string()), // 默认媒体标记
            chat_template: None,                           // 默认聊天模板
            images: Vec::new(),
            audio: Vec::new(),

            // 日志和调试
            verbose: false, // 默认禁用详细日志

            normalise: false, // 默认禁用输入归一化

            // 检索增强生成（RAG）参数
            documents: Vec::new(), // 默认文档列表（空）
            n_len: 128,            // 默认检索结果长度

            model_cache_key: "default_model_cache_key".to_string(),
            model_mtmd_context_cache_key: "default_model_mtmd_context_cache_key".to_string(),
            context_history_cache_key: "default_context_history_cache_key".to_string(),
        }
    }
}
