//! llama.cpp options

use std::path::PathBuf;

use pyo3::pyclass;

use crate::{
    error::Error,
    wrapper::{comfy::folder_paths::FolderPaths, comfyui::PromptServer},
};

const SUBFOLDER: &str = "LLM";

/// Options for the llama_cpp library
#[pyclass(subclass)]
#[derive(Debug, Clone)]
pub struct LlamaCppOptions {
    /// Path to the model file (e.g., "ggml-model.bin")
    pub model_path: String,

    /// Path to the multimodal projection file (e.g., "mmproj-model.bin")
    /// Required for models with multimodal capabilities (e.g., vision or audio).
    pub mmproj_path: String,

    /// Controls diversity via top-k sampling (default: 40).
    /// Higher values mean more diverse outputs.
    pub top_k: i32,

    /// Controls diversity via nucleus sampling (default: 0.8).
    /// Lower values mean more focused outputs.
    pub top_p: f32,

    /// Controls randomness (default: 0.7).
    /// Higher values mean more random outputs.
    pub temperature: f32,

    /// Seed for random number generation (default: 0).
    /// Set to a fixed value for reproducible outputs.
    pub seed: u32,

    /// Number of threads to use during generation (default: all available threads).
    /// Set to a specific value to limit CPU usage.
    pub n_threads: i32,

    /// Number of threads to use during batch and prompt processing (default: all available threads).
    /// Useful for optimizing multi-threaded workloads.
    pub n_threads_batch: u32,

    /// Batch size for prompt processing (default: 512).
    /// Larger values may improve throughput but increase memory usage.
    pub n_batch: u32,

    /// Size of the prompt context window (default: 2048).
    /// Defines the maximum context length the model can handle.
    pub n_ctx: u32,

    /// Index of the main GPU to use (default: 0).
    /// Relevant for multi-GPU systems.
    pub main_gpu: i32,

    /// Number of GPU layers to offload (default: 0, CPU-only).
    /// Higher values offload more work to the GPU.
    pub n_gpu_layers: u32,

    /// If set to `true`, disables GPU offloading for the multimodal projection (mmproj) (default: false).
    /// This forces mmproj computations to run on CPU, even if the main model runs on GPU.
    /// Useful for debugging or compatibility with certain hardware configurations.
    pub no_mmproj_offload: bool,

    /// Enables flash attention for faster inference (default: false).
    /// Requires compatible hardware and model support.
    pub flash_attention: bool,

    /// Pooling type for embeddings (default: "Unspecified").
    /// Options: "None", "Mean", "Cls", "Last", "Rank".
    pub pooling_type: String,

    /// Enables verbose logging from llama.cpp (default: false).
    /// Useful for debugging and performance analysis.
    pub verbose: bool,

    // *************************
    /// Path to image file(s)
    pub images: Vec<String>,
    /// Path to audio file(s)
    pub audio: Vec<String>,
    /// Text prompt to use as input to the model. May include media markers - else they will be added automatically.
    pub prompt: String,
    /// Number of tokens to predict (-1 for unlimited)
    // #[arg(
    //     short = 'n',
    //     long = "n-predict",
    //     value_name = "N",
    //     default_value = "-1"
    // )]
    pub n_predict: i32,

    /// Maximum number of tokens in context
    // #[arg(long = "n-tokens", value_name = "N", default_value = "4096")]
    // pub n_tokens: NonZeroU32,

    /// Chat template to use, default template if not provided
    // #[arg(long = "chat-template", value_name = "TEMPLATE")]
    pub chat_template: Option<String>,

    /// Media marker. If not provided, the default marker will be used.
    pub media_marker: Option<String>,

    /// Whether to normalise the produced embeddings
    normalise: bool,

    /// The query to embed
    query: String,

    /// The documents to embed and compare against
    documents: Vec<String>,

    /// override some parameters of the model
    // #[arg(short = 'o', value_parser = parse_key_val)]
    // key_value_overrides: Vec<(String, ParamOverrideValue)>,

    /// set the length of the prompt + output in tokens
    // #[arg(long, default_value_t = 32)]
    n_len: i32,
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
}

impl Default for LlamaCppOptions {
    fn default() -> Self {
        Self {
            // 模型文件路径（需用户指定，默认留空）
            model_path: String::new(),
            // 多模态投影文件路径（需用户指定，默认留空）
            mmproj_path: String::new(),

            // 采样参数
            top_k: 40,        // 默认 top-k 采样值
            top_p: 0.8,       // 默认 top-p 采样值
            temperature: 0.7, // 默认温度值
            seed: 0,          // 默认随机种子（0 表示随机）

            // 线程和批处理参数
            n_threads: 0,       // 0 表示自动使用所有可用线程
            n_threads_batch: 0, // 0 表示自动使用所有可用线程
            n_batch: 512,       // 默认批处理大小
            n_ctx: 2048,        // 默认上下文窗口大小

            // GPU 相关参数
            main_gpu: 0,              // 默认主 GPU 索引
            n_gpu_layers: 0,          // 默认不启用 GPU 卸载
            no_mmproj_offload: false, // 默认启用 mmproj 的 GPU 卸载
            flash_attention: false,   // 默认禁用 Flash Attention

            // 池化类型（默认未指定）
            pooling_type: String::from("Unspecified"),

            // 日志和调试
            verbose: false, // 默认禁用详细日志

            // 多模态输入（默认留空）
            images: Vec::new(),
            audio: Vec::new(),

            // 文本生成参数
            prompt: String::new(), // 默认空提示
            n_predict: -1,         // -1 表示无限生成
            chat_template: None,   // 默认聊天模板（空）

            // 多模态标记和归一化
            media_marker: None, // 默认媒体标记（空）
            normalise: false,   // 默认禁用输入归一化

            // 检索增强生成（RAG）参数
            query: String::new(),  // 默认查询文本（空）
            documents: Vec::new(), // 默认文档列表（空）
            n_len: 128,            // 默认检索结果长度
        }
    }
}

impl PromptServer for LlamaCppOptions {}
