//! 推理流水线 - 核心推理引擎
//!
//! Pipeline 是 llama_cpp_core 的核心组件，负责：
//! - 模型加载与管理
//! - 文本生成（聊天）
//! - 多模态推理（视觉）
//! - 缓存管理
//! - 上下文管理

use std::sync::Arc;

use llama_cpp_2::{LogOptions, llama_backend::LlamaBackend, send_logs_to_tracing};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::{
    Backend, CacheManager, HistoryMessage, Model, Sampler,
    context::{ContexParams, ContextWrapper},
    error::Error,
    global_cache,
    model::ModelConfig,
    mtmd_context::MtmdContextWrapper,
    sampler::SamplerConfig,
    types::{GenerationOutput, MediaData, PoolingTypeMode},
    utils::image::Image,
};

/// 流水线配置
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PipelineConfig {
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

    /// Min-p 采样阈值
    pub min_p: f32,

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

    /// Micro batch size for prompt processing.
    /// For vision models, should be >= image tokens. Default: 1024
    #[serde(default)]
    pub n_ubatch: u32,

    /// Size of the prompt context window.
    /// Defines the maximum context length the model can handle.
    #[serde(default)]
    pub n_ctx: u32,

    /// Number of tokens to predict (-1 for unlimited)
    #[serde(default)]
    pub n_predict: i32,

    /// Disable offloading layers to the gpu
    #[serde(default)]
    pub disable_gpu: bool,

    /// Index of the main GPU to use.
    /// Relevant for multi-GPU systems.
    #[serde(default)]
    pub main_gpu: i32,

    /// 设备索引列表
    /// This option overrides `main-gpu` and enables multi-GPU.
    /// Set devices to use by index, separated by commas (e.g. --devices 0,1,2). Overrides main-gpu and enables multi-GPU.
    #[serde(default)]
    pub devices: Vec<usize>,

    /// Number of GPU layers to offload.
    /// Higher values offload more work to the GPU.
    #[serde(default)]
    pub n_gpu_layers: u32,

    /// Keep MoE layers on CPU
    #[serde(default)]
    pub cmoe: bool,

    /// Force system to keep model in RAM (use mlock)
    #[serde(default)]
    pub use_mlock: bool,

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
    pub medias: Vec<MediaData>,

    /// Image max resolution
    pub image_max_resolution: u32,

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

    /// Enables verbose logging from llama.cpp.
    /// Useful for debugging and performance analysis.
    #[serde(default)]
    pub verbose: bool,
}

impl Default for PipelineConfig {
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
            min_p: 0.0,       // 最小概率阈值
            seed: -1,         // 默认随机种子（-1 表示随机）

            // 线程和批处理参数
            n_threads: 0,       // 0 表示自动使用所有可用线程
            n_threads_batch: 0, // 0 表示自动使用所有可用线程
            n_batch: 512,       // 默认批处理大小
            n_ubatch: 1024,     // 默认微批处理大小（视觉模型需要较大值）
            n_ctx: 4096,        // 默认上下文窗口大小
            n_predict: 2048,    // 要预测的Token数量， -1 表示无限生成

            // GPU 相关参数
            disable_gpu: true,
            main_gpu: 0,     // 默认主 GPU 索引
            n_gpu_layers: 0, // 默认不启用 GPU 卸载
            // no_mmproj_offload: false, // 默认启用 mmproj 的 GPU 卸载
            flash_attention: false, // 默认禁用 Flash Attention

            cmoe: false,
            use_mlock: false,
            devices: Vec::new(),

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
            medias: Vec::new(),
            image_max_resolution: 768,

            // 日志和调试
            verbose: false, // 默认禁用详细日志

            normalise: false, // 默认禁用输入归一化

            // 检索增强生成（RAG）参数
            documents: Vec::new(),
        }
    }
}

impl PipelineConfig {
    pub fn new(model_path: String, mmproj_path: Option<String>) -> Self {
        Self {
            model_path,
            mmproj_path: mmproj_path.unwrap_or_default(),
            ..Default::default()
        }
    }

    pub fn with_model_path(mut self, model_path: impl Into<String>) -> Self {
        self.model_path = model_path.into();
        self
    }

    pub fn with_mmproj_path(mut self, mmproj_path: impl Into<String>) -> Self {
        self.mmproj_path = mmproj_path.into();
        self
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.system_prompt = system_prompt.into();
        self
    }

    pub fn with_user_prompt(mut self, user_prompt: impl Into<String>) -> Self {
        self.user_prompt = user_prompt.into();
        self
    }

    pub fn with_media(mut self, media: MediaData) -> Self {
        self.medias.push(media);
        self
    }

    pub fn with_medias(mut self, medias: Vec<MediaData>) -> Self {
        self.medias = medias;
        self
    }

    pub fn with_n_threads(mut self, n_threads: u32) -> Self {
        self.n_threads = n_threads as i32;
        self
    }

    pub fn with_image_max_resolution(mut self, image_max_resolution: u32) -> Self {
        self.image_max_resolution = image_max_resolution;
        self
    }

    pub fn with_n_threads_batch(mut self, n_threads_batch: u32) -> Self {
        self.n_threads_batch = n_threads_batch as i32;
        self
    }

    pub fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.n_batch = n_batch;
        self
    }

    pub fn with_n_ubatch(mut self, n_ubatch: u32) -> Self {
        self.n_ubatch = n_ubatch;
        self
    }

    pub fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    pub fn with_n_predict(mut self, n_predict: i32) -> Self {
        self.n_predict = n_predict;
        self
    }

    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }
    pub fn with_disable_gpu(mut self, disable_gpu: bool) -> Self {
        self.disable_gpu = disable_gpu;
        self
    }

    pub fn with_cmoe(mut self, cmoe: bool) -> Self {
        self.cmoe = cmoe;
        self
    }

    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.use_mlock = use_mlock;
        self
    }

    pub fn with_media_marker(mut self, media_marker: impl Into<String>) -> Self {
        self.media_marker = Some(media_marker.into());
        self
    }

    pub fn with_cache_model(mut self, cache_model: bool) -> Self {
        self.cache_model = cache_model;
        self
    }

    pub fn with_keep_context(mut self, keep_context: bool) -> Self {
        self.keep_context = keep_context;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 检查是否为多模态请求
    pub fn is_multimodal(&self) -> bool {
        !self.medias.is_empty()
    }
}

impl From<PipelineConfig> for ContexParams {
    fn from(pipeline_config: PipelineConfig) -> Self {
        ContexParams {
            mmproj_path: pipeline_config.mmproj_path.clone(),
            system_prompt: pipeline_config.system_prompt.clone(),
            user_prompt: pipeline_config.user_prompt.clone(),
            n_threads: pipeline_config.n_threads,
            n_threads_batch: pipeline_config.n_threads_batch,
            n_batch: pipeline_config.n_batch,
            n_ubatch: pipeline_config.n_ubatch,
            n_predict: pipeline_config.n_predict,
            n_ctx: pipeline_config.n_ctx,
            pooling_type: pipeline_config.pooling_type.clone(),
            chat_template: pipeline_config.chat_template.clone(),
            media_marker: pipeline_config.media_marker.clone(),
            disable_gpu: pipeline_config.disable_gpu,
            keep_context: pipeline_config.keep_context,
            cache_model: pipeline_config.cache_model,
            verbose: pipeline_config.verbose,
        }
    }
}

impl From<PipelineConfig> for ModelConfig {
    fn from(pipeline_config: PipelineConfig) -> Self {
        ModelConfig {
            model_path: pipeline_config.model_path.clone(),
            disable_gpu: pipeline_config.disable_gpu,
            cmoe: pipeline_config.cmoe,
            use_mlock: pipeline_config.use_mlock,
            main_gpu: pipeline_config.main_gpu,
            devices: pipeline_config.devices,
            n_gpu_layers: pipeline_config.n_gpu_layers,
            cache_model: pipeline_config.cache_model,
            verbose: pipeline_config.verbose,
        }
    }
}

impl From<PipelineConfig> for SamplerConfig {
    fn from(pipeline_config: PipelineConfig) -> Self {
        SamplerConfig {
            top_k: pipeline_config.top_k,
            top_p: pipeline_config.top_p,
            temperature: pipeline_config.temperature,
            min_p: pipeline_config.min_p,
            seed: pipeline_config.seed,
            penalty_last_n: pipeline_config.penalty_last_n,
            penalty_repeat: pipeline_config.penalty_repeat,
            penalty_freq: pipeline_config.penalty_freq,
            penalty_present: pipeline_config.penalty_present,
        }
    }
}

/// 推理流水线
pub struct Pipeline {
    backend: Arc<LlamaBackend>,
    cache: Arc<CacheManager>,
    config: PipelineConfig,
    history_message: HistoryMessage,
}

impl Pipeline {
    /// 创建新的流水线
    pub fn try_new(config: PipelineConfig) -> Result<Self, Error> {
        // 初始化缓存
        let cache = global_cache().clone();

        // 初始化后端
        let backend = Backend::init_backend()?;

        // 创初始化历史消息
        let history_message = HistoryMessage::new();

        Ok(Self {
            backend: Arc::new(backend),
            cache,
            config,
            history_message,
        })
    }

    pub fn config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// 将日志发送到 tracing
    pub fn send_logs_to_tracing(&self) {
        // llama.cpp 日志
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(self.config.verbose));

        // 初始化 LogTracer 以转发 log 事件到 tracing
        tracing_log::LogTracer::init().expect("Failed to set logger");
    }

    pub fn with_system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.config.system_prompt = system_prompt.into();
        self
    }

    pub fn with_user_prompt(mut self, user_prompt: impl Into<String>) -> Self {
        self.config.user_prompt = user_prompt.into();
        self
    }

    pub fn with_media(mut self, media: MediaData) -> Self {
        self.config.medias.push(media);
        self
    }

    pub fn with_history_message(mut self, history_message: HistoryMessage) -> Self {
        self.history_message = history_message;
        self
    }

    pub fn history_message(&self) -> HistoryMessage {
        self.history_message.clone()
    }

    pub fn clear_media(&mut self) {
        self.config.medias.clear();
    }

    /// Load image from the specified file path
    pub fn load_image_file(&mut self, path: &str) -> Result<(), Error> {
        let mut img = Image::from_file(path)?;

        let max_resolution = img.longest().min(self.config.image_max_resolution);
        img = img.resize_to_longest(max_resolution)?;

        info!(
            "image size width: {}, height: {}",
            img.width(),
            img.height()
        );

        let data = img.to_vec()?;
        self.config.medias.push(MediaData::new_image(data));
        Ok(())
    }

    /// Load image from the specified buffer
    pub fn load_image_buffer(&mut self, data: &[u8]) -> Result<(), Error> {
        self.config.medias.push(MediaData::new_image(data.to_vec()));
        Ok(())
    }

    /// 执行推理
    pub async fn infer(&mut self) -> Result<GenerationOutput, Error> {
        if self.config.is_multimodal() {
            self.infer_multimodal(&self.config.clone()).await
        } else {
            self.infer_text(&self.config.clone()).await
        }
    }

    /// 纯文本推理
    async fn infer_text(&mut self, config: &PipelineConfig) -> Result<GenerationOutput, Error> {
        // Load model
        let model = Model::from_config(config.clone().into()).load_cache_model(&self.backend)?;

        // Load sampler
        let mut sampler = Sampler::load_sampler(&config.clone().into())?;

        // 创建上下文
        let contex_params: ContexParams = config.clone().into();
        let mut ctx = ContextWrapper::try_new(model.clone(), &self.backend, &contex_params)?;

        // 创建消息
        let msgs = ContextWrapper::create_message(
            &contex_params,
            false,
            vec![],
            &self.history_message.messages(),
        )?;

        // 评估消息
        ctx.eval_messages(msgs)?;

        // 生成响应
        let output = ctx.generate_response(&mut sampler)?;

        // 将上下文信息添加到历史消息中
        if config.keep_context {
            // 添加系统提示
            if !self.history_message.messages().is_empty() {
                self.history_message
                    .add_system(contex_params.system_prompt)?;
            }

            // 每轮聊天都添加用户提示和助手响应
            self.history_message.add_user(contex_params.user_prompt)?;
            self.history_message.add_assistant(output.text.clone())?;
        }

        Ok(output)
    }

    /// 多模态推理
    async fn infer_multimodal(
        &mut self,
        config: &PipelineConfig,
    ) -> Result<GenerationOutput, Error> {
        // Load model
        let model = Model::from_config(config.clone().into()).load_cache_model(&self.backend)?;

        // Load sampler
        let mut sampler = Sampler::load_sampler(&config.clone().into())?;

        // 上下文
        let contex_params: ContexParams = config.clone().into();
        let ctx = ContextWrapper::try_new(model.clone(), &self.backend, &contex_params)?;
        let mut mtmd_ctx = MtmdContextWrapper::try_new(model.clone(), ctx, &contex_params)?;

        // Load media files
        // for image_path in &params.images {
        //     info!("Loading image: {image_path}");
        //     mtmd_ctx.load_media_file(image_path)?;
        // }
        // for audio_path in &params.audio {
        //     mtmd_ctx.load_media_file(audio_path)?;
        // }

        for media in &config.medias {
            info!("Loading media: {}", media.media_type);
            mtmd_ctx.load_media_buffer(&media.data)?;
        }

        // 创建消息
        let msgs = ContextWrapper::create_message(
            &contex_params,
            true,
            config.medias.clone(),
            &self.history_message.messages(),
        )?;

        // 评估消息
        mtmd_ctx.eval_messages(msgs)?;

        // 清除媒体缓存
        self.clear_media();

        // 生成响应
        let output = mtmd_ctx.generate_response(&mut sampler)?;

        // 将上下文信息添加到历史消息中
        if config.keep_context {
            // 添加系统提示
            if !self.history_message.messages().is_empty() {
                self.history_message
                    .add_system(contex_params.system_prompt)?;
            }

            // 每轮聊天都添加用户提示和助手响应
            self.history_message.add_user(contex_params.user_prompt)?;
            self.history_message.add_assistant(output.text.clone())?;
        }

        Ok(output)
    }

    /// 清除模型缓存
    pub fn clear_model_cache(&self) -> Result<(), Error> {
        self.cache.clear()
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::log::init_logger;

    use super::*;

    #[tokio::test]
    async fn test_simple() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();
        let pipeline_config = PipelineConfig::new(model_path, None).with_user_prompt("你是谁？");

        let mut pipeline = Pipeline::try_new(pipeline_config)?;

        let results = pipeline.infer().await?;

        println!("{results:?}");
        Ok(())
    }

    #[tokio::test]
    async fn test_simple_vision() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/data/ComfyUI/models/clip/Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf".to_string();
        let mmproj_path =
            "/data/ComfyUI/models/clip/Qwen2.5-VL-7B-Instruct-abliterated.mmproj-f16.gguf"
                .to_string();
        let pipeline_config = PipelineConfig::new(model_path, Some(mmproj_path))
            .with_disable_gpu(false)
            .with_user_prompt("描述这张图片")
            .with_media_marker("<start_of_image>")
            .with_verbose(true);

        let mut pipeline = Pipeline::try_new(pipeline_config)?;

        pipeline.load_image_file("/home/one/Downloads/cy/00089-915810967.png")?;

        let results = pipeline.infer().await?;

        println!("{results:?}");
        Ok(())
    }
}
