//! 流水线配置

use serde::{Deserialize, Serialize};

use crate::{
    context::ContexParams, model::ModelConfig, sampler::SamplerConfig, types::PoolingTypeMode,
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

    /// Chat template to use, default template if not provided
    // #[arg(long = "chat-template", value_name = "TEMPLATE")]
    #[serde(default)]
    pub chat_template: Option<String>,

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

            cache_model: false, // 缓存模型（默认禁用）

            // Penalizes tokens for being present in the context.
            penalty_last_n: 64,   // 重复惩罚的窗口大小
            penalty_repeat: 1.2,  // 重复惩罚系数
            penalty_freq: 1.1,    // 重复频率惩罚系数
            penalty_present: 0.0, // 存在惩罚系数

            // 池化类型（默认未指定）
            pooling_type: PoolingTypeMode::Unspecified.to_string(),

            chat_template: None, // 默认聊天模板

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

    pub fn with_n_threads(mut self, n_threads: u32) -> Self {
        self.n_threads = n_threads as i32;
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

    pub fn with_main_gpu(mut self, main_gpu: i32) -> Self {
        self.main_gpu = main_gpu;
        self
    }

    pub fn with_disable_gpu(mut self, disable_gpu: bool) -> Self {
        self.disable_gpu = disable_gpu;
        self
    }

    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
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

    pub fn with_cache_model(mut self, cache_model: bool) -> Self {
        self.cache_model = cache_model;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl From<PipelineConfig> for ContexParams {
    fn from(pipeline_config: PipelineConfig) -> Self {
        ContexParams {
            n_threads: pipeline_config.n_threads,
            n_threads_batch: pipeline_config.n_threads_batch,
            n_batch: pipeline_config.n_batch,
            n_ubatch: pipeline_config.n_ubatch,
            n_predict: pipeline_config.n_predict,
            n_ctx: pipeline_config.n_ctx,
            pooling_type: pipeline_config.pooling_type.clone(),
            chat_template: pipeline_config.chat_template.clone(),
            verbose: pipeline_config.verbose,
        }
    }
}

impl From<PipelineConfig> for ModelConfig {
    fn from(pipeline_config: PipelineConfig) -> Self {
        ModelConfig {
            model_path: pipeline_config.model_path.clone(),
            mmproj_path: pipeline_config.mmproj_path.clone(),
            disable_gpu: pipeline_config.disable_gpu,
            main_gpu: pipeline_config.main_gpu,
            devices: pipeline_config.devices,
            n_gpu_layers: pipeline_config.n_gpu_layers,
            cmoe: pipeline_config.cmoe,
            use_mlock: pipeline_config.use_mlock,
            n_threads: pipeline_config.n_threads,
            cache_model: pipeline_config.cache_model,
            verbose: pipeline_config.verbose,
            ..Default::default()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple() -> anyhow::Result<()> {
        let pipeline_config = PipelineConfig::default();

        let s = serde_json::to_string(&pipeline_config)?;
        println!("{s:?}\n\n");

        let model_config: ModelConfig = serde_json::from_str(&s)?;
        println!("{model_config:#?}\n\n");

        let sampler_config: SamplerConfig = serde_json::from_str(&s)?;
        println!("{sampler_config:#?}\n\n");

        let contex_params: ContexParams = serde_json::from_str(&s)?;
        println!("{contex_params:#?}\n\n");
        Ok(())
    }
}
