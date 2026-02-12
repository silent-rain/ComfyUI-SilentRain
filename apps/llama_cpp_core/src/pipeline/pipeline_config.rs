//! 流水线配置
//!
//! 使用组合模式，直接包含子配置模块

use async_openai::types::chat::CreateChatCompletionRequest;
use serde::{Deserialize, Serialize};

use crate::{context::ContexParams as ContextConfig, model::ModelConfig, sampler::SamplerConfig};

/// 流水线配置
///
/// 使用组合模式，直接包含三个子配置：
/// - `model`: 模型加载相关配置（路径、GPU、线程等）
/// - `context`: 上下文相关配置（批大小、窗口大小等）
/// - `sampling`: 采样相关配置（温度、top-k/p 等）
///
/// # Example
/// ```rust
/// use llama_cpp_core::PipelineConfig;
///
/// let config = PipelineConfig::new("model.gguf")
///     .with_gpu_layers(10)
///     .with_context_size(8192);
/// ```
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct PipelineConfig {
    /// 模型配置
    #[serde(default)]
    pub model: ModelConfig,

    /// 上下文配置
    #[serde(default)]
    pub context: ContextConfig,

    /// 采样配置
    #[serde(default)]
    pub sampling: SamplerConfig,
}

impl PipelineConfig {
    /// 创建新的流水线配置
    ///
    /// # Arguments
    /// * `model_path` - 模型文件路径
    ///
    /// # Example
    /// ```rust
    /// let config = PipelineConfig::new("model.gguf");
    /// ```
    pub fn new(model_path: impl Into<String>) -> Self {
        let mut config = Self::default();
        config.model.model_path = model_path.into();
        config
    }

    /// 创建带 mmproj 的配置（多模态）
    ///
    /// # Example
    /// ```rust
    /// let config = PipelineConfig::new_with_mmproj("model.gguf", "mmproj.gguf");
    /// ```
    pub fn new_with_mmproj(model_path: impl Into<String>, mmproj_path: impl Into<String>) -> Self {
        let mut config = Self::new(model_path);
        config.model.mmproj_path = mmproj_path.into();
        config
    }
}

/// Builder 风格的便捷方法
impl PipelineConfig {
    /// 设置模型路径
    pub fn with_model_path(mut self, model_path: impl Into<String>) -> Self {
        self.model.model_path = model_path.into();
        self
    }

    /// 设置 mmproj 路径（多模态）
    pub fn with_mmproj_path(mut self, mmproj_path: impl Into<String>) -> Self {
        self.model.mmproj_path = mmproj_path.into();
        self
    }

    /// 设置 GPU 层数（0 = CPU 模式）
    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        self.model.n_gpu_layers = n_gpu_layers;
        self
    }

    /// 设置主 GPU
    pub fn with_main_gpu(mut self, main_gpu: i32) -> Self {
        self.model.main_gpu = main_gpu;
        self
    }

    /// 设置温度
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.sampling.temperature = temperature;
        self
    }

    /// 设置 top_k
    pub fn with_top_k(mut self, top_k: i32) -> Self {
        self.sampling.top_k = top_k;
        self
    }

    /// 设置 top_p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.sampling.top_p = top_p;
        self
    }

    /// 设置随机种子
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.sampling.seed = seed;
        self
    }

    /// 设置线程数
    pub fn with_threads(mut self, n_threads: i32) -> Self {
        self.model.n_threads = n_threads;
        self.context.n_threads = n_threads;
        self
    }

    /// 设置媒体标记
    pub fn with_media_marker(mut self, marker: impl Into<String>) -> Self {
        let marker = marker.into();
        self.context.media_marker = marker.clone();
        self.model.media_marker = Some(marker);
        self
    }

    /// 设置图像最大分辨率
    pub fn with_image_max_resolution(mut self, max_resolution: u32) -> Self {
        self.context.image_max_resolution = max_resolution;
        self
    }

    /// 是否启用详细日志
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.model.verbose = verbose;
        self.context.verbose = verbose;
        self
    }

    /// 设置最大历史消息数
    pub fn with_max_history(mut self, max_history: usize) -> Self {
        self.context.max_history = max_history;
        self
    }
}

impl PipelineConfig {
    /// 设置上下文配置
    pub fn with_context(mut self, context: ContextConfig) -> Self {
        self.context = context;
        self
    }

    /// 设置上下文窗口大小
    pub fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        self.context.n_ctx = n_ctx;
        self
    }

    /// 设置批处理大小
    pub fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.context.n_batch = n_batch;
        self
    }

    /// 设置最大完成令牌数
    pub fn with_max_completion_tokens(mut self, max_completion_tokens: i32) -> Self {
        self.context.n_predict = max_completion_tokens;
        self
    }
}

/// 从 OpenAI 标准请求初始化配置
impl PipelineConfig {
    /// 从请求中提取并应用参数到现有配置
    ///
    /// 这个方法可以多次调用，用于动态更新配置。
    /// 适合在需要合并多个来源参数时使用。
    ///
    /// # Arguments
    /// * `request` - OpenAI 标准聊天完成请求
    ///
    /// # Example
    /// ```rust, no_run
    /// let mut config = PipelineConfig::new("model.gguf");
    /// // 先应用基础配置...
    /// config.apply_request_params(&request);
    /// // 再应用请求特定的参数...
    /// config.apply_request_params(&override_request);
    /// ```
    pub fn apply_request_params(&mut self, request: &CreateChatCompletionRequest) {
        // 提取 max_completion_tokens -> n_predict
        if let Some(max_completion_tokens) = request.max_completion_tokens {
            self.context.n_predict = max_completion_tokens as i32;
        }

        // 提取 temperature
        if let Some(temperature) = request.temperature {
            self.sampling.temperature = temperature;
        }

        // 提取 top_p
        if let Some(top_p) = request.top_p {
            self.sampling.top_p = top_p;
        }

        // 提取 presence_penalty -> penalty_present
        if let Some(presence_penalty) = request.presence_penalty {
            self.sampling.penalty_present = presence_penalty;
        }

        // 提取 frequency_penalty -> penalty_freq
        if let Some(frequency_penalty) = request.frequency_penalty {
            self.sampling.penalty_freq = frequency_penalty;
        }

        // 注意：以下参数在标准 OpenAI API 中没有直接对应，保持默认值
        // - top_k (OpenAI 不支持)
        // - n_gpu_layers (基础设施参数)
        // - n_ctx (基础设施参数)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PipelineConfig::default();
        assert!(config.model.model_path.is_empty());
        assert_eq!(config.context.n_ctx, 4096);
        assert_eq!(config.sampling.temperature, 0.6);
    }

    #[test]
    fn test_new_config() {
        let config = PipelineConfig::new("test.gguf");
        assert_eq!(config.model.model_path, "test.gguf");
    }

    #[test]
    fn test_builder_style() {
        let config = PipelineConfig::new("test.gguf")
            .with_n_gpu_layers(10)
            .with_n_ctx(8192)
            .with_temperature(0.8)
            .with_top_k(50);

        assert_eq!(config.model.n_gpu_layers, 10);
        assert!(config.model.n_gpu_layers > 0);
        assert_eq!(config.context.n_ctx, 8192);
        assert_eq!(config.sampling.temperature, 0.8);
        assert_eq!(config.sampling.top_k, 50);
    }

    #[test]
    fn test_serde_roundtrip() {
        let config = PipelineConfig::new("test.gguf")
            .with_n_gpu_layers(10)
            .with_n_ctx(8192);

        let json = serde_json::to_string(&config).unwrap();
        let restored: PipelineConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.model.model_path, config.model.model_path);
        assert_eq!(restored.model.n_gpu_layers, config.model.n_gpu_layers);
        assert_eq!(restored.context.n_ctx, config.context.n_ctx);
    }
}
