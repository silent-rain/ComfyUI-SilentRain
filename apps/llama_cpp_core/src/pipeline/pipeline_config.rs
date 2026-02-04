//! 流水线配置
//!
//! 使用组合模式，直接包含子配置模块

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

    /// Enables verbose logging from llama.cpp.
    #[serde(default)]
    pub verbose: bool,
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
    pub fn with_seed(mut self, seed: i32) -> Self {
        self.sampling.seed = seed;
        self
    }

    /// 设置线程数
    pub fn with_threads(mut self, n_threads: i32) -> Self {
        self.model.n_threads = n_threads;
        self.context.n_threads = n_threads;
        self
    }

    /// 是否缓存模型
    pub fn with_cache_model(mut self, cache: bool) -> Self {
        self.model.cache_model = cache;
        self
    }

    /// 是否启用详细日志
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self.model.verbose = verbose;
        self.context.verbose = verbose;
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

    /// 设置最大生成 token 数
    pub fn with_max_tokens(mut self, tokens: i32) -> Self {
        self.context.n_predict = tokens;
        self
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
