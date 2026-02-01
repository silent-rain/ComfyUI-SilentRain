//! 多模态统一接口
//!
//! 提供统一的接口支持 chat、vision、video、audio 等多种模态的推理

use async_trait::async_trait;
use std::sync::Arc;

use crate::{
    core::{
        cache::CacheManager,
        error::{Error, Result},
        pipeline::{InferenceRequest, Pipeline, PipelineConfig},
    },
    plugins::{FeaturePlugin, PluginRegistry},
    types::{MediaType, PluginInput, PluginOutput},
};

/// 多模态推理引擎
pub struct MultimodalEngine {
    registry: Arc<PluginRegistry>,
    cache: Arc<CacheManager>,
    pipeline: Arc<Pipeline>,
    stats: Arc<std::sync::RwLock<EngineStats>>,
}

impl MultimodalEngine {
    /// 创建新的引擎
    pub fn try_new() -> Result<Self> {
        let cache = Arc::new(CacheManager::default());
        let pipeline = Arc::new(Pipeline::with_cache(cache.clone())?);

        Ok(Self {
            registry: Arc::new(PluginRegistry::new()),
            cache,
            pipeline,
            stats: Arc::new(std::sync::RwLock::new(EngineStats::default())),
        })
    }

    /// 使用自定义组件创建
    pub fn with_components(
        registry: Arc<PluginRegistry>,
        cache: Arc<CacheManager>,
        config: PipelineConfig,
    ) -> Result<Self> {
        let pipeline = Arc::new(Pipeline::try_new(cache.clone(), config)?);

        Ok(Self {
            registry,
            cache,
            pipeline,
            stats: Arc::new(std::sync::RwLock::new(EngineStats::default())),
        })
    }

    /// 统一推理入口
    ///
    /// 自动根据输入选择最合适的插件
    pub async fn infer(&self, input: PluginInput) -> Result<PluginOutput> {
        let start = std::time::Instant::now();

        // 选择最佳插件
        let plugin = self
            .select_best_plugin(&input)
            .ok_or_else(|| Error::NoSupportedPlugin)?;

        // 验证输入
        self.validate_input(&input, &*plugin)?;

        // 执行推理
        let result = plugin.process(input.clone()).await;

        // 更新统计
        if let Ok(ref output) = result {
            let mut stats = self.stats.write().unwrap();
            stats.total_requests += 1;
            stats.total_tokens += output
                .metadata
                .get("tokens_generated")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            stats.total_duration += start.elapsed();
        }

        result
    }

    /// 批量推理
    ///
    /// 支持混合类型的批量推理，自动分组并行处理
    pub async fn infer_batch(&self, inputs: Vec<PluginInput>) -> Vec<Result<PluginOutput>> {
        use futures::stream::{self, StreamExt};

        stream::iter(inputs)
            .map(|input| self.infer(input))
            .buffer_unordered(4)
            .collect()
            .await
    }

    /// 直接通过 Pipeline 推理（绕过插件系统）
    pub async fn infer_direct(&self, request: InferenceRequest) -> Result<PluginOutput> {
        let output = self.pipeline.infer(request).await?;

        Ok(PluginOutput {
            text: output.text,
            metadata: {
                let mut meta = std::collections::HashMap::new();
                meta.insert(
                    "tokens_generated".to_string(),
                    output.tokens_generated.into(),
                );
                meta.insert(
                    "finish_reason".to_string(),
                    output.finish_reason.to_string().into(),
                );
                meta
            },
        })
    }

    /// 选择最佳插件
    fn select_best_plugin(&self, input: &PluginInput) -> Option<Arc<dyn FeaturePlugin>> {
        self.registry.find_for_input(input)
    }

    /// 验证输入
    fn validate_input(&self, input: &PluginInput, plugin: &dyn FeaturePlugin) -> Result<()> {
        // 检查模型路径
        if input.model_path.is_empty() {
            return Err(Error::MissingInput {
                field: "model_path",
            });
        }

        // 视觉插件需要 mmproj_path
        if plugin.name() == "vision" && input.mmproj_path.is_none() {
            return Err(Error::MissingMmprojPath);
        }

        Ok(())
    }

    /// 获取引擎统计信息
    pub fn stats(&self) -> EngineStats {
        self.stats.read().unwrap().clone()
    }

    /// 获取 Pipeline 引用
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// 获取缓存管理器
    pub fn cache(&self) -> &CacheManager {
        &self.cache
    }

    /// 获取插件注册表
    pub fn registry(&self) -> &PluginRegistry {
        &self.registry
    }

    /// 清除缓存
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
}

impl Default for MultimodalEngine {
    fn default() -> Self {
        Self::try_new().expect("Failed to create default multimodal engine")
    }
}

/// 引擎统计信息
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub total_requests: usize,
    pub total_tokens: usize,
    pub total_duration: std::time::Duration,
    pub registered_plugins: usize,
    pub cache_entries: usize,
}

impl EngineStats {
    /// 获取平均 token 生成速度
    pub fn avg_tokens_per_sec(&self) -> f64 {
        let secs = self.total_duration.as_secs_f64();
        if secs > 0.0 {
            self.total_tokens as f64 / secs
        } else {
            0.0
        }
    }

    /// 获取平均请求处理时间
    pub fn avg_request_duration(&self) -> std::time::Duration {
        if self.total_requests > 0 {
            self.total_duration / self.total_requests as u32
        } else {
            std::time::Duration::default()
        }
    }
}

/// 全局多模态引擎
pub struct GlobalMultimodalEngine;

impl GlobalMultimodalEngine {
    /// 获取全局引擎实例
    pub fn get() -> &'static Arc<MultimodalEngine> {
        use std::sync::OnceLock;
        static ENGINE: OnceLock<Arc<MultimodalEngine>> = OnceLock::new();
        ENGINE.get_or_init(|| Arc::new(MultimodalEngine::default()))
    }
}

/// 便捷函数：获取全局引擎
pub fn engine() -> &'static Arc<MultimodalEngine> {
    GlobalMultimodalEngine::get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_capabilities() {
        let text_caps = PluginCapabilities::text_only();
        assert!(text_caps.supports_text);
        assert!(text_caps.required_media.is_empty());

        let vision_caps = PluginCapabilities::vision();
        assert_eq!(vision_caps.required_media, vec![MediaType::Image]);

        let video_caps = PluginCapabilities::video();
        assert_eq!(video_caps.required_media, vec![MediaType::Video]);
        assert_eq!(video_caps.optional_media, vec![MediaType::Audio]);
    }

    #[test]
    fn test_engine_stats() {
        let stats = EngineStats {
            total_requests: 10,
            total_tokens: 1000,
            total_duration: std::time::Duration::from_secs(10),
            ..Default::default()
        };

        assert_eq!(stats.avg_tokens_per_sec(), 100.0);
        assert_eq!(
            stats.avg_request_duration(),
            std::time::Duration::from_secs(1)
        );
    }

    #[test]
    fn test_multimodal_engine_creation() {
        let engine = MultimodalEngine::default();
        let stats = engine.stats();
        assert_eq!(stats.total_requests, 0);
    }
}
