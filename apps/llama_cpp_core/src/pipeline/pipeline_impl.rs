//! 推理流水线 - 核心推理引擎
//!
//! Pipeline 是 llama_cpp_core 的核心组件，负责：
//! - 模型加载与管理
//! - 文本生成（聊天）
//! - 多模态推理（视觉）
//! - 缓存管理
//! - 上下文管理

use std::sync::Arc;

use llama_cpp_2::model::LlamaChatMessage;
use tokio::sync::mpsc;

use llama_cpp_2::{LogOptions, llama_backend::LlamaBackend, send_logs_to_tracing};
use tracing::{error, info};

use crate::{
    Backend, GenerateRequest, Model, PipelineConfig, Sampler,
    context::{ContexParams, ContextWrapper},
    error::Error,
    mtmd_context::MtmdContextWrapper,
    types::{GenerationOutput, MediaData, StreamToken},
};

/// 推理流水线
///
/// **注意：** Pipeline 是完全无状态的，所有动态数据（提示词、历史消息、媒体）都通过 `GenerateRequest` 传入。
/// 这样可以安全地在并发场景中使用 `Arc<Pipeline>`。
pub struct Pipeline {
    backend: Arc<LlamaBackend>,
    config: PipelineConfig,
}

impl Pipeline {
    /// 创建新的流水线
    pub fn try_new(config: PipelineConfig) -> Result<Self, Error> {
        // 初始化后端
        let backend = Backend::init_backend()?;

        Ok(Self {
            backend: Arc::new(backend),
            config,
        })
    }

    /// 将日志发送到 tracing
    pub fn send_logs_to_tracing(&self) {
        // llama.cpp 日志
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(self.config.verbose));

        // 初始化 LogTracer 以转发 log 事件到 tracing
        tracing_log::LogTracer::init().expect("Failed to set logger");
    }

    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }
}

impl Pipeline {
    /// 执行推理
    ///
    /// # Arguments
    /// * `request` - 生成请求
    ///
    /// # Example
    /// ```rust,ignore
    /// let request = GenerateRequest::text("你好");
    /// let result = pipeline.generate(&request).await?;
    /// ```
    pub async fn generate(&self, request: &GenerateRequest) -> Result<GenerationOutput, Error> {
        let msgs = request.to_messages()?;
        let mut rx = if !request.is_multimodal() {
            self.generate_text_stream(&msgs)?
        } else {
            self.generate_media_stream(&msgs, &request.medias)?
        };

        let mut full_text = String::new();
        while let Some(token) = rx.recv().await {
            match token {
                StreamToken::Content(text) => full_text.push_str(&text),
                StreamToken::Finish(reason) => full_text.push_str(&reason.to_string()),
                StreamToken::Error(msg) => return Ok(GenerationOutput::new(msg)),
            }
        }

        Ok(GenerationOutput::new(full_text))
    }

    /// 执行流式推理
    ///
    /// # Arguments
    /// * `request` - 生成请求
    ///
    /// # Returns
    /// 返回一个 receiver，每次生成一个 token 时会发送一个 StreamToken
    pub async fn generate_stream(
        &self,
        request: &GenerateRequest,
    ) -> Result<mpsc::UnboundedReceiver<StreamToken>, Error> {
        let msgs = request.to_messages()?;

        let rx = if !request.is_multimodal() {
            self.generate_text_stream(&msgs)?
        } else {
            self.generate_media_stream(&msgs, &request.medias)?
        };
        Ok(rx)
    }

    /// 纯文本流式推理内部实现
    fn generate_text_stream(
        &self,
        msgs: &[LlamaChatMessage],
    ) -> Result<mpsc::UnboundedReceiver<StreamToken>, Error> {
        // Load model
        let llama_model = Model::from_config(self.config.clone().into())
            .load_cache_llama_model(&self.backend)
            .map_err(|e| {
                error!("Failed to load model: {}", e);
                e
            })?;

        // Load sampler
        let mut sampler = Sampler::load_sampler(&self.config.clone().into()).map_err(|e| {
            error!("Failed to load sampler: {}", e);
            e
        })?;

        // 创建上下文
        let contex_params: ContexParams = self.config.clone().into();
        let mut ctx = ContextWrapper::try_new(llama_model.clone(), &self.backend, &contex_params)
            .map_err(|e| {
            error!("Failed to create context: {}", e);
            e
        })?;

        // 评估消息
        ctx.eval_messages(msgs.to_vec()).map_err(|e| {
            error!("Failed to eval messages: {}", e);
            e
        })?;

        // 使用 channel 方式生成响应
        Ok(ctx.generate_response(&mut sampler))
    }

    /// 媒体流式推理内部实现
    fn generate_media_stream(
        &self,
        msgs: &[LlamaChatMessage],
        medias: &[MediaData],
    ) -> Result<mpsc::UnboundedReceiver<StreamToken>, Error> {
        // Load model
        let model = Model::from_config(self.config.clone().into());
        let llama_model = model.load_cache_llama_model(&self.backend).map_err(|e| {
            error!("Failed to load llama model: {}", e);
            e
        })?;
        let mtmd_context = model
            .load_cache_mtmd_context(llama_model.clone())
            .map_err(|e| {
                error!("Failed to load mtmd context: {}", e);
                e
            })?;

        // Load sampler
        let mut sampler = Sampler::load_sampler(&self.config.clone().into()).map_err(|e| {
            error!("Failed to load sampler: {}", e);
            e
        })?;

        // 上下文
        let contex_params: ContexParams = self.config.clone().into();
        let ctx = ContextWrapper::try_new(llama_model.clone(), &self.backend, &contex_params)
            .map_err(|e| {
                error!("Failed to create context: {}", e);
                e
            })?;

        let mut mtmd_ctx =
            MtmdContextWrapper::try_new(llama_model.clone(), ctx, mtmd_context, &contex_params)
                .map_err(|e| {
                    error!("Failed to create mtmd context: {}", e);
                    e
                })?;

        // Load media files
        for media in medias {
            info!("Loading media: {}", media.media_type);
            mtmd_ctx.load_media_buffer(&media.data).map_err(|e| {
                error!("Failed to load media: {}", e);
                e
            })?;
        }

        // 评估消息
        mtmd_ctx.eval_messages(msgs.to_vec()).map_err(|e| {
            error!("Failed to eval messages: {}", e);
            e
        })?;

        // 使用 channel 方式生成响应
        Ok(mtmd_ctx.generate_response(&mut sampler))
    }
}

#[cfg(test)]
mod tests {
    use crate::{HistoryMessage, Model, utils::log::init_logger};

    use super::*;

    #[test]
    fn test_check_gpu() -> anyhow::Result<()> {
        init_logger();
        // 检测设备是否可用，如果没有 GPU 则使用 CPU
        let model = Model::new("", None::<String>);
        let devices = model.devices();

        println!("devices list {devices:?}");
        println!("Detected {} GPU device(s)", devices.len());
        Ok(())
    }

    #[tokio::test]
    async fn test_simple() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();

        let pipeline_config = PipelineConfig::new(model_path, None)
            .with_disable_gpu(false)
            .with_main_gpu(0)
            .with_n_gpu_layers(10)
            .with_verbose(true);

        let pipeline = Pipeline::try_new(pipeline_config)?;

        let request = GenerateRequest::text("你是谁？");
        let results = pipeline.generate(&request).await?;

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
            .with_main_gpu(0)
            .with_n_gpu_layers(10)
            .with_media_marker("<start_of_image>")
            .with_verbose(true);

        let pipeline = Pipeline::try_new(pipeline_config)?;

        let request = GenerateRequest::text("描述这张图片")
            .with_media_file("/home/one/Downloads/cy/00089-915810967.png")?
            .with_media_marker("<start_of_image>");

        let results = pipeline.generate(&request).await?;

        println!("{results:?}");
        Ok(())
    }

    #[tokio::test]
    async fn test_concurrent() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();
        let pipeline_config = PipelineConfig::new(model_path, None)
            .with_cache_model(true)
            .with_keep_context(false);

        let pipeline = Arc::new(Pipeline::try_new(pipeline_config)?);

        // 并发执行多个请求
        let tasks = vec![
            tokio::spawn({
                let pipeline = Arc::clone(&pipeline);
                async move {
                    let request = GenerateRequest::text("解释量子力学");
                    pipeline.generate(&request).await
                }
            }),
            tokio::spawn({
                let pipeline = Arc::clone(&pipeline);
                async move {
                    let request = GenerateRequest::text("解释相对论");
                    pipeline.generate(&request).await
                }
            }),
        ];

        let results = futures::future::try_join_all(tasks).await?;
        for result in results {
            match result {
                Ok(output) => println!("Result: {:?}", output),
                Err(e) => eprintln!("Error: {}", e),
            }
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_with_external_history() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();
        let pipeline_config = PipelineConfig::new(model_path, None);

        let pipeline = Pipeline::try_new(pipeline_config)?;

        // 外部管理历史消息
        let mut history = HistoryMessage::new();

        // 第一轮对话
        {
            let request1 =
                GenerateRequest::text("你好，我叫小明").with_system("你是一个 helpful 的助手");
            let result1 = pipeline.generate(&request1).await?;
            println!("Assistant: {}", result1.text);

            // 更新历史（外部管理）
            history.add_user("你好，我叫小明")?;
            history.add_assistant(result1.text.clone())?;
        }

        // 第二轮对话（带历史）
        {
            let request2 = GenerateRequest::text("我叫什么名字？")
                .with_system("你是一个 helpful 的助手")
                .with_history(history.clone());
            let result2 = pipeline.generate(&request2).await?;
            println!("Assistant: {}", result2.text);

            // 验证模型是否记住了名字
            assert!(result2.text.contains("小明"));
        }

        Ok(())
    }
}
