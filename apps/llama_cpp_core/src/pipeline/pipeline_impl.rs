//! 推理流水线 - 核心推理引擎
//!
//! Pipeline 是 llama_cpp_core 的核心组件，负责：
//! - 模型加载与管理
//! - 文本生成（聊天）
//! - 多模态推理（视觉）
//! - 缓存管理
//! - 上下文管理

use std::sync::Arc;

use tokio::sync::mpsc;

use llama_cpp_2::{LogOptions, llama_backend::LlamaBackend, send_logs_to_tracing};
use tracing::{error, info};

use crate::{
    Backend, HistoryMessage, Model, PipelineConfig, Sampler,
    context::{ContexParams, ContextWrapper},
    error::Error,
    mtmd_context::MtmdContextWrapper,
    types::{GenerationOutput, MediaData, StreamToken},
    utils::image::Image,
};

/// 推理流水线
pub struct Pipeline {
    backend: Arc<LlamaBackend>,
    config: PipelineConfig,
    history_message: HistoryMessage,
}

impl Pipeline {
    /// 创建新的流水线
    pub fn try_new(config: PipelineConfig) -> Result<Self, Error> {
        // 初始化后端
        let backend = Backend::init_backend()?;

        // 创初始化历史消息
        let history_message = HistoryMessage::new();

        Ok(Self {
            backend: Arc::new(backend),
            config,
            history_message,
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
}

impl Pipeline {
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
    pub async fn generate(&mut self) -> Result<GenerationOutput, Error> {
        let mut rx = self.generate_stream().await?;

        let mut full_text = String::new();
        while let Some(token) = rx.recv().await {
            match token {
                StreamToken::Content(text) => full_text.push_str(&text),
                StreamToken::Finish(reason) => full_text.push_str(&reason.to_string()),
                StreamToken::Error(msg) => return Ok(GenerationOutput::new(msg)),
            }
        }

        // 清除媒体缓存
        self.clear_media();

        Ok(GenerationOutput::new(full_text))
    }

    /// 执行流式推理
    ///
    /// 返回一个 receiver，每次生成一个 token 时会发送一个 StreamToken
    ///
    /// # Example
    /// ```rust,ignore
    /// let rx = pipeline.generate_stream().await?;
    /// while let Some(token) = rx.recv().await {
    ///     match token {
    ///         StreamToken::Content(text) => print!("{}", text),
    ///         StreamToken::Finish(reason) => println!("\nFinished: {:?}", reason),
    ///         StreamToken::Error(msg) => eprintln!("Error: {}", msg),
    ///     }
    /// }
    /// ```
    pub async fn generate_stream(&mut self) -> Result<mpsc::UnboundedReceiver<StreamToken>, Error> {
        let rx = if self.config.is_media() {
            self.generate_media_stream()?
        } else {
            self.generate_text_stream()?
        };

        // 清除媒体缓存
        self.clear_media();

        Ok(rx)
    }

    /// 纯文本流式推理内部实现
    pub fn generate_text_stream(&self) -> Result<mpsc::UnboundedReceiver<StreamToken>, Error> {
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

        // 创建消息
        let msgs = ContextWrapper::create_message(
            &contex_params,
            false,
            vec![],
            &self.history_message.messages(),
        )
        .map_err(|e| {
            error!("Failed to create message: {}", e);
            e
        })?;

        // 评估消息
        ctx.eval_messages(msgs).map_err(|e| {
            error!("Failed to eval messages: {}", e);
            e
        })?;

        // 使用 channel 方式生成响应
        Ok(ctx.generate_response(&mut sampler))
    }

    /// 媒体流式推理内部实现
    pub fn generate_media_stream(&self) -> Result<mpsc::UnboundedReceiver<StreamToken>, Error> {
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
        for media in &self.config.medias {
            info!("Loading media: {}", media.media_type);
            mtmd_ctx.load_media_buffer(&media.data).map_err(|e| {
                error!("Failed to load media: {}", e);
                e
            })?;
        }

        // 创建消息
        let msgs = ContextWrapper::create_message(
            &contex_params,
            true,
            self.config.medias.clone(),
            &self.history_message.messages(),
        )
        .map_err(|e| {
            error!("Failed to create message: {}", e);
            e
        })?;

        // 评估消息
        mtmd_ctx.eval_messages(msgs).map_err(|e| {
            error!("Failed to eval messages: {}", e);
            e
        })?;

        // 使用 channel 方式生成响应
        Ok(mtmd_ctx.generate_response(&mut sampler))
    }

    /// 根据流式生成的结果更新历史消息
    ///
    /// # Arguments
    /// * `full_text` - 完整文本
    pub fn update_history(&mut self, full_text: &str) -> Result<(), Error> {
        if !self.config.keep_context {
            return Ok(());
        }

        // 添加系统提示（如果历史为空且已有系统消息）
        if self.history_message.messages().is_empty() && !self.config.system_prompt.is_empty() {
            self.history_message
                .add_system(self.config.system_prompt.clone())?;
        }

        // 添加用户提示和助手响应
        self.history_message
            .add_user(self.config.user_prompt.clone())?;
        self.history_message.add_assistant(full_text.to_string())?;

        Ok(())
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

        let results = pipeline.generate().await?;

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

        let results = pipeline.generate().await?;

        println!("{results:?}");
        Ok(())
    }
}
