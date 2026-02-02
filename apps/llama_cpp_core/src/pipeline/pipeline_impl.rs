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
use tracing::info;

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
        if self.config.is_media() {
            self.generate_media(&self.config.clone()).await
        } else {
            self.generate_text(&self.config.clone()).await
        }
    }

    pub async fn generate2(&mut self) -> Result<GenerationOutput, Error> {
        let mut rx = if self.config.is_media() {
            Self::infer_multimodal_stream_blocking_inner(
                &self.config,
                &self.backend,
                &self.history_message,
            )
        } else {
            Self::infer_text_stream_blocking_inner(
                &self.config,
                &self.backend,
                &self.history_message,
            )
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
    /// 返回一个 receiver，每次生成一个 token 时会发送一个 StreamToken
    ///
    /// # Example
    /// ```rust,ignore
    /// let rx = pipeline.infer_stream().await?;
    /// while let Some(token) = rx.recv().await {
    ///     match token {
    ///         StreamToken::Content(text) => print!("{}", text),
    ///         StreamToken::Finish(reason) => println!("\nFinished: {:?}", reason),
    ///         StreamToken::Error(msg) => eprintln!("Error: {}", msg),
    ///     }
    /// }
    /// ```
    pub async fn infer_stream(&self) -> Result<mpsc::UnboundedReceiver<StreamToken>, Error> {
        let config = self.config.clone();
        let backend = self.backend.clone();
        let history_message = self.history_message.clone();

        let rx = if config.is_media() {
            Self::infer_multimodal_stream_blocking_inner(&config, &backend, &history_message)
        } else {
            Self::infer_text_stream_blocking_inner(&config, &backend, &history_message)
        };

        Ok(rx)
    }

    /// 纯文本推理
    async fn generate_text(&mut self, config: &PipelineConfig) -> Result<GenerationOutput, Error> {
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

    /// 媒体内容推理
    async fn generate_media(&mut self, config: &PipelineConfig) -> Result<GenerationOutput, Error> {
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

    /// 纯文本流式推理内部实现（静态方法）
    ///
    /// 使用 ContextWrapper::generate_response_channel 直接获取 channel receiver
    fn infer_text_stream_blocking_inner(
        config: &PipelineConfig,
        backend: &LlamaBackend,
        history_message: &HistoryMessage,
    ) -> mpsc::UnboundedReceiver<StreamToken> {
        // Load model
        let model = Model::from_config(config.clone().into())
            .load_cache_model(backend)
            .expect("Failed to load model");

        // Load sampler
        let mut sampler =
            Sampler::load_sampler(&config.clone().into()).expect("Failed to load sampler");

        // 创建上下文
        let contex_params: ContexParams = config.clone().into();
        let mut ctx = ContextWrapper::try_new(model.clone(), backend, &contex_params)
            .expect("Failed to create context");

        // 创建消息
        let msgs = ContextWrapper::create_message(
            &contex_params,
            false,
            vec![],
            &history_message.messages(),
        )
        .expect("Failed to create message");

        // 评估消息
        ctx.eval_messages(msgs).expect("Failed to eval messages");

        // 使用 channel 方式生成响应
        ctx.generate_response_channel(&mut sampler)
    }

    /// 多模态流式推理内部实现（静态方法）
    fn infer_multimodal_stream_blocking_inner(
        config: &PipelineConfig,
        backend: &LlamaBackend,
        history_message: &HistoryMessage,
    ) -> mpsc::UnboundedReceiver<StreamToken> {
        // Load model
        let model = Model::from_config(config.clone().into())
            .load_cache_model(backend)
            .expect("Failed to load model");

        // Load sampler
        let mut sampler =
            Sampler::load_sampler(&config.clone().into()).expect("Failed to load sampler");

        // 上下文
        let contex_params: ContexParams = config.clone().into();
        let ctx = ContextWrapper::try_new(model.clone(), backend, &contex_params)
            .expect("Failed to create context");
        let mut mtmd_ctx = MtmdContextWrapper::try_new(model.clone(), ctx, &contex_params)
            .expect("Failed to create mtmd context");

        // Load media files
        for media in &config.medias {
            info!("Loading media: {}", media.media_type);
            mtmd_ctx
                .load_media_buffer(&media.data)
                .expect("Failed to load media");
        }

        // 创建消息
        let msgs = ContextWrapper::create_message(
            &contex_params,
            true,
            config.medias.clone(),
            &history_message.messages(),
        )
        .expect("Failed to create message");

        // 评估消息
        mtmd_ctx
            .eval_messages(msgs)
            .expect("Failed to eval messages");

        // 使用 channel 方式生成响应
        mtmd_ctx.generate_response_channel(&mut sampler)
    }

    /// 根据流式生成的结果更新历史消息
    ///
    /// # Arguments
    /// * `full_text` - 流式生成收集到的完整文本
    pub fn update_history_from_stream(&mut self, full_text: &str) -> Result<(), Error> {
        if !self.config.keep_context {
            return Ok(());
        }

        // 添加系统提示（如果历史不为空且已有系统消息）
        if !self.history_message.messages().is_empty() && !self.config.system_prompt.is_empty() {
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
