//! 推理流水线 - 核心推理引擎
//!
//! Pipeline 是 llama_cpp_core 的核心组件，负责：
//! - 模型加载与管理
//! - 文本生成（聊天）
//! - 多模态推理（视觉）
//! - 缓存管理
//! - 上下文管理

use std::sync::Arc;

use async_openai::types::chat::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    FinishReason,
};
use llama_cpp_2::{
    LogOptions, llama_backend::LlamaBackend, model::LlamaChatMessage, send_logs_to_tracing,
};
use tokio::sync::mpsc::UnboundedReceiver;
use tracing::{error, info};

use crate::{
    Backend, HistoryMessage, Model, PipelineConfig, Sampler,
    cache::{CacheType, global_cache},
    context::{ContexParams, ContextWrapper},
    error::Error,
    mtmd_context::MtmdContextWrapper,
    pipeline::{
        ChatStreamBuilder,
        request::{is_multimodal_request, parse_request_input},
    },
    utils::image::{Image, decode_image_sources},
};

/// 推理流水线
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
    pub fn send_logs_to_tracing(enabled: bool) {
        // llama.cpp 日志
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(enabled));
    }

    /// 保存会话历史
    ///
    /// # Arguments
    /// * `session_id` - 会话ID
    /// * `user_prompt` - 用户提示词
    /// * `assistant_response` - 助手回复内容
    fn save_session_history(
        &self,
        session_id: &str,
        user_prompt: &str,
        assistant_response: &str,
    ) -> Result<(), Error> {
        // 尝试加载已有历史，如果不存在则创建新的
        let mut history = match HistoryMessage::from_cache(session_id.to_string()) {
            Ok(existing) => (*existing).clone(),
            Err(_) => HistoryMessage::new(),
        };

        // 添加用户消息
        history.add_user(user_prompt)?;
        // 添加助手回复
        history.add_assistant(assistant_response)?;

        // 保存到缓存
        history.force_update_cache(session_id.to_string())?;

        info!(
            "Session '{}' history saved ({} messages)",
            session_id,
            history.message_count()
        );
        Ok(())
    }

    /// 清除指定会话的历史
    ///
    /// # Arguments
    /// * `session_id` - 会话ID
    pub fn clear_session_history(&self, session_id: &str) -> Result<(), Error> {
        let cache_key = format!("history_message_{}", session_id);
        global_cache().remove(&cache_key)?;
        info!("Session '{}' history cleared", session_id);
        Ok(())
    }

    /// 获取所有会话ID列表
    pub fn list_session_ids(&self) -> Result<Vec<String>, Error> {
        let session_ids = global_cache().get_keys_by_type(CacheType::MessageContext)?;
        Ok(session_ids)
    }
}

impl Pipeline {
    /// 根据请求准备消息
    ///
    /// 转换为 LlamaChatMessage 列表
    pub fn prepare_messages(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<Vec<LlamaChatMessage>, Error> {
        let mut messages = Vec::new();

        // TODO 添加历史消息
        // if let Some(store) = request.store
        //     && store
        // {
        //     messages.extend(request.history.clone());
        // }

        // 解析输入消息
        let parsed_input =
            parse_request_input(request, Some(self.config.context.media_marker.clone()))?;
        messages.extend(parsed_input.messages);

        info!("Prepared messages: {:?}", messages);
        Ok(messages)
    }

    /// 同步推理包装
    pub fn generate_block(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, Error> {
        // 创建 tokio 运行时用于执行异步逻辑
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            error!("Failed to create tokio runtime: {}", e);
            Error::RuntimeError(e.to_string())
        })?;

        let pipeline = Arc::new(self);

        // 在运行时中阻塞执行生成任务
        rt.block_on(async {
            let output = pipeline.generate(request).await?;
            Ok(output)
        })
    }

    /// 执行推理
    pub async fn generate(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, Error> {
        let mut rx = self.generate_stream(request).await?;

        let mut full_text = String::new();
        let mut finish_reason: Option<FinishReason> = None;
        let mut response_model = String::new();
        let mut prompt_tokens: u32 = 0;
        let mut completion_tokens: u32 = 0;

        // 收集所有流式响应
        while let Some(chunk) = rx.recv().await {
            // 保存响应元数据（从第一个 chunk 获取）
            if response_model.is_empty() {
                response_model = chunk.model.clone();
            }

            // 提取 usage（通常在最后一个 chunk）
            if let Some(u) = &chunk.usage {
                prompt_tokens = u.prompt_tokens;
                completion_tokens = u.completion_tokens;
            }

            // 处理 choices
            if let Some(choice) = chunk.choices.first() {
                // 累加文本内容
                if let Some(content) = &choice.delta.content {
                    full_text.push_str(content);
                }
                // 记录结束原因
                if choice.finish_reason.is_some() {
                    finish_reason = choice.finish_reason;
                }
            }
        }

        // 使用 ChatStreamBuilder 构建完整的非流式响应
        let builder = ChatStreamBuilder::new(response_model).with_prompt_tokens(prompt_tokens);
        let response =
            builder.build_non_streaming_response(full_text, finish_reason, completion_tokens);

        Ok(response)
    }

    /// 执行流式推理
    pub async fn generate_stream(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<UnboundedReceiver<CreateChatCompletionStreamResponse>, Error> {
        let rx = if !is_multimodal_request(request) {
            self.generate_text_stream(request).await?
        } else {
            self.generate_media_stream(request).await?
        };

        Ok(rx)
    }

    /// 纯文本流式推理内部实现
    async fn generate_text_stream(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<UnboundedReceiver<CreateChatCompletionStreamResponse>, Error> {
        let msgs = self.prepare_messages(request)?;

        // Load model
        let llama_model = Model::from_config(self.config.model.clone())
            .load_cache_llama_model(&self.backend)
            .map_err(|e| {
                error!("Failed to load model: {}", e);
                e
            })?;

        // Load sampler
        let mut sampler = Sampler::load_sampler(&self.config.sampling.clone()).map_err(|e| {
            error!("Failed to load sampler: {}", e);
            e
        })?;

        // 创建上下文
        let contex_params: ContexParams = self.config.context.clone();
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
        ctx.generate_response(&mut sampler, request.model.clone())
    }

    /// 媒体流式推理内部实现
    async fn generate_media_stream(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<UnboundedReceiver<CreateChatCompletionStreamResponse>, Error> {
        let msgs = self.prepare_messages(request)?;

        // Load model
        let model = Model::from_config(self.config.model.clone());
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
        let mut sampler = Sampler::load_sampler(&self.config.sampling.clone()).map_err(|e| {
            error!("Failed to load sampler: {}", e);
            e
        })?;

        // 上下文
        let contex_params: ContexParams = self.config.context.clone();
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
        let parsed_input =
            parse_request_input(request, Some(self.config.context.media_marker.clone()))?;
        let decoded_images = decode_image_sources(&parsed_input.image_sources).await?;

        // 图片缩放逻辑
        let image_max_resolution = self.config.context.image_max_resolution;

        for data in decoded_images {
            // 从二进制数据创建 Image 对象，并根据配置缩放
            let img = Image::from_bytes(&data)?.resize_with_max_resolution(image_max_resolution)?;
            img.save("/home/one/Downloads/resized.jpg")?;

            // 将缩放后的图像转换为字节数据
            let resized_data = img.to_vec()?;

            mtmd_ctx.load_media_buffer(&resized_data).map_err(|e| {
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
        mtmd_ctx.generate_response(&mut sampler, request.model.clone())
    }
}

#[cfg(test)]
mod tests {
    use async_openai::types::chat::CreateChatCompletionRequestArgs;
    use base64::{Engine, engine::general_purpose};

    use crate::{
        pipeline::{ChatMessagesBuilder, UserMessageBuilder},
        utils::log::init_logger,
    };

    use super::*;

    #[tokio::test]
    async fn test_simple_text() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();

        let pipeline_config = PipelineConfig::new(model_path).with_verbose(false);

        let pipeline = Pipeline::try_new(pipeline_config)?;

        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(2048u32)
            .model("Qwen3-VL-2B-Instruct")
            .messages(
                ChatMessagesBuilder::new()
                    .system("You are a helpful assistant.")
                    .user("Who won the world series in 2020?")
                    .assistant("The Los Angeles Dodgers won the World Series in 2020.")
                    .users(UserMessageBuilder::new().text("Where was it played?"))
                    .build(),
            )
            .build()?;

        info!("{}", serde_json::to_string(&request).unwrap());

        let results = pipeline.generate(&request).await?;

        info!("{:?}", results);
        Ok(())
    }

    #[tokio::test]
    async fn test_simple_vision_for_image_base64() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf"
                .to_string();
        let mmproj_path =
            "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.mmproj-Q8_0.gguf"
                .to_string();

        let pipeline_config =
            PipelineConfig::new_with_mmproj(model_path, mmproj_path).with_verbose(false);

        let pipeline = Pipeline::try_new(pipeline_config)?;

        // 读取图像文件并编码为base64
        let image_url = "/data/cy/rgthree.compare._temp_glbsl_00013_.png";
        let mime_type = infer::get_from_path(image_url)?.unwrap().mime_type();
        let buffer = std::fs::read(image_url)?;
        let base64_data = general_purpose::STANDARD.encode(&buffer);

        // 反推图片
        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(2048u32)
            .model("Qwen3-VL-2B-Instruct")
            .messages(
                ChatMessagesBuilder::new()
                    .system("You are a helpful assistant.")
                    .users(
                        UserMessageBuilder::new()
                            .text("描述这张图片")
                            .image_base64(mime_type, base64_data),
                    )
                    .build(),
            )
            .build()?;

        let results = pipeline.generate(&request).await?;

        info!("{:?}", results);
        Ok(())
    }

    #[tokio::test]
    async fn test_simple_vision_for_image_url() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf"
                .to_string();
        let mmproj_path =
            "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.mmproj-Q8_0.gguf"
                .to_string();

        let pipeline_config =
            PipelineConfig::new_with_mmproj(model_path, mmproj_path).with_verbose(true);

        let pipeline = Pipeline::try_new(pipeline_config)?;

        // 反推图片
        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(2048u32)
            .model("Qwen3-VL-2B-Instruct")
            .messages(
                ChatMessagesBuilder::new()
                .system("You are a helpful assistant.") 
                .users(
                    UserMessageBuilder::new()
                        .text("描述这张图片")
                        .image_url("https://muse-ai.oss-cn-hangzhou.aliyuncs.com/img/ffdebd6731594c7fbef751944dddf1c0.jpeg"),
                )
                .build()
            )
            .build()?;

        let results = pipeline.generate(&request).await?;

        info!("{:?}", results);
        Ok(())
    }
}
