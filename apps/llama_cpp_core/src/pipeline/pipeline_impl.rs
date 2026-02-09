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
use base64::{Engine, engine::general_purpose};
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

        // // 系统消息：优先使用请求中的，其次使用配置中的
        // if let Some(instructions) = &request.instructions
        //     && !instructions.is_empty()
        // {
        //     messages.push(LlamaChatMessage::new(
        //         MessageRole::System.to_string(),
        //         instructions.to_string(),
        //     )?);
        // }

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
    ///
    /// # Arguments
    /// * `request` - 生成请求
    ///
    /// # Example
    /// ```rust,ignore
    /// let request = CreateChatCompletionRequest {
    ///     model: "gpt-4".to_string(),
    ///     messages: vec![...],
    ///     ..Default::default()
    /// };
    /// let result = pipeline.generate(&request).await?;
    /// ```
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
                    finish_reason = choice.finish_reason.clone();
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
            self.generate_text_stream(request)?
        } else {
            self.generate_media_stream(request)?
        };

        Ok(rx)
    }

    /// 纯文本流式推理内部实现
    fn generate_text_stream(
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
    fn generate_media_stream(
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
        for image_source in &parsed_input.image_sources {
            // 处理图像来源
            match image_source {
                crate::pipeline::request::ImageSource::Url(_url) => {
                    unimplemented!()
                }
                crate::pipeline::request::ImageSource::Base64(base64_str) => {
                    // data:{};base64,
                    // 提取base64数据部分
                    let base64_data = if base64_str.starts_with("data:") {
                        base64_str.split_once(',').map(|(_, data)| data)
                    } else {
                        Some(base64_str).map(|x| x.as_str())
                    }
                    .ok_or_else(|| {
                        error!("Invalid base64 format");
                        std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "Invalid base64 format",
                        )
                    })?;

                    // 解析
                    info!("Loading media from base64 string");
                    let data = general_purpose::STANDARD.decode(base64_data)?;
                    mtmd_ctx.load_media_buffer(&data).map_err(|e| {
                        error!("Failed to load media: {}", e);
                        e
                    })?;
                }
            }
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
    use crate::{
        types::Request,
        utils::{image::Image, log::init_logger},
    };

    use super::*;

    #[tokio::test]
    async fn test_simple_text() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();

        let pipeline_config = PipelineConfig::new(model_path).with_verbose(true);

        let pipeline = Pipeline::try_new(pipeline_config)?;

        let request = Request::builder().input("你是谁？").build();
        let results = pipeline.generate(&request).await?;

        println!("{:?}", results);
        Ok(())
    }

    #[tokio::test]
    async fn test_simple_vision() -> anyhow::Result<()> {
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

        // 读取图像文件并编码为base64
        let image_path = "/data/cy/00089-915810967.png";
        let mime_type = infer::get_from_path(image_path)?.unwrap().mime_type();
        let base64_data = Image::from_file(image_path)?
            .resize_to_longest(512)?
            .to_base64()?;

        // 反推图片
        let request = Request::builder()
            .input_items(vec![InputItem::message(
                MessageRole::User.to_string(),
                vec![
                    InputItem::content_text("描述这张图片"),
                    InputItem::content_image_base64(&base64_data, mime_type),
                ],
            )])
            .build();
        let results = pipeline.generate(&request).await?;

        println!("{:?}", results);
        Ok(())
    }
}
