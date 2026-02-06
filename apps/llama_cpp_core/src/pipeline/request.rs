//! Request processing module for pipeline operations
//!
//! ## 架构说明
//!
//! 本模块实现了与 OpenAI API 兼容的请求结构，采用**组合模式**设计：
//!
//! - **标准字段**: 使用 `async_openai::types::CreateChatCompletionRequest` 保证与 OpenAI API 100% 兼容
//! - **扩展字段**: llama.cpp 特有的功能（如会话管理、多模态媒体等）作为扩展字段
//!
//! ### 扩展字段说明（TODO: 后期优化）
//!
//! | 扩展字段 | 用途 | 优化方向 |
//! |---------|------|---------|
//! | `session_id` | 会话隔离与自动历史管理 | 可考虑使用 `metadata` 字段 |
//! | `keep_context` | 控制是否保留会话历史 | 可考虑使用 `metadata` 字段 |
//! | `medias` | 多模态媒体数据（图片/音频）| 标准化为 OpenAI 的 `content` 格式 |
//! | `media_marker` | 媒体占位符标记 | 内部实现细节，对外透明 |
//! | `image_max_resolution` | 图片预处理分辨率限制 | 可作为模型配置参数 |
//! | `history` | 外部历史消息管理 | 标准化为 `messages` 的一部分 |

use llama_cpp_2::{model::LlamaChatMessage, mtmd::mtmd_default_marker};
use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::{
    HistoryMessage,
    error::Error,
    types::{
        ChatCompletionRequestMessage, CreateChatCompletionRequest, MediaData, PromptMessageRole,
    },
    utils::image::Image,
};

/// 生成请求结构体（OpenAI API 兼容 + llama.cpp 扩展）
///
/// ## 设计原则
///
/// 1. **标准兼容**: `standard` 字段完全兼容 OpenAI API，可直接序列化为标准请求
/// 2. **扩展功能**: 本结构体提供 llama.cpp 特有功能（会话管理、媒体处理等）
/// 3. **向后兼容**: 保留原有 Builder API，现有代码无需修改
///
/// ## 使用示例
///
/// ```rust,ignore
/// // 标准 OpenAI 风格
/// let request = GenerateRequest::standard(
///     "gpt-4".to_string(),
///     vec![ChatCompletionRequestUserMessageArgs::default()
///         .content("Hello")
///         .build()?.into()],
/// );
///
/// // llama.cpp 扩展风格
/// let request = GenerateRequest::text("Hello")
///     .with_session_id("user_123")
///     .with_keep_context(true);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    // ============================================================
    // 标准字段（OpenAI API 兼容）
    // 这些字段直接映射到 CreateChatCompletionRequest
    // ============================================================
    /// 标准 OpenAI 请求参数
    ///
    /// 包含: model, messages, temperature, max_tokens, stream 等
    ///
    /// **注意**: 此字段为内部实现，外部通过 builder API 访问
    #[serde(flatten)]
    pub standard: CreateChatCompletionRequest,

    // ============================================================
    // 扩展字段（llama.cpp 特有）
    // TODO: 后期考虑整合到 metadata 或通过其他方式标准化
    // ============================================================
    /// 扩展: 会话ID（用于自动历史管理和并发隔离）
    ///
    /// OpenAI API 无此概念，我们通过 session_id 实现：
    /// - 多用户并发隔离
    /// - 自动历史上下文管理
    /// - 跨请求状态保持
    ///
    /// # TODO 优化方向
    /// 可考虑使用 `metadata` 字段传递，或作为 HTTP Header
    #[serde(default)]
    pub session_id: String,

    /// 扩展: 是否保留上下文历史
    ///
    /// 当 `true` 时，自动将当前对话保存到 session 缓存
    /// 供后续请求使用
    ///
    /// # TODO 优化方向
    /// 与 session_id 合并为会话配置对象
    #[serde(default)]
    pub keep_context: bool,

    /// 扩展: 外部历史消息（优先于 session 缓存）
    ///
    /// 允许调用者完全控制历史管理，绕过内部缓存机制
    ///
    /// # TODO 优化方向
    /// 标准化为 `messages` 字段的一部分，废弃此字段
    #[serde(default)]
    pub history: Option<HistoryMessage>,

    /// 扩展: 多模态媒体数据
    ///
    /// OpenAI 使用 messages[].content 的多模态格式，我们暂时
    /// 使用独立字段存储原始媒体数据，后期需要统一
    ///
    /// # TODO 优化方向
    /// 完全兼容 OpenAI 的 `ChatCompletionRequestMessageContentPartImage` 格式
    #[serde(default)]
    pub medias: Vec<MediaData>,

    /// 扩展: 媒体占位符标记
    ///
    /// 用于在 prompt 中标记媒体位置，如 `<image>`
    ///
    /// # TODO 优化方向
    /// 内部实现细节，对外透明，移除用户可见性
    #[serde(default)]
    pub media_marker: Option<String>,

    /// 扩展: 图片最大分辨率限制
    ///
    /// 防止图片过大导致编码失败或性能问题
    ///
    /// # TODO 优化方向
    /// 作为 PipelineConfig 的模型参数，而非请求参数
    #[serde(default)]
    pub image_max_resolution: u32,

    /// 扩展: 用户提示词（临时兼容字段）
    ///
    /// 在完全迁移到 messages[] 格式前的过渡字段
    ///
    /// # TODO 优化方向
    /// 完全迁移后删除此字段
    #[serde(default)]
    pub user_prompt: String,

    /// 扩展: 系统提示词（临时兼容字段）
    ///
    /// 在完全迁移到 messages[] 格式前的过渡字段
    ///
    /// # TODO 优化方向
    /// 完全迁移后删除此字段，从 messages[] 中提取
    #[serde(default)]
    pub system_prompt: Option<String>,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        let session_id = Uuid::new_v4().to_string();

        Self {
            standard: CreateChatCompletionRequest {
                messages: vec![],
                model: "llama-model".to_string(),
                frequency_penalty: None,
                logit_bias: None,
                logprobs: None,
                max_completion_tokens: None,
                #[allow(deprecated)]
                max_tokens: None,
                n: Some(1),
                presence_penalty: None,
                response_format: None,
                #[allow(deprecated)]
                seed: None,
                stop: None,
                stream: Some(false),
                temperature: Some(0.7),
                top_p: None,
                tools: None,
                tool_choice: None,
                #[allow(deprecated)]
                user: None,
                audio: None,
                modalities: None,
                prediction: None,
                reasoning_effort: None,
                service_tier: None,
                store: None,
                stream_options: None,
                top_logprobs: None,
                web_search_options: None,
                parallel_tool_calls: None,
                verbosity: None,
                safety_identifier: todo!(),
                prompt_cache_key: todo!(),
                function_call: todo!(),
                functions: todo!(),
                metadata: todo!(),
            },
            session_id,
            keep_context: false,
            history: None,
            medias: Vec::new(),
            media_marker: Some("<__media__>".to_string()),
            image_max_resolution: 768,
            user_prompt: String::new(),
            system_prompt: None,
        }
    }
}

impl GenerateRequest {
    // ============================================================
    // 构造函数
    // ============================================================

    /// 创建空请求
    pub fn new() -> Self {
        GenerateRequest::default()
    }

    /// 从标准 OpenAI 请求创建（保留扩展字段默认值）
    pub fn from_standard(standard: CreateChatCompletionRequest) -> Self {
        Self {
            standard,
            ..Default::default()
        }
    }

    /// 创建标准 OpenAI 风格的请求
    ///
    /// 这是最接近原生 OpenAI API 的创建方式
    pub fn standard(model: impl Into<String>, messages: Vec<ChatCompletionRequestMessage>) -> Self {
        let mut request = Self::default();
        request.standard.model = model.into();
        request.standard.messages = messages;
        request
    }

    /// 创建纯文本请求（llama.cpp 简化风格）
    ///
    /// 自动创建单条用户消息，使用 user_prompt 字段
    ///
    /// # TODO 优化
    /// 后期改为直接创建 messages[User]
    pub fn text(user_prompt: impl Into<String>) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            ..Default::default()
        }
    }

    /// 创建多模态请求
    pub fn media(media: MediaData) -> Self {
        Self {
            medias: vec![media],
            ..Default::default()
        }
    }

    // ============================================================
    // 标准字段访问器（OpenAI API 兼容）
    // ============================================================

    /// 获取模型名称
    pub fn model(&self) -> &str {
        &self.standard.model
    }

    /// 设置模型名称
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.standard.model = model.into();
        self
    }

    /// 获取消息列表
    pub fn messages(&self) -> &[ChatCompletionRequestMessage] {
        &self.standard.messages
    }

    /// 设置消息列表
    pub fn with_messages(mut self, messages: Vec<ChatCompletionRequestMessage>) -> Self {
        self.standard.messages = messages;
        self
    }

    /// 获取温度参数
    pub fn temperature(&self) -> f32 {
        self.standard.temperature.unwrap_or(0.7)
    }

    /// 设置温度参数
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.standard.temperature = Some(temperature);
        self
    }

    /// 获取最大 token 数（优先使用新的 max_completion_tokens）
    pub fn max_tokens(&self) -> Option<u32> {
        self.standard
            .max_completion_tokens
            .or(self.standard.max_tokens)
    }

    /// 设置最大 token 数
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.standard.max_completion_tokens = Some(max_tokens);
        self
    }

    /// 是否流式输出
    pub fn is_stream(&self) -> bool {
        self.standard.stream.unwrap_or(false)
    }

    /// 设置流式输出
    pub fn with_stream(mut self, stream: bool) -> Self {
        self.standard.stream = Some(stream);
        self
    }

    // ============================================================
    // 扩展字段访问器（llama.cpp 特有）
    // ============================================================

    /// 设置系统提示词
    ///
    /// # TODO 优化
    /// 改为操作 standard.messages 中的 System 消息
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system_prompt = Some(system.into());
        self
    }

    /// 设置会话ID（扩展功能）
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
        self
    }

    /// 设置历史消息（扩展功能）
    pub fn with_history(mut self, history: HistoryMessage) -> Self {
        self.history = Some(history);
        self
    }

    /// 设置是否保留上下文（扩展功能）
    pub fn with_keep_context(mut self, keep_context: bool) -> Self {
        self.keep_context = keep_context;
        self
    }

    /// 设置媒体标记（扩展功能）
    pub fn with_media_marker(mut self, media_marker: impl Into<String>) -> Self {
        self.media_marker = Some(media_marker.into());
        self
    }

    /// 添加媒体（扩展功能）
    pub fn with_media(mut self, media: MediaData) -> Self {
        self.medias.push(media);
        self
    }

    /// 设置媒体列表（扩展功能）
    pub fn with_medias(mut self, medias: Vec<MediaData>) -> Self {
        self.medias = medias;
        self
    }

    /// 添加媒体缓冲区数据（扩展功能）
    pub fn with_media_buffer(mut self, buf: &[u8]) -> Result<Self, Error> {
        let kind = infer::get(buf).ok_or_else(|| Error::InvalidInput {
            field: "file_type".to_string(),
            message: "Failed to infer file type".to_string(),
        })?;

        if kind.mime_type().starts_with("image/") {
            self.medias.push(MediaData::new_image(buf.to_vec()));
        } else {
            return Err(Error::InvalidInput {
                field: "file_type".to_string(),
                message: format!("Unsupported file type: {}", kind.mime_type()),
            });
        }

        Ok(self)
    }

    /// 设置图片最大分辨率（扩展功能）
    pub fn with_image_max_resolution(mut self, image_max_resolution: u32) -> Self {
        self.image_max_resolution = image_max_resolution;
        self
    }

    /// 添加媒体文件（扩展功能）
    pub fn with_media_file(mut self, path: &str) -> Result<Self, Error> {
        let file_type = infer::get_from_path(path)
            .map_err(|e| {
                error!("Failed to infer file type: {}", e);
                e
            })?
            .ok_or_else(|| Error::InvalidInput {
                field: "file_type".to_string(),
                message: "Failed to infer file type".to_string(),
            })?;

        if file_type.mime_type().starts_with("image/") {
            let media = self.load_image_file(path)?;
            self.medias.push(media);
        } else {
            return Err(Error::InvalidInput {
                field: "file_type".to_string(),
                message: format!("Unsupported file type: {}", file_type.mime_type()),
            });
        }

        Ok(self)
    }

    /// 添加图片 tensor 缓存（扩展功能）
    pub fn load_image_tensor_buffer(
        &self,
        data: Vec<u8>,
        height: u32,
        width: u32,
        channels: usize,
    ) -> Result<MediaData, Error> {
        let mut img = Image::from_tensor(data, height, width, channels)?;
        let mut image_max_resolution = self.image_max_resolution;
        if image_max_resolution < 64 {
            image_max_resolution = 64;
            warn!("image_max_resolution is too small, set to 64");
        }

        let max_resolution = img.longest().min(image_max_resolution);
        img = img.resize_to_longest(max_resolution)?;

        let data = img.to_vec()?;
        let media_data = MediaData::new_image(data);
        Ok(media_data)
    }

    /// Load image from the specified file path（扩展功能）
    pub fn load_image_file(&self, path: &str) -> Result<MediaData, Error> {
        let mut img = Image::from_file(path)?;

        let max_resolution = img.longest().min(self.image_max_resolution);
        img = img.resize_to_longest(max_resolution)?;

        info!(
            "image size width: {}, height: {}, path: {}",
            img.width(),
            img.height(),
            path
        );

        let data = img.to_vec()?;
        Ok(MediaData::new_image(data))
    }

    /// 是否为多模态请求
    pub fn is_multimodal(&self) -> bool {
        !self.medias.is_empty()
    }

    // ============================================================
    // 转换方法
    // ============================================================

    /// 根据请求准备消息（转换为 LlamaChatMessage 列表）
    ///
    /// 将 GenerateRequest 转换为 llama.cpp 内部使用的消息格式
    ///
    /// # TODO 优化方向
    /// 直接处理 standard.messages，而非单独维护历史逻辑
    pub fn to_messages(&self) -> Result<Vec<LlamaChatMessage>, Error> {
        let mut messages = Vec::new();

        // 系统消息 - TODO: 从 standard.messages 中提取
        if let Some(system_prompt) = &self.system_prompt
            && !system_prompt.is_empty()
        {
            messages.push(LlamaChatMessage::new(
                PromptMessageRole::System.to_string(),
                system_prompt.clone(),
            )?);
        }

        // 添加历史消息（外部历史和内部缓存历史二选一，优先外部历史）
        if let Some(history) = &self.history {
            messages.extend(history.to_llama_message()?);
        } else if !self.session_id.is_empty() {
            match HistoryMessage::from_cache(self.session_id.clone()) {
                Ok(history) => messages.extend(history.to_llama_message()?),
                Err(e) => info!("No cached history for session '{}': {}", self.session_id, e),
            }
        }

        // 用户消息：多模态时添加媒体标记
        let user_prompt = if self.is_multimodal() {
            let default_marker = mtmd_default_marker().to_string();
            let media_marker = self.media_marker.as_ref().unwrap_or(&default_marker);
            let markers = media_marker.repeat(self.medias.len());
            format!(
                "{} {}",
                self.user_prompt.replace(&default_marker, ""),
                markers
            )
        } else {
            self.user_prompt.clone()
        };

        if !user_prompt.is_empty() {
            info!("User prompt: {}", user_prompt);
            messages.push(LlamaChatMessage::new(
                PromptMessageRole::User.to_string(),
                user_prompt,
            )?);
        }

        Ok(messages)
    }

    /// 转换为标准 OpenAI 请求
    ///
    /// 用于外部 API 兼容，丢弃 llama.cpp 扩展字段
    pub fn into_standard(self) -> CreateChatCompletionRequest {
        self.standard
    }
}

// ============================================================
// 从标准请求转换
// ============================================================

impl From<CreateChatCompletionRequest> for GenerateRequest {
    fn from(standard: CreateChatCompletionRequest) -> Self {
        Self::from_standard(standard)
    }
}

impl From<GenerateRequest> for CreateChatCompletionRequest {
    fn from(req: GenerateRequest) -> Self {
        req.standard
    }
}
