//! Request processing module for pipeline operations
//!
//! ## 架构说明
//!
//! 本模块使用 `async_openai::CreateChatCompletionRequest` 标准结构

use std::{collections::HashMap, fs};

use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// Re-export async-openai types for OpenAI API compatibility
pub use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionResponseMessage, CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    CreateChatCompletionResponse, FunctionCall, FunctionObject, ImageDetail, ImageUrl, Role,
};

use crate::error::Error;

/// 请求结构别名 - 使用 async-openai 标准请求
pub type Request = CreateChatCompletionRequest;

/// 请求元数据
///
/// 用于从 CreateChatCompletionRequest.metadata 解析自定义字段
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Metadata {
    /// 会话 ID
    pub session_id: Option<String>,
    /// 其他自定义字段
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl From<Metadata> for async_openai::types::Metadata {
    fn from(value: Metadata) -> Self {
        let v = serde_json::to_value(value)
            .map_err(Error::from)
            .unwrap_or_default();

        async_openai::types::Metadata::from(v)
    }
}

impl TryFrom<async_openai::types::Metadata> for Metadata {
    type Error = Error;
    fn try_from(value: async_openai::types::Metadata) -> Result<Self, Self::Error> {
        let v = serde_json::to_value(value)?;
        serde_json::from_value(v).map_err(Error::from)
    }
}

impl TryFrom<&async_openai::types::Metadata> for Metadata {
    type Error = Error;
    fn try_from(value: &async_openai::types::Metadata) -> Result<Self, Self::Error> {
        let v = serde_json::to_value(value)?;
        serde_json::from_value(v).map_err(Error::from)
    }
}

/// 从请求的 metadata 中提取 session_id
pub fn extract_session_id(request: &CreateChatCompletionRequest) -> Option<String> {
    parse_metadata(request)
        .ok()
        .and_then(|meta| meta.map(|m| m.session_id))
        .unwrap_or(None)
}

/// 解析请求 metadata 到 Metadata
pub fn parse_metadata(request: &CreateChatCompletionRequest) -> Result<Option<Metadata>, Error> {
    let metadata = match &request.metadata {
        Some(metadata) => metadata,
        None => return Ok(None),
    };

    let json_value = serde_json::to_value(metadata)?;
    let meta = serde_json::from_value::<Metadata>(json_value)?;
    Ok(Some(meta))
}

/// 用户消息内容构建器
///
/// 用于构建单条用户消息的内容，支持纯文本或多模态（文本+图片）
///
/// # 示例
/// ```
/// use llama_flow::pipeline::{ChatMessagesBuilder, UserMessageBuilder};
///
/// // 纯文本消息
/// let text_only = UserMessageBuilder::new().text("Hello").build();
///
/// // 多模态消息（文本 + 图片）
/// let multimodal = UserMessageBuilder::new()
///     .text("描述这张图片")
///     .image_url("https://example.com/image.jpg")
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct UserMessageBuilder {
    parts: Vec<ChatCompletionRequestUserMessageContentPart>,
}

impl UserMessageBuilder {
    /// 创建新的用户消息构建器
    pub fn new() -> Self {
        Self { parts: Vec::new() }
    }

    /// 添加文本内容
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.parts
            .push(ChatCompletionRequestMessageContentPartText::from(text.into()).into());
        self
    }

    /// 添加图片 URL
    pub fn image_url(mut self, url: impl Into<String>) -> Self {
        self.parts.push(
            ChatCompletionRequestMessageContentPartImage {
                image_url: ImageUrl {
                    url: url.into(),
                    detail: Some(ImageDetail::Auto),
                },
            }
            .into(),
        );
        self
    }

    /// 添加 Base64 图片
    ///
    /// # Arguments
    /// * `mime_type` - MIME 类型，如 "image/png", "image/jpeg"
    /// * `base64_data` - Base64 编码的图片数据
    pub fn image_base64(
        mut self,
        mime_type: impl Into<String>,
        base64_data: impl Into<String>,
    ) -> Self {
        let url = format!("data:{};base64,{}", mime_type.into(), base64_data.into());
        self.parts.push(
            ChatCompletionRequestMessageContentPartImage {
                image_url: ImageUrl {
                    url,
                    detail: Some(ImageDetail::Auto),
                },
            }
            .into(),
        );
        self
    }

    /// 从文件路径添加图片（自动检测 MIME 类型并编码为 Base64）
    #[allow(unused)]
    pub fn image_file(mut self, path: impl AsRef<std::path::Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let data = fs::read(path)?;

        // 尝试推断 MIME 类型
        let mime_type = infer::get_from_path(path)
            .ok()
            .flatten()
            .map(|t| t.mime_type().to_string())
            .unwrap_or_else(|| "image/png".to_string());

        let base64_data = base64::engine::general_purpose::STANDARD.encode(&data);
        Ok(self.image_base64(mime_type, base64_data))
    }

    /// 构建用户消息
    ///
    /// 如果只有一部分且是文本，返回 Text 类型
    /// 否则返回 Array 类型
    pub fn build(self) -> ChatCompletionRequestUserMessage {
        if self.parts.len() == 1 {
            // 检查是否是纯文本
            if let ChatCompletionRequestUserMessageContentPart::Text(text_part) = &self.parts[0] {
                return ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Text(text_part.text.clone()),
                    ..Default::default()
                };
            }
        }

        // 多部分或单图片，使用 Array
        ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(self.parts),
            ..Default::default()
        }
    }
}

/// 聊天消息构建器
///
/// 简化 Vec<ChatCompletionRequestMessage> 的构建
///
/// # 示例
/// ```
/// use llama_flow::pipeline::{ChatMessagesBuilder, UserMessageBuilder};
/// use async_openai::types::chat::CreateChatCompletionRequestArgs;
///
/// let messages = ChatMessagesBuilder::new()
///     .system("You are a helpful assistant.")
///     .user(UserMessageBuilder::new().text("Hello"))
///     .assistant("Hi there!")
///     .user(UserMessageBuilder::new()
///         .text("描述这张图片")
///         .image_url("https://example.com/image.jpg"))
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct ChatMessagesBuilder {
    messages: Vec<ChatCompletionRequestMessage>,
}

impl ChatMessagesBuilder {
    /// 创建新的消息构建器
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// 添加系统消息
    pub fn system(mut self, message: impl Into<String>) -> Self {
        self.messages
            .push(ChatCompletionRequestSystemMessage::from(message.into()).into());
        self
    }

    /// 添加用户消息列表（使用 UserMessageBuilder）
    pub fn users(mut self, builder: UserMessageBuilder) -> Self {
        self.messages.push(builder.build().into());
        self
    }

    /// 添加用户消息
    pub fn user(mut self, message: impl Into<String>) -> Self {
        self.messages
            .push(ChatCompletionRequestUserMessage::from(message.into()).into());
        self
    }

    /// 添加助手消息
    pub fn assistant(mut self, message: impl Into<String>) -> Self {
        self.messages
            .push(ChatCompletionRequestAssistantMessage::from(message.into()).into());
        self
    }

    /// 构建消息列表
    pub fn build(self) -> Vec<ChatCompletionRequestMessage> {
        self.messages
    }
}

impl From<UserMessageBuilder> for ChatCompletionRequestUserMessage {
    fn from(builder: UserMessageBuilder) -> Self {
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_message_builder() {
        // 纯文本 - 应该返回 Text 类型
        let user_msg = UserMessageBuilder::new().text("Hello").build();
        match &user_msg.content {
            ChatCompletionRequestUserMessageContent::Text(text) => {
                assert_eq!(text, "Hello");
            }
            _ => panic!("Expected Text content for single text message"),
        }

        // 多模态 - 应该返回 Array 类型
        let user_msg = UserMessageBuilder::new()
            .text("描述这张图片")
            .image_url("https://example.com/image.jpg")
            .build();
        match &user_msg.content {
            ChatCompletionRequestUserMessageContent::Array(parts) => {
                assert_eq!(parts.len(), 2);
            }
            _ => panic!("Expected Array content for multimodal message"),
        }

        // 单图片 - 应该返回 Array 类型（因为包含图片）
        let user_msg = UserMessageBuilder::new()
            .image_url("https://example.com/image.jpg")
            .build();
        match &user_msg.content {
            ChatCompletionRequestUserMessageContent::Array(parts) => {
                assert_eq!(parts.len(), 1);
            }
            _ => panic!("Expected Array content for image-only message"),
        }
    }

    #[test]
    fn test_chat_messages_builder() {
        // 与原始 API 等价性测试
        let builder_messages = ChatMessagesBuilder::new()
            .system("You are a helpful assistant.")
            .users(UserMessageBuilder::new().text("Who won the world series in 2020?"))
            .assistant("The Los Angeles Dodgers won the World Series in 2020.")
            .users(
                UserMessageBuilder::new()
                    .text("Where was it played?")
                    .image_url("https://example.com/image.png"),
            )
            .build();

        // 期望的消息数量: system + user1 + assistant + user2 = 4
        assert_eq!(builder_messages.len(), 4);

        // 验证第一条用户消息是纯文本
        if let ChatCompletionRequestMessage::User(user_msg) = &builder_messages[1] {
            match &user_msg.content {
                ChatCompletionRequestUserMessageContent::Text(text) => {
                    assert_eq!(text, "Who won the world series in 2020?");
                }
                _ => panic!("First user message should be Text type"),
            }
        } else {
            panic!("Expected User message at index 1");
        }

        // 验证最后一条用户消息是多模态 Array
        if let ChatCompletionRequestMessage::User(user_msg) = &builder_messages[3] {
            match &user_msg.content {
                ChatCompletionRequestUserMessageContent::Array(parts) => {
                    assert_eq!(parts.len(), 2); // 文本 + 图片
                }
                _ => panic!("Last user message should be Array type"),
            }
        } else {
            panic!("Expected User message at index 3");
        }
    }

    #[test]
    fn to_async_openai_metadata() {
        let metadata = Metadata {
            session_id: Some("12345".to_string()),
            ..Default::default()
        };
        let async_metadata: async_openai::types::Metadata = metadata.into();
        println!("{:#?}", async_metadata);
    }
}
