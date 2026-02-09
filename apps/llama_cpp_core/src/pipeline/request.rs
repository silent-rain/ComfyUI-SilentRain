//! Request processing module for pipeline operations
//!
//! ## 架构说明
//!
//! 本模块使用 `async_openai::CreateChatCompletionRequest` 标准结构

use std::fs;

use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestAssistantMessageContentPart, ChatCompletionRequestMessage,
    ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    CreateChatCompletionRequest, ImageDetail, ImageUrl,
};
use base64::Engine;
use llama_cpp_2::{model::LlamaChatMessage, mtmd::mtmd_default_marker};
use serde_json::json;

use crate::{error::Error, types::MessageRole};

// 导出 async-openai 类型
pub use async_openai::types::chat::{
    ChatCompletionResponseMessage, CreateChatCompletionResponse, FunctionCall, FunctionObject,
};

/// 图片来源
#[derive(Debug, Clone)]
pub enum ImageSource {
    /// URL（http/https）
    Url(String),
    /// Base64 数据，包含 (media_type, base64_data)
    Base64(String, String),
}

/// 解析后的输入结果
#[derive(Debug, Clone)]
pub struct ParsedInput {
    /// 转换后的消息列表
    pub messages: Vec<LlamaChatMessage>,
    /// 提取的图片源（供后续下载/编码）
    pub image_sources: Vec<ImageSource>,
}

/// 检查请求是否包含图片（多模态请求）
pub fn is_multimodal_request(request: &CreateChatCompletionRequest) -> bool {
    for msg in request.messages.clone() {
        if let ChatCompletionRequestMessage::User(user_msg) = msg
            && let ChatCompletionRequestUserMessageContent::Array(parts) = &user_msg.content
        {
            for part in parts {
                if matches!(
                    part,
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(_)
                ) {
                    return true;
                }
            }
        }
    }
    false
}

/// 解析 Request 的消息
///
/// 处理标准的 `CreateChatCompletionRequest`，将消息转换为 LlamaChatMessage 列表
/// 支持多模态：图片内容会被提取，并在对应用户消息中添加媒体标记
///
/// # Arguments
/// * `request` - 标准 Chat Completions 请求
/// * `media_marker` - 媒体标记（用于多模态，默认使用 mtmd 默认标记）
pub fn parse_request_input(
    request: &CreateChatCompletionRequest,
    media_marker: Option<impl Into<String>>,
) -> Result<ParsedInput, Error> {
    let mut messages = Vec::new();
    let mut image_sources = Vec::new();

    let media_marker = media_marker
        .map(|s| s.into())
        .unwrap_or(mtmd_default_marker().to_string());

    for msg in request.messages.clone() {
        match msg {
            // 系统消息
            ChatCompletionRequestMessage::System(system_msg) => {
                let content = extract_system_content(&system_msg.content);
                if !content.is_empty() {
                    messages.push(LlamaChatMessage::new(
                        MessageRole::System.to_string(),
                        content,
                    )?);
                }
            }
            // 用户消息：支持文本和多模态（文本+图片）
            ChatCompletionRequestMessage::User(user_msg) => {
                let (user_text, img_sources) =
                    parse_user_message_content(&user_msg.content, &media_marker)?;

                if !user_text.is_empty() {
                    // 添加用户消息（包含媒体标记）
                    messages.push(LlamaChatMessage::new(
                        MessageRole::User.to_string(),
                        user_text,
                    )?);

                    // 收集图片来源
                    image_sources.extend(img_sources);
                }
            }
            // 助手消息
            ChatCompletionRequestMessage::Assistant(assistant_msg) => {
                if let Some(content) = &assistant_msg.content {
                    let text = extract_assistant_content(content);
                    if !text.is_empty() {
                        messages.push(LlamaChatMessage::new(
                            MessageRole::Assistant.to_string(),
                            text,
                        )?);
                    }
                }
            }
            // 工具/函数消息：暂不支持，记录日志后跳过
            _ => {
                tracing::debug!("Skipping unsupported message type in request parsing");
            }
        }
    }

    Ok(ParsedInput {
        messages,
        image_sources,
    })
}

/// 解析用户消息内容
///
/// 处理文本或多模态内容（文本+图片数组）
/// 多模态时，在文本后追加媒体标记（参考 llama.cpp mtmd 规范）
///
/// # Returns
/// - `(用户消息文本, 图片来源列表)`
fn parse_user_message_content(
    content: &ChatCompletionRequestUserMessageContent,
    media_marker: &str,
) -> Result<(String, Vec<ImageSource>), Error> {
    let mut contents = Vec::new();
    let mut image_sources = Vec::new();

    match content {
        // 纯文本消息
        ChatCompletionRequestUserMessageContent::Text(text) => {
            contents.push(json!({
                "type": "text",
                "text": text.replace(media_marker, "").to_string(),
            }));
        }
        // 多模态消息：文本 + 图片数组
        ChatCompletionRequestUserMessageContent::Array(parts) => {
            for part in parts {
                match part {
                    ChatCompletionRequestUserMessageContentPart::Text(text_part) => {
                        // 清理文本中可能存在的默认标记（避免重复）
                        let clean_text = text_part.text.replace(media_marker, "");
                        contents.push(json!({
                            "type": "text",
                            "text": clean_text.to_string(),
                        }));
                    }
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part) => {
                        // 提取图片来源
                        if let Some(source) = extract_image_source(&image_part.image_url.url) {
                            contents.push(json!({
                                "type": "image_url",
                                "image_url": {
                                    "url": media_marker,
                                    "detail": image_part.image_url.detail
                                }
                            }));
                            image_sources.push(source);
                        }
                    }
                    _ => {} // 忽略其他类型
                }
            }
        }
    }

    let user_text = serde_json::to_string(&contents)?;
    Ok((user_text, image_sources))
}

/// 提取系统消息内容
fn extract_system_content(content: &ChatCompletionRequestSystemMessageContent) -> String {
    let contents = match content {
        ChatCompletionRequestSystemMessageContent::Text(text) => {
            ChatCompletionRequestSystemMessageContent::Array(vec![
                ChatCompletionRequestSystemMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: text.to_string(),
                    },
                ),
            ])
        }
        ChatCompletionRequestSystemMessageContent::Array(parts) => {
            ChatCompletionRequestSystemMessageContent::Array(parts.to_vec())
        }
    };

    serde_json::to_string(&contents).unwrap_or_default()
}

/// 提取助手消息内容
fn extract_assistant_content(content: &ChatCompletionRequestAssistantMessageContent) -> String {
    let contents = match content {
        ChatCompletionRequestAssistantMessageContent::Text(text) => {
            ChatCompletionRequestAssistantMessageContent::Array(vec![
                ChatCompletionRequestAssistantMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText {
                        text: text.to_string(),
                    },
                ),
            ])
        }
        ChatCompletionRequestAssistantMessageContent::Array(parts) => {
            ChatCompletionRequestAssistantMessageContent::Array(parts.to_vec())
        }
    };

    serde_json::to_string(&contents).unwrap_or_default()
}

/// 从图片 URL 提取图片来源
fn extract_image_source(url: &str) -> Option<ImageSource> {
    if url.starts_with("http") || url.starts_with("https") {
        Some(ImageSource::Url(url.to_string()))
    } else if url.starts_with("data:") {
        // 从 data URI 中提取 media_type 和 base64 数据
        extract_base64_from_data_uri(url)
            .map(|(media_type, base64_data)| ImageSource::Base64(media_type, base64_data))
    } else {
        None
    }
}

/// 从 data URI 中提取 media_type 和 base64 数据
///
/// data URI 格式: data:[<mediatype>][;base64],<data>
/// 示例: data:image/png;base64,iVBORw0KGgo...
///
/// # Returns
/// - `Some((media_type, base64_data))` - 成功提取媒体类型和 base64 数据
fn extract_base64_from_data_uri(data_uri: &str) -> Option<(String, String)> {
    if !data_uri.starts_with("data:") {
        return None;
    }

    // 找到逗号的位置，逗号后面才是真正的 base64 数据
    data_uri.find(',').map(|comma_pos| {
        let base64_data = data_uri[comma_pos + 1..].to_string();
        // 解析媒体类型部分（data: 和 , 之间的内容）
        let media_part = &data_uri[5..comma_pos];

        // 分割 ; 来获取媒体类型和参数
        let parts: Vec<&str> = media_part.split(';').collect();

        let media_type = if parts.is_empty() || parts[0].is_empty() || parts[0] == "base64" {
            // 没有指定媒体类型，使用默认值
            "application/octet-stream".to_string()
        } else {
            parts[0].to_string()
        };

        (media_type, base64_data)
    })
}

/// 聊天消息构建器
///
/// 简化 Vec<ChatCompletionRequestMessage> 的构建，支持文本和图片
///
/// # 示例
/// ```
/// use llama_cpp_core::pipeline::ChatMessagesBuilder;
/// use async_openai::types::chat::CreateChatCompletionRequestArgs;
///
/// let messages = ChatMessagesBuilder::new()
///     .system("You are a helpful assistant.")
///     .user_text("描述这张图片")
///     .image_url("https://example.com/image.jpg")
///     .build();
///
/// let request = CreateChatCompletionRequestArgs::default()
///     .model("Qwen3-VL-2B-Instruct")
///     .messages(messages)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ChatMessagesBuilder {
    messages: Vec<ChatCompletionRequestMessage>,
    current_user_parts: Vec<ChatCompletionRequestUserMessageContentPart>,
}

impl ChatMessagesBuilder {
    /// 创建新的消息构建器
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            current_user_parts: Vec::new(),
        }
    }

    /// 添加系统消息
    pub fn system(mut self, message: impl Into<String>) -> Self {
        self.messages
            .push(ChatCompletionRequestSystemMessage::from(message.into()).into());
        self
    }

    /// 添加用户文本消息
    pub fn user_text(mut self, text: impl Into<String>) -> Self {
        self.current_user_parts
            .push(ChatCompletionRequestMessageContentPartText::from(text.into()).into());
        self
    }

    /// 添加图片 URL
    pub fn image_url(mut self, url: impl Into<String>) -> Self {
        self.current_user_parts.push(
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
        self.current_user_parts.push(
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

    /// 添加助手消息
    pub fn assistant(mut self, message: impl Into<String>) -> Self {
        self.messages
            .push(ChatCompletionRequestAssistantMessage::from(message.into()).into());
        self
    }

    /// 将当前用户部分内容刷新到消息列表
    fn flush_user_parts(&mut self) {
        if !self.current_user_parts.is_empty() {
            self.messages.push(
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Array(
                        self.current_user_parts.clone(),
                    ),
                    ..Default::default()
                }
                .into(),
            );
        }
    }

    /// 构建消息列表
    pub fn build(mut self) -> Vec<ChatCompletionRequestMessage> {
        self.flush_user_parts();
        self.messages
    }
}

impl Default for ChatMessagesBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use async_openai::types::chat::{
        ChatCompletionRequestMessageContentPartImage, ImageDetail, ImageUrl,
    };

    use super::*;

    #[test]
    fn test_extract_base64_from_data_uri() {
        // 标准 data URI
        let data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let (media_type, base64_data) = extract_base64_from_data_uri(data_uri).unwrap();
        assert_eq!(media_type, "image/png");
        assert_eq!(
            base64_data,
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        );

        // 不带 media type 的 data URI
        let data_uri2 = "data:base64,SGVsbG8gV29ybGQ=";
        let (media_type2, base64_data2) = extract_base64_from_data_uri(data_uri2).unwrap();
        assert_eq!(media_type2, "application/octet-stream");
        assert_eq!(base64_data2, "SGVsbG8gV29ybGQ=");

        // 普通 URL 应该返回 None
        let url = "https://example.com/image.png";
        assert!(extract_base64_from_data_uri(url).is_none());

        // 普通 base64 字符串应该返回 None
        let plain_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        assert!(extract_base64_from_data_uri(plain_base64).is_none());
    }

    #[test]
    fn test_extract_system_content() {
        let system_content = ChatCompletionRequestSystemMessageContent::Text(
            "You are a system assistant".to_string(),
        );
        let result = extract_system_content(&system_content);
        println!("{:?}", result);

        let system_content = ChatCompletionRequestSystemMessageContent::Array(vec![
            ChatCompletionRequestSystemMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText {
                    text: "You are a system assistant".to_string(),
                },
            ),
        ]);
        let result = extract_system_content(&system_content);
        println!("{:?}", result);
    }

    #[test]
    fn test_extract_assistant_content() {
        let assistant_content = ChatCompletionRequestAssistantMessageContent::Text(
            "You are a helpful assistant".to_string(),
        );
        let result = extract_assistant_content(&assistant_content);
        println!("{:?}", result);

        let assistant_content = ChatCompletionRequestAssistantMessageContent::Array(vec![
            ChatCompletionRequestAssistantMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText {
                    text: "You are a helpful assistant".to_string(),
                },
            ),
        ]);
        let result = extract_assistant_content(&assistant_content);
        println!("{:?}", result);
    }

    #[test]
    fn test_parse_user_message_content() -> anyhow::Result<()> {
        let media_marker = mtmd_default_marker().to_string();

        let user_content =
            ChatCompletionRequestUserMessageContent::Text("Hello, how can I help you?".to_string());
        let (user_text, _media_sources) = parse_user_message_content(&user_content, &media_marker)?;
        println!("{:?}", user_text);

        let user_content = ChatCompletionRequestUserMessageContent::Array(vec![
            ChatCompletionRequestUserMessageContentPart::Text(
                ChatCompletionRequestMessageContentPartText {
                    text: "Hello, how can I help you?".to_string(),
                },
            ),
            ChatCompletionRequestUserMessageContentPart::ImageUrl(
                ChatCompletionRequestMessageContentPartImage {
                    image_url:ImageUrl {url:"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==".to_string(), detail: Some(ImageDetail::Auto) },
                },
            ),
        ]);
        let (user_text, _media_sources) = parse_user_message_content(&user_content, &media_marker)?;
        println!("{:?}", user_text);

        Ok(())
    }

    #[test]
    fn test_chat_messages_builder() {
        // 测试纯文本
        let messages = ChatMessagesBuilder::new()
            .system("You are a helpful assistant.")
            .user_text("Hello")
            .build();

        assert_eq!(messages.len(), 2);

        // 测试多模态
        let messages = ChatMessagesBuilder::new()
            .system("You are a helpful assistant.")
            .user_text("描述这张图片")
            .image_url("https://example.com/image.jpg")
            .build();

        assert_eq!(messages.len(), 2);

        // 测试 Base64 图片
        let messages = ChatMessagesBuilder::new()
            .user_text("描述这张图片")
            .image_base64("image/png", "iVBORw0KGgo=")
            .build();

        assert_eq!(messages.len(), 1);

        // 测试多轮对话
        let messages = ChatMessagesBuilder::new()
            .system("You are a helpful assistant.")
            .user_text("Hello")
            .assistant("Hi there!")
            .user_text("描述这张图片")
            .image_url("https://example.com/image.jpg")
            .build();

        assert_eq!(messages.len(), 3);
    }
}
