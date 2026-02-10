//! 统一消息格式模块
//!
//! 提供简化、扁平化的消息结构，替代复杂的 async_openai 嵌套类型
//! 支持标准对话消息、多模态内容和 Tool 调用

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::{MessageRole, error::Error};

/// 图片来源类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageSource {
    /// HTTP/HTTPS URL
    Url(String),
    /// Base64 编码数据 (mime_type, base64_data)
    Base64(String, String),
}

/// 内容块 - 统一用数组表示，单文本是单元素数组
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentBlock {
    /// 纯文本内容
    Text { text: String },
    /// 图片内容（多模态）
    Image {
        source: ImageSource,
        detail: Option<String>, // auto, low, high
    },
    /// Tool 调用请求（由 Assistant 发出）
    ToolCall {
        id: String,
        name: String,
        arguments: String, // JSON 字符串
    },
    /// Tool 执行结果（Tool 返回给 Assistant）
    ToolResult {
        call_id: String, // 关联 ToolCall.id
        content: String,
    },
}

/// 统一消息结构 - 扁平化设计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedMessage {
    /// 消息角色
    pub role: MessageRole,
    /// 内容块列表（数组格式，文本也是数组）
    pub content: Vec<ContentBlock>,
    /// 名称（Tool 调用时的 tool name）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool 调用 ID（Tool 结果消息需要）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl UnifiedMessage {
    /// 创建系统消息
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: vec![ContentBlock::Text { text: text.into() }],
            name: None,
            tool_call_id: None,
        }
    }
    /// 创建带内容块的系统消息
    pub fn system_with_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::System,
            content: blocks,
            name: None,
            tool_call_id: None,
        }
    }

    /// 创建纯文本用户消息
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text { text: text.into() }],
            name: None,
            tool_call_id: None,
        }
    }

    /// 创建带图片的用户消息
    pub fn user_with_image(text: impl Into<String>, image: ImageSource) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![
                ContentBlock::Text { text: text.into() },
                ContentBlock::Image {
                    source: image,
                    detail: Some("auto".to_string()),
                },
            ],
            name: None,
            tool_call_id: None,
        }
    }

    /// 从多个内容块创建用户消息
    pub fn user_with_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::User,
            content: blocks,
            name: None,
            tool_call_id: None,
        }
    }

    /// 创建助手文本消息
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: vec![ContentBlock::Text { text: text.into() }],
            name: None,
            tool_call_id: None,
        }
    }

    /// 从多个内容块创建助手消息
    pub fn assistant_with_blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: blocks,
            name: None,
            tool_call_id: None,
        }
    }

    /// 创建带 Tool 调用的助手消息
    pub fn assistant_with_tool_calls(calls: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: calls,
            name: None,
            tool_call_id: None,
        }
    }

    /// 创建 Tool 结果消息
    pub fn tool_result(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: vec![ContentBlock::Text {
                text: content.into(),
            }],
            name: None,
            tool_call_id: Some(call_id.into()),
        }
    }

    /// 从多个内容块创建 Tool 结果消息
    pub fn tool_result_with_blocks(call_id: impl Into<String>, blocks: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: blocks,
            name: None,
            tool_call_id: Some(call_id.into()),
        }
    }

    /// 检查是否是纯文本消息
    pub fn is_text_only(&self) -> bool {
        self.content.len() == 1 && matches!(self.content.first(), Some(ContentBlock::Text { .. }))
    }

    /// 获取纯文本内容（如果是纯文本消息）
    pub fn get_text(&self) -> Option<&str> {
        match self.content.first() {
            Some(ContentBlock::Text { text }) if self.content.len() == 1 => Some(text),
            _ => None,
        }
    }

    /// 检查是否包含图片
    pub fn has_image(&self) -> bool {
        self.content
            .iter()
            .any(|block| matches!(block, ContentBlock::Image { .. }))
    }

    /// 获取所有图片来源
    pub fn get_image_sources(&self) -> Vec<&ImageSource> {
        self.content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Image { source, .. } => Some(source),
                _ => None,
            })
            .collect()
    }

    /// 清理内容中的媒体标记
    pub fn sanitize_media_marker(&mut self, marker: &str) {
        for block in &mut self.content {
            if let ContentBlock::Text { text } = block {
                *text = text.replace(marker, "[图片]");
            }
        }
    }

    /// 获取消息中的所有 Tool 调用
    pub fn get_tool_calls(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|block| matches!(block, ContentBlock::ToolCall { .. }))
            .collect()
    }

    /// 转换为 llama.cpp 需要的格式字符串
    pub fn to_llama_format(&self, media_marker: &str) -> Result<String, Error> {
        // 如果是纯文本，直接返回
        if let Some(text) = self.get_text() {
            return Ok(text.to_string());
        }

        // 多内容块，序列化为 JSON 数组格式
        let parts: Vec<serde_json::Value> = self
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text } => {
                    serde_json::json!({"type": "text", "text": text})
                }
                ContentBlock::Image { detail, .. } => {
                    serde_json::json!({
                        "type": "image_url",
                        "image_url": {
                            "url": media_marker,
                            "detail": detail.as_deref().unwrap_or("auto")
                        }
                    })
                }
                ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    serde_json::json!({
                        "type": "tool_call",
                        "tool_call": {
                            "id": id,
                            "name": name,
                            "arguments": arguments
                        }
                    })
                }
                ContentBlock::ToolResult { call_id, content } => {
                    serde_json::json!({
                        "type": "tool_result",
                        "tool_result": {
                            "call_id": call_id,
                            "content": content
                        }
                    })
                }
            })
            .collect();

        serde_json::to_string(&parts).map_err(Error::Serde)
    }
}

impl TryFrom<UnifiedMessage> for llama_cpp_2::model::LlamaChatMessage {
    type Error = Error;

    fn try_from(msg: UnifiedMessage) -> Result<Self, Error> {
        let role_str = msg.role.to_string();
        // 默认使用标准媒体标记
        let content = msg.to_llama_format("<image>")?;

        Self::new(role_str, content).map_err(|e| Error::InvalidInput {
            field: "LlamaChatMessage".to_string(),
            message: e.to_string(),
        })
    }
}

// ========== async_openai 类型转换 ==========

use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestAssistantMessageContentPart, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPart,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart,
};

impl TryFrom<ChatCompletionRequestMessage> for UnifiedMessage {
    type Error = Error;

    fn try_from(msg: ChatCompletionRequestMessage) -> Result<Self, Error> {
        match msg {
            ChatCompletionRequestMessage::System(m) => parse_system_message(m),
            ChatCompletionRequestMessage::User(m) => parse_user_message(m),
            ChatCompletionRequestMessage::Assistant(m) => parse_assistant_message(m),
            ChatCompletionRequestMessage::Tool(m) => parse_tool_message(m),
            _ => Err(Error::InvalidInput {
                field: "message".to_string(),
                message: "Unsupported message type".to_string(),
            }),
        }
    }
}

fn parse_system_message(msg: ChatCompletionRequestSystemMessage) -> Result<UnifiedMessage, Error> {
    let mut blocks = Vec::new();
    match msg.content {
        ChatCompletionRequestSystemMessageContent::Text(t) => {
            blocks.push(ContentBlock::Text { text: t });
        }
        ChatCompletionRequestSystemMessageContent::Array(parts) => {
            for part in parts {
                match part {
                    ChatCompletionRequestSystemMessageContentPart::Text(t) => {
                        blocks.push(ContentBlock::Text { text: t.text });
                    }
                }
            }
        }
    };

    Ok(UnifiedMessage::system_with_blocks(blocks))
}

fn parse_user_message(msg: ChatCompletionRequestUserMessage) -> Result<UnifiedMessage, Error> {
    let mut blocks = Vec::new();
    match msg.content {
        ChatCompletionRequestUserMessageContent::Text(text) => {
            blocks.push(ContentBlock::Text { text });
        }
        ChatCompletionRequestUserMessageContent::Array(parts) => {
            for part in parts {
                match part {
                    ChatCompletionRequestUserMessageContentPart::Text(t) => {
                        blocks.push(ContentBlock::Text { text: t.text });
                    }
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(img) => {
                        let source = extract_image_source(&img.image_url.url);
                        if let Some(source) = source {
                            blocks.push(ContentBlock::Image {
                                source,
                                detail: img.image_url.detail.map(|d| format!("{:?}", d)),
                            });
                        }
                    }
                    _ => {
                        warn!("Unsupported user message part");
                    }
                }
            }
        }
    };

    Ok(UnifiedMessage::user_with_blocks(blocks))
}

fn parse_assistant_message(
    msg: ChatCompletionRequestAssistantMessage,
) -> Result<UnifiedMessage, Error> {
    let mut blocks = Vec::new();

    // TODO: 处理 Tool 调用（需要先确定正确的类型）
    // if let Some(tool_calls) = msg.tool_calls {
    //     for call in tool_calls {
    //         blocks.push(ContentBlock::ToolCall { ... });
    //     }
    // }

    // 添加文本内容
    if let Some(content) = msg.content {
        match content {
            ChatCompletionRequestAssistantMessageContent::Text(t) => {
                blocks.push(ContentBlock::Text { text: t });
            }
            ChatCompletionRequestAssistantMessageContent::Array(parts) => {
                for part in parts {
                    match part {
                        ChatCompletionRequestAssistantMessageContentPart::Text(t) => {
                            blocks.push(ContentBlock::Text { text: t.text });
                        }
                        ChatCompletionRequestAssistantMessageContentPart::Refusal(refusal) => {
                            // 处理拒绝消息
                            blocks.push(ContentBlock::Text {
                                text: refusal.refusal,
                            });
                        }
                    }
                }
            }
        };
    }

    if let Some(refusal) = msg.refusal {
        // 处理拒绝消息
        blocks.push(ContentBlock::Text { text: refusal });
    }

    Ok(UnifiedMessage::assistant_with_blocks(blocks))
}

fn parse_tool_message(msg: ChatCompletionRequestToolMessage) -> Result<UnifiedMessage, Error> {
    let mut blocks = Vec::new();

    match msg.content {
        ChatCompletionRequestToolMessageContent::Text(t) => {
            blocks.push(ContentBlock::ToolResult {
                call_id: msg.tool_call_id.clone(),
                content: t,
            });
        }
        ChatCompletionRequestToolMessageContent::Array(parts) => {
            for part in parts {
                match part {
                    ChatCompletionRequestToolMessageContentPart::Text(t) => {
                        blocks.push(ContentBlock::ToolResult {
                            call_id: msg.tool_call_id.clone(),
                            content: t.text,
                        });
                    }
                }
            }
        }
    }

    Ok(UnifiedMessage::tool_result_with_blocks(
        msg.tool_call_id.clone(),
        blocks,
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_system_message() {
        let msg = UnifiedMessage::system("You are a helpful assistant");
        assert_eq!(msg.role, MessageRole::System);
        assert!(msg.is_text_only());
        assert_eq!(msg.get_text(), Some("You are a helpful assistant"));
    }

    #[test]
    fn test_create_user_text_message() {
        let msg = UnifiedMessage::user_text("Hello");
        assert_eq!(msg.role, MessageRole::User);
        assert!(msg.is_text_only());
        assert!(!msg.has_image());
    }

    #[test]
    fn test_create_multimodal_message() {
        let msg = UnifiedMessage::user_with_image(
            "Describe this",
            ImageSource::Url("https://example.com/image.png".to_string()),
        );
        assert_eq!(msg.role, MessageRole::User);
        assert!(!msg.is_text_only());
        assert!(msg.has_image());
        assert_eq!(msg.get_image_sources().len(), 1);
    }

    #[test]
    fn test_sanitize_media_marker() {
        let mut msg = UnifiedMessage::user_text("See this image: <image> what is it?");
        msg.sanitize_media_marker("<image>");
        assert_eq!(msg.get_text(), Some("See this image: [图片] what is it?"));
    }

    #[test]
    fn test_tool_result_message() {
        let msg = UnifiedMessage::tool_result("call_123", "The weather is sunny");
        assert_eq!(msg.role, MessageRole::Tool);
        assert_eq!(msg.tool_call_id, Some("call_123".to_string()));
    }

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
}
