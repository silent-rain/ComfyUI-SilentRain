//! 统一消息格式模块
//!
//! 提供简化、扁平化的消息结构，替代复杂的 async_openai 嵌套类型
//! 支持标准对话消息、多模态内容和 Tool 调用

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::{error::Error, types::MessageRole, utils::image::extract_image_source};

/// 图片来源类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ImageSource {
    /// HTTP/HTTPS URL
    Url(String),
    /// Base64 编码数据 (mime_type, base64_data)
    Base64(String, String),
}

/// 内容块 - 统一用数组表示，单文本是单元素数组
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
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
    pub fn user(text: impl Into<String>) -> Self {
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
    pub fn sanitize_media_marker(&self) -> Vec<ContentBlock> {
        self.content
            .clone()
            .iter()
            .map(|block| {
                if let ContentBlock::Image { detail, .. } = block {
                    ContentBlock::Image {
                        source: ImageSource::Url("[image]".to_string()),
                        detail: detail.clone(),
                    }
                } else {
                    block.clone()
                }
            })
            .collect()
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

// ========== async_openai 类型转换 ==========

use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestAssistantMessageContentPart, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPart,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest,
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

/// 解析系统消息，支持文本内容
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

/// 解析用户消息，支持文本和图片（图片需要提取 URL 作为来源）
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

/// 解析助手消息，支持文本和 Tool 调用（Tool 调用需要根据实际结构调整）
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

/// 解析 Tool 消息（目前仅处理文本结果，Tool 调用消息需要根据实际结构调整）
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

/// 从请求中解析媒体资源
pub fn extract_media_sources_from_request(
    request: &CreateChatCompletionRequest,
) -> Result<Vec<ImageSource>, Error> {
    let mut sources = Vec::new();

    for msg in &request.messages {
        let unified_msg = UnifiedMessage::try_from(msg.clone())?;
        sources.extend(unified_msg.get_image_sources().into_iter().cloned());
    }

    Ok(sources)
}

/// 检查请求是否包含图片（多模态请求）
pub fn is_multimodal_request(request: &CreateChatCompletionRequest) -> bool {
    for msg in &request.messages {
        let unified_msg = UnifiedMessage::try_from(msg.clone());
        if let Ok(msg) = unified_msg
            && msg.has_image()
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use async_openai::types::chat::CreateChatCompletionRequestArgs;
    use llama_cpp_2::{model::LlamaChatMessage, mtmd::mtmd_default_marker};

    use super::*;
    use crate::request::{ChatMessagesBuilder, UserMessageBuilder};

    #[test]
    fn test_create_system_message() {
        let msg = UnifiedMessage::system("You are a helpful assistant");
        assert_eq!(msg.role, MessageRole::System);
    }

    #[test]
    fn test_create_user_text_message() {
        let msg = UnifiedMessage::user("Hello");
        assert_eq!(msg.role, MessageRole::User);
        assert!(!msg.has_image());
    }

    #[test]
    fn test_create_multimodal_message() {
        let msg = UnifiedMessage::user_with_image(
            "Describe this",
            ImageSource::Url("https://example.com/image.png".to_string()),
        );
        assert_eq!(msg.role, MessageRole::User);
        assert!(msg.has_image());
        assert_eq!(msg.get_image_sources().len(), 1);
    }

    #[test]
    fn test_sanitize_media_marker() {
        let default_marker = mtmd_default_marker().to_string();
        let msg = UnifiedMessage::user_with_blocks(vec![
            ContentBlock::Text {
                text: "See this image what is it?".to_string(),
            },
            ContentBlock::Image {
                source: ImageSource::Url(default_marker.clone()),
                detail: None,
            },
        ]);

        let new_content = msg.sanitize_media_marker();
        println!("{:#?}", new_content);
    }

    #[test]
    fn test_tool_result_message() {
        let msg = UnifiedMessage::tool_result("call_123", "The weather is sunny");
        assert_eq!(msg.role, MessageRole::Tool);
        assert_eq!(msg.tool_call_id, Some("call_123".to_string()));
    }

    #[test]
    fn test_chat_completion_request_message_to_unified_message() -> anyhow::Result<()> {
        let request = CreateChatCompletionRequestArgs::default()
                    .max_completion_tokens(2048u32)
                    .model("Qwen3-VL-2B-Instruct")
                    .messages(ChatMessagesBuilder::new()
                        .system("你是专注生成套图模特提示词专家，用于生成9个同人物，同场景，同服装，不同的模特照片，需要保持专业性。")
                        .users(
                            UserMessageBuilder::new()
                                .text("描述这张图片")
                                .image_url("https://muse-ai.oss-cn-hangzhou.aliyuncs.com/img/ffdebd6731594c7fbef751944dddf1c0.jpeg"),
                        )
                    .build())
                    .build()?;

        println!("request: {:#?}", serde_json::to_string(&request)?);

        // 将请求消息转换为 UnifiedMessage
        let unified_msg: Vec<UnifiedMessage> = request
            .messages
            .iter()
            .cloned()
            .map(UnifiedMessage::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        println!("unified_msg: {:#?}", serde_json::to_string(&unified_msg)?);

        Ok(())
    }

    #[test]
    fn test_unified_message_to_llama_messages() -> anyhow::Result<()> {
        let request = CreateChatCompletionRequestArgs::default()
                    .max_completion_tokens(2048u32)
                    .model("Qwen3-VL-2B-Instruct")
                    .messages(ChatMessagesBuilder::new()
                        .system("你是专注生成套图模特提示词专家，用于生成9个同人物，同场景，同服装，不同的模特照片，需要保持专业性。")
                        .users(
                            UserMessageBuilder::new()
                                .text("描述这张图片")
                                .image_url("https://muse-ai.oss-cn-hangzhou.aliyuncs.com/img/ffdebd6731594c7fbef751944dddf1c0.jpeg"),
                        )
                    .build())
                    .build()?;

        println!("request: {:#?}\n", serde_json::to_string(&request)?);

        // 将请求消息转换为 UnifiedMessage
        let unified_msg: Vec<UnifiedMessage> = request
            .messages
            .iter()
            .cloned()
            .map(UnifiedMessage::try_from)
            .collect::<Result<Vec<_>, _>>()?;

        println!("unified_msg: {:#?}\n", serde_json::to_string(&unified_msg)?);

        // 5. 转换为 LlamaChatMessage
        let media_marker = mtmd_default_marker().to_string();
        let llama_messages: Vec<LlamaChatMessage> = unified_msg
            .into_iter()
            .map(|msg| {
                let content = msg.to_llama_format(&media_marker)?;
                LlamaChatMessage::new(msg.role.to_string(), content).map_err(|e| {
                    Error::InvalidInput {
                        field: "LlamaChatMessage".to_string(),
                        message: e.to_string(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        println!("llama_messages: {:#?}\n", llama_messages);
        Ok(())
    }

    #[test]
    fn test_extract_media_sources_from_request() -> anyhow::Result<()> {
        let request = CreateChatCompletionRequestArgs::default()
                    .max_completion_tokens(2048u32)
                    .model("Qwen3-VL-2B-Instruct")
                    .messages(ChatMessagesBuilder::new()
                        .system("你是专注生成套图模特提示词专家，用于生成9个同人物，同场景，同服装，不同的模特照片，需要保持专业性。")
                        .users(
                            UserMessageBuilder::new()
                                .text("描述这张图片")
                                .image_url("https://muse-ai.oss-cn-hangzhou.aliyuncs.com/img/ffdebd6731594c7fbef751944dddf1c0.jpeg")
                                .image_url("https://muse-ai.oss-cn-hangzhou.aliyuncs.com/img/ffdebd6731594c7fbef751944dddf1c0.jpeg"),
                        )
                    .build())
                    .build()?;

        let media_sources = extract_media_sources_from_request(&request)?;
        println!("Extracted media sources: {:#?}", media_sources);
        Ok(())
    }
}
