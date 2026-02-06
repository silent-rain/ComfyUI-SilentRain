//! 统一类型定义
//!
//! 此模块集中定义所有公共类型，避免分散在多个文件中

use llama_cpp_2::context::params::LlamaPoolingType;
use serde::{Deserialize, Serialize};

// Re-export async-openai types for OpenAI API compatibility
pub use async_openai::types::chat::{
    ChatChoice, ChatChoiceLogprobs, ChatChoiceStream, ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta, CompletionTokensDetails, CompletionUsage,
    CreateChatCompletionResponse, CreateChatCompletionStreamResponse, FinishReason,
    PromptTokensDetails, Role,
};

/// 对话消息角色枚举
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PromptMessageRole {
    /// 系统角色（用于初始化或系统级指令）
    System,
    /// 用户角色（人类用户输入）
    User,
    /// AI 助手角色（模型生成的回复）
    Assistant,
    /// 可选：自定义角色（如多AI代理场景）
    Custom(String),
}

impl std::fmt::Display for PromptMessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromptMessageRole::System => write!(f, "System"),
            PromptMessageRole::User => write!(f, "User"),
            PromptMessageRole::Assistant => write!(f, "Assistant"),
            PromptMessageRole::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl PromptMessageRole {
    pub fn custom(role: &str) -> Self {
        Self::Custom(role.to_string())
    }
}

/// 处理模式枚举
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingTypeMode {
    /// 无模式
    None,
    /// 均值模式
    Mean,
    /// 分类模式
    Cls,
    /// 最后模式
    Last,
    /// 排序模式
    Rank,
    /// 未指定模式
    #[default]
    Unspecified,
}

impl std::fmt::Display for PoolingTypeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolingTypeMode::None => write!(f, "None"),
            PoolingTypeMode::Mean => write!(f, "Mean"),
            PoolingTypeMode::Cls => write!(f, "Cls"),
            PoolingTypeMode::Last => write!(f, "Last"),
            PoolingTypeMode::Rank => write!(f, "Rank"),
            PoolingTypeMode::Unspecified => write!(f, "Unspecified"),
        }
    }
}

impl From<PoolingTypeMode> for LlamaPoolingType {
    fn from(value: PoolingTypeMode) -> Self {
        match value {
            PoolingTypeMode::None => LlamaPoolingType::None,
            PoolingTypeMode::Mean => LlamaPoolingType::Mean,
            PoolingTypeMode::Cls => LlamaPoolingType::Cls,
            PoolingTypeMode::Last => LlamaPoolingType::Last,
            PoolingTypeMode::Rank => LlamaPoolingType::Rank,
            PoolingTypeMode::Unspecified => LlamaPoolingType::Unspecified,
        }
    }
}

/// 媒体类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaType {
    Image,
    Audio,
    Video,
}

impl std::fmt::Display for MediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MediaType::Image => write!(f, "image"),
            MediaType::Audio => write!(f, "audio"),
            MediaType::Video => write!(f, "video"),
        }
    }
}

/// 媒体数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaData {
    pub media_type: MediaType,
    pub data: Vec<u8>,
    pub mime_type: Option<String>,
}

impl MediaData {
    pub fn new_image(data: Vec<u8>) -> Self {
        Self {
            media_type: MediaType::Image,
            data,
            mime_type: Some("image/png".to_string()),
        }
    }

    pub fn new_audio(data: Vec<u8>) -> Self {
        Self {
            media_type: MediaType::Audio,
            data,
            mime_type: Some("audio/wav".to_string()),
        }
    }

    pub fn new_video(data: Vec<u8>) -> Self {
        Self {
            media_type: MediaType::Video,
            data,
            mime_type: Some("video/mp4".to_string()),
        }
    }
}

/// 流式响应构建器
/// 用于构建标准的 CreateChatCompletionStreamResponse
#[derive(Debug, Clone)]
pub struct StreamResponseBuilder {
    id: String,
    model: String,
    created: u32,
    index: u32,
}

impl StreamResponseBuilder {
    /// 创建新的流式响应构建器
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            model: model.into(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as u32,
            index: 0,
        }
    }

    /// 设置 choice index
    pub fn with_index(mut self, index: u32) -> Self {
        self.index = index;
        self
    }

    /// 构建内容块响应
    pub fn build_content(&self, content: impl Into<String>) -> CreateChatCompletionStreamResponse {
        let delta = ChatCompletionStreamResponseDelta {
            content: Some(content.into()),
            #[allow(deprecated)]
            function_call: None,
            tool_calls: None,
            role: Some(Role::Assistant),
            refusal: None,
        };

        let choice = ChatChoiceStream {
            index: self.index,
            delta,
            finish_reason: None,
            logprobs: None,
        };

        CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            choices: vec![choice],
            created: self.created,
            model: self.model.clone(),
            service_tier: None,
            #[allow(deprecated)]
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        }
    }

    /// 构建结束响应
    pub fn build_finish(&self, finish_reason: FinishReason) -> CreateChatCompletionStreamResponse {
        self.build_finish_with_usage(finish_reason, 0, 0)
    }

    /// 构建错误响应（将错误信息作为内容发送，然后正常结束）
    pub fn build_error(&self, error_msg: impl Into<String>) -> CreateChatCompletionStreamResponse {
        let delta = ChatCompletionStreamResponseDelta {
            content: Some(format!("\n[Error: {}]", error_msg.into())),
            #[allow(deprecated)]
            function_call: None,
            tool_calls: None,
            role: Some(Role::Assistant),
            refusal: None,
        };

        let choice = ChatChoiceStream {
            index: self.index,
            delta,
            finish_reason: Some(FinishReason::Stop),
            logprobs: None,
        };

        CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            choices: vec![choice],
            created: self.created,
            model: self.model.clone(),
            service_tier: None,
            #[allow(deprecated)]
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        }
    }

    /// 构建带 usage 的最终响应（用于流式传输的最后一条消息，包含完整统计）
    pub fn build_finish_with_usage(
        &self,
        finish_reason: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> CreateChatCompletionStreamResponse {
        let delta = ChatCompletionStreamResponseDelta {
            content: None,
            #[allow(deprecated)]
            function_call: None,
            tool_calls: None,
            role: None,
            refusal: None,
        };

        let choice = ChatChoiceStream {
            index: self.index,
            delta,
            finish_reason: Some(finish_reason),
            logprobs: None,
        };

        let usage = if prompt_tokens > 0 || completion_tokens > 0 {
            Some(CompletionUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            })
        } else {
            None
        };

        CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            choices: vec![choice],
            created: self.created,
            model: self.model.clone(),
            service_tier: None,
            #[allow(deprecated)]
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage,
        }
    }
}

/// 构建标准聊天完成响应
pub fn build_chat_completion_response(
    id: impl Into<String>,
    model: impl Into<String>,
    content: impl Into<String>,
    finish_reason: FinishReason,
) -> CreateChatCompletionResponse {
    build_chat_completion_response_with_usage(id, model, content, finish_reason, None)
}

/// 构建带 usage 统计的标准聊天完成响应
pub fn build_chat_completion_response_with_usage(
    id: impl Into<String>,
    model: impl Into<String>,
    content: impl Into<String>,
    finish_reason: FinishReason,
    usage: Option<CompletionUsage>,
) -> CreateChatCompletionResponse {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as u32;

    let message = ChatCompletionResponseMessage {
        content: Some(content.into()),
        refusal: None,
        role: Role::Assistant,
        audio: None,
        #[allow(deprecated)]
        function_call: None,
        tool_calls: None,
        annotations: None,
    };

    let choice = ChatChoice {
        index: 0,
        message,
        finish_reason: Some(finish_reason),
        logprobs: None,
    };

    CreateChatCompletionResponse {
        id: id.into(),
        choices: vec![choice],
        created,
        model: model.into(),
        service_tier: None,
        #[allow(deprecated)]
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage,
    }
}

/// 从响应中提取内容文本
pub fn chat_completion_response_extract_content(response: &CreateChatCompletionResponse) -> String {
    response
        .choices
        .first()
        .and_then(|c| c.message.content.as_ref())
        .cloned()
        .unwrap_or_default()
}
