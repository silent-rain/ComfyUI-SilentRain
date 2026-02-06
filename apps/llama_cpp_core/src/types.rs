//! 统一类型定义
//!
//! 此模块集中定义所有公共类型，避免分散在多个文件中

use llama_cpp_2::context::params::LlamaPoolingType;
use serde::{Deserialize, Serialize};

// Re-export async-openai types for OpenAI API compatibility
pub use async_openai::types::chat::{
    ChatChoice,
    ChatChoiceLogprobs,
    ChatChoiceStream,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    CompletionTokensDetails,
    CompletionUsage,
    // Standard request types
    CreateChatCompletionRequest,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    FinishReason,
    PromptTokensDetails,
    Role,
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
