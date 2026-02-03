//! 统一类型定义
//!
//! 此模块集中定义所有公共类型，避免分散在多个文件中

use serde::{Deserialize, Serialize};
use strum_macros::{Display, EnumString};

/// 对话消息角色枚举
#[derive(Debug, Clone, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum PromptMessageRole {
    /// 系统角色（用于初始化或系统级指令）
    #[strum(to_string = "System")]
    System,
    /// 用户角色（人类用户输入）
    #[strum(to_string = "User")]
    User,
    /// AI 助手角色（模型生成的回复）
    #[strum(to_string = "Assistant")]
    Assistant,
    /// 可选：自定义角色（如多AI代理场景）
    #[strum(transparent)]
    Custom(String),
}

impl PromptMessageRole {
    pub fn custom(role: &str) -> Self {
        Self::Custom(role.to_string())
    }
}

/// 处理模式枚举
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum PoolingTypeMode {
    /// 无模式
    #[strum(to_string = "None")]
    None,

    /// 均值模式
    #[strum(to_string = "Mean")]
    Mean,

    /// 分类模式
    #[strum(to_string = "Cls")]
    Cls,

    /// 最后模式
    #[strum(to_string = "Last")]
    Last,

    /// 排序模式
    #[strum(to_string = "Rank")]
    Rank,

    /// 未指定模式
    #[strum(to_string = "Unspecified")]
    Unspecified,
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

/// 生成输出
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GenerationOutput {
    pub text: String,
    pub tokens_generated: usize,
    pub finish_reason: FinishReason,
}

impl GenerationOutput {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens_generated: 0,
            finish_reason: FinishReason::Stop,
        }
    }

    pub fn with_tokens(mut self, tokens: usize) -> Self {
        self.tokens_generated = tokens;
        self
    }

    pub fn with_finish_reason(mut self, reason: FinishReason) -> Self {
        self.finish_reason = reason;
        self
    }
}

/// 生成结束原因
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    #[default]
    Stop,
    Length,
    Error,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::Error => write!(f, "error"),
        }
    }
}

/// 生成请求结构体
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// 系统提示词（可选）
    pub system_prompt: Option<String>,
    /// 用户提示词
    pub user_prompt: String,
    /// 历史消息（可选）
    pub history: Vec<llama_cpp_2::model::LlamaChatMessage>,
    /// 媒体数据（多模态场景）
    pub medias: Vec<MediaData>,
}

impl GenerateRequest {
    /// 创建纯文本请求
    pub fn text(user_prompt: impl Into<String>) -> Self {
        Self {
            system_prompt: None,
            user_prompt: user_prompt.into(),
            history: Vec::new(),
            medias: Vec::new(),
        }
    }

    /// 创建多模态请求
    pub fn media(user_prompt: impl Into<String>, medias: Vec<MediaData>) -> Self {
        Self {
            system_prompt: None,
            user_prompt: user_prompt.into(),
            history: Vec::new(),
            medias,
        }
    }

    /// 设置系统提示词
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system_prompt = Some(system.into());
        self
    }

    /// 设置历史消息
    pub fn with_history(mut self, history: Vec<llama_cpp_2::model::LlamaChatMessage>) -> Self {
        self.history = history;
        self
    }

    /// 添加媒体
    pub fn with_media(mut self, media: MediaData) -> Self {
        self.medias.push(media);
        self
    }

    /// 是否为多模态请求
    pub fn is_multimodal(&self) -> bool {
        !self.medias.is_empty()
    }
}

/// 流式生成 Token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamToken {
    /// 生成的文本内容
    Content(String),
    /// 生成完成，包含结束原因
    Finish(FinishReason),
    /// 生成过程中发生错误
    Error(String),
}

impl StreamToken {
    /// 创建内容 Token
    pub fn content(text: impl Into<String>) -> Self {
        Self::Content(text.into())
    }

    /// 创建完成 Token
    pub fn finish(reason: FinishReason) -> Self {
        Self::Finish(reason)
    }

    /// 创建错误 Token
    pub fn error(msg: impl Into<String>) -> Self {
        Self::Error(msg.into())
    }

    /// 获取内容（如果是 Content 类型）
    pub fn as_content(&self) -> Option<&str> {
        match self {
            Self::Content(text) => Some(text),
            _ => None,
        }
    }

    /// 是否为完成 Token
    pub fn is_finish(&self) -> bool {
        matches!(self, Self::Finish(_))
    }

    /// 是否为错误 Token
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}
