//! 统一类型定义
//!
//! 此模块集中定义所有公共类型，避免分散在多个文件中

use llama_cpp_2::context::params::LlamaPoolingType;
use serde::{Deserialize, Serialize};

/// 消息角色枚举 - 清晰明确的角色定义
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageRole {
    /// 系统角色 - 用于系统提示和全局指令
    System,
    /// 用户角色 - 人类用户输入
    #[default]
    User,
    /// 助手角色 - AI 生成的回复
    Assistant,
    /// Tool 角色 - Tool 执行结果
    Tool,
    /// 可选：自定义角色（如多AI代理场景）
    Custom(String),
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
            MessageRole::Tool => write!(f, "tool"),
            MessageRole::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl MessageRole {
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
