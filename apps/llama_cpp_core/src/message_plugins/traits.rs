//! 消息处理插件系统
//!
//! 定义了消息处理插件的trait和相关功能

use crate::{error::Error, message_plugins::unified_message::UnifiedMessage};

/// 消息处理插件 trait
///
/// 所有消息处理插件都需要实现此 trait
pub trait MessagePlugin: Send + Sync + std::fmt::Debug {
    /// 插件名称
    fn name(&self) -> &str;

    /// 执行消息转换
    ///
    /// # Arguments
    /// * `messages` - 当前消息列表
    /// * `context` - 处理上下文
    ///
    /// # Returns
    /// 转换后的消息列表
    fn process(
        &self,
        messages: Vec<UnifiedMessage>,
        context: &MessageContext,
    ) -> Result<Vec<UnifiedMessage>, Error>;

    /// 插件优先级（数字越小越早执行）
    ///
    /// 默认优先级：
    /// - 60: 标准化插件 (NormalizePlugin)
    /// - 70: 系统消息插件 (SystemPlugin)
    /// - 80: 历史消息插件 (HistoryPlugin)
    /// - 90: 当前输入插件 (CurrentInputPlugin)
    /// - 100: Tool 插件 (ToolsPlugin)
    fn priority(&self) -> i32 {
        50
    }
}

/// 消息处理上下文
/// 在插件链中传递的上下文信息
#[derive(Debug, Clone)]
pub struct MessageContext {
    /// 会话 ID（用于加载历史）
    pub session_id: Option<String>,
    /// 媒体标记（用于多模态）
    pub media_marker: String,
    /// 最大历史消息数
    pub max_history: usize,
    /// 是否启用 Tool 支持
    pub enable_tools: bool,
}

impl Default for MessageContext {
    fn default() -> Self {
        Self {
            session_id: None,
            media_marker: "<image>".to_string(),
            max_history: 100,
            enable_tools: false,
        }
    }
}

impl MessageContext {
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    pub fn with_media_marker(mut self, marker: impl Into<String>) -> Self {
        self.media_marker = marker.into();
        self
    }

    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    pub fn with_tools(mut self, enable: bool) -> Self {
        self.enable_tools = enable;
        self
    }
}
