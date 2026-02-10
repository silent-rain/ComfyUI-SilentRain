//! 历史消息插件

use crate::{
    HistoryMessage, MessageRole,
    error::Error,
    message_plugins::{MessageContext, MessagePlugin, unified_message::UnifiedMessage},
};

/// 历史消息插件
///
/// 负责：
/// 1. 从缓存加载历史消息
/// 2. 清理历史消息中的媒体标记
/// 3. 将历史消息插入到系统消息之后、当前消息之前
#[derive(Debug)]
pub struct HistoryPlugin {
    max_history: usize,
}

impl Default for HistoryPlugin {
    fn default() -> Self {
        Self { max_history: 100 }
    }
}

impl HistoryPlugin {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// 从历史记录创建统一消息
    ///
    /// 过滤掉系统消息（由 SystemPlugin 处理）
    /// 清理媒体标记为描述文本
    fn history_to_unified(
        &self,
        history: &HistoryMessage,
        media_marker: &str,
    ) -> Vec<UnifiedMessage> {
        let entries = history.entries();
        // 只取最近的 N 条
        let start_idx = entries.len().saturating_sub(self.max_history);

        entries[start_idx..]
            .iter()
            .filter(|entry| entry.role != MessageRole::System)
            .cloned()
            .map(|mut msg| {
                // 清理媒体标记
                msg.sanitize_media_marker(media_marker);
                msg
            })
            .collect()
    }
}

impl MessagePlugin for HistoryPlugin {
    fn name(&self) -> &str {
        "HistoryPlugin"
    }

    fn priority(&self) -> i32 {
        80
    }

    fn process(
        &self,
        messages: Vec<UnifiedMessage>,
        context: &MessageContext,
    ) -> Result<Vec<UnifiedMessage>, Error> {
        // 如果没有 session_id，直接返回
        let session_id = match &context.session_id {
            Some(id) => id,
            None => return Ok(messages),
        };

        // 尝试加载历史
        let history = match HistoryMessage::from_cache(session_id.clone()) {
            Ok(h) => h,
            Err(_) => return Ok(messages), // 没有历史记录，直接返回
        };

        // 转换历史消息
        let history_msgs = self.history_to_unified(&history, &context.media_marker);

        if history_msgs.is_empty() {
            return Ok(messages);
        }

        // 分离系统消息和当前消息
        let (system_msgs, current_msgs): (Vec<_>, Vec<_>) = messages
            .into_iter()
            .partition(|msg| msg.role == MessageRole::System);

        // 组装：系统消息 + 历史消息 + 当前消息
        let mut result = system_msgs;
        result.extend(history_msgs);
        result.extend(current_msgs);

        Ok(result)
    }
}
