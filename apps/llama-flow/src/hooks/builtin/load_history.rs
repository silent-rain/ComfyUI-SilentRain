//! 加载历史消息 Hook

use crate::{
    chat_history,
    error::Error,
    hooks::{HookContext, InferenceHook, priorities},
    types::MessageRole,
    unified_message::UnifiedMessage,
};

/// 加载历史消息 Hook
///
/// 负责：
/// 1. 从缓存加载历史消息
/// 2. 清理历史消息中的媒体标记
/// 3. 将历史消息保存到 pipeline_state.loaded_history
#[derive(Debug)]
pub struct LoadHistoryHook {
    priority: i32,
    max_history: usize,
}

impl Default for LoadHistoryHook {
    fn default() -> Self {
        Self {
            priority: priorities::LOAD_HISTORY,
            max_history: 100,
        }
    }
}

impl LoadHistoryHook {
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// 从会话消息创建统一消息
    ///
    /// 过滤掉系统消息，清理媒体标记为描述文本
    fn messages_to_unified(&self, messages: &[UnifiedMessage]) -> Vec<UnifiedMessage> {
        // 只取最近的 N 条
        let start_idx = messages.len().saturating_sub(self.max_history);

        messages[start_idx..]
            .iter()
            .filter(|entry| entry.role != MessageRole::System)
            .cloned()
            .map(|mut msg| {
                // 清理媒体标记
                msg.content = msg.sanitize_media_marker();
                msg
            })
            .collect()
    }
}

#[async_trait::async_trait]
impl InferenceHook for LoadHistoryHook {
    fn name(&self) -> &str {
        "LoadHistoryHook"
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    async fn on_prepare(&self, ctx: &mut HookContext) -> Result<(), Error> {
        let session_id = match &ctx.session_id {
            Some(id) => id,
            None => {
                return Ok(());
            }
        };

        // 尝试加载历史
        let history_messages = match chat_history().get_messages(session_id) {
            Some(msgs) => msgs,
            None => {
                return Ok(());
            }
        };

        // 转换历史消息
        let history_msgs = self.messages_to_unified(&history_messages);

        // 保存到状态
        ctx.pipeline_state.loaded_history = history_msgs;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_2::mtmd::mtmd_default_marker;

    use crate::unified_message::{ContentBlock, ImageSource};

    use super::*;

    #[test]
    fn test_messages_to_unified() {
        let default_marker = mtmd_default_marker().to_string();
        let hook = LoadHistoryHook::new();

        let history = vec![
            UnifiedMessage::user("Hello"),
            UnifiedMessage::user_with_blocks(vec![
                ContentBlock::Text {
                    text: "Test message".to_string(),
                },
                ContentBlock::Image {
                    source: ImageSource::Url(default_marker),
                    detail: None,
                },
            ]),
            UnifiedMessage::assistant("Hi there"),
        ];

        let results = hook.messages_to_unified(&history);

        println!("{:?}", results);
    }
}
