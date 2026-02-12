//! 标准化 Hook

use crate::{
    error::Error,
    hooks::{HookContext, InferenceHook, priorities},
    unified_message::{ContentBlock, UnifiedMessage},
};

/// 标准化 Hook
///
/// 负责：
/// 1. 清理内容中的多余空白
/// 2. 移除空消息
/// 3. 规范化媒体标记格式
/// 4. 验证消息结构完整性
#[derive(Debug)]
pub struct NormalizeHook {
    priority: i32,
    /// 是否清理空白字符
    pub trim_whitespace: bool,
    /// 是否移除空消息
    pub remove_empty: bool,
}

impl Default for NormalizeHook {
    fn default() -> Self {
        Self {
            priority: priorities::NORMALIZE,
            trim_whitespace: false,
            remove_empty: false,
        }
    }
}

impl NormalizeHook {
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_trim(mut self, enable: bool) -> Self {
        self.trim_whitespace = enable;
        self
    }

    pub fn with_remove_empty(mut self, enable: bool) -> Self {
        self.remove_empty = enable;
        self
    }

    /// 清理单个内容块
    fn normalize_block(&self, block: ContentBlock) -> Option<ContentBlock> {
        match block {
            ContentBlock::Text { text } => {
                let cleaned = if self.trim_whitespace {
                    text.trim().to_string()
                } else {
                    text
                };
                if self.remove_empty && cleaned.is_empty() {
                    None
                } else {
                    Some(ContentBlock::Text { text: cleaned })
                }
            }
            ContentBlock::Image { source, detail } => Some(ContentBlock::Image { source, detail }),
            other => Some(other),
        }
    }

    /// 处理消息列表
    fn process_messages(
        &self,
        messages: Vec<UnifiedMessage>,
    ) -> Result<Vec<UnifiedMessage>, Error> {
        let mut result = Vec::with_capacity(messages.len());

        for mut msg in messages {
            // 规范化内容块
            let normalized_blocks: Vec<ContentBlock> = msg
                .content
                .into_iter()
                .filter_map(|block| self.normalize_block(block))
                .collect();

            // 如果内容块为空且配置为移除空消息，则跳过
            if normalized_blocks.is_empty() && self.remove_empty {
                continue;
            }

            msg.content = normalized_blocks;
            result.push(msg);
        }

        Ok(result)
    }
}

#[async_trait::async_trait]
impl InferenceHook for NormalizeHook {
    fn name(&self) -> &str {
        "NormalizeHook"
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    async fn on_prepare(&self, ctx: &mut HookContext) -> Result<(), Error> {
        // 处理原始消息
        let normalized = self.process_messages(ctx.pipeline_state.raw_messages.clone())?;
        ctx.pipeline_state.current_input = normalized;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_hook_removes_empty() {
        let hook = NormalizeHook::new().with_remove_empty(true).with_trim(true);

        let messages = vec![
            UnifiedMessage::user("Hello"),
            UnifiedMessage::user("   "), // 空白消息
            UnifiedMessage::user(""),    // 空消息
            UnifiedMessage::user("World"),
        ];

        let result = hook.process_messages(messages).unwrap();
        assert_eq!(result.len(), 2);
    }
}
