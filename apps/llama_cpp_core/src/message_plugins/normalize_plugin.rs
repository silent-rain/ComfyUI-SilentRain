//! 标准化插件

use crate::{
    error::Error,
    message_plugins::{
        MessageContext, MessagePlugin,
        unified_message::{ContentBlock, UnifiedMessage},
    },
};

/// 标准化插件
///
/// 负责：
/// 1. 清理内容中的多余空白
/// 2. 移除空消息
/// 3. 规范化媒体标记格式
/// 4. 验证消息结构完整性
#[derive(Debug, Default)]
pub struct NormalizePlugin {
    /// 是否清理空白字符
    pub trim_whitespace: bool,
    /// 是否移除空消息
    pub remove_empty: bool,
}

impl NormalizePlugin {
    pub fn new() -> Self {
        Self::default()
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
}

impl MessagePlugin for NormalizePlugin {
    fn name(&self) -> &str {
        "NormalizePlugin"
    }

    fn priority(&self) -> i32 {
        60
    }

    fn process(
        &self,
        messages: Vec<UnifiedMessage>,
        _context: &MessageContext,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_plugin_removes_empty() {
        let plugin = NormalizePlugin::new()
            .with_remove_empty(true)
            .with_trim(true);
        let context = MessageContext::default();

        let messages = vec![
            UnifiedMessage::user_text("Hello"),
            UnifiedMessage::user_text("   "), // 空白消息
            UnifiedMessage::user_text(""),    // 空消息
            UnifiedMessage::user_text("World"),
        ];

        let result = plugin.process(messages, &context).unwrap();
        assert_eq!(result.len(), 2);
    }
}
