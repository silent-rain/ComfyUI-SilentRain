//! 系统消息插件

use crate::{
    error::Error,
    message_plugins::{MessageContext, MessagePlugin},
    types::MessageRole,
    unified_message::UnifiedMessage,
};

/// 系统消息处理策略
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SystemStrategy {
    /// 仅保留第一个系统消息
    KeepFirst,
    /// 合并所有系统消息为一个
    Merge,
    /// 保留最后一个系统消息
    KeepLast,
}

/// 系统消息插件
///
/// 负责：
/// 1. 根据策略处理多个系统消息
/// 2. 确保系统消息始终在最前面
/// 3. 分离系统消息与后续消息
#[derive(Debug)]
pub struct SystemPromptPlugin {
    strategy: SystemStrategy,
    default_system: Option<String>,
}

impl Default for SystemPromptPlugin {
    fn default() -> Self {
        Self {
            strategy: SystemStrategy::KeepFirst,
            default_system: None,
        }
    }
}

impl SystemPromptPlugin {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn keep_first() -> Self {
        Self {
            strategy: SystemStrategy::KeepFirst,
            default_system: None,
        }
    }

    pub fn merge() -> Self {
        Self {
            strategy: SystemStrategy::Merge,
            default_system: None,
        }
    }

    pub fn keep_last() -> Self {
        Self {
            strategy: SystemStrategy::KeepLast,
            default_system: None,
        }
    }

    pub fn with_default_system(mut self, system: impl Into<String>) -> Self {
        self.default_system = Some(system.into());
        self
    }

    /// 应用策略处理系统消息
    fn apply_strategy(&self, system_messages: Vec<UnifiedMessage>) -> Option<UnifiedMessage> {
        if system_messages.is_empty() {
            return None;
        }

        match self.strategy {
            SystemStrategy::KeepFirst => system_messages.into_iter().next(),
            SystemStrategy::KeepLast => system_messages.into_iter().last(),
            SystemStrategy::Merge => {
                let merged_blocks = system_messages
                    .iter()
                    .filter_map(|msg| {
                        if msg.role == MessageRole::System {
                            Some(msg.content.clone())
                        } else {
                            None
                        }
                    })
                    .flatten()
                    .collect::<Vec<_>>();
                if merged_blocks.is_empty() {
                    None
                } else {
                    Some(UnifiedMessage::system_with_blocks(merged_blocks))
                }
            }
        }
    }
}

impl MessagePlugin for SystemPromptPlugin {
    fn name(&self) -> &str {
        "SystemPlugin"
    }

    fn priority(&self) -> i32 {
        70
    }

    fn process(
        &self,
        messages: Vec<UnifiedMessage>,
        _context: &MessageContext,
    ) -> Result<Vec<UnifiedMessage>, Error> {
        // 分离系统消息和其他消息
        let (system_msgs, other_msgs): (Vec<_>, Vec<_>) = messages
            .into_iter()
            .partition(|msg| msg.role == MessageRole::System);

        // 处理系统消息
        let processed_system = self.apply_strategy(system_msgs);

        // 如果没有系统消息但有默认值，使用默认值
        let final_system = processed_system.or_else(|| {
            self.default_system
                .as_ref()
                .map(|text| UnifiedMessage::system(text.clone()))
        });

        // 重新组装：系统消息在最前，然后是其他消息
        let mut result = Vec::new();
        if let Some(sys) = final_system {
            result.push(sys);
        }
        result.extend(other_msgs);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_plugin_keep_first() {
        let plugin = SystemPromptPlugin::keep_first();
        let context = MessageContext::default();

        let messages = vec![
            UnifiedMessage::system("First system"),
            UnifiedMessage::user_text("Hello"),
            UnifiedMessage::system("Second system"), // 应该被过滤
        ];

        let result = plugin.process(messages, &context).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].role, MessageRole::System);
    }

    #[test]
    fn test_system_plugin_merge() {
        let plugin = SystemPromptPlugin::merge();
        let context = MessageContext::default();

        let messages = vec![
            UnifiedMessage::system("First"),
            UnifiedMessage::user_text("Hello"),
            UnifiedMessage::system("Second"),
        ];

        let result = plugin.process(messages, &context).unwrap();
        assert_eq!(result.len(), 2);
    }
}
