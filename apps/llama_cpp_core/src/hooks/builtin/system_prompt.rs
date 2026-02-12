//! 系统提示词 Hook

use crate::{
    error::Error,
    hooks::{HookContext, InferenceHook, priorities},
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

/// 系统提示词 Hook
///
/// 负责：
/// 1. 根据策略处理多个系统消息
/// 2. 确保系统消息始终在最前面
/// 3. 将系统消息分离到 pipeline_state.system_prompt
#[derive(Debug)]
pub struct SystemPromptHook {
    priority: i32,
    strategy: SystemStrategy,
    default_system: Option<String>,
}

impl Default for SystemPromptHook {
    fn default() -> Self {
        Self {
            priority: priorities::SYSTEM_PROMPT,
            strategy: SystemStrategy::KeepFirst,
            default_system: None,
        }
    }
}

impl SystemPromptHook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn keep_first() -> Self {
        Self {
            priority: priorities::SYSTEM_PROMPT,
            strategy: SystemStrategy::KeepFirst,
            default_system: None,
        }
    }

    pub fn merge() -> Self {
        Self {
            priority: priorities::SYSTEM_PROMPT,
            strategy: SystemStrategy::Merge,
            default_system: None,
        }
    }

    pub fn keep_last() -> Self {
        Self {
            priority: priorities::SYSTEM_PROMPT,
            strategy: SystemStrategy::KeepLast,
            default_system: None,
        }
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
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

#[async_trait::async_trait]
impl InferenceHook for SystemPromptHook {
    fn name(&self) -> &str {
        "SystemPromptHook"
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    async fn on_prepare(&self, ctx: &mut HookContext) -> Result<(), Error> {
        // 从用户输入的消息中分离系统消息
        let (system_msgs, other_msgs): (Vec<_>, Vec<_>) = ctx
            .pipeline_state
            .current_input
            .clone()
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

        // 保存系统提示词
        ctx.pipeline_state.system_prompt = final_system;

        // 更新用户输入的消息（移除系统消息）
        ctx.pipeline_state.current_input = other_msgs;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_system_hook_keep_first() {
        let hook = SystemPromptHook::keep_first();

        let mut ctx = HookContext::default();
        ctx.pipeline_state.current_input = vec![
            UnifiedMessage::system("First system"),
            UnifiedMessage::user("Hello"),
            UnifiedMessage::system("Second system"),
        ];

        hook.on_prepare(&mut ctx).await.unwrap();

        assert!(ctx.pipeline_state.system_prompt.is_some());
        assert_eq!(ctx.pipeline_state.current_input.len(), 1);
    }
}
