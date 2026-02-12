//! Tool 调用消息插件

use std::collections::HashSet;

use crate::{
    error::Error,
    message_plugins::{MessageContext, MessagePlugin},
    types::MessageRole,
    unified_message::{ContentBlock, UnifiedMessage},
};

/// Tool 调用消息插件
///
/// 负责：
/// 1. 验证 Tool 调用链的完整性
/// 2. 确保 Tool 结果与调用请求匹配
/// 3. 格式化 Tool 相关消息
#[derive(Debug, Default)]
pub struct ToolsPlugin {
    validate_chains: bool,
}

impl ToolsPlugin {
    pub fn new() -> Self {
        Self {
            validate_chains: true,
        }
    }

    pub fn without_validation(mut self) -> Self {
        self.validate_chains = false;
        self
    }

    /// 验证 Tool 调用链
    fn validate_tool_chains(&self, messages: &[UnifiedMessage]) -> Result<(), Error> {
        let mut pending_calls: HashSet<String> = HashSet::new();

        for msg in messages {
            match msg.role {
                MessageRole::Assistant => {
                    // 记录新的 Tool 调用
                    for block in &msg.content {
                        if let ContentBlock::ToolCall { id, .. } = block {
                            pending_calls.insert(id.clone());
                        }
                    }
                }
                MessageRole::Tool => {
                    // 检查 Tool 结果是否有对应的调用
                    if let Some(ref call_id) = msg.tool_call_id {
                        if !pending_calls.contains(call_id) {
                            return Err(Error::InvalidInput {
                                field: "tool_call_id".to_string(),
                                message: format!("No matching tool call for id: {}", call_id),
                            });
                        }
                        pending_calls.remove(call_id);
                    }
                }
                _ => {}
            }
        }

        // 检查是否有未完成的调用
        if !pending_calls.is_empty() {
            tracing::warn!("Uncompleted tool calls: {:?}", pending_calls);
        }

        Ok(())
    }
}

impl MessagePlugin for ToolsPlugin {
    fn name(&self) -> &str {
        "ToolsPlugin"
    }

    fn priority(&self) -> i32 {
        100
    }

    fn process(
        &self,
        _context: &MessageContext,
        messages: Vec<UnifiedMessage>,
    ) -> Result<Vec<UnifiedMessage>, Error> {
        if self.validate_chains && !messages.is_empty() {
            self.validate_tool_chains(&messages)?;
        }
        Ok(messages)
    }
}
