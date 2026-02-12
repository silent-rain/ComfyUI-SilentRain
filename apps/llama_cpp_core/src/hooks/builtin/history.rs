//! 历史记录保存 Hook

use tracing::{debug, error};

use crate::{
    chat_history,
    error::Error,
    hooks::{HookContext, InferenceHook, priorities},
};

/// 历史记录保存 Hook
///
/// 在推理前保存当前输入，在推理后保存助手回复
#[derive(Debug, Clone)]
pub struct HistoryHook {
    priority: i32,
}

impl Default for HistoryHook {
    fn default() -> Self {
        Self {
            priority: priorities::HISTORY,
        }
    }
}

impl HistoryHook {
    /// 创建新的历史钩子
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

#[async_trait::async_trait]
impl InferenceHook for HistoryHook {
    fn name(&self) -> &str {
        "HistoryHook"
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    async fn on_before(&self, ctx: &mut HookContext) -> Result<(), Error> {
        let session_id = match &ctx.session_id {
            Some(id) => id,
            None => {
                debug!("No session_id, skipping history save");
                return Ok(());
            }
        };

        debug!("Saving current input for session: {}", session_id);

        let manager = chat_history();
        let _ = manager.get_or_create(session_id);

        // 只保存当前输入的消息
        // 避免重复保存历史消息
        let current_input = &ctx.pipeline_state.current_input;

        if !current_input.is_empty() {
            if let Err(e) = manager.add_messages(session_id, current_input) {
                error!("Failed to add user messages: {}", e);
            } else {
                debug!("Saved {} user message(s)", current_input.len());
            }
        }

        Ok(())
    }

    async fn on_after(&self, ctx: &mut HookContext) -> Result<(), Error> {
        let session_id = match &ctx.session_id {
            Some(id) => id,
            None => {
                debug!("No session_id, skipping history save");
                return Ok(());
            }
        };

        debug!("Saving assistant response for session: {}", session_id);

        let manager = chat_history();
        let _ = manager.get_or_create(session_id);

        // 提取助手回复
        let assistant_content = ctx
            .stream_collected_text
            .clone()
            .or_else(|| {
                ctx.response.as_ref().and_then(|resp| {
                    resp.choices
                        .first()
                        .and_then(|choice| choice.message.content.clone())
                })
            })
            .unwrap_or_default();

        if assistant_content.is_empty() {
            debug!("No assistant content to save");
            return Ok(());
        }

        // 添加助手回复
        if let Err(e) = manager.add_assistant_text(session_id, &assistant_content) {
            error!("Failed to add assistant message: {}", e);
        } else {
            debug!("History saved successfully for session: {}", session_id);
        }

        Ok(())
    }
}
