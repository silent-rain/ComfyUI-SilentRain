//! 历史记录保存钩子

use async_trait::async_trait;
use tracing::{debug, error};

use crate::{
    chat_history,
    error::Error,
    hooks::{HookContext, InferenceHook},
};

/// 历史记录保存钩子
///
/// 在推理结束后自动保存对话历史到缓存
#[derive(Debug, Clone, Default)]
pub struct HistoryHook;

impl HistoryHook {
    /// 创建新的历史钩子
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl InferenceHook for HistoryHook {
    fn name(&self) -> &str {
        "HistoryHook"
    }

    fn priority(&self) -> i32 {
        90 // 低优先级，最后执行
    }

    async fn on_before(&self, ctx: &mut HookContext) -> Result<(), Error> {
        let session_id = match &ctx.session_id {
            Some(id) => id,
            None => {
                debug!("No session_id, skipping history save");
                return Ok(());
            }
        };

        debug!("Saving history for session: {}", session_id);

        let manager = chat_history();

        let _ = manager.get_or_create(session_id);

        if let Err(e) = manager.add_messages(session_id, &ctx.unified_messages) {
            error!("Failed to add user message: {}", e);
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

        debug!("Saving history for session: {}", session_id);

        let manager = chat_history();

        // 获取或创建会话
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
