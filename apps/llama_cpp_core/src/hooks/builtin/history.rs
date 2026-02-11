//! 历史记录保存钩子

use std::sync::Arc;

use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageContent,
};
use async_trait::async_trait;
use tracing::{debug, error};

use crate::cache::{CacheManager, CacheType};
use crate::error::Error;
use crate::history::HistoryMessage;
use crate::hooks::{HookContext, InferenceHook};

/// 历史记录保存钩子
///
/// 在推理结束后自动保存对话历史到缓存
#[derive(Debug, Clone)]
pub struct HistoryHook {
    cache: Arc<CacheManager>,
}

impl HistoryHook {
    /// 创建新的历史钩子
    pub fn new() -> Self {
        Self {
            cache: Arc::new(CacheManager::with_capacity(1000)),
        }
    }

    /// 使用共享的缓存管理器创建
    pub fn with_shared_cache(cache: Arc<CacheManager>) -> Self {
        Self { cache }
    }
}

impl Default for HistoryHook {
    fn default() -> Self {
        Self::new()
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

    async fn on_after(&self, ctx: &mut HookContext) -> Result<(), Error> {
        let session_id = match &ctx.session_id {
            Some(id) => id,
            None => {
                debug!("No session_id, skipping history save");
                return Ok(());
            }
        };

        let request = match &ctx.request {
            Some(req) => req,
            None => {
                debug!("No request in context, skipping history save");
                return Ok(());
            }
        };

        debug!("Saving history for session: {}", session_id);

        // 从缓存获取或创建历史记录
        let mut history = match self.cache.get_data::<HistoryMessage>(session_id) {
            Ok(Some(h)) => (*h).clone(),
            _ => HistoryMessage::new(),
        };

        // 添加用户消息
        for msg in &request.messages {
            if let ChatCompletionRequestMessage::User(user_msg) = msg {
                let content = match &user_msg.content {
                    ChatCompletionRequestUserMessageContent::Text(text) => text.clone(),
                    ChatCompletionRequestUserMessageContent::Array(parts) => {
                        parts
                            .iter()
                            .filter_map(|p| match p {
                                async_openai::types::chat::ChatCompletionRequestUserMessageContentPart::Text(t) => Some(t.text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                };
                history.add_user_text(&content);
            }
        }

        // 提取助手回复
        let assistant_content = ctx
            .stream_collected_text
            .clone()
            .or_else(|| {
                ctx.response.as_ref().and_then(|resp| {
                    resp.choices.first().and_then(|choice| {
                        choice.message.content.clone()
                    })
                })
            })
            .unwrap_or_default();

        if assistant_content.is_empty() {
            debug!("No assistant content to save");
            return Ok(());
        }

        // 添加助手回复
        history.add_assistant_text(&assistant_content);

        // 保存回缓存 - 使用 insert_or_update
        let history_arc: Arc<dyn std::any::Any + Send + Sync> = Arc::new(history);
        if let Err(e) = self.cache.insert_or_update(
            session_id,
            CacheType::Custom("history".to_string()),
            &[session_id.to_string()],
            history_arc,
        ) {
            error!("Failed to save history: {}", e);
        } else {
            debug!("History saved successfully for session: {}", session_id);
        }

        Ok(())
    }
}
