//! 组装最终工作消息列表 Hook

use crate::{
    error::Error,
    hooks::{HookContext, InferenceHook, priorities},
};

/// 当前输入处理 Hook
///
/// 负责：
/// 1. 组装最终工作消息列表
/// 2. 确保消息顺序正确：系统提示 + 历史 + 当前输入
/// 3. 设置最终的 pipeline_state.working_messages
#[derive(Debug)]
pub struct AssembleMessagesHook {
    priority: i32,
}

impl Default for AssembleMessagesHook {
    fn default() -> Self {
        Self {
            priority: priorities::ASSEMBLE_MESSAGES,
        }
    }
}

impl AssembleMessagesHook {
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
impl InferenceHook for AssembleMessagesHook {
    fn name(&self) -> &str {
        "CurrentInputHook"
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    async fn on_prepare(&self, ctx: &mut HookContext) -> Result<(), Error> {
        // 组装最终工作消息
        ctx.pipeline_state.assemble_working_messages();

        // 检查是否包含多模态内容
        ctx.pipeline_state.has_multimodal = ctx
            .pipeline_state
            .working_messages
            .iter()
            .any(|msg| msg.has_image());

        Ok(())
    }
}
