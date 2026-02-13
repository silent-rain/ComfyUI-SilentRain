//! 错误日志钩子

use async_trait::async_trait;
use tracing::{error, warn};

use crate::error::Error;
use crate::hooks::priorities::ERROR_LOG;
use crate::hooks::{HookContext, InferenceHook};

/// 错误日志钩子
///
/// 在发生错误时记录详细日志
#[derive(Debug)]
pub struct ErrorLogHook {
    priority: i32,
    /// 是否记录请求详情
    log_request: bool,
    /// 是否记录上下文信息
    log_context: bool,
}

impl Default for ErrorLogHook {
    fn default() -> Self {
        Self {
            priority: ERROR_LOG, // 高优先级，确保尽早记录错误
            log_request: true,
            log_context: true,
        }
    }
}

impl ErrorLogHook {
    /// 创建新的错误日志钩子
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// 设置是否记录请求
    pub fn with_log_request(mut self, enable: bool) -> Self {
        self.log_request = enable;
        self
    }

    /// 设置是否记录上下文
    pub fn with_log_context(mut self, enable: bool) -> Self {
        self.log_context = enable;
        self
    }
}

#[async_trait]
impl InferenceHook for ErrorLogHook {
    fn name(&self) -> &str {
        "ErrorLogHook"
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    async fn on_prepare(&self, _ctx: &mut HookContext) -> Result<(), Error> {
        // 错误日志钩子不需要处理消息准备阶段
        Ok(())
    }

    async fn on_before(&self, _ctx: &mut HookContext) -> Result<(), Error> {
        // 错误日志钩子不需要处理推理前阶段
        Ok(())
    }

    async fn on_after(&self, _ctx: &mut HookContext) -> Result<(), Error> {
        // 错误日志钩子不需要处理推理后阶段
        Ok(())
    }

    async fn on_error(&self, ctx: &HookContext, error: &Error) -> Result<(), Error> {
        if self.log_context {
            // 记录额外的上下文信息
            if let Some(elapsed) = ctx.elapsed_ms() {
                error!(elapsed_ms = elapsed, "Error occurred after");
            }

            warn!(
                message_count = ctx.request.messages.len(),
                max_completion_tokens = ?ctx.request.max_completion_tokens,
                temperature = ?ctx.request.temperature,
                error = ?error
            );
        }

        Ok(())
    }
}
