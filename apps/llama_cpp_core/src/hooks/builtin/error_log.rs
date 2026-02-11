//! 错误日志钩子

use async_trait::async_trait;
use tracing::{error, warn};

use crate::error::Error;
use crate::hooks::{HookContext, InferenceHook};

/// 错误日志钩子
///
/// 在发生错误时记录详细日志
#[derive(Debug, Default)]
pub struct ErrorLogHook {
    /// 是否记录请求详情
    log_request: bool,
    /// 是否记录上下文信息
    log_context: bool,
}

impl ErrorLogHook {
    /// 创建新的错误日志钩子
    pub fn new() -> Self {
        Self {
            log_request: true,
            log_context: true,
        }
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
        5 // 高优先级，确保尽早记录错误
    }

    async fn on_error(&self, ctx: &HookContext, error: &Error) -> Result<(), Error> {
        if self.log_request {
            error!(
                error = ?error,
                session_id = ?ctx.session_id,
                model = ctx.request.as_ref().map(|r| &r.model),
                "Inference error occurred"
            );
        }

        if self.log_context {
            // 记录额外的上下文信息
            if let Some(elapsed) = ctx.elapsed_ms() {
                error!(elapsed_ms = elapsed, "Error occurred after");
            }

            if let Some(req) = &ctx.request {
                warn!(
                    message_count = req.messages.len(),
                    max_completion_tokens = ?req.max_completion_tokens,
                    temperature = ?req.temperature,
                    "Request details"
                );
            }
        }

        Ok(())
    }
}
