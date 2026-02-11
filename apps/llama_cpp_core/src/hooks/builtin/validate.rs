//! 参数验证钩子

use async_trait::async_trait;
use tracing::debug;

use crate::error::Error;
use crate::hooks::{HookContext, InferenceHook};

/// 参数验证钩子
///
/// 在推理前验证请求参数的有效性
#[derive(Debug, Default)]
pub struct ValidateHook {
    /// 最大允许的 tokens
    max_allowed_tokens: Option<u32>,
    /// 最小允许的 tokens
    min_allowed_tokens: Option<u32>,
    /// 是否允许空消息
    allow_empty_messages: bool,
}

impl ValidateHook {
    /// 创建新的验证钩子
    pub fn new() -> Self {
        Self {
            max_allowed_tokens: None,
            min_allowed_tokens: Some(1),
            allow_empty_messages: false,
        }
    }

    /// 设置最大允许的 tokens
    pub fn with_max_tokens(mut self, max: u32) -> Self {
        self.max_allowed_tokens = Some(max);
        self
    }

    /// 设置最小允许的 tokens
    pub fn with_min_tokens(mut self, min: u32) -> Self {
        self.min_allowed_tokens = Some(min);
        self
    }

    /// 设置是否允许空消息
    pub fn with_allow_empty_messages(mut self, allow: bool) -> Self {
        self.allow_empty_messages = allow;
        self
    }
}

#[async_trait]
impl InferenceHook for ValidateHook {
    fn name(&self) -> &str {
        "ValidateHook"
    }

    fn priority(&self) -> i32 {
        10 // 高优先级，尽早执行
    }

    async fn on_before(&self, ctx: &mut HookContext) -> Result<(), Error> {
        let request = ctx.request.as_ref().ok_or_else(|| Error::InvalidInput {
            field: "request".to_string(),
            message: "Request is required".to_string(),
        })?;

        debug!("Validating request parameters");

        // 验证消息不为空
        if !self.allow_empty_messages && request.messages.is_empty() {
            return Err(Error::InvalidInput {
                field: "messages".to_string(),
                message: "Message list cannot be empty".to_string(),
            });
        }

        // 验证 max_completion_tokens
        if let Some(max_completion_tokens) = request.max_completion_tokens {
            // 检查最大值
            if let Some(max_allowed) = self.max_allowed_tokens
                && max_completion_tokens > max_allowed
            {
                return Err(Error::InvalidInput {
                    field: "max_completion_tokens".to_string(),
                    message: format!(
                        "max_completion_tokens ({}) exceeds maximum allowed ({})",
                        max_completion_tokens, max_allowed
                    ),
                });
            }

            // 检查最小值
            if let Some(min_allowed) = self.min_allowed_tokens
                && max_completion_tokens < min_allowed
            {
                return Err(Error::InvalidInput {
                    field: "max_completion_tokens".to_string(),
                    message: format!(
                        "max_completion_tokens ({}) is less than minimum allowed ({})",
                        max_completion_tokens, min_allowed
                    ),
                });
            }
        }

        // 验证 temperature 范围
        if let Some(temp) = request.temperature
            && !(0.0..=2.0).contains(&temp)
        {
            return Err(Error::InvalidInput {
                field: "temperature".to_string(),
                message: format!("temperature ({}) must be between 0.0 and 2.0", temp),
            });
        }

        // 验证 top_p 范围
        if let Some(top_p) = request.top_p
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err(Error::InvalidInput {
                field: "top_p".to_string(),
                message: format!("top_p ({}) must be between 0.0 and 1.0", top_p),
            });
        }

        debug!("Request validation passed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::CreateChatCompletionRequestArgs;

    #[tokio::test]
    async fn test_validate_empty_messages() {
        let hook = ValidateHook::new();
        let request = CreateChatCompletionRequestArgs::default()
            .model("test")
            .build()
            .unwrap();

        let mut ctx = HookContext::new().with_request(&request);
        let result = hook.on_before(&mut ctx).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_validate_max_tokens() {
        let hook = ValidateHook::new().with_max_tokens(100);

        // 应该通过的请求
        let request = CreateChatCompletionRequestArgs::default()
            .model("test")
            .max_completion_tokens(50u32)
            .messages(vec![async_openai::types::chat::ChatCompletionRequestMessage::User(
                async_openai::types::chat::ChatCompletionRequestUserMessage {
                    content: async_openai::types::chat::ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                    ..Default::default()
                }
            )])
            .build()
            .unwrap();

        let mut ctx = HookContext::new().with_request(&request);
        assert!(hook.on_before(&mut ctx).await.is_ok());

        // 应该失败的请求
        let request = CreateChatCompletionRequestArgs::default()
            .model("test")
            .max_completion_tokens(200u32)
            .messages(vec![async_openai::types::chat::ChatCompletionRequestMessage::User(
                async_openai::types::chat::ChatCompletionRequestUserMessage {
                    content: async_openai::types::chat::ChatCompletionRequestUserMessageContent::Text("Hello".to_string()),
                    ..Default::default()
                }
            )])
            .build()
            .unwrap();

        let mut ctx = HookContext::new().with_request(&request);
        assert!(hook.on_before(&mut ctx).await.is_err());
    }
}
