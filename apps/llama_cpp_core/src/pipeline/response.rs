//! 流式响应构建器

// Re-export async-openai types for OpenAI API compatibility
pub use async_openai::types::chat::{
    ChatChoice,
    ChatChoiceStream,
    ChatCompletionResponseMessage,
    ChatCompletionStreamResponseDelta,
    CompletionUsage,
    // Standard request types
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
    FinishReason,
    Role,
};

/// 流式响应构建器
/// 用于构建标准的 CreateChatCompletionStreamResponse
#[derive(Debug, Clone)]
pub struct StreamResponseBuilder {
    id: String,
    model: String,
    created: u32,
    index: u32,
}

impl StreamResponseBuilder {
    /// 创建新的流式响应构建器
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            model: model.into(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as u32,
            index: 0,
        }
    }

    /// 设置 choice index
    pub fn with_index(mut self, index: u32) -> Self {
        self.index = index;
        self
    }

    /// 构建内容块响应
    pub fn build_content(&self, content: impl Into<String>) -> CreateChatCompletionStreamResponse {
        let delta = ChatCompletionStreamResponseDelta {
            content: Some(content.into()),
            #[allow(deprecated)]
            function_call: None,
            tool_calls: None,
            role: Some(Role::Assistant),
            refusal: None,
        };

        let choice = ChatChoiceStream {
            index: self.index,
            delta,
            finish_reason: None,
            logprobs: None,
        };

        CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            choices: vec![choice],
            created: self.created,
            model: self.model.clone(),
            service_tier: None,
            #[allow(deprecated)]
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        }
    }

    /// 构建结束响应
    pub fn build_finish(&self, finish_reason: FinishReason) -> CreateChatCompletionStreamResponse {
        self.build_finish_with_usage(finish_reason, 0, 0)
    }

    /// 构建错误响应（将错误信息作为内容发送，然后正常结束）
    pub fn build_error(&self, error_msg: impl Into<String>) -> CreateChatCompletionStreamResponse {
        let delta = ChatCompletionStreamResponseDelta {
            content: Some(format!("\n[Error: {}]", error_msg.into())),
            #[allow(deprecated)]
            function_call: None,
            tool_calls: None,
            role: Some(Role::Assistant),
            refusal: None,
        };

        let choice = ChatChoiceStream {
            index: self.index,
            delta,
            finish_reason: Some(FinishReason::Stop),
            logprobs: None,
        };

        CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            choices: vec![choice],
            created: self.created,
            model: self.model.clone(),
            service_tier: None,
            #[allow(deprecated)]
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        }
    }

    /// 构建带 usage 的最终响应（用于流式传输的最后一条消息，包含完整统计）
    pub fn build_finish_with_usage(
        &self,
        finish_reason: FinishReason,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) -> CreateChatCompletionStreamResponse {
        let delta = ChatCompletionStreamResponseDelta {
            content: None,
            #[allow(deprecated)]
            function_call: None,
            tool_calls: None,
            role: None,
            refusal: None,
        };

        let choice = ChatChoiceStream {
            index: self.index,
            delta,
            finish_reason: Some(finish_reason),
            logprobs: None,
        };

        let usage = if prompt_tokens > 0 || completion_tokens > 0 {
            Some(CompletionUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            })
        } else {
            None
        };

        CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            choices: vec![choice],
            created: self.created,
            model: self.model.clone(),
            service_tier: None,
            #[allow(deprecated)]
            system_fingerprint: None,
            object: "chat.completion.chunk".to_string(),
            usage,
        }
    }
}

/// 构建标准聊天完成响应
pub fn build_chat_completion_response(
    id: impl Into<String>,
    model: impl Into<String>,
    content: impl Into<String>,
    finish_reason: FinishReason,
) -> CreateChatCompletionResponse {
    build_chat_completion_response_with_usage(id, model, content, finish_reason, None)
}

/// 构建带 usage 统计的标准聊天完成响应
pub fn build_chat_completion_response_with_usage(
    id: impl Into<String>,
    model: impl Into<String>,
    content: impl Into<String>,
    finish_reason: FinishReason,
    usage: Option<CompletionUsage>,
) -> CreateChatCompletionResponse {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as u32;

    let message = ChatCompletionResponseMessage {
        content: Some(content.into()),
        refusal: None,
        role: Role::Assistant,
        audio: None,
        #[allow(deprecated)]
        function_call: None,
        tool_calls: None,
        annotations: None,
    };

    let choice = ChatChoice {
        index: 0,
        message,
        finish_reason: Some(finish_reason),
        logprobs: None,
    };

    CreateChatCompletionResponse {
        id: id.into(),
        choices: vec![choice],
        created,
        model: model.into(),
        service_tier: None,
        #[allow(deprecated)]
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage,
    }
}

/// 从响应中提取内容文本
pub fn chat_completion_response_extract_content(response: &CreateChatCompletionResponse) -> String {
    response
        .choices
        .first()
        .and_then(|c| c.message.content.as_ref())
        .cloned()
        .unwrap_or_default()
}
