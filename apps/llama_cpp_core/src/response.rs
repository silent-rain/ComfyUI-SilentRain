//! 流式响应构建器 - OpenAI API 兼容

use async_openai::types::chat::{
    ChatChoice, ChatChoiceStream, ChatCompletionResponseMessage, ChatCompletionStreamResponseDelta,
    CompletionUsage, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    FinishReason, Role,
};

/// 响应结构别名 - 使用 async-openai 标准响应
pub type Response = CreateChatCompletionResponse;

/// 流式聊天响应构建器
/// 用于高效构建 OpenAI Chat Completions API 流式响应
/// 复用共享字段，减少每次创建时的内存分配
#[derive(Debug, Clone)]
pub struct ChatStreamBuilder {
    /// 响应 ID（所有 chunk 共享）
    id: String,
    /// 模型名称（所有 chunk 共享）
    model: String,
    /// 创建时间戳（所有 chunk 共享）
    created: u32,
    /// 当前 choice 索引
    choice_index: u32,
    /// 是否已发送第一个 chunk（用于发送 role）
    has_started: bool,
    /// 累计生成的 token 数
    tokens_generated: u32,
    /// 输入 prompt tokens 数
    prompt_tokens: u32,
}

impl ChatStreamBuilder {
    /// 创建新的流式响应构建器
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            model: model.into(),
            created: chrono::Utc::now().timestamp() as u32,
            choice_index: 0,
            has_started: false,
            tokens_generated: 0,
            prompt_tokens: 0,
        }
    }

    /// 设置 prompt tokens 数量（用于 usage 计算）
    pub fn with_prompt_tokens(mut self, tokens: u32) -> Self {
        self.prompt_tokens = tokens;
        self
    }

    /// 设置 choice 索引
    pub fn with_choice_index(mut self, index: u32) -> Self {
        self.choice_index = index;
        self
    }

    /// 构建内容增量 chunk
    /// 第一个 chunk 会包含 role 字段
    pub fn build_content_chunk(
        &mut self,
        content: impl Into<String>,
    ) -> CreateChatCompletionStreamResponse {
        let content = content.into();

        let delta = if !self.has_started {
            self.has_started = true;
            ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(content),
                #[allow(deprecated)]
                function_call: None,
                tool_calls: None,
                refusal: None,
            }
        } else {
            ChatCompletionStreamResponseDelta {
                role: None,
                content: Some(content),
                #[allow(deprecated)]
                function_call: None,
                tool_calls: None,
                refusal: None,
            }
        };

        self.tokens_generated += 1;

        CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            #[allow(deprecated)]
            system_fingerprint: None,
            choices: vec![ChatChoiceStream {
                index: self.choice_index,
                delta,
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
            service_tier: None,
        }
    }

    /// 构建停止 chunk（正常结束）
    pub fn build_stop_chunk(&self) -> CreateChatCompletionStreamResponse {
        CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            #[allow(deprecated)]
            system_fingerprint: None,
            choices: vec![ChatChoiceStream {
                index: self.choice_index,
                delta: ChatCompletionStreamResponseDelta {
                    role: None,
                    content: None,
                    #[allow(deprecated)]
                    function_call: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: Some(FinishReason::Stop),
                logprobs: None,
            }],
            usage: None,
            service_tier: None,
        }
    }

    /// 构建长度限制停止 chunk（达到最大 token 数）
    pub fn build_length_chunk(&self) -> CreateChatCompletionStreamResponse {
        CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            #[allow(deprecated)]
            system_fingerprint: None,
            choices: vec![ChatChoiceStream {
                index: self.choice_index,
                delta: ChatCompletionStreamResponseDelta {
                    role: None,
                    content: None,
                    #[allow(deprecated)]
                    function_call: None,
                    tool_calls: None,
                    refusal: None,
                },
                finish_reason: Some(FinishReason::Length),
                logprobs: None,
            }],
            usage: None,
            service_tier: None,
        }
    }

    /// 构建包含 usage 的最终 chunk
    /// 根据 OpenAI API 规范，usage 只在最后一个 chunk 中返回（如果设置了 stream_options.include_usage）
    pub fn build_final_chunk_with_usage(&self) -> CreateChatCompletionStreamResponse {
        CreateChatCompletionStreamResponse {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            #[allow(deprecated)]
            system_fingerprint: None,
            choices: vec![], // usage chunk 的 choices 为空数组
            usage: Some(CompletionUsage {
                prompt_tokens: self.prompt_tokens,
                completion_tokens: self.tokens_generated,
                total_tokens: self.prompt_tokens + self.tokens_generated,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            service_tier: None,
        }
    }

    /// 获取响应 ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// 获取生成的 token 数量
    pub fn tokens_generated(&self) -> u32 {
        self.tokens_generated
    }

    /// 获取总 token 数量（prompt + completion）
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens + self.tokens_generated
    }

    /// 构建完整的非流式响应
    /// 将流式响应收集的文本组装成标准的 CreateChatCompletionResponse
    pub fn build_non_streaming_response(
        &self,
        full_text: impl Into<String>,
        finish_reason: Option<FinishReason>,
        completion_tokens: u32,
    ) -> CreateChatCompletionResponse {
        let full_text = full_text.into();
        let finish_reason = finish_reason.unwrap_or(FinishReason::Stop);

        CreateChatCompletionResponse {
            id: self.id.clone(),
            object: "chat.completion".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChatChoice {
                index: self.choice_index,
                message: ChatCompletionResponseMessage {
                    role: Role::Assistant,
                    content: Some(full_text),
                    #[allow(deprecated)]
                    function_call: None,
                    tool_calls: None,
                    refusal: None,
                    annotations: None,
                    audio: None,
                },
                finish_reason: Some(finish_reason),
                logprobs: None,
            }],
            usage: Some(CompletionUsage {
                prompt_tokens: self.prompt_tokens,
                completion_tokens,
                total_tokens: self.prompt_tokens + completion_tokens,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            #[allow(deprecated)]
            system_fingerprint: None,
            service_tier: None,
        }
    }
}

/// 从响应中提取内容文本
pub fn response_extract_content(response: &CreateChatCompletionResponse) -> String {
    response
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default()
}
