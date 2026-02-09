//! Pipeline module for processing tasks

mod pipeline_config;
mod pipeline_impl;
mod request;
mod response;

pub use pipeline_config::PipelineConfig;
pub use pipeline_impl::Pipeline;
pub use request::{
    // async-openai types
    ChatCompletionResponseMessage,
    CreateChatCompletionResponse,
    FunctionCall,
    FunctionObject,
    ImageSource,
    ParsedInput,
    is_multimodal_request,
    parse_request_input,
};
pub use response::{ChatStreamBuilder, response_extract_content};

// 重新导出 async-openai 的核心类型
pub use crate::types::{
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest as Request,
};
