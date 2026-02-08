//! Pipeline module for processing tasks

mod pipeline_config;
mod pipeline_impl;
mod request;
mod response;

pub use pipeline_config::PipelineConfig;
pub use pipeline_impl::Pipeline;
pub use request::{
    Input,
    InputItem,
    MessageContent,
    Model,
    // 标准 OpenAI Responses API 类型
    Request,
    RequestBuilder,
    Response,
    ResponseItem,
    StreamEvent,
    Tool,
    ToolChoice,
};
pub use response::{
    StreamResponseBuilder, create_model, create_text_input, create_vision_input,
    response_extract_content,
};
