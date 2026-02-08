//! 流式响应构建器 - OpenAI Responses API 兼容

// Re-export open-ai-rust-responses-by-sshift types for OpenAI Responses API compatibility
pub use open_ai_rust_responses_by_sshift::{InputItem, Model, Response, StreamEvent};

/// 流式响应构建器
/// 用于构建标准的 OpenAI Responses API 流式响应
#[derive(Debug, Clone)]
pub struct StreamResponseBuilder {
    id: String,
    index: u32,
}
impl Default for StreamResponseBuilder {
    fn default() -> Self {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        Self { id, index: 0 }
    }
}
impl StreamResponseBuilder {
    /// 创建新的流式响应构建器
    pub fn new() -> Self {
        StreamResponseBuilder::default()
    }

    /// 设置响应 ID
    pub fn new_with_id(index: u32) -> Self {
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        Self { id, index }
    }

    /// 构建响应创建事件
    pub fn build_created_event(&self) -> StreamEvent {
        StreamEvent::ResponseCreated {
            id: self.id.clone(),
        }
    }

    /// 构建内容块响应 (流式事件)
    pub fn build_content_event(&self, content: impl Into<String>) -> StreamEvent {
        StreamEvent::TextDelta {
            content: content.into(),
            index: self.index,
        }
    }

    /// 构建文本停止事件
    /// 按照 OpenAI Responses API 标准，每个 choice 结束后应发送 TextStop 事件
    pub fn build_text_stop_event(&self) -> StreamEvent {
        StreamEvent::TextStop { index: self.index }
    }

    /// 构建完成事件
    pub fn build_done_event(&self) -> StreamEvent {
        StreamEvent::Done
    }
}

/// 从响应中提取内容文本
pub fn response_extract_content(response: &Response) -> String {
    response.output_text()
}

/// 创建简单的文本输入项
pub fn create_text_input(content: impl Into<String>) -> Vec<InputItem> {
    vec![InputItem::message(
        "user",
        vec![InputItem::content_text(content)],
    )]
}

/// 创建带图像的输入项
pub fn create_vision_input(
    text: impl Into<String>,
    image_url: impl Into<String>,
) -> Vec<InputItem> {
    vec![InputItem::message(
        "user",
        vec![
            InputItem::content_text(text),
            InputItem::content_image_with_detail(image_url, "auto"),
        ],
    )]
}

/// 创建模型枚举
pub fn create_model(model_name: &str) -> Model {
    match model_name.to_lowercase().as_str() {
        "gpt-5" => Model::GPT5,
        "gpt-5-mini" => Model::GPT5Mini,
        "gpt-5-nano" => Model::GPT5Nano,
        "gpt-4o" => Model::GPT4o,
        "gpt-4o-mini" => Model::GPT4oMini,
        "o4-mini" => Model::O4Mini,
        "o3" => Model::O3,
        "o1" => Model::O1,
        "o1-mini" => Model::O1Mini,
        "o1-preview" => Model::O1Preview,
        _ => Model::GPT4oMini, // 默认使用 GPT4oMini
    }
}
