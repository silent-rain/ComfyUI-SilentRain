//! Request processing module for pipeline operations
//!
//! ## 架构说明
//!
//! 本模块直接使用 `open-ai-rust-responses-by-sshift::Request` 标准结构

use llama_cpp_2::{model::LlamaChatMessage, mtmd::mtmd_default_marker};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{error::Error, types::MessageRole};

// Re-export open-ai-rust-responses-by-sshift types for OpenAI Responses API compatibility
pub use open_ai_rust_responses_by_sshift::{
    Input, InputItem, MessageContent, Model, Request, RequestBuilder, Response, ResponseItem,
    StreamEvent, Tool, ToolChoice,
};

/// 内容片段（用于反序列化 message content）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    /// 内容类型: input_image/input_text
    pub content_type: String,
    /// 图片 URL（可能是 path 或 http 或 data:image/...）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// file_id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    /// 图片URL (标准字段)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    /// 细节级别
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// 文本内容
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// 图片来源
#[derive(Debug, Clone)]
pub enum ImageSource {
    /// URL（http/https 或 data:image base64）
    Url(String),
    /// data:{};base64,{}
    Base64(String),
    /// OpenAI File ID
    FileId(String),
}

/// 解析后的输入结果
#[derive(Debug, Clone)]
pub struct ParsedInput {
    /// 转换后的消息列表
    pub messages: Vec<LlamaChatMessage>,
    /// 提取的图片源（供后续下载/编码）
    pub image_sources: Vec<ImageSource>,
}

/// 检查请求是否包含图片（多模态请求）
pub fn is_multimodal_request(request: &Request) -> bool {
    // 2. 检查 InputItem 中的图片
    if let Input::Items(ref items) = request.input {
        for item in items {
            // 直接是 input_image 类型
            if item.item_type == "input_image" {
                return true;
            }
            // 或者是 message 类型中包含 input_image
            if item.item_type == "message"
                && let Some(Value::Array(contents)) = &item.content
            {
                for content in contents {
                    if let Some(type_val) = content.get("type")
                        && type_val == "input_image"
                    {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// 解析 Request 的 input 字段
///
/// 处理各种 InputItem 类型，返回结构化的解析结果
///
/// # Arguments
/// * `request` - 请求
/// ```
pub fn parse_request_input(
    request: &Request,
    media_marker: Option<impl Into<String>>,
) -> Result<ParsedInput, Error> {
    let mut messages = Vec::new();
    let mut image_sources = Vec::new();

    let media_marker = media_marker
        .map(|s| s.into())
        .unwrap_or(mtmd_default_marker().to_string());

    match &request.input {
        Input::Text(text) => {
            messages.push(LlamaChatMessage::new(
                MessageRole::User.to_string(),
                text.to_string(),
            )?);
        }
        Input::Items(input_items) => {
            for input_item in input_items {
                match input_item.item_type.as_str() {
                    // 纯文本类型
                    "text" => {
                        let text = input_item
                            .content
                            .as_ref()
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .unwrap_or_default();
                        messages.push(LlamaChatMessage::new(MessageRole::User.to_string(), text)?);
                    }
                    // 图片类型
                    "input_image" => {
                        if let Some(image) = extract_image_source(input_item) {
                            messages.push(LlamaChatMessage::new(
                                MessageRole::User.to_string(),
                                media_marker.clone(),
                            )?);
                            image_sources.push(image);
                        }
                    }
                    // 消息类型（包含 role 和 content 数组）
                    "message" => {
                        let role = input_item
                            .role
                            .clone()
                            .unwrap_or_else(|| "user".to_string());
                        if let Some(contents) = &input_item.content {
                            let message_content = parse_message_content(
                                role,
                                contents.clone(),
                                Some(media_marker.clone()),
                            )?;
                            messages.extend(message_content.messages);
                            image_sources.extend(message_content.image_sources);
                        }
                    }
                    _ => {
                        tracing::warn!("Unknown content type in message: {}", input_item.item_type);
                    }
                }
            }
        }
    }

    Ok(ParsedInput {
        messages,
        image_sources,
    })
}

/// 解析消息内容
fn parse_message_content(
    role: String,
    contents: Value,
    media_marker: Option<impl Into<String>>,
) -> Result<ParsedInput, Error> {
    let mut messages = Vec::new();
    let mut image_sources = Vec::new();

    let media_marker = media_marker
        .map(|s| s.into())
        .unwrap_or(mtmd_default_marker().to_string());

    let contents: Vec<ContentPart> = serde_json::from_value(contents)?;

    for content in contents {
        match content.content_type.as_str() {
            // 文本内容处理
            "input_text" => {
                if let Some(text) = content.text {
                    messages.push(LlamaChatMessage::new(role.clone(), text)?);
                }
            }
            // 图片类型
            "input_image" => {
                if let Some(image) = extract_image_source_from_content(&content) {
                    messages.push(LlamaChatMessage::new(
                        MessageRole::User.to_string(),
                        media_marker.clone(),
                    )?);
                    image_sources.push(image);
                }
            }
            _ => {
                tracing::warn!("Unknown content type in message: {}", content.content_type);
            }
        }
    }

    Ok(ParsedInput {
        messages,
        image_sources,
    })
}

/// 从 data URI 中提取纯 base64 数据
///
/// data URI 格式: data:[<mediatype>][;base64],<data>
/// 示例: data:image/png;base64,iVBORw0KGgo...
fn extract_base64_from_data_uri(data_uri: &str) -> Option<String> {
    if !data_uri.starts_with("data:") {
        return None;
    }

    // 找到逗号的位置，逗号后面才是真正的 base64 数据
    data_uri
        .find(',')
        .map(|comma_pos| data_uri[comma_pos + 1..].to_string())
}

/// 从 InputItem 提取图片来源
fn extract_image_source(item: &InputItem) -> Option<ImageSource> {
    // http/https
    if let Some(url) = &item.image_url
        && (url.starts_with("http") || url.starts_with("https"))
    {
        return Some(ImageSource::Url(url.clone()));
    }
    // base64 (data URI 格式)
    if let Some(url) = &item.image_url
        && url.starts_with("data:")
    {
        // 从 data URI 中提取纯 base64 数据
        if let Some(base64_data) = extract_base64_from_data_uri(url) {
            return Some(ImageSource::Base64(base64_data));
        }
    }
    // file id
    if let Some(file_id) = &item.text {
        return Some(ImageSource::FileId(file_id.clone()));
    }
    None
}

/// 从 ContentPart 提取图片来源
fn extract_image_source_from_content(content: &ContentPart) -> Option<ImageSource> {
    // http/https
    if let Some(url) = &content.image_url
        && (url.starts_with("http") || url.starts_with("https"))
    {
        return Some(ImageSource::Url(url.clone()));
    }
    // base64 (data URI 格式)
    if let Some(url) = &content.image_url
        && url.starts_with("data:")
    {
        // 从 data URI 中提取纯 base64 数据
        if let Some(base64_data) = extract_base64_from_data_uri(url) {
            return Some(ImageSource::Base64(base64_data));
        }
    }
    // file id
    if let Some(file_id) = &content.file_id {
        return Some(ImageSource::FileId(file_id.clone()));
    }
    None
}

// impl MultimodalRequest {
//     pub fn new() -> Self {
//         Self::default()
//     }

//     /// 从标准 Request 创建
//     pub fn from_request(request: Request) -> Self {
//         Self {
//             request,
//             medias: Vec::new(),
//             image_max_resolution: 768,
//         }
//     }

//     /// 是否为多模态请求
//     pub fn is_multimodal(&self) -> bool {
//         !self.medias.is_empty()
//     }

//     /// 添加媒体
//     pub fn with_media(mut self, media: MediaData) -> Self {
//         self.medias.push(media);
//         self
//     }

//     /// 设置媒体列表
//     pub fn with_medias(mut self, medias: Vec<MediaData>) -> Self {
//         self.medias = medias;
//         self
//     }

//     /// 添加媒体文件
//     pub fn with_media_file(mut self, path: &str) -> Result<Self, Error> {
//         let file_type = infer::get_from_path(path)
//             .map_err(|e| {
//                 error!("Failed to infer file type: {}", e);
//                 e
//             })?
//             .ok_or_else(|| Error::InvalidInput {
//                 field: "file_type".to_string(),
//                 message: "Failed to infer file type".to_string(),
//             })?;

//         if file_type.mime_type().starts_with("image/") {
//             let media = self.load_image_file(path)?;
//             self.medias.push(media);
//         } else {
//             return Err(Error::InvalidInput {
//                 field: "file_type".to_string(),
//                 message: format!("Unsupported file type: {}", file_type.mime_type()),
//             });
//         }

//         Ok(self)
//     }

//     /// 加载图片文件
//     fn load_image_file(&self, path: &str) -> Result<MediaData, Error> {
//         let mut img = Image::from_file(path)?;

//         let max_resolution = img.longest().min(self.image_max_resolution);
//         img = img.resize_to_longest(max_resolution)?;

//         info!(
//             "image size width: {}, height: {}, path: {}",
//             img.width(),
//             img.height(),
//             path
//         );

//         let data = img.to_vec()?;
//         Ok(MediaData::new_image(data))
//     }

// /// 根据请求准备消息（转换为 LlamaChatMessage 列表）
// /// TODO: 多模态消息处理需要重构以支持标准 InputItem
// pub fn to_messages(&self) -> Result<Vec<LlamaChatMessage>, Error> {
//     use crate::types::MessageRole;
//     let mut messages = Vec::new();

//     // 系统消息
//     if let Some(system_prompt) = &self.request.instructions
//         && !system_prompt.is_empty()
//     {
//         messages.push(LlamaChatMessage::new(
//             MessageRole::System.to_string(),
//             system_prompt.clone(),
//         )?);
//     }

//     // 添加历史消息（外部历史和内部缓存历史二选一，优先外部历史）
//     if let Some(history) = &self.history {
//         messages.extend(history.to_llama_message()?);
//     } else if !self.session_id.is_empty() {
//         match HistoryMessage::from_cache(self.session_id.clone()) {
//             Ok(history) => messages.extend(history.to_llama_message()?),
//             Err(e) => info!("No cached history for session '{}': {}", self.session_id, e),
//         }
//     }

//     // 用户消息：多模态时添加媒体标记
//     let user_prompt = if self.is_multimodal() {
//         let default_marker = mtmd_default_marker().to_string();
//         let media_marker = self.media_marker.as_ref().unwrap_or(&default_marker);
//         let markers = media_marker.repeat(self.medias.len());
//         format!(
//             "{} {}",
//             self.request.user_prompt().replace(&default_marker, ""),
//             markers
//         )
//     } else {
//         self.request.user_prompt()
//     };

//     if !user_prompt.is_empty() {
//         info!("User prompt: {}", user_prompt);
//         messages.push(LlamaChatMessage::new(
//             MessageRole::User.to_string(),
//             user_prompt,
//         )?);
//     }

//     Ok(messages)
// }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_base64_from_data_uri() {
        // 标准 data URI
        let data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let result = extract_base64_from_data_uri(data_uri).unwrap();
        assert_eq!(
            result,
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        );

        // 不带 media type 的 data URI
        let data_uri2 = "data:base64,SGVsbG8gV29ybGQ=";
        let result2 = extract_base64_from_data_uri(data_uri2).unwrap();
        assert_eq!(result2, "SGVsbG8gV29ybGQ=");

        // 普通 URL 应该返回 None
        let url = "https://example.com/image.png";
        assert!(extract_base64_from_data_uri(url).is_none());

        // 普通 base64 字符串应该返回 None
        let plain_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        assert!(extract_base64_from_data_uri(plain_base64).is_none());
    }

    #[test]
    fn test_extract_base64_from_data_uri_with_newlines() {
        // 测试包含换行符的 data URI (某些系统会添加换行)
        let data_uri = "data:image/png;base64,iVBORw0KGgo\nAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ\nAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";
        let result = extract_base64_from_data_uri(data_uri).unwrap();
        // 注意：换行符会被保留在 base64 字符串中，解码时需要处理
        assert!(result.contains("\n"));
    }
}
