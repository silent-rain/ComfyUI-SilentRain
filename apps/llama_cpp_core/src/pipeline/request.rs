//! Request processing module for pipeline operations
//!
//! ## 架构说明
//!
//! 本模块使用 `async_openai::CreateChatCompletionRequest` 标准结构

use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, CreateChatCompletionRequest,
};
use llama_cpp_2::{model::LlamaChatMessage, mtmd::mtmd_default_marker};

use crate::{error::Error, types::MessageRole};

// 导出 async-openai 类型
pub use async_openai::types::chat::{
    ChatCompletionResponseMessage, CreateChatCompletionResponse, FunctionCall, FunctionObject,
};

/// 图片来源
#[derive(Debug, Clone)]
pub enum ImageSource {
    /// URL（http/https 或 data:image base64）
    Url(String),
    /// data:{};base64,{}
    Base64(String),
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
pub fn is_multimodal_request(request: &CreateChatCompletionRequest) -> bool {
    for msg in request.messages.clone() {
        if let ChatCompletionRequestMessage::User(user_msg) = msg
            && let ChatCompletionRequestUserMessageContent::Array(parts) = &user_msg.content
        {
            for part in parts {
                if matches!(
                    part,
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(_)
                ) {
                    return true;
                }
            }
        }
    }
    false
}

/// 解析 Request 的消息
///
/// 处理标准的 `CreateChatCompletionRequest`，将消息转换为 LlamaChatMessage 列表
/// 支持多模态：图片内容会被提取，并在对应用户消息中添加媒体标记
///
/// # Arguments
/// * `request` - 标准 Chat Completions 请求
/// * `media_marker` - 媒体标记（用于多模态，默认使用 mtmd 默认标记）
pub fn parse_request_input(
    request: &CreateChatCompletionRequest,
    media_marker: Option<impl Into<String>>,
) -> Result<ParsedInput, Error> {
    let mut messages = Vec::new();
    let mut image_sources = Vec::new();

    let media_marker = media_marker
        .map(|s| s.into())
        .unwrap_or(mtmd_default_marker().to_string());

    for msg in request.messages.clone() {
        match msg {
            // 系统消息
            ChatCompletionRequestMessage::System(system_msg) => {
                let content = extract_system_content(&system_msg.content);
                if !content.is_empty() {
                    messages.push(LlamaChatMessage::new(
                        MessageRole::System.to_string(),
                        content,
                    )?);
                }
            }
            // 用户消息：支持文本和多模态（文本+图片）
            ChatCompletionRequestMessage::User(user_msg) => {
                let (user_text, img_sources) =
                    parse_user_message(&user_msg.content, &media_marker)?;

                // 收集图片来源
                image_sources.extend(img_sources);

                // 添加用户消息（包含媒体标记）
                if !user_text.is_empty() {
                    messages.push(LlamaChatMessage::new(
                        MessageRole::User.to_string(),
                        user_text,
                    )?);
                }
            }
            // 助手消息
            ChatCompletionRequestMessage::Assistant(assistant_msg) => {
                if let Some(content) = &assistant_msg.content {
                    let text = extract_assistant_content(content);
                    if !text.is_empty() {
                        messages.push(LlamaChatMessage::new(
                            MessageRole::Assistant.to_string(),
                            text,
                        )?);
                    }
                }
            }
            // 工具/函数消息：暂不支持，记录日志后跳过
            _ => {
                tracing::debug!("Skipping unsupported message type in request parsing");
            }
        }
    }

    Ok(ParsedInput {
        messages,
        image_sources,
    })
}

/// 提取系统消息内容
fn extract_system_content(content: &ChatCompletionRequestSystemMessageContent) -> String {
    match content {
        ChatCompletionRequestSystemMessageContent::Text(text) => text.clone(),
        ChatCompletionRequestSystemMessageContent::Array(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ChatCompletionRequestSystemMessageContentPart::Text(t) => Some(t.text.clone()),
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

/// 解析用户消息内容
///
/// 处理文本或多模态内容（文本+图片数组）
/// 多模态时，在文本后追加媒体标记（参考 llama.cpp mtmd 规范）
///
/// # Returns
/// - `(用户消息文本, 图片来源列表)`
fn parse_user_message(
    content: &ChatCompletionRequestUserMessageContent,
    media_marker: &str,
) -> Result<(String, Vec<ImageSource>), Error> {
    match content {
        // 纯文本消息
        ChatCompletionRequestUserMessageContent::Text(text) => Ok((text.clone(), Vec::new())),
        // 多模态消息：文本 + 图片数组
        ChatCompletionRequestUserMessageContent::Array(parts) => {
            let mut text_parts = Vec::new();
            let mut image_count = 0;
            let mut image_sources = Vec::new();

            for part in parts {
                match part {
                    ChatCompletionRequestUserMessageContentPart::Text(text_part) => {
                        // 清理文本中可能存在的默认标记（避免重复）
                        let clean_text = text_part.text.replace(media_marker, "");
                        text_parts.push(clean_text);
                    }
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(image_part) => {
                        image_count += 1;
                        // 提取图片来源
                        if let Some(source) = extract_image_source(&image_part.image_url.url) {
                            image_sources.push(source);
                        }
                    }
                    _ => {} // 忽略其他类型
                }
            }

            // 合并文本，添加媒体标记
            let mut user_text = text_parts.join(" ");

            // 多模态时，在消息末尾添加媒体标记
            if image_count > 0 {
                let markers = media_marker.repeat(image_count);
                user_text = format!("{} {}", user_text.trim(), markers);
            }

            Ok((user_text, image_sources))
        }
    }
}

/// 提取助手消息内容
fn extract_assistant_content(content: &ChatCompletionRequestAssistantMessageContent) -> String {
    match content {
        ChatCompletionRequestAssistantMessageContent::Text(text) => text.clone(),
        ChatCompletionRequestAssistantMessageContent::Array(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ChatCompletionRequestAssistantMessageContentPart::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

/// 从图片 URL 提取图片来源
fn extract_image_source(url: &str) -> Option<ImageSource> {
    if url.starts_with("http") || url.starts_with("https") {
        Some(ImageSource::Url(url.to_string()))
    } else if url.starts_with("data:") {
        // 从 data URI 中提取纯 base64 数据
        extract_base64_from_data_uri(url).map(ImageSource::Base64)
    } else {
        None
    }
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
}
