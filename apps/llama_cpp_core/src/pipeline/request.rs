//! Request processing module for pipeline operations
use llama_cpp_2::{model::LlamaChatMessage, mtmd::mtmd_default_marker};
use serde::{Deserialize, Serialize};
use tracing::{error, info};
use uuid::Uuid;

use crate::{HistoryMessage, MediaData, PromptMessageRole, error::Error, utils::image::Image};

/// 生成请求结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// 会话ID（可选）
    /// 用于在 Pipeline 内部自动管理历史上下文，实现并发隔离
    /// 不同 session_id 的对话历史完全隔离
    #[serde(default)]
    pub session_id: String,

    /// 历史消息
    #[serde(default)]
    pub history: Option<HistoryMessage>,

    /// 系统提示词（可选）
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// 用户提示词
    #[serde(default)]
    pub user_prompt: String,
    /// 媒体数据（多模态场景）
    #[serde(default)]
    pub medias: Vec<MediaData>,

    /// Media marker. If not provided, the default marker will be used.
    #[serde(default)]
    pub media_marker: Option<String>,

    /// Image max resolution, default is 768，防止图片过大导致编码失败
    #[serde(default)]
    pub image_max_resolution: u32,

    /// Whether to keep context between requests
    #[serde(default)]
    pub keep_context: bool,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        let session_id = Uuid::new_v4().to_string();
        Self {
            session_id,
            history: None,
            system_prompt: None,
            user_prompt: String::new(),
            medias: Vec::new(),
            media_marker: Some("<__media__>".to_string()),
            image_max_resolution: 768,
            keep_context: false,
        }
    }
}
impl GenerateRequest {
    pub fn new() -> Self {
        GenerateRequest::default()
    }

    /// 创建纯文本请求
    pub fn text(user_prompt: impl Into<String>) -> Self {
        Self {
            user_prompt: user_prompt.into(),
            ..Default::default()
        }
    }

    /// 创建多模态请求
    pub fn media(media: MediaData) -> Self {
        Self {
            medias: vec![media],
            ..Default::default()
        }
    }

    /// 设置系统提示词
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system_prompt = Some(system.into());
        self
    }

    /// 设置会话ID，用于自动管理历史上下文
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = session_id.into();
        self
    }

    /// 设置历史消息
    pub fn with_history(mut self, history: HistoryMessage) -> Self {
        self.history = Some(history);
        self
    }

    /// 设置媒体标记
    pub fn with_media_marker(mut self, media_marker: impl Into<String>) -> Self {
        self.media_marker = Some(media_marker.into());
        self
    }

    /// 添加媒体
    pub fn with_media(mut self, media: MediaData) -> Self {
        self.medias.push(media);
        self
    }

    /// 设置媒体列表
    pub fn with_medias(mut self, medias: Vec<MediaData>) -> Self {
        self.medias = medias;
        self
    }

    /// 添加媒体缓冲区数据
    pub fn with_media_buffer(mut self, buf: &[u8]) -> Result<Self, Error> {
        let kind = infer::get(buf).ok_or_else(|| Error::InvalidInput {
            field: "file_type".to_string(),
            message: "Failed to infer file type".to_string(),
        })?;

        if kind.mime_type().starts_with("image/") {
            self.medias.push(MediaData::new_image(buf.to_vec()));
        } else {
            // 暂不支持的文件类型
            return Err(Error::InvalidInput {
                field: "file_type".to_string(),
                message: format!("Unsupported file type: {}", kind.mime_type()),
            });
        }

        Ok(self)
    }

    /// 设置图片最大分辨率
    pub fn with_image_max_resolution(mut self, image_max_resolution: u32) -> Self {
        self.image_max_resolution = image_max_resolution;
        self
    }

    pub fn with_keep_context(mut self, keep_context: bool) -> Self {
        self.keep_context = keep_context;
        self
    }

    /// 添加媒体文件
    pub fn with_media_file(mut self, path: &str) -> Result<Self, Error> {
        // 判断文件类型
        let file_type = infer::get_from_path(path)
            .map_err(|e| {
                error!("Failed to infer file type: {}", e);
                e
            })?
            .ok_or_else(|| Error::InvalidInput {
                field: "file_type".to_string(),
                message: "Failed to infer file type".to_string(),
            })?;

        if file_type.mime_type().starts_with("image/") {
            let media = self.load_image_file(path)?;
            self.medias.push(media);
        } else {
            // 暂不支持的文件类型
            return Err(Error::InvalidInput {
                field: "file_type".to_string(),
                message: format!("Unsupported file type: {}", file_type.mime_type()),
            });
        }

        Ok(self)
    }

    /// 添加图片 tensor 缓存
    pub fn load_image_tensor_buffer(
        &self,
        data: Vec<u8>,
        height: u32,
        width: u32,
        channels: usize,
    ) -> Result<MediaData, Error> {
        let mut img = Image::from_tensor(data, height, width, channels)?;

        let max_resolution = img.longest().min(self.image_max_resolution);
        img = img.resize_to_longest(max_resolution)?;

        let data = img.to_vec()?;
        let media_data = MediaData::new_image(data);
        Ok(media_data)
    }

    /// Load image from the specified file path
    pub fn load_image_file(&self, path: &str) -> Result<MediaData, Error> {
        let mut img = Image::from_file(path)?;

        let max_resolution = img.longest().min(self.image_max_resolution);
        img = img.resize_to_longest(max_resolution)?;

        info!(
            "image size width: {}, height: {}, path: {}",
            img.width(),
            img.height(),
            path
        );

        let data = img.to_vec()?;
        Ok(MediaData::new_image(data))
    }

    /// 是否为多模态请求
    pub fn is_multimodal(&self) -> bool {
        !self.medias.is_empty()
    }
}

impl GenerateRequest {
    /// 根据请求准备消息
    ///
    /// 将 GenerateRequest 转换为 LlamaChatMessage 列表
    pub fn to_messages(&self) -> Result<Vec<LlamaChatMessage>, Error> {
        let mut messages = Vec::new();

        // 系统消息
        if let Some(system_prompt) = &self.system_prompt
            && !system_prompt.is_empty()
        {
            messages.push(LlamaChatMessage::new(
                PromptMessageRole::System.to_string(),
                system_prompt.clone(),
            )?);
        }

        // 添加历史消息（外部历史和内部缓存历史二选一，优先外部历史）
        if let Some(history) = &self.history {
            // 使用外部传入的历史
            messages.extend(history.to_llama_message()?);
        } else if !self.session_id.is_empty() {
            // 从缓存加载历史
            match HistoryMessage::from_cache(self.session_id.clone()) {
                Ok(history) => messages.extend(history.to_llama_message()?),
                Err(e) => info!("No cached history for session '{}': {}", self.session_id, e),
            }
        }

        // 用户消息：多模态时添加媒体标记
        let user_prompt = if self.is_multimodal() {
            // 媒体标记
            let default_marker = mtmd_default_marker().to_string();
            let media_marker = self.media_marker.as_ref().unwrap_or(&default_marker);
            let markers = media_marker.repeat(self.medias.len());
            format!(
                "{} {}",
                self.user_prompt.replace(&default_marker, ""),
                markers
            )
        } else {
            self.user_prompt.clone()
        };
        info!("User prompt: {}", user_prompt);
        messages.push(LlamaChatMessage::new(
            PromptMessageRole::User.to_string(),
            user_prompt,
        )?);

        Ok(messages)
    }
}
