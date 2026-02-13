//! 钩子上下文

use std::collections::HashMap;

use async_openai::types::chat::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
};
use serde_json::Value;
use tracing::error;

use crate::hooks::pipeline_state::PipelineState;
use crate::unified_message::ImageSource;
use crate::{PipelineConfig, request::extract_session_id, unified_message::UnifiedMessage};

/// 钩子上下文
///
/// 在钩子执行期间传递的上下文信息，包含请求、响应、会话 ID 等
#[derive(Debug, Default, Clone)]
pub struct HookContext {
    /// 推理请求
    pub request: CreateChatCompletionRequest,
    /// Pipeline 配置
    pub config: PipelineConfig,
    /// 会话 ID
    pub session_id: Option<String>,
    /// 推理响应（非流式）
    pub response: Option<CreateChatCompletionResponse>,
    /// 流式响应的最后一块（用于流式推理）
    pub stream_last_chunk: Option<CreateChatCompletionStreamResponse>,
    /// 流式推理收集的完整文本
    pub stream_collected_text: Option<String>,
    /// 额外元数据（用于钩子间传递数据）
    pub metadata: HashMap<String, Value>,
    /// 推理开始时间
    pub start_time: Option<std::time::Instant>,
    /// Pipeline 状态（新增）
    pub pipeline_state: PipelineState,
}

impl HookContext {
    /// 创建新的上下文
    ///
    /// 从请求和配置创建上下文
    pub fn new(request: &CreateChatCompletionRequest, config: &PipelineConfig) -> Self {
        let session_id = extract_session_id(request);

        let unified_messages: Vec<UnifiedMessage> = request
            .messages
            .iter()
            .cloned()
            .map(UnifiedMessage::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                error!("Failed to convert messages: {}", e);
                e
            })
            .unwrap_or_default();

        Self {
            request: request.clone(),
            response: None,
            stream_last_chunk: None,
            stream_collected_text: None,
            session_id,
            config: config.clone(),
            metadata: Default::default(),
            start_time: None,
            pipeline_state: PipelineState::new(unified_messages),
        }
    }

    /// 设置配置
    pub fn with_config(mut self, config: &PipelineConfig) -> Self {
        self.config = config.clone();
        self
    }

    /// 设置会话 ID
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// 设置响应
    pub fn set_response(&mut self, response: CreateChatCompletionResponse) {
        self.response = Some(response);
    }

    /// 设置流式最后一块
    pub fn set_stream_last_chunk(&mut self, chunk: CreateChatCompletionStreamResponse) {
        self.stream_last_chunk = Some(chunk);
    }

    /// 设置流式收集的文本
    pub fn set_stream_collected_text(&mut self, text: String) {
        self.stream_collected_text = Some(text);
    }

    /// 添加元数据
    pub fn insert_metadata(&mut self, key: impl Into<String>, value: Value) {
        self.metadata.insert(key.into(), value);
    }

    /// 获取元数据
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// 获取推理耗时（毫秒）
    pub fn elapsed_ms(&self) -> Option<u64> {
        self.start_time
            .map(|start| start.elapsed().as_millis() as u64)
    }

    /// 获取当前请求的媒体资源
    pub fn media_sources(&self) -> Vec<ImageSource> {
        let mut sources = Vec::new();

        for msg in &self.pipeline_state.raw_messages {
            sources.extend(msg.get_image_sources().into_iter().cloned());
        }

        sources
    }

    /// 检查请求是否包含图片（多模态请求）
    pub fn is_multimodal_request(&self) -> bool {
        for msg in &self.pipeline_state.raw_messages {
            if msg.has_image() {
                return true;
            }
        }
        false
    }
}
