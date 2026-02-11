//! 历史消息管理
//!
//! 提供 HistoryMessage 结构体，支持多模态和 Tool 调用
//! 底层使用 ChatHistoryManager 进行持久化

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{error, info, warn};

use crate::error::Error;
use crate::types::MessageRole;
use crate::unified_message::{ImageSource, UnifiedMessage};

use super::manager::chat_history;
use super::session::SessionContext;

/// 默认最大历史消息数（保留最近100条）
const DEFAULT_MAX_HISTORY_MESSAGES: usize = 100;
/// 警告阈值，超过此值会触发日志警告
const WARN_HISTORY_THRESHOLD: usize = 80;

/// 历史消息管理器
///
/// 内部直接使用 UnifiedMessage 存储，支持：
/// - 文本消息
/// - 多模态消息（图片）
/// - Tool 调用和结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryMessage {
    /// 消息列表（UnifiedMessage 格式）
    entries: Vec<UnifiedMessage>,
    /// 最大消息数限制
    max_entries: usize,
}

impl Default for HistoryMessage {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            max_entries: DEFAULT_MAX_HISTORY_MESSAGES,
        }
    }
}

impl HistoryMessage {
    pub fn new() -> Self {
        HistoryMessage::default()
    }

    /// 创建指定大小的历史消息管理器
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// 从 SessionContext 创建
    pub(crate) fn from_session(ctx: &SessionContext) -> Self {
        Self {
            entries: ctx.entries().to_vec(),
            max_entries: ctx.max_entries(),
        }
    }

    /// 转换为 SessionContext
    pub(crate) fn to_session(&self, session_id: impl Into<String>) -> SessionContext {
        let mut ctx = SessionContext::with_capacity(session_id, self.max_entries);
        for entry in &self.entries {
            ctx.add_message(entry.clone());
        }
        ctx
    }

    /// 添加消息条目
    ///
    /// 当超过最大限制时，移除最旧的消息（保留系统消息）
    pub fn add_message(&mut self, message: UnifiedMessage) {
        // 检查是否需要淘汰旧消息
        if self.entries.len() >= self.max_entries {
            self.truncate_oldest();
        }

        // 警告检查
        if self.entries.len() >= WARN_HISTORY_THRESHOLD && self.entries.len() < self.max_entries {
            info!(
                "History message count ({}) approaching limit ({})",
                self.entries.len(),
                self.max_entries
            );
        }

        self.entries.push(message);
    }

    /// 添加条目列表
    pub fn add_messages(&mut self, messages: Vec<UnifiedMessage>) {
        for msg in messages {
            self.add_message(msg);
        }
    }

    /// 添加用户文本消息
    pub fn add_user_text(&mut self, text: impl Into<String>) {
        self.add_message(UnifiedMessage::user_text(text));
    }

    /// 添加用户消息（支持多模态）
    pub fn add_user(&mut self, message: UnifiedMessage) {
        assert!(
            message.role == MessageRole::User,
            "Expected user role message"
        );
        self.add_message(message);
    }

    /// 添加助手文本消息
    pub fn add_assistant_text(&mut self, text: impl Into<String>) {
        self.add_message(UnifiedMessage::assistant(text));
    }

    /// 添加助手消息（支持 Tool 调用）
    pub fn add_assistant(&mut self, message: UnifiedMessage) {
        assert!(
            message.role == MessageRole::Assistant,
            "Expected assistant role message"
        );
        self.add_message(message);
    }

    /// 添加系统消息
    pub fn add_system_text(&mut self, text: impl Into<String>) {
        self.add_message(UnifiedMessage::system(text));
    }

    /// 添加系统消息（支持内容块）
    pub fn add_system(&mut self, message: UnifiedMessage) {
        assert!(
            message.role == MessageRole::System,
            "Expected system role message"
        );
        self.add_message(message);
    }

    /// 添加 Tool 结果消息
    pub fn add_tool_result(&mut self, call_id: impl Into<String>, content: impl Into<String>) {
        self.add_message(UnifiedMessage::tool_result(call_id, content));
    }

    /// 淘汰最旧的消息（保留系统消息）
    fn truncate_oldest(&mut self) {
        let first_non_system = self
            .entries
            .iter()
            .position(|e| e.role != MessageRole::System)
            .unwrap_or(0);

        if first_non_system < self.entries.len() {
            let removed = self.entries.remove(first_non_system);
            info!(
                "History truncated: removed oldest {:?} message (count: {})",
                removed.role,
                self.entries.len()
            );
        } else {
            self.entries.remove(0);
        }
    }

    /// 获取所有消息条目
    pub fn entries(&self) -> &[UnifiedMessage] {
        &self.entries
    }

    /// 获取所有消息（克隆）
    pub fn to_messages(&self) -> Vec<UnifiedMessage> {
        self.entries.clone()
    }

    /// 获取最后 n 条消息
    pub fn get_last_n(&self, n: usize) -> Vec<UnifiedMessage> {
        let start = self.entries.len().saturating_sub(n);
        self.entries[start..].to_vec()
    }

    /// 获取当前最大消息数限制
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// 设置最大消息数限制
    pub fn set_max_entries(&mut self, max_entries: usize) {
        self.max_entries = max_entries;
        while self.entries.len() > self.max_entries {
            self.truncate_oldest();
        }
    }

    /// 获取历史消息数量
    pub fn message_count(&self) -> usize {
        self.entries.len()
    }

    /// 清空历史
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// 清理所有消息中的媒体标记
    pub fn sanitize_media_markers(&mut self, marker: &str) {
        for msg in &mut self.entries {
            msg.content = msg.sanitize_media_marker(marker);
        }
    }

    /// 过滤掉包含图片的消息
    pub fn to_text_only_history(&self) -> Vec<UnifiedMessage> {
        self.entries
            .iter()
            .filter(|msg| !msg.has_image())
            .cloned()
            .collect()
    }

    /// 获取所有图片资源引用
    pub fn get_image_references(&self) -> Vec<&ImageSource> {
        self.entries
            .iter()
            .flat_map(|msg| msg.get_image_sources())
            .collect()
    }

    /// 序列化为 JSON 字符串
    pub fn to_json(&self) -> Result<String, Error> {
        serde_json::to_string(self).map_err(Error::Serde)
    }

    /// 从 JSON 字符串反序列化
    pub fn from_json(json: &str) -> Result<Self, Error> {
        serde_json::from_str(json).map_err(Error::Serde)
    }

    /// 计算内容哈希
    fn compute_content_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.entries.len().hash(&mut hasher);
        for entry in &self.entries {
            entry.role.to_string().hash(&mut hasher);
            let content_str = serde_json::to_string(&entry.content).unwrap_or_default();
            content_str.hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }
}

/// 缓存操作（使用 ChatHistoryManager）
impl HistoryMessage {
    /// 从缓存加载历史消息
    pub fn from_cache(session_id: impl AsRef<str>) -> Result<Arc<Self>, Error> {
        let session_id = session_id.as_ref();
        let manager = chat_history();

        match manager.get(session_id) {
            Some(ctx) => {
                info!("Loaded history from cache for session: {}", session_id);
                Ok(Arc::new(HistoryMessage::from_session(&ctx)))
            }
            None => {
                error!("History not found in cache for session: {}", session_id);
                Err(Error::CacheNotFound {
                    key: session_id.to_string(),
                })
            }
        }
    }

    /// 强制更新缓存
    pub fn force_update_cache(&mut self, session_id: impl AsRef<str>) -> Result<(), Error> {
        let session_id = session_id.as_ref();
        let manager = chat_history();

        // 转换为 SessionContext 并保存
        let ctx = self.to_session(session_id);
        manager.save(session_id, ctx)?;

        info!("History saved to cache for session: {}", session_id);
        Ok(())
    }

    /// 保存到缓存
    pub fn save_to_cache(&mut self, session_id: impl AsRef<str>) -> Result<(), Error> {
        self.force_update_cache(session_id)
    }

    /// 从缓存删除
    pub fn remove_from_cache(session_id: impl AsRef<str>) -> Result<(), Error> {
        let session_id = session_id.as_ref();
        let manager = chat_history();

        if manager.remove(session_id).is_some() {
            info!("Removed history from cache for session: {}", session_id);
        } else {
            warn!("No history found to remove for session: {}", session_id);
        }
        Ok(())
    }

    /// 检查缓存是否存在
    pub fn exists_in_cache(session_id: impl AsRef<str>) -> bool {
        chat_history().exists(session_id)
    }
}

/// 会话管理
impl HistoryMessage {
    /// 创建新会话（清空旧历史）
    pub fn new_session(session_id: impl AsRef<str>) -> Self {
        let _ = Self::remove_from_cache(session_id.as_ref());
        Self::new()
    }

    /// 获取所有会话ID列表（从缓存）
    pub fn list_session_ids() -> Result<Vec<String>, Error> {
        Ok(chat_history().list_sessions())
    }

    /// 合并另一个历史记录到当前记录
    pub fn merge(&mut self, other: HistoryMessage) {
        for msg in other.entries {
            self.add_message(msg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_messages() {
        let mut history = HistoryMessage::new();

        history.add_system_text("You are a helpful assistant");
        history.add_user_text("Hello");
        history.add_assistant_text("Hi there!");

        assert_eq!(history.message_count(), 3);
    }

    #[test]
    fn test_truncate_oldest() {
        let mut history = HistoryMessage::with_capacity(3);

        history.add_system_text("System prompt");
        history.add_user_text("Message 1");
        history.add_assistant_text("Response 1");
        history.add_user_text("Message 2");

        assert_eq!(history.message_count(), 3);
        assert_eq!(history.entries[0].role, MessageRole::System);
    }

    #[test]
    fn test_to_text_only_history() {
        let mut history = HistoryMessage::new();

        history.add_user_text("Hello");
        let img_msg = UnifiedMessage::user_with_image(
            "Describe this",
            ImageSource::Url("http://example.com/img.png".to_string()),
        );
        history.add_user(img_msg);

        let text_only = history.to_text_only_history();
        assert_eq!(text_only.len(), 1);
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut history = HistoryMessage::new();
        history.add_system_text("System");
        history.add_user_text("User message");
        history.add_assistant_text("Assistant response");

        let json = history.to_json().unwrap();
        let restored = HistoryMessage::from_json(&json).unwrap();

        assert_eq!(restored.message_count(), history.message_count());
    }

    #[test]
    fn test_session_conversion() {
        let mut history = HistoryMessage::with_capacity(50);
        history.add_system_text("System");
        history.add_user_text("User");
        history.add_assistant_text("Assistant");

        let ctx = history.to_session("test_session");
        assert_eq!(ctx.session_id(), "test_session");
        assert_eq!(ctx.message_count(), 3);
        assert_eq!(ctx.max_entries(), 50);

        let restored = HistoryMessage::from_session(&ctx);
        assert_eq!(restored.message_count(), 3);
    }
}
