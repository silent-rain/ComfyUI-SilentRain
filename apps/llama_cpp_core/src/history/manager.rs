//! 聊天历史管理器
//!
//! 专用全局缓存，管理服务多用户的会话历史
//! 完全独立于 global_cache，只存储 session_id -> SessionContext

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use dashmap::DashMap;
use tracing::{info, warn};

use crate::error::Error;
use crate::unified_message::{ImageSource, UnifiedMessage};

use super::session::SessionContext;

/// 默认最大会话数
const DEFAULT_MAX_SESSIONS: usize = 1000;
/// 默认会话TTL（1小时，毫秒）
const DEFAULT_SESSION_TTL_MS: u64 = 3600 * 1000;
/// 默认清理间隔（5分钟，毫秒）
const DEFAULT_CLEANUP_INTERVAL_MS: u64 = 300 * 1000;

/// 获取当前时间戳（毫秒）
fn now_ms() -> u64 {
    chrono::Utc::now().timestamp_millis() as u64
}

/// 全局单例
static CHAT_HISTORY: OnceLock<Arc<ChatHistoryManager>> = OnceLock::new();

/// 聊天历史管理器
///
/// 专用全局缓存，只存储会话历史，与其他缓存隔离
#[derive(Debug)]
pub struct ChatHistoryManager {
    /// 会话存储: session_id -> SessionContext
    sessions: DashMap<String, SessionContext>,
    /// 最大会话数
    max_sessions: usize,
    /// 会话TTL（毫秒）
    ttl_ms: u64,
    /// 最后清理时间（毫秒）
    last_cleanup: std::sync::RwLock<u64>,
}

impl Default for ChatHistoryManager {
    fn default() -> Self {
        Self {
            sessions: DashMap::new(),
            max_sessions: DEFAULT_MAX_SESSIONS,
            ttl_ms: DEFAULT_SESSION_TTL_MS,
            last_cleanup: std::sync::RwLock::new(now_ms()),
        }
    }
}

impl ChatHistoryManager {
    /// 获取全局单例
    pub fn global() -> Arc<ChatHistoryManager> {
        CHAT_HISTORY
            .get_or_init(|| Arc::new(ChatHistoryManager::default()))
            .clone()
    }

    /// 创建自定义配置的管理器
    pub fn from_config(max_sessions: usize, ttl: Duration) -> Self {
        Self {
            sessions: DashMap::new(),
            max_sessions,
            ttl_ms: ttl.as_millis() as u64,
            last_cleanup: std::sync::RwLock::new(now_ms()),
        }
    }

    /// 创建自定义配置的管理器（毫秒版本）
    pub fn from_config_ms(max_sessions: usize, ttl_ms: u64) -> Self {
        Self {
            sessions: DashMap::new(),
            max_sessions,
            ttl_ms,
            last_cleanup: std::sync::RwLock::new(now_ms()),
        }
    }

    /// 获取或创建会话
    pub fn get_or_create(&self, session_id: impl AsRef<str>) -> SessionContext {
        let session_id = session_id.as_ref();

        self.maybe_cleanup();

        if let Some(mut entry) = self.sessions.get_mut(session_id) {
            entry.touch();
            return entry.clone();
        }

        // 检查是否需要淘汰
        self.evict_if_needed();

        // 创建新会话
        let ctx = SessionContext::new(session_id);
        self.sessions.insert(session_id.to_string(), ctx.clone());
        info!("Created new session context: {}", session_id);
        ctx
    }

    /// 获取会话（如果不存在返回None）
    pub fn get(&self, session_id: impl AsRef<str>) -> Option<SessionContext> {
        let session_id = session_id.as_ref();

        self.sessions.get_mut(session_id).map(|mut entry| {
            entry.touch();
            entry.clone()
        })
    }

    /// 保存会话
    pub fn save(&self, session_id: impl AsRef<str>, ctx: SessionContext) -> Result<(), Error> {
        let session_id = session_id.as_ref();

        self.maybe_cleanup();
        self.evict_if_needed();

        self.sessions.insert(session_id.to_string(), ctx);
        Ok(())
    }

    /// 删除会话
    pub fn remove(&self, session_id: impl AsRef<str>) -> Option<SessionContext> {
        let session_id = session_id.as_ref();
        self.sessions.remove(session_id).map(|(_, v)| v)
    }

    /// 检查会话是否存在
    pub fn exists(&self, session_id: impl AsRef<str>) -> bool {
        self.sessions.contains_key(session_id.as_ref())
    }

    /// 列出所有会话ID
    pub fn list_sessions(&self) -> Vec<String> {
        self.sessions
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// 获取会话数量
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// 清空所有会话
    pub fn clear(&self) {
        self.sessions.clear();
        info!("All session histories cleared");
    }

    /// 清理过期会话
    pub fn clear_expired(&self) -> usize {
        let now = now_ms();
        let expired_keys: Vec<String> = self
            .sessions
            .iter()
            .filter(|entry| {
                let elapsed = now.saturating_sub(entry.last_accessed());
                elapsed > self.ttl_ms
            })
            .map(|entry| entry.key().clone())
            .collect();

        let count = expired_keys.len();
        for key in expired_keys {
            self.sessions.remove(&key);
            info!("Removed expired session: {}", key);
        }

        if count > 0 {
            info!("Cleared {} expired sessions", count);
        }

        count
    }

    /// 获取会话统计信息
    pub fn get_stats(&self, session_id: impl AsRef<str>) -> Option<SessionStats> {
        self.sessions
            .get(session_id.as_ref())
            .map(|entry| SessionStats {
                session_id: entry.key().clone(),
                message_count: entry.message_count(),
                created_at: entry.created_at(),
                last_accessed: entry.last_accessed(),
                idle_duration_ms: entry.idle_duration_ms(),
                access_count: entry.access_count(),
                version: entry.version(),
            })
    }

    /// 获取所有会话统计
    pub fn get_all_stats(&self) -> Vec<SessionStats> {
        self.sessions
            .iter()
            .filter_map(|entry| self.get_stats(entry.key()))
            .collect()
    }

    /// 添加消息到会话
    pub fn add_message(
        &self,
        session_id: impl AsRef<str>,
        message: UnifiedMessage,
    ) -> Result<(), Error> {
        let session_id = session_id.as_ref();

        if let Some(mut entry) = self.sessions.get_mut(session_id) {
            entry.add_message(message);
            entry.increment_version();
            Ok(())
        } else {
            Err(Error::CacheNotFound {
                key: session_id.to_string(),
            })
        }
    }

    /// 添加消息到会话列表
    pub fn add_messages(
        &self,
        session_id: impl AsRef<str>,
        messages: &Vec<UnifiedMessage>,
    ) -> Result<(), Error> {
        let session_id = session_id.as_ref();

        if let Some(mut entry) = self.sessions.get_mut(session_id) {
            for message in messages {
                entry.add_message(message.clone());
            }
            entry.increment_version();
            Ok(())
        } else {
            Err(Error::CacheNotFound {
                key: session_id.to_string(),
            })
        }
    }

    /// 添加用户文本消息
    pub fn add_user_text(
        &self,
        session_id: impl AsRef<str>,
        text: impl Into<String>,
    ) -> Result<(), Error> {
        self.add_message(session_id, UnifiedMessage::user_text(text))
    }

    /// 添加助手文本消息
    pub fn add_assistant_text(
        &self,
        session_id: impl AsRef<str>,
        text: impl Into<String>,
    ) -> Result<(), Error> {
        self.add_message(session_id, UnifiedMessage::assistant(text))
    }

    /// 添加系统文本消息
    pub fn add_system_text(
        &self,
        session_id: impl AsRef<str>,
        text: impl Into<String>,
    ) -> Result<(), Error> {
        self.add_message(session_id, UnifiedMessage::system(text))
    }

    /// 添加 Tool 结果消息
    pub fn add_tool_result(
        &self,
        session_id: impl AsRef<str>,
        call_id: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<(), Error> {
        self.add_message(session_id, UnifiedMessage::tool_result(call_id, content))
    }

    /// 获取会话消息列表
    pub fn get_messages(&self, session_id: impl AsRef<str>) -> Option<Vec<UnifiedMessage>> {
        self.sessions
            .get(session_id.as_ref())
            .map(|entry| entry.entries().to_vec())
    }

    /// 获取最后 n 条消息
    pub fn get_last_n_messages(
        &self,
        session_id: impl AsRef<str>,
        n: usize,
    ) -> Option<Vec<UnifiedMessage>> {
        self.sessions
            .get(session_id.as_ref())
            .map(|entry| entry.get_last_n(n))
    }

    /// 清空会话消息
    pub fn clear_messages(&self, session_id: impl AsRef<str>) -> Result<(), Error> {
        let session_id = session_id.as_ref();
        if let Some(mut entry) = self.sessions.get_mut(session_id) {
            entry.clear();
            entry.increment_version();
            Ok(())
        } else {
            Err(Error::CacheNotFound {
                key: session_id.to_string(),
            })
        }
    }

    /// 获取会话中所有图片资源引用
    pub fn get_image_references(&self, session_id: impl AsRef<str>) -> Vec<ImageSource> {
        self.sessions
            .get(session_id.as_ref())
            .map(|entry| {
                entry
                    .entries()
                    .iter()
                    .flat_map(|msg| msg.get_image_sources().into_iter().cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// 私有：检查并执行清理
    fn maybe_cleanup(&self) {
        let should_cleanup = {
            let last = self.last_cleanup.read().unwrap();
            now_ms().saturating_sub(*last) > DEFAULT_CLEANUP_INTERVAL_MS
        };

        if should_cleanup {
            self.clear_expired();
            if let Ok(mut last) = self.last_cleanup.write() {
                *last = now_ms();
            }
        }
    }

    /// 私有：如果需要则淘汰会话
    fn evict_if_needed(&self) {
        if self.sessions.len() < self.max_sessions {
            return;
        }

        // LRU 淘汰：移除最久未访问的
        let oldest_key = self
            .sessions
            .iter()
            .min_by_key(|entry| entry.last_accessed())
            .map(|entry| entry.key().clone());

        if let Some(key) = oldest_key {
            self.sessions.remove(&key);
            warn!(
                "Evicted oldest session due to capacity limit: {} (capacity: {})",
                key, self.max_sessions
            );
        }
    }
}

/// 会话统计信息
#[derive(Debug, Clone)]
pub struct SessionStats {
    pub session_id: String,
    pub message_count: usize,
    /// 创建时间（Unix 时间戳，毫秒）
    pub created_at: u64,
    /// 最后访问时间（Unix 时间戳，毫秒）
    pub last_accessed: u64,
    /// 空闲时长（毫秒）
    pub idle_duration_ms: u64,
    pub access_count: u64,
    pub version: u64,
}

/// 便捷函数：获取全局聊天历史管理器
pub fn chat_history() -> Arc<ChatHistoryManager> {
    ChatHistoryManager::global()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_history_manager() {
        let manager = ChatHistoryManager::from_config_ms(10, 60_000);

        // 创建会话
        let ctx = manager.get_or_create("test_session");
        assert_eq!(ctx.session_id(), "test_session");
        assert_eq!(manager.session_count(), 1);

        // 获取已存在的会话
        let ctx2 = manager.get_or_create("test_session");
        assert_eq!(ctx2.access_count(), 2); // 访问计数增加

        // 列出会话
        let sessions = manager.list_sessions();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0], "test_session");

        // 删除会话
        manager.remove("test_session");
        assert_eq!(manager.session_count(), 0);
    }

    #[test]
    fn test_lru_eviction() {
        let manager = ChatHistoryManager::from_config_ms(3, 60_000);

        manager.get_or_create("session_1");
        manager.get_or_create("session_2");
        manager.get_or_create("session_3");
        assert_eq!(manager.session_count(), 3);

        // 访问 session_1 更新其访问时间
        std::thread::sleep(std::time::Duration::from_millis(10));
        manager.get_or_create("session_1");

        // 创建第4个会话，应该淘汰最久未访问的 session_2
        manager.get_or_create("session_4");
        assert_eq!(manager.session_count(), 3);
        assert!(manager.exists("session_1"));
        assert!(!manager.exists("session_2"));
        assert!(manager.exists("session_3"));
        assert!(manager.exists("session_4"));
    }

    #[test]
    fn test_chat_history_manager_api() {
        let manager = chat_history();

        // 清理
        let _ = manager.remove("test_api");

        // 创建会话
        let ctx = manager.get_or_create("test_api");
        assert_eq!(ctx.session_id(), "test_api");

        // 获取统计
        let stats = manager.get_stats("test_api").unwrap();
        assert_eq!(stats.session_id, "test_api");
        assert_eq!(stats.message_count, 0);

        // 获取所有统计
        let all_stats = manager.get_all_stats();
        assert!(!all_stats.is_empty());

        // 清理
        manager.remove("test_api");
    }

    #[test]
    fn test_add_messages() {
        let manager = chat_history();
        let session_id = "test_add_msgs";

        // 清理测试会话
        let _ = manager.remove(session_id);

        // 先创建会话
        let _ = manager.get_or_create(session_id);

        // 添加消息
        manager
            .add_system_text(session_id, "You are a helpful assistant")
            .unwrap();
        manager.add_user_text(session_id, "Hello").unwrap();
        manager.add_assistant_text(session_id, "Hi there!").unwrap();

        let ctx = manager.get(session_id).unwrap();
        assert_eq!(ctx.message_count(), 3);

        // 清理
        manager.remove(session_id);
    }

    #[test]
    fn test_get_image_references() {
        use crate::unified_message::{ImageSource, UnifiedMessage};

        let manager = chat_history();
        let session_id = "test_images";

        // 清理测试会话
        let _ = manager.remove(session_id);

        // 创建会话
        let _ = manager.get_or_create(session_id);

        // 添加带图片的消息
        let img_msg = UnifiedMessage::user_with_image(
            "Describe this",
            ImageSource::Url("http://example.com/img.png".to_string()),
        );
        manager.add_message(session_id, img_msg).unwrap();

        // 获取图片引用
        let images = manager.get_image_references(session_id);
        assert_eq!(images.len(), 1);
        match &images[0] {
            ImageSource::Url(url) => assert_eq!(url, "http://example.com/img.png"),
            _ => panic!("Expected URL image source"),
        }

        // 清理
        manager.remove(session_id);
    }
}
