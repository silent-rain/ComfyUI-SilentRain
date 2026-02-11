//! 会话上下文条目
//!
//! 存储单个会话的历史消息和元数据

use std::time::Instant;

use crate::unified_message::UnifiedMessage;

/// 默认最大历史消息数
const DEFAULT_MAX_HISTORY_MESSAGES: usize = 100;

/// 会话上下文条目
///
/// 包含会话的历史消息和元数据（创建时间、最后访问时间等）
///
/// 注意：SessionContext 不实现 Serialize/Deserialize，因为它包含 Instant
/// 如果需要持久化，请使用 HistoryMessage 进行序列化
#[derive(Debug, Clone)]
pub struct SessionContext {
    /// 会话ID
    session_id: String,
    /// 消息列表
    entries: Vec<UnifiedMessage>,
    /// 最大消息数限制
    max_entries: usize,
    /// 创建时间
    created_at: Instant,
    /// 最后访问时间
    last_accessed: Instant,
    /// 访问计数
    access_count: u64,
    /// 版本号（用于乐观锁）
    version: u64,
}

impl SessionContext {
    /// 创建新的会话上下文
    pub fn new(session_id: impl Into<String>) -> Self {
        let now = Instant::now();
        Self {
            session_id: session_id.into(),
            entries: Vec::new(),
            max_entries: DEFAULT_MAX_HISTORY_MESSAGES,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            version: 1,
        }
    }

    /// 创建指定容量的会话上下文
    pub fn with_capacity(session_id: impl Into<String>, max_entries: usize) -> Self {
        let mut ctx = Self::new(session_id);
        ctx.max_entries = max_entries;
        ctx
    }

    /// 获取会话ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// 获取消息列表
    pub fn entries(&self) -> &[UnifiedMessage] {
        &self.entries
    }

    /// 获取消息数量
    pub fn message_count(&self) -> usize {
        self.entries.len()
    }

    /// 获取最大消息数限制
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// 设置最大消息数限制
    pub fn set_max_entries(&mut self, max_entries: usize) {
        self.max_entries = max_entries;
        self.truncate_if_needed();
    }

    /// 添加消息
    pub fn add_message(&mut self, message: UnifiedMessage) {
        self.truncate_if_needed();
        self.entries.push(message);
        self.touch();
    }

    /// 添加多条消息
    pub fn add_messages(&mut self, messages: Vec<UnifiedMessage>) {
        for msg in messages {
            self.add_message(msg);
        }
    }

    /// 清空历史
    pub fn clear(&mut self) {
        self.entries.clear();
        self.touch();
    }

    /// 获取最后 n 条消息
    pub fn get_last_n(&self, n: usize) -> Vec<UnifiedMessage> {
        let start = self.entries.len().saturating_sub(n);
        self.entries[start..].to_vec()
    }

    /// 获取创建时间
    pub fn created_at(&self) -> Instant {
        self.created_at
    }

    /// 获取最后访问时间
    pub fn last_accessed(&self) -> Instant {
        self.last_accessed
    }

    /// 获取访问计数
    pub fn access_count(&self) -> u64 {
        self.access_count
    }

    /// 获取版本号
    pub fn version(&self) -> u64 {
        self.version
    }

    /// 更新访问时间和计数
    pub fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// 递增版本号
    pub fn increment_version(&mut self) {
        self.version += 1;
    }

    /// 检查是否需要截断并执行
    fn truncate_if_needed(&mut self) {
        while self.entries.len() >= self.max_entries {
            self.truncate_oldest();
        }
    }

    /// 淘汰最旧的消息（保留系统消息）
    fn truncate_oldest(&mut self) {
        use crate::types::MessageRole;

        let first_non_system = self
            .entries
            .iter()
            .position(|e| e.role != MessageRole::System)
            .unwrap_or(0);

        if first_non_system < self.entries.len() {
            self.entries.remove(first_non_system);
        } else {
            self.entries.remove(0);
        }
    }

    /// 计算内容哈希
    pub fn compute_content_hash(&self) -> String {
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

impl Default for SessionContext {
    fn default() -> Self {
        Self::new("default")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MessageRole;
    use crate::unified_message::UnifiedMessage;

    #[test]
    fn test_session_context_basic() {
        let mut ctx = SessionContext::new("test_session");
        assert_eq!(ctx.session_id(), "test_session");
        assert_eq!(ctx.message_count(), 0);

        ctx.add_message(UnifiedMessage::user_text("Hello"));
        assert_eq!(ctx.message_count(), 1);
        assert_eq!(ctx.access_count(), 1);
    }

    #[test]
    fn test_truncate_keeps_system() {
        let mut ctx = SessionContext::with_capacity("test", 3);

        ctx.add_message(UnifiedMessage::system("System prompt"));
        ctx.add_message(UnifiedMessage::user_text("Message 1"));
        ctx.add_message(UnifiedMessage::assistant("Response 1"));
        ctx.add_message(UnifiedMessage::user_text("Message 2"));

        assert_eq!(ctx.message_count(), 3);
        assert_eq!(ctx.entries[0].role, MessageRole::System);
    }
}
