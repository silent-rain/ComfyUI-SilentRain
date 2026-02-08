//! History Context

use std::sync::Arc;

use llama_cpp_2::model::LlamaChatMessage;
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::{cache::CacheType, error::Error, global_cache, types::MessageRole};

/// 默认最大历史消息数（保留最近100条）
const DEFAULT_MAX_HISTORY_MESSAGES: usize = 100;
/// 警告阈值，超过此值会触发日志警告
const WARN_HISTORY_THRESHOLD: usize = 80;

/// 内部消息条目，支持序列化
/// 用于存储历史消息，可持久化到 SQLite 或缓存
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEntry {
    role: MessageRole,
    content: String,
}

impl MessageEntry {
    fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    /// 转换为 LlamaChatMessage
    fn to_llama_message(&self) -> Result<LlamaChatMessage, Error> {
        LlamaChatMessage::new(self.role.to_string(), self.content.clone()).map_err(|e| {
            Error::InvalidInput {
                field: "LlamaChatMessage".to_string(),
                message: e.to_string(),
            }
        })
    }
}

/// 历史消息管理器
/// 内部使用可序列化的 MessageEntry 存储，仅在导出时转换为 LlamaChatMessage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryMessage {
    entries: Vec<MessageEntry>,
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

    /// 添加内部消息条目
    /// 当超过最大限制时，移除最旧的消息（保留系统消息）
    fn add_entry(&mut self, entry: MessageEntry) {
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

        self.entries.push(entry);
    }

    /// 淘汰最旧的消息（保留系统消息）
    fn truncate_oldest(&mut self) {
        // 找到第一个非系统消息的索引
        let first_non_system = self
            .entries
            .iter()
            .position(|e| e.role != MessageRole::System)
            .unwrap_or(0);

        if first_non_system < self.entries.len() {
            // 移除最旧的非系统消息（或如果全是系统消息，移除最旧的）
            self.entries.remove(first_non_system);
            info!(
                "History truncated: removed oldest message (count: {})",
                self.entries.len()
            );
        } else {
            // 如果只有系统消息，移除最旧的
            self.entries.remove(0);
        }
    }

    pub fn add_assistant(&mut self, msg: impl Into<String>) -> Result<(), Error> {
        let entry = MessageEntry::new(MessageRole::Assistant, msg);
        self.add_entry(entry);
        Ok(())
    }

    pub fn add_user(&mut self, msg: impl Into<String>) -> Result<(), Error> {
        let entry = MessageEntry::new(MessageRole::User, msg);
        self.add_entry(entry);
        Ok(())
    }

    pub fn add_system(&mut self, msg: impl Into<String>) -> Result<(), Error> {
        let entry = MessageEntry::new(MessageRole::System, msg);
        self.add_entry(entry);
        Ok(())
    }

    pub fn add_custom(
        &mut self,
        role: MessageRole,
        msg: impl Into<String>,
    ) -> Result<(), Error> {
        let entry = MessageEntry::new(role, msg);
        self.add_entry(entry);
        Ok(())
    }

    /// 导出为 LlamaChatMessage 列表
    /// 此方法会创建新的 LlamaChatMessage 对象
    pub fn to_llama_message(&self) -> Result<Vec<LlamaChatMessage>, Error> {
        self.entries
            .iter()
            .map(|entry| entry.to_llama_message())
            .collect()
    }

    /// 获取所有消息条目
    pub fn entries(&self) -> Vec<MessageEntry> {
        self.entries.clone()
    }

    /// 获取最后n条消息
    pub fn get_last_n(&self, n: usize) -> Result<Vec<LlamaChatMessage>, Error> {
        let n = n.min(self.entries.len()); // 确保不越界
        let start = self.entries.len().saturating_sub(n);
        self.entries[start..]
            .iter()
            .map(|entry| entry.to_llama_message())
            .collect()
    }

    /// 获取当前最大消息数限制
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// 设置最大消息数限制
    pub fn set_max_entries(&mut self, max_entries: usize) {
        self.max_entries = max_entries;
        // 如果当前消息数超过新限制，立即截断
        while self.entries.len() > self.max_entries {
            self.truncate_oldest();
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl HistoryMessage {
    pub fn from_cache(session_id: String) -> Result<Arc<Self>, Error> {
        let cache = global_cache();
        let data = cache
            .get_data::<HistoryMessage>(&session_id)
            .map_err(|e| {
                error!("get cache error: {e:?}");
                e
            })?
            .ok_or_else(|| {
                error!("Cache data not found");
                Error::CacheNotFound {
                    key: session_id.clone(),
                }
            })?;

        Ok(data.clone())
    }

    /// 计算历史消息的内容哈希，用于缓存参数
    /// 使用消息内容和角色进行哈希
    fn compute_content_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        // 使用消息数量作为哈希的一部分
        self.entries.len().hash(&mut hasher);
        // 使用每条消息的角色和内容进行哈希
        for entry in &self.entries {
            entry.role.to_string().hash(&mut hasher);
            entry.content.hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }

    /// 获取历史消息数量
    pub fn message_count(&self) -> usize {
        self.entries.len()
    }

    /// 序列化为 JSON 字符串（用于 SQLite 存储）
    pub fn to_json(&self) -> Result<String, Error> {
        serde_json::to_string(self).map_err(Error::Serde)
    }

    /// 从 JSON 字符串反序列化（用于 SQLite 恢复）
    pub fn from_json(json: &str) -> Result<Self, Error> {
        serde_json::from_str(json).map_err(Error::Serde)
    }

    pub fn force_update_cache(&mut self, session_id: String) -> Result<(), Error> {
        let cache = global_cache();

        // 使用内容哈希 + 消息数量作为参数，确保：
        // 1. 相同内容可以命中缓存
        // 2. 内容变化时哈希变化，触发更新
        let content_hash = self.compute_content_hash();
        let msg_count = self.entries.len() as u64;
        let params = vec![content_hash, msg_count.to_string()];

        cache.force_update(
            &session_id,
            CacheType::MessageContext,
            &params,
            Arc::new(self.clone()),
        )?;

        Ok(())
    }
}
