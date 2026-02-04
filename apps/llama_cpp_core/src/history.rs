//! History Context

use std::sync::Arc;

use llama_cpp_2::model::LlamaChatMessage;
use tracing::error;

use crate::{cache::CacheType, error::Error, global_cache, types::PromptMessageRole};

#[derive(Debug, Default, Clone)]
pub struct HistoryMessage {
    msgs: Vec<LlamaChatMessage>,
}

impl HistoryMessage {
    pub fn new() -> Self {
        HistoryMessage::default()
    }

    pub fn from_vec(msgs: Vec<LlamaChatMessage>) -> Self {
        Self { msgs }
    }

    pub fn add(&mut self, msg: LlamaChatMessage) {
        self.msgs.push(msg);
    }

    pub fn adds(&mut self, msgs: Vec<LlamaChatMessage>) {
        self.msgs.extend(msgs);
    }

    pub fn add_assistant(&mut self, msg: impl Into<String>) -> Result<(), Error> {
        let assistant_message =
            LlamaChatMessage::new(PromptMessageRole::Assistant.to_string(), msg.into())?;
        self.add(assistant_message);
        Ok(())
    }

    pub fn add_user(&mut self, msg: impl Into<String>) -> Result<(), Error> {
        let user_message = LlamaChatMessage::new(PromptMessageRole::User.to_string(), msg.into())?;
        self.add(user_message);
        Ok(())
    }

    pub fn add_system(&mut self, msg: impl Into<String>) -> Result<(), Error> {
        let system_message =
            LlamaChatMessage::new(PromptMessageRole::System.to_string(), msg.into())?;
        self.add(system_message);
        Ok(())
    }

    pub fn add_custom(
        &mut self,
        role: PromptMessageRole,
        msg: impl Into<String>,
    ) -> Result<(), Error> {
        let custom_message = LlamaChatMessage::new(role.to_string(), msg.into())?;
        self.add(custom_message);
        Ok(())
    }

    pub fn messages(&self) -> Vec<LlamaChatMessage> {
        self.msgs.clone()
    }

    /// 获取最后n条消息
    pub fn get_last_n(&self, n: usize) -> Vec<LlamaChatMessage> {
        self.msgs
            .clone()
            .into_iter()
            .skip(self.msgs.len() - n)
            .collect()
    }

    pub fn clear(&mut self) {
        self.msgs.clear();
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
    /// 使用消息数量和总内容长度作为简单哈希
    fn compute_content_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        // 使用消息数量作为哈希的一部分
        self.msgs.len().hash(&mut hasher);
        // 使用每条消息的内存地址（指针）作为唯一标识
        for msg in &self.msgs {
            let ptr = std::ptr::addr_of!(msg);
            ptr.hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }

    /// 获取历史消息数量，用于版本控制
    pub fn message_count(&self) -> usize {
        self.msgs.len()
    }

    pub fn force_update_cache(&mut self, session_id: String) -> Result<(), Error> {
        let cache = global_cache();

        // 使用内容哈希 + 消息数量作为参数，确保：
        // 1. 相同内容可以命中缓存
        // 2. 内容变化时哈希变化，触发更新
        let content_hash = self.compute_content_hash();
        let msg_count = self.msgs.len() as u64;
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
