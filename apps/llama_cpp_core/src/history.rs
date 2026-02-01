//! History Context

use std::sync::Arc;

use llama_cpp_2::model::LlamaChatMessage;
use rand::TryRngCore;
use tracing::error;

use crate::{error::Error, global_cache, types::PromptMessageRole};

#[derive(Debug, Default, Clone)]
pub struct HistoryMessage {
    megs: Vec<LlamaChatMessage>,
}

impl HistoryMessage {
    pub fn new() -> Self {
        HistoryMessage::default()
    }

    pub fn from_vec(megs: Vec<LlamaChatMessage>) -> Self {
        Self { megs }
    }

    pub fn add(&mut self, msg: LlamaChatMessage) {
        self.megs.push(msg);
    }

    pub fn adds(&mut self, msgs: Vec<LlamaChatMessage>) {
        self.megs.extend(msgs);
    }

    pub fn add_assistant(&mut self, msg: String) -> Result<(), Error> {
        let assistant_message =
            LlamaChatMessage::new(PromptMessageRole::Assistant.to_string(), msg.clone())?;
        self.add(assistant_message);
        Ok(())
    }

    pub fn add_user(&mut self, msg: String) -> Result<(), Error> {
        let user_message = LlamaChatMessage::new(PromptMessageRole::User.to_string(), msg.clone())?;
        self.add(user_message);
        Ok(())
    }

    pub fn add_system(&mut self, msg: String) -> Result<(), Error> {
        let system_message =
            LlamaChatMessage::new(PromptMessageRole::System.to_string(), msg.clone())?;
        self.add(system_message);
        Ok(())
    }

    pub fn add_custom(&mut self, role: PromptMessageRole, msg: String) -> Result<(), Error> {
        let custom_message = LlamaChatMessage::new(role.to_string(), msg.clone())?;
        self.add(custom_message);
        Ok(())
    }

    pub fn messages(&self) -> Vec<LlamaChatMessage> {
        self.megs.clone()
    }

    /// 获取最后n条消息
    pub fn get_last_n(&self, n: usize) -> Vec<LlamaChatMessage> {
        self.megs
            .clone()
            .into_iter()
            .skip(self.megs.len() - n)
            .collect()
    }

    pub fn clear(&mut self) {
        self.megs.clear();
    }

    pub fn from_cache(cache_key: String) -> Result<Self, Error> {
        let cache = global_cache();
        let cache_key = format!("history_message_{}", cache_key);
        let megs = cache
            .get_data::<Vec<LlamaChatMessage>>(&cache_key)
            .map_err(|e| {
                error!("get cache error: {e:?}");
                e
            })?;
        if let Some(data) = megs {
            return Ok(Self {
                megs: data.to_vec(),
            });
        }

        Ok(Self { megs: Vec::new() })
    }

    pub fn force_update_cache(
        &mut self,
        cache_key: String,
    ) -> Result<Vec<LlamaChatMessage>, Error> {
        let cache = global_cache();
        let cache_key = format!("history_message_{}", cache_key);

        // 随机值
        let random = rand::rng().try_next_u32().unwrap_or(0);
        let params = vec![random.to_string()];

        cache.force_update(&cache_key, &params, Arc::new(self.megs.clone()))?;

        if let Some(data) = cache.get_data::<Vec<LlamaChatMessage>>(&cache_key)? {
            self.megs = data.to_vec();
            return Ok(self.megs.clone());
        }
        Ok(self.megs.clone())
    }
}
