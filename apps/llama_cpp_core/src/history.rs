//! History Context

use llama_cpp_2::model::LlamaChatMessage;

use crate::{error::Error, types::PromptMessageRole};

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

    pub fn get_all(&self) -> Vec<LlamaChatMessage> {
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
}
