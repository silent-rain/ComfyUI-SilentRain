//! 当前输入插件

use crate::{
    error::Error,
    message_plugins::{MessageContext, MessagePlugin},
    unified_message::UnifiedMessage,
};

/// 当前输入插件
///
/// 负责：
/// 1. 处理当前轮次的用户输入
/// 2. 确保用户消息格式正确
/// 3. 可选：添加当前时间等上下文
#[derive(Debug, Default)]
pub struct CurrentInputPlugin;

impl CurrentInputPlugin {
    pub fn new() -> Self {
        Self
    }
}

impl MessagePlugin for CurrentInputPlugin {
    fn name(&self) -> &str {
        "CurrentInputPlugin"
    }

    fn priority(&self) -> i32 {
        90
    }

    fn process(
        &self,
        _context: &MessageContext,
        messages: Vec<UnifiedMessage>,
    ) -> Result<Vec<UnifiedMessage>, Error> {
        // 当前输入通常已经通过其他方式添加
        // 此插件可用于添加额外处理，如时间戳、元数据等
        Ok(messages)
    }
}
