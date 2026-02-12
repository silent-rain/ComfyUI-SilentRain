//! 消息处理插件系统
//!
//! 提供可扩展的插件架构，用于处理消息标准化、系统消息去重、
//! 历史上下文加载、Tool 调用处理等

use std::sync::Arc;

use tracing::debug;

use crate::{
    error::Error,
    message_plugins::{MessageContext, MessagePlugin},
    unified_message::UnifiedMessage,
};

/// 插件管道
/// 管理和执行一系列消息处理插件
#[derive(Debug, Default)]
pub struct MessagePipeline {
    plugins: Vec<Arc<dyn MessagePlugin>>,
}

impl MessagePipeline {
    /// 创建空的插件管道
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// 添加插件
    pub fn add_plugin(mut self, plugin: impl MessagePlugin + 'static) -> Self {
        self.plugins.push(Arc::new(plugin));
        // 按优先级排序
        self.plugins.sort_by_key(|p| p.priority());
        self
    }

    /// 批量添加插件
    pub fn add_plugins(mut self, plugins: Vec<Box<dyn MessagePlugin>>) -> Self {
        for plugin in plugins {
            self.plugins.push(plugin.into());
        }
        self.plugins.sort_by_key(|p| p.priority());
        self
    }

    /// 执行所有插件处理消息
    pub fn process(
        &self,
        messages: Vec<UnifiedMessage>,
        context: &MessageContext,
    ) -> Result<Vec<UnifiedMessage>, Error> {
        let mut result = messages;

        for plugin in &self.plugins {
            debug!(
                "Running plugin '{}' with priority {}",
                plugin.name(),
                plugin.priority()
            );
            result = plugin.process(context, result)?;
        }

        Ok(result)
    }

    /// 获取插件数量
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }
}

/// 构建器模式创建管道
pub struct MessagePipelineBuilder {
    plugins: Vec<Arc<dyn MessagePlugin>>,
}

impl Default for MessagePipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MessagePipelineBuilder {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// 添加插件
    pub fn with_plugin(mut self, plugin: impl MessagePlugin + 'static) -> Self {
        self.plugins.push(Arc::new(plugin));
        self
    }

    /// 构建管道
    pub fn build(self) -> MessagePipeline {
        let mut plugins = self.plugins;
        plugins.sort_by_key(|p| p.priority());
        MessagePipeline { plugins }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct TestPlugin {
        name: String,
        prio: i32,
    }

    impl MessagePlugin for TestPlugin {
        fn name(&self) -> &str {
            &self.name
        }

        fn process(
            &self,
            _context: &MessageContext,
            messages: Vec<UnifiedMessage>,
        ) -> Result<Vec<UnifiedMessage>, Error> {
            Ok(messages)
        }

        fn priority(&self) -> i32 {
            self.prio
        }
    }

    #[test]
    fn test_pipeline_priority_ordering() {
        let pipeline = MessagePipeline::new()
            .add_plugin(TestPlugin {
                name: "High".to_string(),
                prio: 10,
            })
            .add_plugin(TestPlugin {
                name: "Low".to_string(),
                prio: 100,
            })
            .add_plugin(TestPlugin {
                name: "Mid".to_string(),
                prio: 50,
            });

        assert_eq!(pipeline.plugins[0].priority(), 10);
        assert_eq!(pipeline.plugins[1].priority(), 50);
        assert_eq!(pipeline.plugins[2].priority(), 100);
    }
}
