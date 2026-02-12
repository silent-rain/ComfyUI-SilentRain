//! 工具调用处理 Hook

use crate::{
    error::Error,
    hooks::{HookContext, InferenceHook, priorities},
};

/// 工具调用处理 Hook
///
/// 负责：
/// 1. 检测并解析工具调用请求
/// 2. 执行工具调用
/// 3. 将工具结果添加到消息上下文
///
/// 注意：此 Hook 在 on_prepare 阶段主要进行初始化，
/// 实际的工具调用在 generate 过程中处理
#[derive(Debug)]
pub struct ToolsHook {
    priority: i32,
    enabled: bool,
}

impl Default for ToolsHook {
    fn default() -> Self {
        Self {
            priority: priorities::TOOLS,
            enabled: true,
        }
    }
}

impl ToolsHook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn disabled() -> Self {
        Self {
            priority: priorities::TOOLS,
            enabled: false,
        }
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

#[async_trait::async_trait]
impl InferenceHook for ToolsHook {
    fn name(&self) -> &str {
        "ToolsHook"
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    async fn on_prepare(&self, ctx: &mut HookContext) -> Result<(), Error> {
        // 设置工具启用状态
        ctx.pipeline_state.tools_enabled = self.enabled;
        
        // 这里可以初始化工具相关的元数据
        // 实际的工具调用在推理过程中处理
        
        Ok(())
    }
}
