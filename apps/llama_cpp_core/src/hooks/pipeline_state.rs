//! Pipeline 状态管理
//!
//! 在消息处理管道中维护各阶段的状态数据

use crate::unified_message::UnifiedMessage;

/// Pipeline 处理状态
///
/// 维护消息处理管道中的各个阶段数据
#[derive(Debug, Default, Clone)]
pub struct PipelineState {
    /// 原始请求消息（未经任何处理）
    pub raw_messages: Vec<UnifiedMessage>,

    /// 系统提示词（单独管理）
    pub system_prompt: Option<UnifiedMessage>,

    /// 从历史存储加载的消息
    pub loaded_history: Vec<UnifiedMessage>,

    /// 用户输入的消息，不含历史
    pub current_input: Vec<UnifiedMessage>,

    /// 工具调用相关的消息（ToolCall + ToolResult）
    pub tool_messages: Vec<UnifiedMessage>,

    /// 最终工作消息（组装完成，用于模型推理）
    pub working_messages: Vec<UnifiedMessage>,

    /// 是否启用工具
    pub tools_enabled: bool,

    /// 是否包含多模态内容
    pub has_multimodal: bool,
}

impl PipelineState {
    /// 创建新的状态实例
    pub fn new(raw_messages: Vec<UnifiedMessage>) -> Self {
        Self {
            raw_messages,
            ..Default::default()
        }
    }

    /// 组装最终消息列表
    ///
    /// 按顺序组装：系统提示 + 历史消息 + 当前输入 + 工具消息
    pub fn assemble_working_messages(&mut self) {
        let mut result = Vec::new();

        // 1. 系统提示（放在第一位）
        if let Some(sys) = &self.system_prompt {
            result.push(sys.clone());
        }

        // 2. 历史消息
        result.extend(self.loaded_history.clone());

        // 3. 当前输入
        result.extend(self.current_input.clone());

        // 4. 工具消息（如果有）
        result.extend(self.tool_messages.clone());

        self.working_messages = result;
    }

    /// 清空状态
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// 优先级常量定义
pub mod priorities {
    /// 标准化插件优先级
    pub const NORMALIZE: i32 = 10;
    /// 系统提示词插件优先级
    pub const SYSTEM_PROMPT: i32 = 20;
    /// 加载历史插件优先级
    pub const LOAD_HISTORY: i32 = 30;
    /// 当前输入插件优先级
    pub const CURRENT_INPUT: i32 = 40;
    /// 工具插件优先级
    pub const TOOLS: i32 = 50;
    /// 历史保存插件优先级
    pub const HISTORY: i32 = 60;
    /// 错误记录插件优先级
    pub const ERROR_LOG: i32 = 70;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assemble_working_messages() {
        let mut state = PipelineState::new(vec![]);

        // 验证结果
        // println!("{:?}", state.working_messages);
        assert_eq!(state.working_messages.len(), 0);

        // 添加测试数据
        state.system_prompt = Some(UnifiedMessage::system("系统: 你好"));

        state
            .loaded_history
            .push(UnifiedMessage::user("用户: 你好"));
        state
            .loaded_history
            .push(UnifiedMessage::assistant("助手: 早上好"));

        state
            .current_input
            .push(UnifiedMessage::user("用户: 请帮我总结一下"));

        state
            .tool_messages
            .push(UnifiedMessage::tool_result("tool_id", "工具: 数据分析"));

        // 调用被测试函数
        state.assemble_working_messages();

        // 验证结果
        // println!("{:?}", state.working_messages);
        assert_eq!(state.working_messages.len(), 5);
    }
}
