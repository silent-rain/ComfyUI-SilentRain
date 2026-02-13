//! 钩子相关类型定义

/// 钩子类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookType {
    /// 推理前钩子
    BeforeInference,
    /// 推理后钩子
    AfterInference,
    /// 错误钩子
    OnError,
}
