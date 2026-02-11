//! 推理钩子相关trait和类型定义

use std::sync::Arc;

use crate::{error::Error, hooks::HookContext};

/// 推理钩子 trait
///
/// 实现此 trait 可在推理生命周期的关键节点插入自定义逻辑
#[async_trait::async_trait]
pub trait InferenceHook: Send + Sync + std::fmt::Debug {
    /// 推理前钩子
    ///
    /// 在推理开始前执行，可用于：
    /// - 参数验证
    /// - 限流检查
    /// - 上下文加载
    /// - 日志记录
    async fn on_before(&self, _ctx: &mut HookContext) -> Result<(), Error> {
        Ok(())
    }

    /// 推理后钩子
    ///
    /// 在推理结束后执行，可用于：
    /// - 保存历史上下文
    /// - 更新缓存
    /// - 审计日志
    /// - 结果后处理
    async fn on_after(&self, _ctx: &mut HookContext) -> Result<(), Error> {
        Ok(())
    }

    /// 错误钩子
    ///
    /// 在推理发生错误时执行，可用于：
    /// - 错误日志记录
    /// - 告警通知
    /// - 降级处理
    async fn on_error(&self, _ctx: &HookContext, _error: &Error) -> Result<(), Error> {
        Ok(())
    }

    /// 钩子名称
    ///
    /// 用于日志记录和调试
    fn name(&self) -> &str;

    /// 执行优先级
    ///
    /// 数字越小越早执行，默认值为 50
    fn priority(&self) -> i32 {
        50
    }
}

/// 动态钩子类型（用于存储）
pub(crate) type DynHook = Arc<dyn InferenceHook>;
