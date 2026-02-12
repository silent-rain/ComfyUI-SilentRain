//! 钩子注册表

use std::sync::Arc;

use tracing::{debug, error, warn};

use crate::{error::Error, hooks::InferenceHook};

use super::{DynHook, HookContext};

/// 钩子注册表
///
/// 只维护一个钩子列表，按需调用三个方法
#[derive(Debug, Default, Clone)]
pub struct HookRegistry {
    hooks: Vec<DynHook>,
}

impl HookRegistry {
    /// 创建空的注册表
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// 注册钩子
    pub fn register<H>(&mut self, hook: H)
    where
        H: super::InferenceHook + 'static,
    {
        let hook: DynHook = Arc::new(hook);

        self.hooks.push(hook.clone());

        // 按优先级排序
        self.sort_hooks();

        debug!("Registered hook: {}", hook.name());
    }

    /// 注册装箱的钩子（用于动态创建的钩子）
    pub fn register_boxed(&mut self, hook: Box<dyn InferenceHook>) {
        let hook: DynHook = hook.into();

        self.hooks.push(hook.clone());

        // 按优先级排序
        self.sort_hooks();

        debug!("Registered hook: {}", hook.name());
    }

    /// 批量注册钩子
    pub fn register_many<H>(&mut self, hooks: Vec<H>)
    where
        H: InferenceHook + 'static,
    {
        for hook in hooks {
            self.register(hook);
        }
    }

    /// 执行消息准备钩子
    ///
    /// 按优先级顺序执行，如果某个钩子失败，会执行错误钩子并返回错误
    pub async fn run_prepare(&self, ctx: &mut HookContext) -> Result<(), Error> {
        if self.hooks.is_empty() {
            debug!("No hooks to run for prepare");
            return Ok(());
        }

        for hook in &self.hooks {
            debug!(
                "Running prepare hook: {} (priority: {})",
                hook.name(),
                hook.priority()
            );

            if let Err(e) = hook.on_prepare(ctx).await {
                error!("Prepare hook '{}' failed: {}", hook.name(), e);
                // 执行错误钩子
                let _ = hook.on_error(ctx, &e).await;
                return Err(e);
            }
        }
        Ok(())
    }

    /// 执行推理前钩子
    ///
    /// 按优先级顺序执行，如果某个钩子失败，会执行错误钩子并返回错误
    pub async fn run_before(&self, ctx: &mut HookContext) -> Result<(), Error> {
        if self.hooks.is_empty() {
            debug!("No hooks to run before inference");
            return Ok(());
        }

        for hook in &self.hooks {
            debug!(
                "Running before hook: {} (priority: {})",
                hook.name(),
                hook.priority()
            );

            if let Err(e) = hook.on_before(ctx).await {
                error!("Before hook '{}' failed: {}", hook.name(), e);
                // 执行错误钩子
                let _ = hook.on_error(ctx, &e).await;
                return Err(e);
            }
        }
        Ok(())
    }

    /// 执行推理后钩子
    ///
    /// 按优先级顺序执行，如果某个钩子失败，会执行错误钩子并返回错误
    pub async fn run_after(&self, ctx: &mut HookContext) -> Result<(), Error> {
        if self.hooks.is_empty() {
            debug!("No hooks to run after inference");
            return Ok(());
        }

        for hook in &self.hooks {
            debug!(
                "Running after hook: {} (priority: {})",
                hook.name(),
                hook.priority()
            );

            if let Err(e) = hook.on_after(ctx).await {
                error!("After hook '{}' failed: {}", hook.name(), e);
                // 执行错误钩子
                let _ = hook.on_error(ctx, &e).await;
                return Err(e);
            }
        }
        Ok(())
    }

    /// 执行错误钩子
    ///
    /// 按优先级顺序执行，错误不会中断其他钩子执行
    pub async fn run_on_error(&self, ctx: &HookContext, error: &Error) -> Result<(), Error> {
        for hook in &self.hooks {
            if let Err(e) = hook.on_error(ctx, error).await {
                warn!("Error hook '{}' failed: {}", hook.name(), e);
            }
        }
        Ok(())
    }

    /// 按优先级排序钩子
    fn sort_hooks(&mut self) {
        self.hooks.sort_by_key(|a| a.priority());
    }

    /// 检查是否有钩子
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// 获取钩子数量
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// 检查是否有钩子
    pub fn has_hooks(&self) -> bool {
        !self.hooks.is_empty()
    }

    /// 清空所有钩子
    pub fn clear(&mut self) {
        self.hooks.clear();
    }
}

/// 构建器模式创建 HookRegistry
#[derive(Debug, Default)]
pub struct HookRegistryBuilder {
    hooks: Vec<DynHook>,
}

impl HookRegistryBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// 添加钩子
    pub fn with_hook(mut self, hook: impl InferenceHook + 'static) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    /// 构建注册表
    #[allow(unused_mut)]
    pub fn build(mut self) -> HookRegistry {
        let mut registry = HookRegistry::new();

        for hook in self.hooks {
            registry.hooks.push(hook);
        }

        registry
    }
}
