//! 推理钩子系统
//!
//! 提供推理生命周期钩子，支持在推理前、推理后和错误时执行自定义逻辑。
//!
//! # 使用示例
//!
//! ```rust
//! use llama_cpp_core::hooks::{InferenceHook, HookContext, HookRegistry};
//!
//! // 定义钩子
//! #[derive(Debug)]
//! struct MyHook;
//!
//! #[async_trait::async_trait]
//! impl InferenceHook for MyHook {
//!     fn name(&self) -> &str { "MyHook" }
//!
//!     async fn on_before(&self, ctx: &mut HookContext) -> Result<(), Error> {
//!         println!("推理开始: {:?}", ctx.session_id);
//!         Ok(())
//!     }
//!
//!     async fn on_after(&self, ctx: &mut HookContext) -> Result<(), Error> {
//!         println!("推理结束");
//!         Ok(())
//!     }
//! }
//! ```

pub mod builtin;

mod context;
mod registry;
mod traits;
mod types;

pub use context::HookContext;
pub use registry::{HookRegistry, HookRegistryBuilder};
use traits::DynHook;
pub use traits::InferenceHook;
pub use types::HookType;
