//! 推理钩子系统
//!
//! 提供统一的消息处理和推理生命周期钩子，支持在消息准备、推理前、推理后执行自定义逻辑。
//!
//! # 使用示例
//!
//! ```rust
//! use llama_cpp_core::hooks::{InferenceHook, HookContext, priorities};
//!
//! // 定义钩子
//! #[derive(Debug)]
//! struct MyHook;
//!
//! #[async_trait::async_trait]
//! impl InferenceHook for MyHook {
//!     fn name(&self) -> &str { "MyHook" }
//!
//!     async fn on_prepare(&self, ctx: &mut HookContext) -> Result<(), Error> {
//!         println!("消息准备: {:?}", ctx.session_id);
//!         Ok(())
//!     }
//!
//!     async fn on_before(&self, ctx: &mut HookContext) -> Result<(), Error> {
//!         println!("推理开始: {:?}", ctx.session_id);
//!         Ok(())
//!    }
//!
//!     async fn on_after(&self, ctx: &mut HookContext, output: &str) -> Result<(), Error> {
//!         println!("推理结束: {}", output);
//!         Ok(())
//!     }
//! }
//! ```

pub mod builtin;

mod context;
mod pipeline_state;
mod registry;
mod traits;
mod types;

pub use context::HookContext;
pub use pipeline_state::{PipelineState, priorities};
pub use registry::{HookRegistry, HookRegistryBuilder};
pub use traits::{DynHook, InferenceHook};
pub use types::HookType;
