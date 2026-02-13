//! 内置钩子
//!
//! 提供常用的钩子实现

mod assemble_messages;
mod error_log;
mod load_history;
mod normalize;
mod save_history;
mod system_prompt;
mod tools;
mod validate;

pub use assemble_messages::AssembleMessagesHook;
pub use error_log::ErrorLogHook;
pub use load_history::LoadHistoryHook;
pub use normalize::NormalizeHook;
pub use save_history::SaveHistoryHook;
pub use system_prompt::{SystemPromptHook, SystemStrategy};
pub use tools::ToolsHook;
pub use validate::ValidateHook;
