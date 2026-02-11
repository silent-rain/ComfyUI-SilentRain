//! 内置钩子
//!
//! 提供常用的钩子实现

mod error_log;
mod history;
mod validate;

pub use error_log::ErrorLogHook;
pub use history::HistoryHook;
pub use validate::ValidateHook;
