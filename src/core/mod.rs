//! 公共库

mod always_equal_proxy;
pub use always_equal_proxy::{any_type, AlwaysEqualProxy};

mod prompt_server;
pub use prompt_server::PromptServer;
