//! 消息处理插件系统
//!
//! 包含各类消息处理插件的实现：
//! - NormalizePlugin: 消息标准化预处理
//! - SystemPlugin: 系统消息去重与处理
//! - HistoryPlugin: 历史上下文加载
//! - ToolsPlugin: Tool 调用消息处理

mod traits;
mod unified_message;

mod current_input_plugin;
mod history_plugin;
mod message_pipeline;
mod normalize_plugin;
mod system_prompt_plugin;
mod tools_plugin;

pub use traits::{MessageContext, MessagePlugin};
pub use unified_message::{ContentBlock, ImageSource, UnifiedMessage};

pub use current_input_plugin::CurrentInputPlugin;
pub use history_plugin::HistoryPlugin;
pub use message_pipeline::{MessagePipeline, MessagePipelineBuilder};
pub use normalize_plugin::NormalizePlugin;
pub use system_prompt_plugin::SystemPromptPlugin;
pub use tools_plugin::ToolsPlugin;
