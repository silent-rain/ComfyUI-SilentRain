//! 历史上下文管理模块
//!
//! 提供服务多用户的会话历史缓存管理，完全独立于 global_cache
//!
//! # 核心组件
//!
//! - `ChatHistoryManager`: 全局会话管理器（主要入口）
//! - `SessionContext`: 单个会话的上下文数据
//!
//! # 使用示例
//!
//! ```rust
//! use llama_flow::history::chat_history;
//!
//! // 获取管理器实例
//! let manager = chat_history();
//!
//! // 添加消息
//! manager.add_user_text("user_123", "Hello").unwrap();
//! manager.add_assistant_text("user_123", "Hi!").unwrap();
//!
//! // 获取消息列表
//! let messages = manager.get_messages("user_123").unwrap();
//!
//! // 获取统计信息
//! let stats = manager.get_stats("user_123");
//! ```

mod manager;
mod session;

// 公开导出
pub use manager::{ChatHistoryManager, SessionStats, chat_history};
pub use session::SessionContext;
