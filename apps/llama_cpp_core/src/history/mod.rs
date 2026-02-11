//! 历史上下文管理模块
//!
//! 提供服务多用户的会话历史缓存管理，完全独立于 global_cache
//!
//! # 核心组件
//!
//! - `HistoryMessage`: 历史消息操作接口，保持向后兼容
//! - `ChatHistoryManager`: 全局会话管理器
//! - `SessionContext`: 单个会话的上下文数据
//!
//! # 使用示例
//!
//! ```rust
//! use llama_cpp_core::history::{HistoryMessage, chat_history};
//!
//! // 从缓存加载历史
//! let history = HistoryMessage::from_cache("user_123").unwrap_or_default();
//!
//! // 添加消息并保存
//! let mut history = HistoryMessage::new();
//! history.add_user_text("Hello");
//! history.add_assistant_text("Hi!");
//! history.save_to_cache("user_123").unwrap();
//!
//! // 使用管理器获取统计信息
//! let manager = chat_history();
//! let stats = manager.get_stats("user_123");
//! ```

mod manager;
mod message;
mod session;

// 公开导出
pub use manager::{chat_history, ChatHistoryManager, SessionStats};
pub use message::HistoryMessage;
pub use session::SessionContext;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration() {
        // 清理测试会话
        let _ = HistoryMessage::remove_from_cache("test_integration");

        // 创建并保存历史
        let mut history = HistoryMessage::new();
        history.add_system_text("You are a test assistant");
        history.add_user_text("Test message");
        history.add_assistant_text("Test response");

        history.save_to_cache("test_integration").unwrap();

        // 从缓存加载
        let loaded = HistoryMessage::from_cache("test_integration").unwrap();
        assert_eq!(loaded.message_count(), 3);

        // 检查会话存在
        assert!(HistoryMessage::exists_in_cache("test_integration"));

        // 列出会话
        let sessions = HistoryMessage::list_session_ids().unwrap();
        assert!(sessions.contains(&"test_integration".to_string()));

        // 删除会话
        HistoryMessage::remove_from_cache("test_integration").unwrap();
        assert!(!HistoryMessage::exists_in_cache("test_integration"));
    }

    #[test]
    fn test_chat_history_manager_api() {
        let manager = chat_history();

        // 清理
        let _ = manager.remove("test_api");

        // 创建会话
        let ctx = manager.get_or_create("test_api");
        assert_eq!(ctx.session_id(), "test_api");

        // 获取统计
        let stats = manager.get_stats("test_api").unwrap();
        assert_eq!(stats.session_id, "test_api");
        assert_eq!(stats.message_count, 0);

        // 获取所有统计
        let all_stats = manager.get_all_stats();
        assert!(!all_stats.is_empty());

        // 清理
        manager.remove("test_api");
    }
}
