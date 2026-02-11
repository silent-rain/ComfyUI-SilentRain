//! Llama.cpp Core - 底层推理引擎
//!
//! 提供统一的推理接口，支持多种后端（llama.cpp 等）

pub mod backend;
pub mod cache;
pub mod context;
pub mod error;
pub mod history;
pub mod hooks;
pub mod message_plugins;
pub mod model;
pub mod mtmd_context;
pub mod pipeline;
pub mod request;
pub mod response;
pub mod sampler;
pub mod types;
pub mod unified_message;
pub mod utils;

pub use backend::Backend;
pub use cache::{CacheManager, global_cache};
pub use context::ContexParams;
pub use history::HistoryMessage;
pub use model::Model;
pub use pipeline::{Pipeline, PipelineConfig};
pub use sampler::Sampler;
