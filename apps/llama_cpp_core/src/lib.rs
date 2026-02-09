//! Llama.cpp Core - 底层推理引擎
//!
//! 提供统一的推理接口，支持多种后端（llama.cpp 等）

pub mod backend;
pub mod cache;
pub mod context;
pub mod error;
pub mod history;
pub mod model;
pub mod mtmd_context;
pub mod pipeline;
pub mod sampler;
pub mod types;
pub mod utils;

// Re-export llama_cpp_2 types for building history messages
pub use llama_cpp_2::model::LlamaChatMessage;

pub use backend::Backend;
pub use cache::{CacheManager, global_cache};
pub use context::ContexParams;
pub use history::HistoryMessage;
pub use model::Model;
pub use pipeline::{ChatStreamBuilder, Pipeline, PipelineConfig};
pub use sampler::Sampler;

// Re-export internal types
pub use types::{
    CreateChatCompletionRequest as Request, CreateChatCompletionResponse as Response, MediaData,
    MediaType, MessageRole, PoolingTypeMode,
};
