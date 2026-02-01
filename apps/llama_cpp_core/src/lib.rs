//! Llama.cpp Core - 底层推理引擎
//!
//! 提供统一的推理接口，支持多种后端（llama.cpp 等）

pub mod backend;
pub mod cache;
pub mod config;
pub mod context;
pub mod error;
pub mod history;
pub mod model;
pub mod mtmd_context;
pub mod pipeline;
pub mod sampler;
pub mod types;
pub mod utils;

pub use backend::Backend;
pub use cache::{CacheManager, global_cache};
pub use history::HistoryMessage;
pub use model::Model;
pub use pipeline::{Pipeline, PipelineConfig};
pub use sampler::Sampler;
