//! Pipeline module for processing tasks

mod pipeline_config;
mod pipeline_impl;
mod request;

pub use pipeline_config::PipelineConfig;
pub use pipeline_impl::Pipeline;
pub use request::GenerateRequest;
