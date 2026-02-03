//! 基础使用示例 - 新版API

use std::sync::Arc;

use llama_cpp_core::{GenerateRequest, Pipeline, PipelineConfig, utils::log::init_logger};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();

    let model_path =
        "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();
    let pipeline_config = PipelineConfig::new(model_path, None).with_verbose(true);

    let pipeline = Arc::new(Pipeline::try_new(pipeline_config)?);

    let request = GenerateRequest::text("你是谁？");
    let results = pipeline.generate(&request).await?;

    println!("{results:?}");
    Ok(())
}
