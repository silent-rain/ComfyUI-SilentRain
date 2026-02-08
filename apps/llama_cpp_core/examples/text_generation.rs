//! 基础使用示例 - 新版API

use std::sync::Arc;

use llama_cpp_core::{Pipeline, PipelineConfig, Request, utils::log::init_logger};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();

    let model_path =
        "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();
    let pipeline_config = PipelineConfig::new(model_path).with_verbose(false);

    let pipeline = Arc::new(Pipeline::try_new(pipeline_config)?);

    let request = Request::builder().input("你是谁？").build();
    let results = pipeline.generate(&request).await?;

    println!("{results:?}");
    Ok(())
}
