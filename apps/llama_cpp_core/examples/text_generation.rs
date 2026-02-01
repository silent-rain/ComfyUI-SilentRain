//! 基础使用示例

use llama_cpp_core::{Pipeline, PipelineConfig, utils::log::init_logger};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();

    let model_path =
        "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();
    let pipeline_config = PipelineConfig::new(model_path, None).with_user_prompt("你是谁？");

    let mut pipeline = Pipeline::try_new(pipeline_config)?;

    let results = pipeline.infer().await?;

    println!("{results:?}");
    Ok(())
}
