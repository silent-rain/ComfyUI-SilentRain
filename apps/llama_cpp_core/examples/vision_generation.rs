//! 视觉推理示例 - 新版API

use std::sync::Arc;

use async_openai::types::chat::CreateChatCompletionRequestArgs;
use base64::Engine;
use llama_cpp_core::{
    Pipeline, PipelineConfig,
    request::{ChatMessagesBuilder, Metadata, UserMessageBuilder},
    response::response_extract_content,
    utils::log::init_logger,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logger();

    let model_path =
        "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();
    let mmproj_path =
        "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.mmproj-Q8_0.gguf"
            .to_string();

    let pipeline_config = PipelineConfig::new_with_mmproj(model_path, mmproj_path)
        .with_n_gpu_layers(10) // GPU 层数配置 - 影响 GPU 加速
        // .with_n_threads(8) // 线程配置 - 影响 mmproj 编码速度
        .with_n_batch(1024) // 批处理配置 - 影响图像解码
        .with_media_marker("<start_of_image>") // 媒体标记配置
        .with_image_max_resolution(768) // 图片最大分辨率配置
        .with_verbose(false);

    // 创建 Pipeline（注意：是 Arc，支持并发共享）
    let pipeline = Arc::new(Pipeline::try_new(pipeline_config)?);

    // 读取图片文件
    let image_path = "/data/cy/ComfyUI_01908_.png"; // 请替换为实际图片路径
    let image_data = std::fs::read(image_path)?;
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&image_data);
    let mime_type = infer::get_from_path(image_path)
        .ok()
        .flatten()
        .map(|t| t.mime_type().to_string())
        .unwrap_or_else(|| "image/jpeg".to_string());

    // 使用新的构建器 API 构建请求
    let metadata = Metadata {
        session_id: Some("12345".to_string()),
        ..Default::default()
    };
    let messages = ChatMessagesBuilder::new()
        .system("你是专注生成套图模特提示词专家，用于生成3同人物，同场景，同服装，不同的模特照片，需要保持专业性。")
        .users(
            UserMessageBuilder::new()
                .text("请描述这张图片中的人物")
                .image_base64(mime_type, base64_data),
        )
        .build();

    let request = CreateChatCompletionRequestArgs::default()
        .max_completion_tokens(2048u32)
        .metadata(metadata)
        .model("Qwen3-VL-2B-Instruct")
        .messages(messages)
        .build()?;

    // 执行推理
    let results = pipeline.generate(&request).await?;
    println!("Response: {}", response_extract_content(&results));

    Ok(())
}
