//! 视觉推理示例 - 并发版本

use std::sync::Arc;

use async_openai::types::chat::CreateChatCompletionRequestArgs;
use futures::future::join_all;
use llama_flow::{
    Pipeline, PipelineConfig,
    error::Error,
    request::{ChatMessagesBuilder, UserMessageBuilder},
    response::response_extract_content,
    utils::log::init_logger,
};
use tokio::sync::Semaphore;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();

    let model_path =
        "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();
    let mmproj_path =
        "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.mmproj-Q8_0.gguf"
            .to_string();

    let pipeline_config = PipelineConfig::new_with_mmproj(model_path, mmproj_path)
        // .with_n_threads(8) // 线程配置 - 影响 mmproj 编码速度
        .with_n_batch(1024) // 批处理配置 - 影响图像解码
        .with_image_max_resolution(768) // 图片最大分辨率配置
        .with_verbose(true);

    // 创建 Pipeline（注意：是 Arc，支持并发共享）
    let pipeline = Arc::new(Pipeline::try_new(pipeline_config)?);
    let semaphore = Arc::new(Semaphore::new(2)); // 限制最大并发数为 2

    let user_prompts = vec![
        "修改表情, 例如：魅惑地笑/捂嘴偷笑/平静地笑容等, 要求9个图像都有不同的表情",
        "修改姿势, 例如：双手叉腰，比心的手势等, 要求9个图像都有不同的动作，动作变化幅度不应很小",
        "修改拍摄景别, 例如：特写，中景等, 要求9个图像根据不同动作有合适的拍摄景别",
        "修改拍摄角度, 例如：微俯拍30度，正面拍摄等, 要求9个图像根据不同动作有合适的拍摄角度",
    ];

    // 生成所有并发
    let futures = user_prompts.into_iter().enumerate().map(|(i, user_prompt)| {
        let semaphore = Arc::clone(&semaphore);
        let pipeline_clone = pipeline.clone();

        {
            async move {
                let _permit = semaphore.acquire().await.map_err(|e| {
                    error!("获取Semaphore许可失败, err: {:#?}", e);
                    Error::AcquireError(e.to_string())
                })?;

                 let request1 = CreateChatCompletionRequestArgs::default()
                    .max_completion_tokens(2048u32)
                    .model("Qwen3-VL-2B-Instruct")
                    .messages(ChatMessagesBuilder::new()
                        .system("你是专注生成套图模特提示词专家，用于生成9个同人物，同场景，同服装，不同的模特照片，需要保持专业性。")
                        .users(
                            UserMessageBuilder::new()
                                .text(user_prompt)
                                .image_file("/home/one/Downloads/cy/ComfyUI_01918_.png")?,
                        )
                    .build())
                    .build()?;


                let output = pipeline_clone.generate(&request1).await?;
                let result = response_extract_content(&output);

                info!("iamge {i} prcessing completed");
                Ok::<_, Error>(result)
            }
        }
    });

    // 并发执行所有 Future
    let results: Vec<String> = join_all(futures)
        .await
        .into_iter()
        .collect::<Result<Vec<String>, _>>()?
        .into_iter()
        .collect();

    info!("Final results: {:#?}", results);
    Ok(())
}
