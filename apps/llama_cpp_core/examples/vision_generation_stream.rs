//! 视觉推理示例 - 新版API

use std::sync::Arc;

use async_openai::types::chat::CreateChatCompletionRequestArgs;
use base64::Engine;
use llama_cpp_core::{
    Pipeline, PipelineConfig,
    pipeline::{ChatMessagesBuilder, UserMessageBuilder},
    utils::log::init_logger,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let image_path = "/path/to/your/image.jpg"; // 请替换为实际图片路径
    let image_data = std::fs::read(image_path)?;
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&image_data);
    let mime_type = infer::get_from_path(image_path)
        .ok()
        .flatten()
        .map(|t| t.mime_type().to_string())
        .unwrap_or_else(|| "image/jpeg".to_string());

    let user_prompt = {
        r#"
任务：根据提供的单张人物图片，生成9个结构化的提示词，要求人物一致性不变，场景不变，服装不变，生成的照片要风格写实，符合专业摄影，光线和原图一致

### 提示词生成规则
获取图片内容，按照整体规则生成合适的提示词；

按以下模板生成9条不重复提示词，每条包含以下部分，同时保证摄影的专业性和观赏性：  

【修改指令】

##修改表情：

示例：

如：魅惑地笑/捂嘴偷笑/平静地笑容等
要求9个图像都有不同的表情

##修改姿势：

示例：

如：双手叉腰，比心的手势等
要求9个图像都有不同的动作，动作变化幅度不应很小


##修改拍摄景别：

示例

如：特写，中景等
要求9个图像根据不同动作有合适的拍摄景别

##修改拍摄角度

示例

如：微俯拍30度，正面拍摄等
要求9个图像根据不同动作有合适的拍摄角度


写实风格，人物轮廓与原图一致，光线柔和无畸变，背景细节保留原图特征。  


### 输出要求  
仅返回10条提示词，每条独立成段，用换行分隔，无其他内容。  
输出格式：【prompt_1】,【prompt_2】,【prompt_3】...


### 示例如下：

【prompt_1】同一角色、服装、场景一致，写实风格，光影一致，仅改表情/姿势/视角：中景拍摄+抿嘴偷笑+眼睛弯弯+双手背后+微俯拍30度，8K

...（共9条）"#
    };

    // 使用新的构建器 API 构建请求
    let messages = ChatMessagesBuilder::new()
        .system("你是专注生成套图模特提示词专家，用于生成3同人物，同场景，同服装，不同的模特照片，需要保持专业性。")
        .users(
            UserMessageBuilder::new()
                .text(user_prompt)
                .image_base64(mime_type, base64_data),
        )
        .build();

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(2048u32)
        .model("Qwen3-VL-2B-Instruct")
        .messages(messages)
        .build()?;

    // 执行推理
    let mut rx = pipeline.generate_stream(&request).await?;

    // 接收响应
    let mut full_text = String::new();
    while let Some(chunk) = rx.recv().await {
        if let Some(choice) = chunk.choices.first() {
            if let Some(content) = &choice.delta.content {
                full_text.push_str(content);
            }
            if choice.finish_reason.is_some() {
                break;
            }
        }
    }
    println!("{full_text:?}");
    Ok(())
}
