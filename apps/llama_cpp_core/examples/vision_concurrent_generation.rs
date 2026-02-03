//! 视觉推理示例 - 并发版本

use std::sync::Arc;

use llama_cpp_core::{GenerateRequest, Pipeline, PipelineConfig, utils::log::init_logger};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();

    let model_path =
        "/data/ComfyUI/models/clip/Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf".to_string();
    let mmproj_path =
        "/data/ComfyUI/models/clip/Qwen2.5-VL-7B-Instruct-abliterated.mmproj-f16.gguf".to_string();

    let pipeline_config = PipelineConfig::new(model_path, Some(mmproj_path))
        .with_disable_gpu(false) // GPU 配置 - 启用 GPU 加速
        .with_n_gpu_layers(10) // GPU 层数配置 - 影响 GPU 加速
        // .with_n_threads(8) // 线程配置 - 影响 mmproj 编码速度
        // .with_n_threads_batch(8) // 批处理线程配置 - 影响处理速度
        .with_n_batch(1024) // 批处理配置 - 影响图像解码
        .with_n_ubatch(2048) // 对于大图像需要较大的 n_ubatch
        // .with_use_mlock(true)  // 内存锁定配置 - 影响处理速度
        .with_cache_model(true) // 模型缓存配置
        .with_verbose(true);

    // 创建 Pipeline（注意：是 Arc，支持并发共享）
    let pipeline = Arc::new(Pipeline::try_new(pipeline_config)?);

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

    // 第一个推理请求
    let request1 = GenerateRequest::text(user_prompt).with_system("你是专注生成套图模特提示词专家，用于生成9个同人物，同场景，同服装，不同的模特照片，需要保持专业性。")
    .with_image_max_resolution(768) // 图片最大分辨率配置
    .with_media_file("/home/one/Downloads/cy/ComfyUI_01918_.png")?;

    // 第二个推理请求（可以并发执行）
    let request2 = GenerateRequest::text(
        "再生成9个提示词，保持写实风格，人物轮廓与原图一致，光线柔和无畸变，背景细节保留原图特征",
    )
    .with_system("你是专注生成套图模特提示词专家。")
    .with_image_max_resolution(768)
    .with_media_file("/home/one/Downloads/cy/ComfyUI_01918_.png")?;

    // 执行第一个推理
    let results1 = pipeline.generate(&request1).await?;
    println!("Response 1: {}", results1.text);

    println!("========================== 并发测试 ==========================");

    // 并发执行多个请求
    let pipeline_clone = Arc::clone(&pipeline);
    let request_clone = request2.clone();
    let task1 = tokio::spawn(async move { pipeline_clone.generate(&request_clone).await });

    let pipeline_clone2 = Arc::clone(&pipeline);
    let request_clone2 = request2.clone();
    let task2 = tokio::spawn(async move { pipeline_clone2.generate(&request_clone2).await });

    // 等待并发结果
    let (result1, result2) = tokio::try_join!(task1, task2)?;

    match result1 {
        Ok(output) => println!("并发结果 1: {}", output.text),
        Err(e) => eprintln!("并发错误 1: {}", e),
    }

    match result2 {
        Ok(output) => println!("并发结果 2: {}", output.text),
        Err(e) => eprintln!("并发错误 2: {}", e),
    }

    Ok(())
}
