//! 对比示例：Session 自动管理 vs 手动管理历史
//!
//! 这个示例展示了两种管理历史上下文的方式：
//! 1. 手动管理：需要在外部维护 HistoryMessage，每次请求时传入
//! 2. Session 自动管理：只需设置 session_id，Pipeline 自动处理历史

use std::sync::Arc;

use llama_cpp_core::{
    GenerateRequest, HistoryMessage, Pipeline, PipelineConfig,
    utils::log::init_logger,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logger();

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string());

    let pipeline_config = PipelineConfig::new(model_path, None)
        .with_cache_model(true);

    let pipeline = Arc::new(Pipeline::try_new(pipeline_config)?);

    println!("========================================");
    println!("方法 1: 手动管理历史 (传统方式)");
    println!("========================================\n");

    // 传统方式：需要在外部维护 history 变量
    {
        let mut history = HistoryMessage::new();

        // 第一轮
        println!("[User] 你好，我叫小明");
        let request = GenerateRequest::text("你好，我叫小明")
            .with_system("你是一个 helpful 的助手");
        let result = pipeline.generate(&request).await?;
        println!("[Assistant] {}\n", result.text);

        // 手动更新历史
        history.add_user("你好，我叫小明")?;
        history.add_assistant(&result.text)?;

        // 第二轮：需要手动传入 history
        println!("[User] 我叫什么名字？");
        let request = GenerateRequest::text("我叫什么名字？")
            .with_system("你是一个 helpful 的助手")
            .with_history(history.clone()); // <-- 需要手动传入
        let result = pipeline.generate(&request).await?;
        println!("[Assistant] {}\n", result.text);

        // 再次手动更新历史
        history.add_user("我叫什么名字？")?;
        history.add_assistant(&result.text)?;

        println!("✓ 特点：需要手动维护 history 变量\n");
    }

    println!("========================================");
    println!("方法 2: Session 自动管理 (新方式)");
    println!("========================================\n");

    // 新方式：只需设置 session_id
    {
        let session_id = "demo_user_001";

        // 第一轮：设置 session_id
        println!("[User] 你好，我叫小红");
        let request = GenerateRequest::text("你好，我叫小红")
            .with_system("你是一个 helpful 的助手")
            .with_session_id(session_id); // <-- 只需设置 session_id
        let result = pipeline.generate(&request).await?;
        println!("[Assistant] {}\n", result.text);

        // 第二轮：自动加载历史，无需传入
        println!("[User] 我叫什么名字？");
        let request = GenerateRequest::text("我叫什么名字？")
            .with_system("你是一个 helpful 的助手")
            .with_session_id(session_id); // <-- 只需 session_id，自动携带历史
        let result = pipeline.generate(&request).await?;
        println!("[Assistant] {}\n", result.text);

        // 第三轮：继续自动累积历史
        println!("[User] 我喜欢吃西瓜");
        let request = GenerateRequest::text("我喜欢吃西瓜")
            .with_system("你是一个 helpful 的助手")
            .with_session_id(session_id);
        let result = pipeline.generate(&request).await?;
        println!("[Assistant] {}\n", result.text);

        // 第四轮：验证长期记忆
        println!("[User] 我的名字和喜欢的水果是什么？");
        let request = GenerateRequest::text("我的名字和喜欢的水果是什么？")
            .with_system("你是一个 helpful 的助手")
            .with_session_id(session_id);
        let result = pipeline.generate(&request).await?;
        println!("[Assistant] {}\n", result.text);

        println!("✓ 特点：无需维护 history，Pipeline 自动管理\n");
    }

    println!("========================================");
    println!("多用户隔离示例");
    println!("========================================\n");

    // 演示多用户隔离
    let sessions = vec![
        ("alice", "Alice", "喜欢编程"),
        ("bob", "Bob", "喜欢音乐"),
        ("carol", "Carol", "喜欢绘画"),
    ];

    // 每个用户进行自我介绍
    for (id, name, hobby) in &sessions {
        println!("[{}] 你好，我叫{}，{}", id, name, hobby);
        let request = GenerateRequest::text(format!("你好，我叫{}，{}", name, hobby))
            .with_system("你是一个 helpful 的助手")
            .with_session_id(*id);
        let result = pipeline.generate(&request).await?;
        println!("[Assistant] {}\n", result.text);
    }

    // 验证每个用户的记忆是独立的
    println!("--- 验证记忆隔离 ---\n");
    for (id, name, _) in &sessions {
        let request = GenerateRequest::text("我叫什么名字？")
            .with_system("你是一个 helpful 的助手")
            .with_session_id(*id);
        let result = pipeline.generate(&request).await?;
        println!("[{}] 我叫什么名字？", id);
        println!("[Assistant] {}\n", result.text);
        assert!(result.text.contains(*name), "应该记住名字 {}", name);
    }

    println!("✓ 所有用户的记忆都是隔离的！\n");

    // 列出所有活跃会话
    let active_sessions = pipeline.list_session_ids()?;
    println!("活跃会话列表: {:?}", active_sessions);

    Ok(())
}
