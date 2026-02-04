//! 基于会话的历史上下文示例
//!
//! 这个示例展示了如何使用 session_id 来实现：
//! - 自动历史管理（无需手动维护 history）
//! - 多会话隔离（不同用户的对话互不干扰）
//! - 并发安全（多个会话可以并发处理）

use std::sync::Arc;

use llama_cpp_core::{GenerateRequest, Pipeline, PipelineConfig, utils::log::init_logger};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logger();

    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string()
    });

    let pipeline_config = PipelineConfig::new(model_path)
        .with_cache_model(true)
        .with_verbose(true);

    let pipeline = Arc::new(Pipeline::try_new(pipeline_config)?);

    println!("=== 单会话对话示例 ===\n");

    // 示例 1: 单用户多轮对话
    let user_session = "user_001";

    // 第一轮对话
    println!("[User] 你好，我叫小明");
    let request = GenerateRequest::text("你好，我叫小明")
        .with_system("你是一个 helpful 的助手")
        .with_session_id(user_session);
    let result = pipeline.generate(&request).await?;
    println!("[Assistant] {}\n", result.text);

    // 第二轮对话（自动携带历史）
    println!("[User] 我叫什么名字？");
    let request = GenerateRequest::text("我叫什么名字？")
        .with_system("你是一个 helpful 的助手")
        .with_session_id(user_session);
    let result = pipeline.generate(&request).await?;
    println!("[Assistant] {}\n", result.text);

    // 第三轮对话
    println!("[User] 我喜欢吃苹果");
    let request = GenerateRequest::text("我喜欢吃苹果")
        .with_system("你是一个 helpful 的助手")
        .with_session_id(user_session);
    let result = pipeline.generate(&request).await?;
    println!("[Assistant] {}\n", result.text);

    // 第四轮对话（测试长期记忆）
    println!("[User] 我的名字和喜欢的水果是什么？");
    let request = GenerateRequest::text("我的名字和喜欢的水果是什么？")
        .with_system("你是一个 helpful 的助手")
        .with_session_id(user_session);
    let result = pipeline.generate(&request).await?;
    println!("[Assistant] {}\n", result.text);

    println!("\n=== 多会话隔离示例 ===\n");

    // 示例 2: 多用户会话隔离
    let user_a = "user_a";
    let user_b = "user_b";

    // 用户 A 的自我介绍
    println!("[User A] 你好，我是 Alice");
    let request = GenerateRequest::text("你好，我是 Alice")
        .with_system("你是一个 helpful 的助手")
        .with_session_id(user_a);
    let result = pipeline.generate(&request).await?;
    println!("[Assistant] {}\n", result.text);

    // 用户 B 的自我介绍
    println!("[User B] 你好，我是 Bob");
    let request = GenerateRequest::text("你好，我是 Bob")
        .with_system("你是一个 helpful 的助手")
        .with_session_id(user_b);
    let result = pipeline.generate(&request).await?;
    println!("[Assistant] {}\n", result.text);

    // 验证用户 A 的记忆
    println!("[User A] 我叫什么名字？");
    let request = GenerateRequest::text("我叫什么名字？")
        .with_system("你是一个 helpful 的助手")
        .with_session_id(user_a);
    let result = pipeline.generate(&request).await?;
    println!("[Assistant] {}\n", result.text);

    // 验证用户 B 的记忆（应该是 Bob，不是 Alice）
    println!("[User B] 我叫什么名字？");
    let request = GenerateRequest::text("我叫什么名字？")
        .with_system("你是一个 helpful 的助手")
        .with_session_id(user_b);
    let result = pipeline.generate(&request).await?;
    println!("[Assistant] {}\n", result.text);

    println!("\n=== 并发会话示例 ===\n");

    // 示例 3: 并发处理多个会话
    let mut tasks = vec![];

    for i in 0..3 {
        let pipeline = Arc::clone(&pipeline);
        let task = tokio::spawn(async move {
            let session_id = format!("concurrent_user_{}", i);
            let name = format!("User{}", i);

            println!("[{}] 开始对话", session_id);

            // 自我介绍
            let request = GenerateRequest::text(format!("你好，我叫{}", name))
                .with_system("你是一个 helpful 的助手")
                .with_session_id(&session_id);
            let result = pipeline.generate(&request).await?;
            println!("[{}] 第一轮: {}", session_id, result.text);

            // 验证记忆
            let request = GenerateRequest::text("请重复我的名字")
                .with_system("你是一个 helpful 的助手")
                .with_session_id(&session_id);
            let result = pipeline.generate(&request).await?;
            println!("[{}] 第二轮: {}", session_id, result.text);

            anyhow::Ok(())
        });
        tasks.push(task);
    }

    // 等待所有任务完成
    for task in tasks {
        task.await??;
    }

    println!("\n=== 会话管理示例 ===\n");

    // 示例 4: 会话管理
    let sessions = pipeline.list_session_ids()?;
    println!("当前所有会话: {:?}", sessions);

    // 清除特定会话
    println!("\n清除会话 '{}'...", user_session);
    pipeline.clear_session_history(user_session)?;

    let sessions = pipeline.list_session_ids()?;
    println!("清除后剩余会话: {:?}", sessions);

    println!("\n=== 使用流式 API ===\n");

    // 示例 5: 流式 API 与 session 结合
    let stream_session = "stream_user";
    println!("[User] 讲一个短故事");

    let request = GenerateRequest::text("讲一个短故事")
        .with_system("你是一个 creative 的助手")
        .with_session_id(stream_session);

    let mut rx = pipeline.generate_stream(&request).await?;
    let mut full_response = String::new();

    println!("[Assistant] ");
    while let Some(token) = rx.recv().await {
        match token {
            llama_cpp_core::types::StreamToken::Content(text) => {
                print!("{}", text);
                full_response.push_str(&text);
            }
            llama_cpp_core::types::StreamToken::Finish(_) => break,
            llama_cpp_core::types::StreamToken::Error(msg) => {
                eprintln!("Error: {}", msg);
                break;
            }
        }
    }
    println!("\n");

    // 手动保存流式会话历史
    pipeline.save_or_clear_session_history(&request, &full_response)?;

    // 验证流式会话的历史已保存
    let request = GenerateRequest::text("刚才讲了什么故事？简要概括")
        .with_system("你是一个 helpful 的助手")
        .with_session_id(stream_session);
    let result = pipeline.generate(&request).await?;
    println!("[Assistant] {}\n", result.text);

    println!("\n所有示例完成！");

    Ok(())
}
