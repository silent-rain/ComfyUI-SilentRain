//! 并发推理 + 外部历史管理示例
//!
//! 本示例展示如何：
//! 1. 使用 Arc<Pipeline> 实现并发推理
//! 2. 在外部管理历史消息，实现多轮对话
//! 3. 不同请求使用不同的历史上下文

use std::sync::Arc;

use llama_cpp_core::{
    GenerateRequest, LlamaChatMessage, Pipeline, PipelineConfig, PromptMessageRole,
    utils::log::init_logger,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();

    let model_path =
        "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();
    let pipeline_config = PipelineConfig::new(model_path, None)
        .with_system_prompt("你是一个 helpful 的助手")
        .with_cache_model(true);

    // 创建 Pipeline（Arc 包装，支持并发共享）
    let pipeline = Arc::new(Pipeline::try_new(pipeline_config)?);

    // ========== 场景1：单用户多轮对话 ==========
    println!("========== 场景1：单用户多轮对话 ==========");
    {
        let mut history = Vec::new();

        // 第一轮
        let request1 = GenerateRequest::text("你好，我叫小明");
        let response1 = pipeline.generate(&request1).await?;
        println!("User: 你好，我叫小明");
        println!("Assistant: {}", response1.text);

        // 更新历史（外部管理）
        history.push(LlamaChatMessage::new(
            PromptMessageRole::User.to_string(),
            "你好，我叫小明".to_string(),
        )?);
        history.push(LlamaChatMessage::new(
            PromptMessageRole::Assistant.to_string(),
            response1.text.clone(),
        )?);

        // 第二轮（带历史）
        let request2 = GenerateRequest::text("我叫什么名字？").with_history(history.clone());
        let response2 = pipeline.generate(&request2).await?;
        println!("User: 我叫什么名字？");
        println!("Assistant: {}", response2.text);

        // 更新历史
        history.push(LlamaChatMessage::new(
            PromptMessageRole::User.to_string(),
            "我叫什么名字？".to_string(),
        )?);
        history.push(LlamaChatMessage::new(
            PromptMessageRole::Assistant.to_string(),
            response2.text.clone(),
        )?);

        // 第三轮（带完整历史）
        let request3 = GenerateRequest::text("我们刚才聊了什么？").with_history(history);
        let response3 = pipeline.generate(&request3).await?;
        println!("User: 我们刚才聊了什么？");
        println!("Assistant: {}", response3.text);
    }

    // ========== 场景2：并发多用户（每个用户独立上下文） ==========
    println!("\n========== 场景2：并发多用户 ==========");
    {
        // 用户A的上下文
        let user_a_history = vec![
            LlamaChatMessage::new(
                PromptMessageRole::User.to_string(),
                "我喜欢Python".to_string(),
            )?,
            LlamaChatMessage::new(
                PromptMessageRole::Assistant.to_string(),
                "那很棒！Python 是一门优秀的编程语言。".to_string(),
            )?,
        ];

        // 用户B的上下文
        let user_b_history = vec![
            LlamaChatMessage::new(
                PromptMessageRole::User.to_string(),
                "我喜欢Rust".to_string(),
            )?,
            LlamaChatMessage::new(
                PromptMessageRole::Assistant.to_string(),
                "太棒了！Rust 以性能和安全性著称。".to_string(),
            )?,
        ];

        // 并发执行两个用户的请求
        let pipeline_a = Arc::clone(&pipeline);
        let task_a = tokio::spawn(async move {
            let request =
                GenerateRequest::text("我应该学习什么编程语言？").with_history(user_a_history);
            pipeline_a.generate(&request).await
        });

        let pipeline_b = Arc::clone(&pipeline);
        let task_b = tokio::spawn(async move {
            let request =
                GenerateRequest::text("我应该学习什么编程语言？").with_history(user_b_history);
            pipeline_b.generate(&request).await
        });

        // 获取结果
        let (result_a, result_b) = tokio::try_join!(task_a, task_b)?;

        println!("User A (喜欢Python) 问：我应该学习什么编程语言？");
        println!("Assistant: {}", result_a?.text);

        println!("\nUser B (喜欢Rust) 问：我应该学习什么编程语言？");
        println!("Assistant: {}", result_b?.text);
    }

    // ========== 场景3：批量处理（无历史，纯并发） ==========
    println!("\n========== 场景3：批量处理 ==========");
    {
        let questions = vec![
            "什么是机器学习？",
            "什么是深度学习？",
            "什么是神经网络？",
            "什么是自然语言处理？",
        ];

        let tasks: Vec<_> = questions
            .into_iter()
            .map(|question| {
                let pipeline = Arc::clone(&pipeline);
                tokio::spawn(async move {
                    let request = GenerateRequest::text(question);
                    pipeline.generate(&request).await
                })
            })
            .collect();

        let results = futures::future::try_join_all(tasks).await?;
        for (i, result) in results.iter().enumerate() {
            match result {
                Ok(output) => println!("问题{}: {}", i + 1, output.text),
                Err(e) => eprintln!("错误{}: {}", i + 1, e),
            }
        }
    }

    Ok(())
}
