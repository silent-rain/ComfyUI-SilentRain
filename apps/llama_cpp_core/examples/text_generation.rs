//! 基础使用示例 - 新版API

use std::sync::Arc;

use async_openai::types::chat::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestSystemMessage,
    CreateChatCompletionRequestArgs, ImageDetail, ImageUrl,
};
use llama_cpp_core::{
    Pipeline, PipelineConfig,
    pipeline::{ChatMessagesBuilder, UserMessageBuilder},
    types::{
        ChatCompletionRequestMessageContentPartImage, ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    },
    utils::log::init_logger,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_logger();

    let model_path =
        "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();
    let pipeline_config = PipelineConfig::new(model_path).with_verbose(false);

    let pipeline = Arc::new(Pipeline::try_new(pipeline_config)?);

    // 原始请求体构建
    {
        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(2048u32)
            .model("Qwen3-VL-2B-Instruct")
            .messages([
                // Can also use ChatCompletionRequest<Role>MessageArgs for builder pattern
                ChatCompletionRequestSystemMessage::from("You are a helpful assistant.").into(),
                ChatCompletionRequestUserMessage::from("Who won the world series in 2020?").into(),
                ChatCompletionRequestAssistantMessage::from(
                    "The Los Angeles Dodgers won the World Series in 2020.",
                )
                .into(),
                // ChatCompletionRequestUserMessage::from("Where was it played?").into(),
                ChatCompletionRequestUserMessage {
                    content: ChatCompletionRequestUserMessageContent::Array(vec![
                        ChatCompletionRequestMessageContentPartText::from("Where was it played?")
                            .into(),
                        ChatCompletionRequestMessageContentPartImage::from(ImageUrl {
                            url: "https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png".to_string(),
                            detail: Some(ImageDetail::Auto),
                        })
                        .into(),
                    ]),
                    ..Default::default()
                }
                .into(),
            ])
            .build()?;

        println!("{}", serde_json::to_string(&request).unwrap());

        let results = pipeline.generate(&request).await?;

        println!("{results:?}");
    }

    // 原始请求体包装
    {
        let  messages = ChatMessagesBuilder::new()
            .system("You are a helpful assistant.")
            .user(UserMessageBuilder::new().text("Who won the world series in 2020?"))
            .assistant("The Los Angeles Dodgers won the World Series in 2020.")
            .user(UserMessageBuilder::new().text("Where was it played?").image_url("https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"))
           .build();

        let request = CreateChatCompletionRequestArgs::default()
            .max_tokens(2048u32)
            .model("Qwen3-VL-2B-Instruct")
            .messages(messages)
            .build()?;

        println!("{}", serde_json::to_string(&request).unwrap());

        let results = pipeline.generate(&request).await?;

        println!("{results:?}");
    }
    Ok(())
}
