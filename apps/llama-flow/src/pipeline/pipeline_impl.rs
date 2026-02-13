//! 推理流水线 - 核心推理引擎
//!
//! Pipeline 是 llama-flow 的核心组件，负责：
//! - 模型加载与管理
//! - 文本生成（聊天）
//! - 多模态推理（视觉）
//! - 缓存管理
//! - 上下文管理

use std::sync::Arc;

use async_openai::types::chat::{
    CreateChatCompletionRequest, CreateChatCompletionResponse, CreateChatCompletionStreamResponse,
    FinishReason,
};
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{LlamaChatMessage, LlamaModel},
    mtmd::MtmdContext,
};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};
use tracing::{error, info, warn};

use crate::{
    Backend, Model, PipelineConfig, Sampler,
    context::{ContexParams, ContextWrapper},
    error::Error,
    hooks::{
        DynHook, HookContext, InferenceHook,
        builtin::{
            AssembleMessagesHook, LoadHistoryHook, NormalizeHook, SaveHistoryHook, SystemPromptHook,
        },
    },
    mtmd_context::MtmdContextWrapper,
    response::ChatCompletionBuilder,
    utils::image::{Image, decode_image_sources},
};

/// 推理流水线
pub struct Pipeline {
    backend: Arc<LlamaBackend>,
    llama_model: Arc<LlamaModel>,
    mtmd_context: Option<Arc<MtmdContext>>,
    config: PipelineConfig,
    hooks: Vec<DynHook>,
}

unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

impl Pipeline {
    /// 创建新的流水线
    pub fn try_new(config: PipelineConfig) -> Result<Self, Error> {
        // 初始化后端
        let backend = Arc::new(Backend::init_backend()?);

        // Load model
        let model = Model::from_config(config.model.clone());
        let llama_model = model.load_llama_model(&backend).map_err(|e| {
            error!("Failed to load model: {}", e);
            e
        })?;

        // Load mtmd model
        let mtmd_context = if !config.model.mmproj_path.is_empty() {
            let mtmd_context = model.load_mtmd_context(llama_model.clone()).map_err(|e| {
                error!("Failed to load mtmd context: {}", e);
                e
            })?;

            Some(mtmd_context)
        } else {
            None
        };

        // load hooks
        let hooks: Vec<DynHook> = vec![
            Arc::new(NormalizeHook::new().with_trim(true).with_remove_empty(true)), // 消息标准化（优先级 10）
            Arc::new(
                SystemPromptHook::keep_first().with_default_system("You are a helpful assistant."),
            ), // 系统提示词处理（优先级 20）
            Arc::new(LoadHistoryHook::new().with_max_history(100)), // 加载历史消息（优先级 30）
            Arc::new(AssembleMessagesHook::new()),                  // 组装最终消息（优先级 40）
            Arc::new(SaveHistoryHook::new()),                       // 保存历史（优先级 60）
        ];

        Ok(Self {
            backend,
            llama_model,
            mtmd_context,
            config,
            hooks,
        })
    }

    /// 注册钩子
    pub fn with_hook(mut self, hook: impl InferenceHook + 'static) -> Self {
        self.hooks.push(Arc::new(hook));
        self
    }

    /// 获取排序后的钩子列表
    fn sorted_hooks(&self) -> Vec<DynHook> {
        let mut hooks: Vec<_> = self.hooks.clone().into_iter().collect();
        hooks.sort_by_key(|h| h.priority());
        hooks
    }
}

impl Pipeline {
    /// 阻塞推理包装
    pub fn generate_block(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, Error> {
        // 创建 tokio 运行时用于执行异步逻辑
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            error!("Failed to create tokio runtime: {}", e);
            Error::RuntimeError(e.to_string())
        })?;

        let pipeline = Arc::new(self);

        // 在运行时中阻塞执行生成任务
        rt.block_on(async {
            let output = pipeline.generate_with_hooks(request).await?;
            Ok(output)
        })
    }

    /// 执行推理
    pub async fn generate(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, Error> {
        self.generate_with_hooks(request).await
    }

    /// 执行推理（带钩子）
    async fn generate_with_hooks(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, Error> {
        let config = PipelineConfig::apply_request_params(&self.config, request)?;

        // 1. 创建上下文
        let mut hook_ctx = HookContext::new(request, &config);
        let hooks = self.sorted_hooks();

        // 2. 执行 on_prepare（准备消息）
        self.on_prepare(&mut hook_ctx).await?;

        // 3. 执行推理
        let result = self.generate_internal(request, &mut hook_ctx).await;

        // 4. 执行 on_after 或 on_error
        match result {
            Ok(response) => {
                hook_ctx.set_response(response.clone());

                // 执行 on_after
                let _ = Self::on_after(hooks.clone(), &mut hook_ctx).await;

                Ok(response)
            }
            Err(e) => {
                for hook in &hooks {
                    let _ = hook.on_error(&hook_ctx, &e).await;
                }
                Err(e)
            }
        }
    }

    /// 执行内部推理
    ///
    /// 将流式推理结果收集成完整响应，返回响应和输出文本
    ///
    /// 直接使用未包装钩子的流式推理，避免流式转发，减少开销。
    async fn generate_internal(
        &self,
        request: &CreateChatCompletionRequest,
        hook_ctx: &mut HookContext,
    ) -> Result<CreateChatCompletionResponse, Error> {
        let mut rx = self.generate_multimodal_stream(hook_ctx).await?;

        let model = request.model.clone();

        let mut full_text = String::new();
        let mut finish_reason: Option<FinishReason> = None;
        let mut prompt_tokens: u32 = 0;
        let mut completion_tokens: u32 = 0;

        // 收集所有流式响应
        while let Some(chunk) = rx.recv().await {
            // 提取 usage（通常在最后一个 chunk）
            if let Some(u) = &chunk.usage {
                prompt_tokens = u.prompt_tokens;
                completion_tokens = u.completion_tokens;
            }

            // 处理 choices
            if let Some(choice) = chunk.choices.first() {
                // 累加文本内容
                if let Some(content) = &choice.delta.content {
                    full_text.push_str(content);
                }
                // 记录结束原因
                if choice.finish_reason.is_some() {
                    finish_reason = choice.finish_reason;
                }
            }
        }

        // 构建完整的非流式响应
        let response = ChatCompletionBuilder::new(model)
            .with_prompt_tokens(prompt_tokens)
            .build(full_text.clone(), finish_reason, completion_tokens);

        Ok(response)
    }

    /// 执行流式推理
    ///
    /// 根据有无钩子决定是否包装钩子的响应流；如果没有钩子则直接返回原始流，避免流式转发，减少开销。
    pub async fn generate_stream(
        &self,
        request: &CreateChatCompletionRequest,
    ) -> Result<UnboundedReceiver<CreateChatCompletionStreamResponse>, Error> {
        let config = PipelineConfig::apply_request_params(&self.config, request)?;

        // 如果没有钩子，直接返回原始流（零开销）
        let mut hook_ctx = HookContext::new(request, &config);
        if self.hooks.is_empty() {
            return self.generate_multimodal_stream(&mut hook_ctx).await;
        }

        // 有钩子时需要包装流
        self.generate_multimodal_stream_with_hooks(request, &mut hook_ctx)
            .await
    }

    /// 带钩子的流式推理
    ///
    /// 流式输出时，会将消息进行转发，有一定性能损耗
    async fn generate_multimodal_stream_with_hooks(
        &self,
        request: &CreateChatCompletionRequest,
        hook_ctx: &mut HookContext,
    ) -> Result<UnboundedReceiver<CreateChatCompletionStreamResponse>, Error> {
        // 1. 执行 on_prepare
        self.on_prepare(hook_ctx).await?;

        // 2. 获取原始流
        let mut inner_rx = self.generate_multimodal_stream(hook_ctx).await?;

        // 3. 创建新通道
        let (tx, rx) = unbounded_channel();

        // 4. 克隆必要的数据用于异步任务
        let hooks_clone = self.sorted_hooks();
        let request_clone = request.clone();
        let config_clone = hook_ctx.config.clone();

        // 5. 启动中转任务
        tokio::spawn(async move {
            let mut full_text = String::new();
            let mut last_chunk: Option<CreateChatCompletionStreamResponse> = None;
            let mut success = true;

            // 中转所有数据
            while let Some(chunk) = inner_rx.recv().await {
                // 收集完整响应文本
                if let Some(choice) = chunk.choices.first()
                    && let Some(content) = &choice.delta.content
                {
                    full_text.push_str(content);
                }
                last_chunk = Some(chunk.clone());

                // 转发给外部通道
                if tx.send(chunk).is_err() {
                    tracing::warn!("Stream receiver dropped");
                    success = false;
                    break;
                }
            }

            // 6. 流结束，执行 on_after 钩子
            let mut hook_ctx = HookContext::new(&request_clone, &config_clone);
            hook_ctx.set_stream_collected_text(full_text.clone());

            if let Some(chunk) = last_chunk {
                hook_ctx.set_stream_last_chunk(chunk);
            }

            if success {
                // 执行 on_after
                let _ = Self::on_after(hooks_clone.clone(), &mut hook_ctx).await;
            }
        });

        Ok(rx)
    }
}

/// 生命周期钩子调用
impl Pipeline {
    /// 执行 on_prepare 钩子
    async fn on_prepare(&self, ctx: &mut HookContext) -> Result<(), Error> {
        let hooks: Vec<_> = self.sorted_hooks();

        for hook in &hooks {
            let result: Result<(), Error> = hook.on_prepare(ctx).await;
            if let Err(e) = result {
                error!("Prepare hook '{}' failed: {}", hook.name(), e);
                let _ = hook.on_error(ctx, &e).await;
                return Err(e);
            }
        }
        Ok(())
    }

    /// 执行 on_before 钩子
    async fn on_before(&self, ctx: &mut HookContext) -> Result<(), Error> {
        let hooks: Vec<_> = self.sorted_hooks();

        for hook in &hooks {
            let result: Result<(), Error> = hook.on_before(ctx).await;
            if let Err(e) = result {
                error!("Before hook '{}' failed: {}", hook.name(), e);
                let _ = hook.on_error(ctx, &e).await;
                return Err(e);
            }
        }
        Ok(())
    }

    /// 执行 on_after 钩子
    async fn on_after(hooks: Vec<DynHook>, ctx: &mut HookContext) -> Result<(), Error> {
        for hook in &hooks {
            let result: Result<(), Error> = hook.on_after(ctx).await;
            if let Err(e) = result {
                error!("After hook '{}' failed: {}", hook.name(), e);
                let _ = hook.on_error(ctx, &e).await;
                return Err(e);
            }
        }

        Ok(())
    }
}

/// 流水线实现细节
impl Pipeline {
    /// 根据请求准备消息
    ///
    /// 流程：
    /// 1. 从 pipeline_state 获取处理后的消息
    /// 2. 转换为 LlamaChatMessage 列表
    ///
    /// # Arguments
    /// * `request` - OpenAI 标准请求
    /// * `hook_ctx` - HookContext（已初始化）
    pub async fn prepare_messages(
        &self,
        hook_ctx: &mut HookContext,
    ) -> Result<Vec<LlamaChatMessage>, Error> {
        if hook_ctx.pipeline_state.working_messages.is_empty() {
            warn!("No messages to prepare");
            return Ok(Vec::new());
        }

        // 从 pipeline_state 获取处理后的消息
        let processed_messages = &hook_ctx.pipeline_state.working_messages;

        info!(
            "Prepared {} messages for inference",
            processed_messages.len()
        );

        // 转换为 LlamaChatMessage
        let llama_messages: Vec<LlamaChatMessage> = processed_messages
            .iter()
            .map(|msg| {
                let content = msg.to_llama_format(&hook_ctx.config.context.media_marker)?;
                LlamaChatMessage::new(msg.role.to_string(), content).map_err(|e| {
                    Error::InvalidInput {
                        field: "LlamaChatMessage".to_string(),
                        message: e.to_string(),
                    }
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        info!("Prepared llama messages: {:?}", llama_messages);

        Ok(llama_messages)
    }

    /// 多模态流式推理
    async fn generate_multimodal_stream(
        &self,
        hook_ctx: &mut HookContext,
    ) -> Result<UnboundedReceiver<CreateChatCompletionStreamResponse>, Error> {
        let rx = if !hook_ctx.is_multimodal_request() {
            self.generate_text_stream(hook_ctx).await?
        } else {
            self.generate_media_stream(hook_ctx).await?
        };

        Ok(rx)
    }

    /// 纯文本流式推理内部实现
    async fn generate_text_stream(
        &self,
        hook_ctx: &mut HookContext,
    ) -> Result<UnboundedReceiver<CreateChatCompletionStreamResponse>, Error> {
        let msgs = self.prepare_messages(hook_ctx).await?;

        // Load sampler
        let mut sampler =
            Sampler::load_sampler(&hook_ctx.config.sampling.clone()).map_err(|e| {
                error!("Failed to load sampler: {}", e);
                e
            })?;

        // 创建上下文
        let contex_params: ContexParams = hook_ctx.config.context.clone();
        let mut ctx =
            ContextWrapper::try_new(self.llama_model.clone(), &self.backend, &contex_params)
                .map_err(|e| {
                    error!("Failed to create context: {}", e);
                    e
                })?;

        // 评估消息
        ctx.eval_messages(msgs.to_vec()).map_err(|e| {
            error!("Failed to eval messages: {}", e);
            e
        })?;

        // 执行 on_before
        self.on_before(hook_ctx).await?;

        // 使用 channel 方式生成响应
        ctx.generate_response(&mut sampler, hook_ctx.config.model.model_name.clone())
    }

    /// 媒体流式推理内部实现
    async fn generate_media_stream(
        &self,
        hook_ctx: &mut HookContext,
    ) -> Result<UnboundedReceiver<CreateChatCompletionStreamResponse>, Error> {
        let msgs = self.prepare_messages(hook_ctx).await?;

        let mtmd_context = self.mtmd_context.clone().ok_or_else(|| {
            error!("MTMD context is not initialized");
            Error::MtmdContextNotInitialized
        })?;

        // Load sampler
        let mut sampler =
            Sampler::load_sampler(&hook_ctx.config.sampling.clone()).map_err(|e| {
                error!("Failed to load sampler: {}", e);
                e
            })?;

        // 上下文
        let contex_params: ContexParams = hook_ctx.config.context.clone();
        let ctx = ContextWrapper::try_new(self.llama_model.clone(), &self.backend, &contex_params)
            .map_err(|e| {
                error!("Failed to create context: {}", e);
                e
            })?;

        let mut mtmd_ctx = MtmdContextWrapper::try_new(
            self.llama_model.clone(),
            ctx,
            mtmd_context,
            &contex_params,
        )
        .map_err(|e| {
            error!("Failed to create mtmd context: {}", e);
            e
        })?;

        // Load media files
        let media_sources = hook_ctx.media_sources();
        let decoded_images = decode_image_sources(&media_sources).await?;

        // 将解码后的图像加载到 MTMD 上下文中
        for data in decoded_images {
            // 从二进制数据创建 Image 对象，并根据配置缩放
            let img = Image::from_bytes(&data)?;

            let max_resolution = img
                .longest()
                .min(hook_ctx.config.context.image_max_resolution);
            img.resize_to_longest(max_resolution)?;
            // img.save("/home/one/Downloads/resized.jpg")?;

            // 将缩放后的图像转换为字节数据
            let resized_data = img.to_vec()?;

            mtmd_ctx.load_media_buffer(&resized_data).map_err(|e| {
                error!("Failed to load media: {}", e);
                e
            })?;
        }

        // 评估消息
        mtmd_ctx.eval_messages(msgs.to_vec()).map_err(|e| {
            error!("Failed to eval messages: {}", e);
            e
        })?;

        // 执行 on_before
        self.on_before(hook_ctx).await?;

        // 使用 channel 方式生成响应
        mtmd_ctx.generate_response(&mut sampler, hook_ctx.config.model.model_name.clone())
    }
}

#[cfg(test)]
mod tests {
    use async_openai::types::chat::CreateChatCompletionRequestArgs;
    use base64::{Engine, engine::general_purpose};

    use crate::{
        hooks::builtin::{ErrorLogHook, ToolsHook},
        request::{ChatMessagesBuilder, UserMessageBuilder},
        utils::log::init_logger,
    };

    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_simple_text() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/dataEtx/models/LLM/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf".to_string();

        let pipeline_config = PipelineConfig::new(model_path).with_verbose(false);

        let pipeline = Pipeline::try_new(pipeline_config)?;

        let request = CreateChatCompletionRequestArgs::default()
            .max_completion_tokens(2048u32)
            .model("Qwen3-VL-2B-Instruct")
            .messages(
                ChatMessagesBuilder::new()
                    .system("You are a helpful assistant.")
                    .user("Who won the world series in 2020?")
                    .assistant("The Los Angeles Dodgers won the World Series in 2020.")
                    .users(UserMessageBuilder::new().text("Where was it played?"))
                    .build(),
            )
            .build()?;

        info!("{}", serde_json::to_string(&request).unwrap());

        let results = pipeline.generate(&request).await?;

        info!("{:?}", results);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_simple_vision_for_image_file() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf"
                .to_string();
        let mmproj_path =
            "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.mmproj-Q8_0.gguf"
                .to_string();

        let pipeline_config =
            PipelineConfig::new_with_mmproj(model_path, mmproj_path).with_verbose(false);

        let pipeline = Pipeline::try_new(pipeline_config)?;

        // 反推图片
        let request = CreateChatCompletionRequestArgs::default()
            .max_completion_tokens(2048u32)
            .model("Qwen3-VL-2B-Instruct")
            .messages(
                ChatMessagesBuilder::new()
                    .system("You are a helpful assistant.")
                    .users(
                        UserMessageBuilder::new()
                            .text("描述这张图片")
                            .image_file("/data/cy/rgthree.compare._temp_glbsl_00013_.png")?,
                    )
                    .build(),
            )
            .build()?;

        let results = pipeline.generate(&request).await?;

        info!("{:?}", results);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_simple_vision_for_image_base64() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf"
                .to_string();
        let mmproj_path =
            "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.mmproj-Q8_0.gguf"
                .to_string();

        let pipeline_config =
            PipelineConfig::new_with_mmproj(model_path, mmproj_path).with_verbose(false);

        let pipeline = Pipeline::try_new(pipeline_config)?;

        // 读取图像文件并编码为base64
        let image_url = "/data/cy/rgthree.compare._temp_glbsl_00013_.png";
        let mime_type = infer::get_from_path(image_url)?.unwrap().mime_type();
        let buffer = std::fs::read(image_url)?;
        let base64_data = general_purpose::STANDARD.encode(&buffer);

        // 反推图片
        let request = CreateChatCompletionRequestArgs::default()
            .max_completion_tokens(2048u32)
            .model("Qwen3-VL-2B-Instruct")
            .messages(
                ChatMessagesBuilder::new()
                    .system("You are a helpful assistant.")
                    .users(
                        UserMessageBuilder::new()
                            .text("描述这张图片")
                            .image_base64(mime_type, base64_data),
                    )
                    .build(),
            )
            .build()?;

        let results = pipeline.generate(&request).await?;

        info!("{:?}", results);
        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_simple_vision_for_image_url() -> anyhow::Result<()> {
        init_logger();

        let model_path =
            "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.Q6_K.gguf"
                .to_string();
        let mmproj_path =
            "/data/ComfyUI/models/LLM/GGUF/Qwen3-VL-2B-Instruct-abliterated-v1.mmproj-Q8_0.gguf"
                .to_string();

        let pipeline_config =
            PipelineConfig::new_with_mmproj(model_path, mmproj_path).with_verbose(true);

        let pipeline = Pipeline::try_new(pipeline_config)?;

        // 反推图片
        let request = CreateChatCompletionRequestArgs::default()
            .max_completion_tokens(2048u32)
            .model("Qwen3-VL-2B-Instruct")
            .messages(
                ChatMessagesBuilder::new()
                    .system("You are a helpful assistant.") 
                    .users(
                        UserMessageBuilder::new()
                            .text("描述这张图片")
                            .image_url("https://muse-ai.oss-cn-hangzhou.aliyuncs.com/img/ffdebd6731594c7fbef751944dddf1c0.jpeg"),
                    )
                    .build()
            )
            .build()?;

        let results = pipeline.generate(&request).await?;

        info!("{:?}", results);
        Ok(())
    }

    /// Hook 使用示例测试
    ///
    /// 演示如何使用标准 hooks 构建 pipeline（仅验证 hooks 注册和排序，不执行推理）
    #[test]
    fn test_pipeline_with_standard_hooks() -> anyhow::Result<()> {
        use crate::hooks::builtin::{
            AssembleMessagesHook, LoadHistoryHook, NormalizeHook, SaveHistoryHook,
            SystemPromptHook, ValidateHook,
        };

        let pipeline_config = PipelineConfig::new("/path/to/model.gguf".to_string());
        let pipeline = Pipeline::try_new(pipeline_config)?;

        // 构建带标准 hooks 的 pipeline
        let pipeline = pipeline
            // 1. 参数验证（优先级 10）
            .with_hook(
                ValidateHook::new()
                    .with_max_tokens(2048)
                    .with_allow_empty_messages(false),
            )
            // 2. 消息标准化（优先级 10，与 ValidateHook 同级，按添加顺序执行）
            .with_hook(NormalizeHook::new().with_trim(true).with_remove_empty(true))
            // 3. 系统提示词处理（优先级 20）
            .with_hook(
                SystemPromptHook::keep_first().with_default_system("You are a helpful assistant."),
            )
            // 4. 加载历史消息（优先级 30）
            .with_hook(LoadHistoryHook::new().with_max_history(100))
            // 5. 组装最终消息（优先级 40）
            .with_hook(AssembleMessagesHook::new())
            // 6. 保存历史（优先级 60）
            .with_hook(SaveHistoryHook::new())
            // 7. 工具处理（优先级 50）
            .with_hook(ToolsHook::new())
            // 8. 错误日志（优先级 70）
            .with_hook(ErrorLogHook::new());

        // 验证 hooks 已注册
        assert_eq!(pipeline.hooks.len(), 8);

        // 验证排序后的优先级顺序
        let sorted = pipeline.sorted_hooks();
        assert_eq!(sorted[0].priority(), 10); // ValidateHook
        assert_eq!(sorted[1].priority(), 10); // NormalizeHook
        assert_eq!(sorted[2].priority(), 20); // SystemPromptHook
        assert_eq!(sorted[3].priority(), 30); // LoadHistoryHook
        assert_eq!(sorted[4].priority(), 40); // CurrentInputHook
        assert_eq!(sorted[5].priority(), 60); // HistoryHook

        Ok(())
    }

    /// 自定义优先级示例测试
    ///
    /// 演示如何调整 hooks 的执行顺序
    #[test]
    fn test_pipeline_with_custom_priorities() -> anyhow::Result<()> {
        use crate::hooks::builtin::{LoadHistoryHook, SaveHistoryHook, SystemPromptHook};

        let pipeline_config = PipelineConfig::new("/path/to/model.gguf".to_string());
        let pipeline = Pipeline::try_new(pipeline_config)?;

        // 自定义优先级：先保存历史，再加载历史（特殊场景）
        let pipeline = pipeline
            // 系统提示词提前到第一位
            .with_hook(
                SystemPromptHook::keep_first().with_priority(5), // 默认是 20，现在提前到 5
            )
            // 历史保存提前执行
            .with_hook(
                SaveHistoryHook::new().with_priority(15), // 默认是 60，现在提前到 15
            )
            // 历史加载延后执行
            .with_hook(
                LoadHistoryHook::new()
                    .with_priority(25) // 默认是 30，稍微延后
                    .with_max_history(50),
            );

        // 验证排序后的优先级顺序（从小到大）
        let sorted = pipeline.sorted_hooks();
        assert_eq!(sorted[0].priority(), 5); // SystemPromptHook (自定义)
        assert_eq!(sorted[1].priority(), 15); // HistoryHook (自定义)
        assert_eq!(sorted[2].priority(), 25); // LoadHistoryHook (自定义)

        Ok(())
    }
}
