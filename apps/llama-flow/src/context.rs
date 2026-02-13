//! llama.cpp context
//!
//!
//! pub fn chat_template(&self, name: Option<&str>) -> Result<LlamaChatTemplate, ChatTemplateError>
//!     - https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
//!     - https://github.com/ggml-org/llama.cpp/blob/master/src/llama-chat.cpp#L28
//!

use std::{io::Write, num::NonZero, sync::Arc, time::Duration};

use llama_cpp_2::{
    context::{LlamaContext, params::LlamaContextParams},
    ggml_time_us,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel},
    mtmd::mtmd_default_marker,
    sampling::LlamaSampler,
    timing::LlamaTimings,
    token::LlamaToken,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{self, UnboundedReceiver};
use tracing::{error, info};

use crate::{error::Error, response::ChatStreamBuilder, types::PoolingTypeMode};

use async_openai::types::chat::CreateChatCompletionStreamResponse;

/// Context Params
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContexParams {
    /// Number of threads to use during generation.
    /// Set to a specific value to limit CPU usage.
    #[serde(default)]
    pub n_threads: i32,

    /// Batch size for prompt processing.
    /// Larger values may improve throughput but increase memory usage.
    #[serde(default)]
    pub n_batch: u32,

    /// Number of tokens to predict (0 for unlimited)
    #[serde(default)]
    pub n_predict: i32,

    /// Size of the prompt context window.
    /// Defines the maximum context length the model can handle.
    #[serde(default)]
    pub n_ctx: u32,

    /// Pooling type for embeddings.
    /// Options: "None", "Mean", "Cls", "Last", "Rank", "Unspecified".
    #[serde(default)]
    pub pooling_type: PoolingTypeMode,

    /// 媒体占位符标记
    #[serde(default)]
    pub media_marker: String,

    /// Chat template to use, default template if not provided
    #[serde(default)]
    pub chat_template: Option<String>,

    /// 图片最大分辨率限制
    #[serde(default)]
    pub image_max_resolution: u32,

    /// 最大历史消息数
    #[serde(default = "default_max_history")]
    pub max_history: usize,

    /// 是否存储此聊天
    #[serde(default)]
    pub store: bool,

    /// Enables verbose logging from llama.cpp.
    #[serde(default)]
    pub verbose: bool,
}

fn default_max_history() -> usize {
    100
}

impl Default for ContexParams {
    fn default() -> Self {
        Self {
            // 线程和批处理参数
            n_threads: 0,
            n_batch: 512,
            n_ctx: 4096,
            n_predict: 2048,

            pooling_type: PoolingTypeMode::Unspecified,

            media_marker: mtmd_default_marker().to_string(),
            chat_template: None,

            image_max_resolution: 768,
            max_history: 100,

            store: false,
            verbose: false,
        }
    }
}

// 生成便捷方法
impl ContexParams {
    /// 设置线程数
    pub fn with_n_threads(mut self, n_threads: i32) -> Self {
        self.n_threads = n_threads;
        self
    }

    /// 设置批处理大小
    pub fn with_n_batch(mut self, n_batch: u32) -> Self {
        self.n_batch = n_batch;
        self
    }

    /// 设置最大完成令牌数
    pub fn with_max_completion_tokens(mut self, max_completion_tokens: i32) -> Self {
        self.n_predict = max_completion_tokens;
        self
    }

    /// 设置上下文窗口大小
    pub fn with_n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    /// 设置池化类型
    pub fn with_pooling_type(mut self, pooling_type: PoolingTypeMode) -> Self {
        self.pooling_type = pooling_type;
        self
    }

    /// 设置媒体标记
    pub fn with_media_marker(mut self, marker: impl Into<String>) -> Self {
        self.media_marker = marker.into();
        self
    }

    /// 设置聊天模板
    pub fn with_chat_template(mut self, chat_template: String) -> Self {
        self.chat_template = Some(chat_template);
        self
    }

    /// 设置图像最大分辨率
    pub fn with_image_max_resolution(mut self, max_resolution: u32) -> Self {
        self.image_max_resolution = max_resolution;
        self
    }

    /// 设置是否打印日志
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

impl ContexParams {
    pub fn max_predict(&self) -> i32 {
        if self.n_predict <= 0 {
            i32::MAX
        } else {
            self.n_predict
        }
    }
}

/// Llama Cpp Context
#[derive(Debug)]
pub struct ContextWrapper {
    llama_model: Arc<LlamaModel>,
    /// Context parameters
    contex_params: ContexParams,
    /// The context for multimodal processing.
    pub context: LlamaContext<'static>,
    /// The batch used for processing tokens.
    pub batch: LlamaBatch<'static>,
}

unsafe impl Send for ContextWrapper {}
unsafe impl Sync for ContextWrapper {}

impl ContextWrapper {
    /// Creates a new context
    pub fn try_new(
        llama_model: Arc<LlamaModel>,
        backend: &LlamaBackend,
        contex_params: &ContexParams,
    ) -> Result<Self, Error> {
        let batch = LlamaBatch::new(contex_params.max_predict() as usize, 1);

        let context = Self::load_context(llama_model.clone(), backend, contex_params)?;

        Ok(Self {
            llama_model,
            contex_params: contex_params.clone(),
            context,
            batch,
        })
    }

    /// Setup context parameters
    pub fn load_context(
        model: Arc<LlamaModel>,
        backend: &LlamaBackend,
        contex_params: &ContexParams,
    ) -> Result<LlamaContext<'static>, Error> {
        let context_params = LlamaContextParams::default()
            .with_n_threads(contex_params.n_threads)
            .with_n_batch(contex_params.n_batch)
            // .with_n_ubatch(contex_params.n_ubatch)
            .with_n_ctx(NonZero::new(contex_params.n_ctx))
            // .with_embeddings(true)
            .with_pooling_type(contex_params.pooling_type.into());

        // Create context
        let context = model.new_context(backend, context_params)?;

        // 确保 context 的生命周期不依赖于 model
        let context = unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'_>>(context) };

        Ok(context)
    }

    /// Evaluates a chat message, tokenizing and processing it through the model
    pub fn eval_messages(&mut self, msgs: Vec<LlamaChatMessage>) -> Result<(), Error> {
        info!("eval messages ...");
        let chat_template =
            ContextWrapper::chat_template(self.llama_model.clone(), &self.contex_params)?;

        let (_formatted_chat_template, tokens) =
            ContextWrapper::apply_chat_template(self.llama_model.clone(), &chat_template, &msgs)?;

        ContextWrapper::validate_tokens_size(
            tokens.len(),
            self.contex_params.n_ctx,
            self.contex_params.max_predict(),
        )?;

        self.rest_batch(self.contex_params.max_predict() as usize)?;

        self.process_user_tokens(tokens)?;

        Ok(())
    }

    /// 流式生成响应，返回 async_openai 标准流式结构
    ///
    /// 返回 CreateChatCompletionStreamResponse 的流，与 OpenAI Chat Completions API 兼容
    pub fn generate_response(
        &mut self,
        sampler: &mut LlamaSampler,
        model_id: impl Into<String>,
    ) -> Result<UnboundedReceiver<CreateChatCompletionStreamResponse>, Error> {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let max_completion_tokens = self.contex_params.max_predict();
        let eos_token = self.llama_model.token_eos();
        let verbose = self.contex_params.verbose;
        let t_main_start = ggml_time_us();

        // 记录 prompt tokens（已经处理的输入 token 数）
        let prompt_tokens = self.batch.n_tokens() as u32;
        let mut n_cur = self.batch.n_tokens();

        // 创建流式响应构建器
        let mut builder = ChatStreamBuilder::new(model_id).with_prompt_tokens(prompt_tokens);

        // 循环生成 token，直到达到最大长度 max_completion_tokens
        while n_cur < max_completion_tokens {
            // 采样下一个 token
            let token = sampler.sample(&self.context, self.batch.n_tokens() - 1);
            sampler.accept(token);

            // 检查是否为结束标记
            if token == eos_token {
                // 记录结束时间并计算持续时间
                let t_main_end = ggml_time_us();
                let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
                let tokens_generated = builder.tokens_generated();
                info!(
                    "Generated {} tokens in {:.2}s ({:.2} t/s)",
                    tokens_generated,
                    duration.as_secs_f32(),
                    tokens_generated as f32 / duration.as_secs_f32().max(0.001)
                );
                info!("{}", self.context.timings());

                // 发送停止标记的 chunk
                if let Err(e) = tx.send(builder.build_stop_chunk()) {
                    error!("Failed to send stop chunk: {}", e);
                    break;
                }
                // 发送包含 usage 的最终 chunk
                if let Err(e) = tx.send(builder.build_final_chunk_with_usage()) {
                    error!("Failed to send final usage chunk: {}", e);
                }
                break;
            }

            // 将 token 转换为字符串
            let piece = match self
                .llama_model
                .token_to_piece(token, &mut decoder, true, None)
            {
                Ok(bytes) => bytes,
                Err(e) => {
                    error!("Token decode error: {}", e);
                    break;
                }
            };

            if verbose {
                let _ = std::io::stdout().write_all(piece.as_bytes());
                let _ = std::io::stdout().flush();
            }

            // 使用构建器创建并发送流式响应 chunk
            let chunk = builder.build_content_chunk(piece);
            if tx.send(chunk).is_err() {
                error!("Failed to send chunk");
                break;
            }

            // 准备下一个 batch
            self.batch.clear();
            if let Err(e) = self.batch.add(token, n_cur, &[0], true) {
                error!("Batch add error: {}", e);
                break;
            }

            // 解码
            if let Err(e) = self.context.decode(&mut self.batch) {
                error!("Decode error: {}", e);
                break;
            }

            n_cur += 1;

            // 检查是否达到最大 token 数
            if n_cur >= max_completion_tokens {
                // 发送长度限制停止 chunk
                if let Err(e) = tx.send(builder.build_length_chunk()) {
                    error!("Failed to send length chunk: {}", e);
                    break;
                }
                // 发送包含 usage 的最终 chunk
                if let Err(e) = tx.send(builder.build_final_chunk_with_usage()) {
                    error!("Failed to send final usage chunk: {}", e);
                }
                break;
            }
        }

        Ok(rx)
    }
}

impl ContextWrapper {
    /// Decodes the batch.
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), Error> {
        self.context.decode(batch).map_err(|e| {
            error!("llama_decode failed: {}", e);
            e
        })?;

        Ok(())
    }

    pub fn rest_batch(&mut self, n_tokens: usize) -> Result<(), Error> {
        let batch = LlamaBatch::new(n_tokens, 1);
        self.batch = batch;
        Ok(())
    }

    /// Clear the KV cache
    pub fn clear_kv_cache(&mut self) {
        self.context.clear_kv_cache();
    }
    pub fn timings(&mut self) -> LlamaTimings {
        self.context.timings()
    }
}

impl ContextWrapper {
    /// Process user tokens
    fn process_user_tokens(&mut self, tokens: Vec<LlamaToken>) -> Result<(), Error> {
        // 计算最后一个索引
        let last_index = tokens.len().saturating_sub(1);

        // 遍历所有 token，i 是索引，token 是实际的 token 值
        for (i, token) in tokens.into_iter().enumerate() {
            // llama_decode 只会为提示的最后一个 token 输出 logits（概率分布）
            // 判断当前 token 是否是最后一个
            let is_last = i == last_index;

            // 将 token 添加到批处理中
            // 参数：token值, 位置索引, 序列ID数组, 是否是最后一个token
            self.batch.add(token, i as i32, &[0], is_last)?;
        }

        // 使用上下文解码批处理中的 token
        self.context.decode(&mut self.batch).map_err(|e| {
            error!("llama_decode failed: {}", e);
            e
        })?;

        Ok(())
    }
}

impl ContextWrapper {
    /// 模板处理
    pub fn chat_template(
        model: Arc<LlamaModel>,
        contex_params: &ContexParams,
    ) -> Result<LlamaChatTemplate, Error> {
        info!("prepare chat template ...");

        // 通过预定义模板进行格式化, 当前模板读取存在问题
        // let chat_template = model
        //     .chat_template(params.chat_template.as_deref())
        //     .map_err(|e| {
        //         error!("Failed to get chat template: {e}");
        //         e
        //     })?;

        // 优先使用自定义模板
        if let Some(ref chat_template) = contex_params.chat_template
            && !chat_template.is_empty()
        {
            return LlamaChatTemplate::new(chat_template).map_err(|e| Error::InvalidInput {
                field: "chat_template".into(),
                message: e.to_string(),
            });
        }

        // 使用模型内置模板
        model.chat_template(None).map_err(|e| Error::InvalidInput {
            field: "chat_template".into(),
            message: e.to_string(),
        })
    }

    /// Apply chat template
    pub fn apply_chat_template(
        model: Arc<LlamaModel>,
        chat_template: &LlamaChatTemplate,
        msgs: &[LlamaChatMessage],
    ) -> Result<(String, Vec<LlamaToken>), Error> {
        info!("apply chat template...");

        // Format the message using chat template (simplified)
        let formatted_template = model
            .apply_chat_template(chat_template, msgs, true)
            .map_err(|e| {
                error!("Failed to apply chat template: {e}");
                e
            })?;

        // 将提示文本转换为 token 列表，AddBos::Always 表示始终在开头添加 BOS (Beginning of Sequence) token
        let tokens = model
            .str_to_token(&formatted_template, AddBos::Always)
            .map_err(|e| {
                error!("failed to tokenize {:?}, err: {}", formatted_template, e);
                e
            })?;

        // let tokens_list = self
        //     .context
        //     .model
        //     .str_to_token(&params.user_prompt, AddBos::Always)?;

        info!("Formatted template: {}", formatted_template);
        Ok((formatted_template, tokens))
    }

    /// Verify tokens size
    pub fn validate_tokens_size(
        tokens_size: usize,
        n_ctx: u32,
        n_predict: i32,
    ) -> Result<(), Error> {
        info!("validate tokens size ...");
        // 验证 token 数量
        if n_predict < 0 {
            if tokens_size > n_ctx as usize {
                return Err(Error::InvalidParameter(
                    "the prompt is too long, it has more tokens than n_ctx".to_string(),
                ));
            }

            return Ok(());
        }

        let n_kv_req = tokens_size as i32 + (n_predict - tokens_size as i32);
        info!("n_predict = {n_predict}, n_ctx = {n_ctx}, k_kv_req = {n_kv_req}");

        if n_kv_req > n_ctx as i32 {
            return Err(Error::InvalidParameter(
                "n_kv_req > n_ctx, the required kv cache size is not big enough either reduce n_predict or increase n_ctx"
                    .to_string(),
            ));
        }
        if tokens_size >= n_predict as usize {
            return Err(Error::InvalidParameter(
                "the prompt is too long, it has more tokens than n_predict".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Backend, Model, Sampler, chat_history, sampler::SamplerConfig, types::MessageRole,
    };

    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_simple() -> anyhow::Result<()> {
        // Initialize backend
        let backend = Backend::init_backend()?;

        // Load model
        let model_id = "7b-chat".to_string();
        let model_path = String::new();
        let llama_model = Model::new(model_path, None).load_llama_model(&backend)?;

        // Load sampler
        let sampler_config = SamplerConfig::default();
        let mut sampler = Sampler::load_sampler(&sampler_config)?;

        // 上下文
        let contex_params = ContexParams::default();
        let mut ctx = ContextWrapper::try_new(llama_model.clone(), &backend, &contex_params)?;

        // 获取历史管理器
        let history = chat_history();
        let session_id = "test_session";

        // 创建消息
        let system_prompt = "You are a helpful assistant".to_string();
        let user_prompt = "Hello, how are you?".to_string();
        let msgs = vec![
            LlamaChatMessage::new(MessageRole::System.to_string(), system_prompt.clone())?,
            LlamaChatMessage::new(MessageRole::User.to_string(), user_prompt.clone())?,
        ];

        // 评估消息
        ctx.eval_messages(msgs)?;

        // 生成响应
        let mut rx = ctx.generate_response(&mut sampler, model_id)?;

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

        // 将上下文信息添加到历史消息中
        history.add_system_text(session_id, system_prompt)?;
        history.add_user_text(session_id, user_prompt)?;
        history.add_assistant_text(session_id, full_text.clone())?;

        println!("{full_text:?}");
        Ok(())
    }
}
