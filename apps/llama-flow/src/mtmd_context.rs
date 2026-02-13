//! llama.cpp mtmd context
//!
//!
//! pub fn chat_template(&self, name: Option<&str>) -> Result<LlamaChatTemplate, ChatTemplateError>
//!     - https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
//!     - https://github.com/ggml-org/llama.cpp/blob/master/src/llama-chat.cpp#L28
//!
//!

use std::{io::Write, sync::Arc, time::Duration};

use async_openai::types::chat::CreateChatCompletionStreamResponse;
use llama_cpp_2::{
    ggml_time_us,
    llama_batch::LlamaBatch,
    model::{LlamaChatMessage, LlamaModel},
    mtmd::{MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdInputText},
    sampling::LlamaSampler,
};
use log::{error, info};
use tokio::sync::mpsc::{self, UnboundedReceiver};

use crate::{
    context::{ContexParams, ContextWrapper},
    error::Error,
    response::ChatStreamBuilder,
};

/// State of the MTMD application.
pub struct MtmdContextWrapper {
    llama_model: Arc<LlamaModel>,
    /// Context parameters
    contex_params: ContexParams,
    /// Base context
    pub base_context: ContextWrapper,
    /// The MTMD context for multimodal processing.
    pub mtmd_context: Arc<MtmdContext>,
    /// The batch used for processing tokens.
    pub batch: LlamaBatch<'static>,
    /// The list of loaded bitmaps (images/audio).
    pub bitmaps: Vec<MtmdBitmap>,
    /// The number of past tokens processed.
    pub n_past: i32,
}

unsafe impl Send for MtmdContextWrapper {}
unsafe impl Sync for MtmdContextWrapper {}

impl MtmdContextWrapper {
    /// Creates a new MTMD context
    pub fn try_new(
        llama_model: Arc<LlamaModel>,
        base_context: ContextWrapper,
        mtmd_context: Arc<MtmdContext>,
        contex_params: &ContexParams,
    ) -> Result<Self, Error> {
        // Create batch for token processing
        let batch = LlamaBatch::new(contex_params.n_predict as usize, 1);

        Ok(Self {
            llama_model,
            contex_params: contex_params.clone(),
            base_context,
            mtmd_context,
            batch,
            bitmaps: Vec::new(),
            n_past: 0,
        })
    }

    /// Loads media (image or audio) from the specified file path
    pub fn load_media_file(&mut self, path: &str) -> Result<(), MtmdBitmapError> {
        let bitmap = MtmdBitmap::from_file(&self.mtmd_context, path)?;
        self.bitmaps.push(bitmap);
        Ok(())
    }

    pub fn load_media_buffer(&mut self, image: &[u8]) -> Result<(), MtmdBitmapError> {
        let bitmap = MtmdBitmap::from_buffer(&self.mtmd_context, image)?;
        self.bitmaps.push(bitmap);
        Ok(())
    }

    /// Evaluates a chat message, tokenizing and processing it through the model
    pub fn eval_messages(&mut self, msgs: Vec<LlamaChatMessage>) -> Result<(), Error> {
        info!("eval messages ...");

        let chat_template =
            ContextWrapper::chat_template(self.llama_model.clone(), &self.contex_params)?;

        let (formatted_chat_template, tokens) =
            ContextWrapper::apply_chat_template(self.llama_model.clone(), &chat_template, &msgs)?;

        ContextWrapper::validate_tokens_size(
            tokens.len(),
            self.contex_params.n_ctx,
            self.contex_params.max_predict(),
        )?;

        self.rest_batch(self.contex_params.n_predict as usize)?;

        self.process_multimodal_input(formatted_chat_template)?;

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
        let verbose = self.contex_params.verbose;
        let t_main_start = ggml_time_us();

        // 记录 prompt tokens（已经处理的输入 token 数）
        let prompt_tokens = self.batch.n_tokens() as u32;
        let mut n_past = self.n_past;
        let mut n_cur = self.batch.n_tokens();

        // 创建流式响应构建器
        let mut builder = ChatStreamBuilder::new(model_id).with_prompt_tokens(prompt_tokens);

        // 循环生成 token，直到达到最大长度 max_completion_tokens
        while n_cur < max_completion_tokens {
            // 采样下一个 token
            let token = sampler.sample(&self.base_context.context, -1);
            sampler.accept(token);

            // 检查是否为结束标记
            if self.llama_model.is_eog_token(token) {
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
                info!("{}", self.base_context.timings());

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
            if let Err(e) = tx.send(chunk) {
                error!("Failed to send chunk: {}", e);
                break;
            }

            // 准备下一个 batch
            self.batch.clear();
            if let Err(e) = self.batch.add(token, n_past, &[0], true) {
                error!("Batch add error: {}", e);
                break;
            }

            // 解码
            if let Err(e) = self.base_context.decode(&mut self.batch) {
                error!("Decode error: {}", e);
                break;
            }

            n_past += 1;
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

impl MtmdContextWrapper {
    pub fn rest_batch(&mut self, n_tokens: usize) -> Result<(), Error> {
        let batch = LlamaBatch::new(n_tokens, 1);
        self.batch = batch;
        Ok(())
    }

    /// Clear the KV cache
    pub fn clear_kv_cache(&mut self) {
        self.base_context.clear_kv_cache();
    }
}

impl MtmdContextWrapper {
    /// Processes and evaluates input text with optional media (images/audio) through the model
    ///
    /// # Arguments
    /// * `formatted_chat_template` - Template for formatting chat messages
    fn process_multimodal_input(&mut self, formatted_chat_template: String) -> Result<(), Error> {
        // 准备输入
        let input_text = MtmdInputText {
            text: formatted_chat_template,
            add_special: true,
            parse_special: true,
        };
        info!("MtmdInputText: {input_text:?}");

        // 处理位图
        let bitmap_refs: Vec<&MtmdBitmap> = self.bitmaps.iter().collect();
        if bitmap_refs.is_empty() {
            info!("No bitmaps loaded");
        } else {
            info!("Tokenizing with {} bitmaps", bitmap_refs.len());
        }

        // Tokenize
        let chunks = self.mtmd_context.tokenize(input_text, &bitmap_refs)?;
        info!("Tokenization complete, {} chunks created", chunks.len());

        // 清空 KV cache
        self.bitmaps.clear();
        self.base_context.clear_kv_cache();

        // 评估 chunks
        self.n_past = chunks.eval_chunks(
            &self.mtmd_context,
            &self.base_context.context,
            0,
            0,
            self.contex_params.n_batch as i32,
            true,
        )?;
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
        let model_path = String::new();
        let mmproj_path = String::new();
        let model = Model::new(model_path, mmproj_path);
        let llama_model = model.load_llama_model(&backend)?;
        let mtmd_context = model.load_mtmd_context(llama_model.clone())?;

        // Load sampler
        let sampler_config = SamplerConfig::default();
        let mut sampler = Sampler::load_sampler(&sampler_config)?;

        // 上下文
        let contex_params = ContexParams::default();
        let ctx = ContextWrapper::try_new(llama_model.clone(), &backend, &contex_params)?;
        let mut mtmd_ctx =
            MtmdContextWrapper::try_new(llama_model.clone(), ctx, mtmd_context, &contex_params)?;

        // Load media files
        // for image_path in &params.images {
        //     info!("Loading image: {image_path}");
        //     mtmd_ctx.load_media_file(image_path)?;
        // }
        // for audio_path in &params.audio {
        //     mtmd_ctx.load_media_file(audio_path)?;
        // }

        // Load media files
        let medias = vec!["".as_bytes().to_vec()];
        for media in &medias {
            mtmd_ctx.load_media_buffer(media)?;
        }

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
        mtmd_ctx.eval_messages(msgs)?;

        // 生成响应
        let mut rx = mtmd_ctx.generate_response(&mut sampler, "test-model")?;

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
