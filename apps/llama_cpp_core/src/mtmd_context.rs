//! llama.cpp mtmd context
//!
//!
//! pub fn chat_template(&self, name: Option<&str>) -> Result<LlamaChatTemplate, ChatTemplateError>
//!     - https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
//!     - https://github.com/ggml-org/llama.cpp/blob/master/src/llama-chat.cpp#L28
//!
//!

use std::{io::Write, sync::Arc, time::Duration};

use tokio::sync::mpsc;

use llama_cpp_2::{
    ggml_time_us,
    llama_batch::LlamaBatch,
    model::{LlamaChatMessage, LlamaModel},
    mtmd::{MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdInputText},
    sampling::LlamaSampler,
};
use log::{error, info};

use crate::{
    context::{ContexParams, ContextWrapper},
    error::Error,
    pipeline::StreamResponseBuilder,
    types::FinishReason,
};
use async_openai::types::chat::CreateChatCompletionStreamResponse;

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

    /// 流式生成响应，通过 channel 返回标准 OpenAI 格式的响应
    pub fn generate_response(
        &mut self,
        sampler: &mut LlamaSampler,
    ) -> mpsc::UnboundedReceiver<CreateChatCompletionStreamResponse> {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let max_tokens = self.contex_params.max_predict();
        let verbose = self.contex_params.verbose;
        let t_main_start = ggml_time_us();

        // 记录 prompt tokens（已经处理的输入 token 数）
        let prompt_tokens = self.batch.n_tokens() as u32;
        let mut n_past = self.n_past;
        let mut n_cur = self.batch.n_tokens();
        let mut tokens_generated: u32 = 0;
        let model_name = "llama-model".to_string(); // 可以从配置中获取

        // 生成唯一ID
        let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let builder = StreamResponseBuilder::new(&id, &model_name);

        while n_cur < max_tokens {
            // 采样下一个 token
            let token = sampler.sample(&self.base_context.context, -1);
            sampler.accept(token);

            // 检查是否为结束标记
            if self.llama_model.is_eog_token(token) {
                {
                    // 记录结束时间并计算持续时间
                    let t_main_end = ggml_time_us();
                    let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
                    info!(
                        "Generated {} tokens in {:.2}s ({:.2} t/s)",
                        tokens_generated,
                        duration.as_secs_f32(),
                        tokens_generated as f32 / duration.as_secs_f32().max(0.001)
                    );

                    info!("{}", self.base_context.timings());
                }
                // 发送带 usage 统计的结束响应
                let response = builder.build_finish_with_usage(
                    FinishReason::Stop,
                    prompt_tokens,
                    tokens_generated,
                );
                let _ = tx.send(response);
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
                    // 发送错误响应给调用者
                    let error_response = builder.build_error(format!("Token decode failed: {}", e));
                    let _ = tx.send(error_response);
                    break;
                }
            };

            if verbose {
                let _ = std::io::stdout().write_all(piece.as_bytes());
                let _ = std::io::stdout().flush();
            }

            // 发送标准的 OpenAI 格式响应
            let response = builder.build_content(piece);
            if tx.send(response).is_err() {
                break;
            }

            // 准备下一个 batch
            self.batch.clear();
            if let Err(e) = self.batch.add(token, n_past, &[0], true) {
                error!("Batch add error: {}", e);
                let error_response = builder.build_error(format!("Batch add failed: {}", e));
                let _ = tx.send(error_response);
                break;
            }

            // 解码
            if let Err(e) = self.base_context.decode(&mut self.batch) {
                error!("Decode error: {}", e);
                let error_response = builder.build_error(format!("Decode failed: {}", e));
                let _ = tx.send(error_response);
                break;
            }

            n_past += 1;
            n_cur += 1;
            tokens_generated += 1;

            // 检查是否达到最大 token 数
            if n_cur >= max_tokens {
                let response = builder.build_finish_with_usage(
                    FinishReason::Length,
                    prompt_tokens,
                    tokens_generated,
                );
                let _ = tx.send(response);
                break;
            }
        }

        rx
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
        Backend, HistoryMessage, Model, Sampler,
        sampler::SamplerConfig,
        types::{MediaData, PromptMessageRole},
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
        let llama_model = model.load_cache_llama_model(&backend)?;
        let mtmd_context = model.load_cache_mtmd_context(llama_model.clone())?;

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
        let medias = vec![MediaData::new_image("".as_bytes().to_vec())];
        for media in &medias {
            info!("Loading media: {}", media.media_type);
            mtmd_ctx.load_media_buffer(&media.data)?;
        }

        // 创建历史消息
        let mut history_message = HistoryMessage::new();

        // 创建消息
        let system_prompt = "You are a helpful assistant".to_string();
        let user_prompt = "Hello, how are you?".to_string();
        let msgs = vec![
            LlamaChatMessage::new(PromptMessageRole::System.to_string(), system_prompt.clone())?,
            LlamaChatMessage::new(PromptMessageRole::User.to_string(), user_prompt.clone())?,
        ];

        // 评估消息
        mtmd_ctx.eval_messages(msgs)?;

        // 生成响应
        let mut rx = mtmd_ctx.generate_response(&mut sampler);

        // 接收响应
        let mut full_text = String::new();
        while let Some(response) = rx.recv().await {
            for choice in &response.choices {
                if let Some(content) = &choice.delta.content {
                    full_text.push_str(content);
                }
            }
        }

        // 将上下文信息添加到历史消息中
        {
            history_message.add_system(system_prompt)?; // 仅添加系统提示

            // 每轮聊天都添加用户提示和助手响应
            history_message.add_user(user_prompt)?;
            history_message.add_assistant(full_text.clone())?;
        }

        println!("{full_text:?}");
        Ok(())
    }
}
