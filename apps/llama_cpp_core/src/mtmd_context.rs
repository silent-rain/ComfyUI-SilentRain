//! llama.cpp mtmd context
//!
//!
//! pub fn chat_template(&self, name: Option<&str>) -> Result<LlamaChatTemplate, ChatTemplateError>
//!     - https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
//!     - https://github.com/ggml-org/llama.cpp/blob/master/src/llama-chat.cpp#L28
//!
//!

use std::{ffi::CString, io::Write, sync::Arc, time::Duration};

use llama_cpp_2::{
    ggml_time_us,
    llama_batch::LlamaBatch,
    model::{LlamaChatMessage, LlamaModel, Special},
    mtmd::{
        MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdContextParams, MtmdInputText,
        mtmd_default_marker,
    },
    sampling::LlamaSampler,
};
use log::info;

use crate::{
    context::{ContexParams, ContextWrapper},
    error::Error,
    types::{FinishReason, GenerationOutput},
};

/// State of the MTMD CLI application.
#[allow(missing_debug_implementations)]
pub struct MtmdContextWrapper {
    model: Arc<LlamaModel>,
    /// Context parameters
    contex_params: ContexParams,
    /// Base context
    base_context: ContextWrapper,
    /// The MTMD context for multimodal processing.
    pub mtmd_context: MtmdContext,
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
        model: Arc<LlamaModel>,
        base_context: ContextWrapper,
        contex_params: &ContexParams,
    ) -> Result<Self, Error> {
        // Initialize MTMD context
        let mtmd_context = Self::init_mtmd_context(model.clone(), contex_params)?;

        // Create batch for token processing
        let batch = LlamaBatch::new(contex_params.n_ctx as usize, 1);

        Ok(Self {
            model,
            contex_params: contex_params.clone(),
            base_context,
            mtmd_context,
            batch,
            bitmaps: Vec::new(),
            n_past: 0,
        })
    }

    /// Initialize MTMD context
    pub fn init_mtmd_context(
        model: Arc<LlamaModel>,
        contex_params: &ContexParams,
    ) -> Result<MtmdContext, Error> {
        let use_gpu = !contex_params.disable_gpu;

        // Create media marker CString
        let media_marker = CString::new(
            contex_params
                .media_marker
                .as_ref()
                .unwrap_or(&mtmd_default_marker().to_string())
                .clone(),
        )
        .map_err(|e| Error::InvalidInput {
            field: "media_marker".into(),
            message: e.to_string(),
        })?;

        // 确保 n_threads 有合理的值（mmproj 编码需要足够线程）
        let n_threads = if contex_params.n_threads > 0 {
            contex_params.n_threads
        } else {
            std::thread::available_parallelism()
                .map(|p| p.get() as i32)
                .unwrap_or(4)
        };

        let mtmd_params = MtmdContextParams {
            use_gpu,
            print_timings: contex_params.verbose,
            n_threads,
            media_marker,
        };
        let mtmd_context =
            MtmdContext::init_from_file(&contex_params.mmproj_path, &model, &mtmd_params)?;

        Ok(mtmd_context)
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

        let chat_template = ContextWrapper::chat_template(self.model.clone(), &self.contex_params)?;

        let (formatted_chat_template, tokens) =
            ContextWrapper::apply_chat_template(self.model.clone(), &chat_template, &msgs)?;

        ContextWrapper::validate_tokens_size(
            tokens.len(),
            self.contex_params.n_ctx,
            self.contex_params.max_predict(),
        )?;

        self.rest_batch(self.contex_params.n_batch as usize)?;

        self.process_multimodal_input(formatted_chat_template)?;

        Ok(())
    }

    /// Generates a response by sampling tokens from the model
    pub fn generate_response(
        &mut self,
        sampler: &mut LlamaSampler,
    ) -> Result<GenerationOutput, Error> {
        // The `Decoder`
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let max_tokens = self.contex_params.max_predict();
        let t_main_start = ggml_time_us();

        let mut n_cur = self.batch.n_tokens();
        let mut tokens_generated = 0;
        let mut results = String::new();

        while n_cur < max_tokens {
            // Sample next token
            let token = sampler.sample(&self.base_context.context, -1);
            sampler.accept(token);

            // Check for end of generation
            if self.model.is_eog_token(token) {
                println!();
                break;
            }

            // Print token
            // let output_string = self.model.token_to_str(token, Special::Tokenize)?;
            // let output_string = self
            //     .model
            //     .tokens_to_str(&[token], Special::Tokenize)
            //     .unwrap_or("口".to_string());

            // let output_string = self.model.token_to_str(token, Special::Tokenize)?;

            let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
            let mut output_string = String::with_capacity(32);
            let _ = decoder.decode_to_string(&output_bytes, &mut output_string, false);
            results.push_str(&output_string);

            if self.contex_params.verbose {
                // 打印解码后的字符串，不换行
                print!("{output_string}");
                // 刷新标准输出，确保文本立即显示
                std::io::stdout().flush()?;
            }

            // Prepare next batch
            self.batch.clear();
            self.batch.add(token, self.n_past, &[0], true)?;
            self.n_past += 1;

            // Decode
            self.base_context.decode(&mut self.batch)?;

            n_cur += 1;
            tokens_generated += 1;
        }

        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
        info!(
            "Generated {} tokens in {:.2}s ({:.2} t/s)",
            tokens_generated,
            duration.as_secs_f32(),
            tokens_generated as f32 / duration.as_secs_f32().max(0.001)
        );

        Ok(GenerationOutput {
            text: results,
            tokens_generated,
            finish_reason: if n_cur >= max_tokens {
                FinishReason::Length
            } else {
                FinishReason::Stop
            },
        })
    }
}

impl MtmdContextWrapper {
    pub fn rest_batch(&mut self, n_batch: usize) -> Result<(), Error> {
        let batch = LlamaBatch::new(n_batch, 1);
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
        Backend, HistoryMessage, Model, Sampler, sampler::SamplerConfig, types::MediaData,
    };

    use super::*;

    #[test]
    #[ignore]
    fn test_simple() -> anyhow::Result<()> {
        // Initialize backend
        let backend = Backend::init_backend()?;

        // Load model
        let model_path = String::new();
        let model = Model::new(model_path).load_cache_model(&backend)?;

        // Load sampler
        let sampler_config = SamplerConfig::default();
        let mut sampler = Sampler::load_sampler(&sampler_config)?;

        // 上下文
        let contex_params = ContexParams::default();
        let ctx = ContextWrapper::try_new(model.clone(), &backend, &contex_params)?;
        let mut mtmd_ctx = MtmdContextWrapper::try_new(model.clone(), ctx, &contex_params)?;

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
        let msgs = ContextWrapper::create_message(
            &contex_params,
            true,
            medias,
            &history_message.messages(),
        )?;

        // 评估消息
        mtmd_ctx.eval_messages(msgs)?;

        // 生成响应
        let results = mtmd_ctx.generate_response(&mut sampler)?;

        // 将上下文信息添加到历史消息中
        {
            history_message.add_system(contex_params.system_prompt)?; // 仅添加系统提示

            // 每轮聊天都添加用户提示和助手响应
            history_message.add_user(contex_params.user_prompt)?;
            history_message.add_assistant(results.text.clone())?;
        }

        println!("{results:?}");
        Ok(())
    }
}
