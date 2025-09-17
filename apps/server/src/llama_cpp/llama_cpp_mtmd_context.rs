//! llama.cpp mtmd context
//!
//!
//! pub fn chat_template(&self, name: Option<&str>) -> Result<LlamaChatTemplate, ChatTemplateError>
//!     - https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
//!     - https://github.com/ggml-org/llama.cpp/blob/master/src/llama-chat.cpp#L28
//!
//!

use std::{ffi::CString, io::Write, sync::Arc};

use llama_cpp_2::{
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, Special},
    mtmd::{
        mtmd_default_marker, MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdContextParams,
        MtmdInputText,
    },
    sampling::LlamaSampler,
    token::LlamaToken,
};
use log::{error, info};

use crate::{
    error::Error,
    llama_cpp::{LlamaCppBaseContext, LlamaCppOptions, PromptMessageRole},
};

/// State of the MTMD CLI application.
#[allow(missing_debug_implementations)]
pub struct LlamaCppMtmdContext {
    model: Arc<LlamaModel>,
    base_context: LlamaCppBaseContext,
    /// The MTMD context for multimodal processing.
    pub mtmd_context: MtmdContext,
    /// The batch used for processing tokens.
    pub batch: LlamaBatch,
    /// The list of loaded bitmaps (images/audio).
    pub bitmaps: Vec<MtmdBitmap>,
    /// The number of past tokens processed.
    pub n_past: i32,
    /// Enables verbose logging from llama.cpp.
    verbose: bool,
}

impl LlamaCppMtmdContext {
    /// Creates a new MTMD context
    ///
    /// # Errors
    pub fn new(
        model: Arc<LlamaModel>,
        backend: &LlamaBackend,
        params: &LlamaCppOptions,
    ) -> Result<Self, Error> {
        let use_gpu = { params.n_gpu_layers > 0 };

        let base_context = LlamaCppBaseContext::new(model.clone(), backend, params)?;

        // Initialize MTMD context
        let mtmd_params = MtmdContextParams {
            use_gpu,
            print_timings: params.verbose,
            n_threads: params.n_threads,
            media_marker: CString::new(
                params
                    .media_marker
                    .as_ref()
                    .unwrap_or(&llama_cpp_2::mtmd::mtmd_default_marker().to_string())
                    .clone(),
            )?,
        };

        let mtmd_context =
            MtmdContext::init_from_file(&params.get_mmproj_path()?, &model, &mtmd_params)?;
        info!("Loading mtmd projection: {}", params.mmproj_path);

        let batch = LlamaBatch::new(params.n_ctx as usize, 1);

        Ok(Self {
            model,
            base_context,
            mtmd_context,
            batch,
            bitmaps: Vec::new(),
            n_past: 0,
            verbose: params.verbose,
        })
    }

    /// Loads media (image or audio) from the specified file path
    /// # Errors
    pub fn load_media_file(&mut self, path: &str) -> Result<(), MtmdBitmapError> {
        let bitmap = MtmdBitmap::from_file(&self.mtmd_context, path)?;
        self.bitmaps.push(bitmap);
        Ok(())
    }

    pub fn load_data_buffer(&mut self, image: &[u8]) -> Result<(), MtmdBitmapError> {
        let bitmap = MtmdBitmap::from_buffer(&self.mtmd_context, image)?;
        self.bitmaps.push(bitmap);
        Ok(())
    }

    /// Evaluates a chat message, tokenizing and processing it through the model
    /// # Errors
    pub fn eval_message(
        &mut self,
        params: &LlamaCppOptions,
        add_bos: bool,
        context_history: &[LlamaChatMessage],
    ) -> Result<(), Error> {
        let msgs = self.create_message(params, context_history)?;
        let chat_template = self.chat_template(params)?;

        let (formatted_chat_template, token_list) =
            self.apply_chat_template(&msgs, &chat_template)?;

        self.validate_tokens_size(token_list.len(), params.n_ctx, params.n_predict)?;

        self.process_multimodal_input(params, add_bos, formatted_chat_template)?;

        self.rest_batch(params.n_batch as usize)?;

        Ok(())
    }

    /// Generates a response by sampling tokens from the model
    /// # Errors
    pub fn generate_response(
        &mut self,
        sampler: &mut LlamaSampler,
        n_predict: i32,
    ) -> Result<String, Error> {
        // The `Decoder`
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        let mut generated_tokens = Vec::new();
        let max_predict = if n_predict < 0 { i32::MAX } else { n_predict };
        let mut results = String::new();

        for _i in 0..max_predict {
            // Sample next token
            let token = sampler.sample(&self.base_context.context, -1);
            generated_tokens.push(token);
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

            let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
            // use `Decoder.decode_to_string()` to avoid the intermediate buffer
            let mut output_string = String::with_capacity(32);
            let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);

            if self.verbose {
                // 打印解码后的字符串，不换行
                print!("{output_string}");
                // 刷新标准输出，确保文本立即显示
                std::io::stdout().flush()?;
            }

            results.push_str(&output_string);

            // Prepare next batch
            self.batch.clear();
            self.batch.add(token, self.n_past, &[0], true)?;
            self.n_past += 1;

            // Decode
            self.base_context.context.decode(&mut self.batch)?;
        }

        Ok(results)
    }
}

impl LlamaCppMtmdContext {
    pub fn rest_batch(&mut self, n_batch: usize) -> Result<(), Error> {
        let batch = LlamaBatch::new(n_batch, 1);
        self.batch = batch;
        Ok(())
    }
}

impl LlamaCppMtmdContext {
    /// Create message
    fn create_message(
        &self,
        params: &LlamaCppOptions,
        context_history: &[LlamaChatMessage],
    ) -> Result<Vec<LlamaChatMessage>, Error> {
        // Add media marker if not present
        let mut user_prompt = params.user_prompt.clone();
        let default_marker = mtmd_default_marker().to_string();
        let media_marker = params.media_marker.as_ref().unwrap_or(&default_marker);
        if !user_prompt.contains(media_marker) {
            user_prompt.push_str(media_marker);
        }

        // Create message
        let mut msgs = vec![
            LlamaChatMessage::new(
                PromptMessageRole::System.to_string(),
                params.system_prompt.clone(),
            )?,
            LlamaChatMessage::new(PromptMessageRole::User.to_string(), user_prompt.clone())?,
        ];

        // Add history messages
        if params.keep_context {
            msgs.extend(context_history.to_owned());
        }

        Ok(msgs)
    }

    /// 模板处理
    fn chat_template(&self, params: &LlamaCppOptions) -> Result<LlamaChatTemplate, Error> {
        // 通过预定义模板进行格式化, 当前模板读取存在问题
        // let chat_template = model
        //     .chat_template(params.chat_template.as_deref())
        //     .map_err(|e| {
        //         error!("Failed to get chat template: {e}");
        //         e
        //     })?;

        // 默认模板
        let mut chat_template = self.model.chat_template(None).map_err(|e| {
            error!("Failed to get chat template: {e}");
            e
        })?;

        // 通过自定义模板进行格式化
        match params.chat_template.clone() {
            Some(inner_chat_template) if !inner_chat_template.is_empty() => {
                chat_template = LlamaChatTemplate::new(&inner_chat_template)?;
            }
            _ => (),
        }
        Ok(chat_template)
    }

    /// Apply chat template
    fn apply_chat_template(
        &self,
        msgs: &[LlamaChatMessage],
        chat_template: &LlamaChatTemplate,
    ) -> Result<(String, Vec<LlamaToken>), Error> {
        // Format the message using chat template (simplified)
        let formatted_template = self
            .model
            .apply_chat_template(chat_template, msgs, true)
            .map_err(|e| {
                error!("Failed to apply chat template: {e}");
                e
            })?;

        // 将提示文本转换为 token 列表，AddBos::Always 表示始终在开头添加 BOS (Beginning of Sequence) token
        let tokens_list = self
            .model
            .str_to_token(&formatted_template, AddBos::Always)
            .map_err(|e| {
                error!("failed to tokenize {:?}, err: {}", formatted_template, e);
                e
            })?;

        // let tokens_list = self
        //     .context
        //     .model
        //     .str_to_token(&params.user_prompt, AddBos::Always)?;

        Ok((formatted_template, tokens_list))
    }

    /// Verify tokens size
    pub fn validate_tokens_size(
        &self,
        tokens_size: usize,
        n_ctx: u32,
        n_predict: i32,
    ) -> Result<(), Error> {
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

    /// Processes and evaluates input text with optional media (images/audio) through the model
    ///
    /// # Arguments
    /// * `params` - Configuration options for the LLM
    /// * `add_bos` - Whether to add beginning-of-sequence token
    /// * `msgs` - Chat messages to process
    /// * `chat_template` - Template for formatting chat messages
    ///
    /// # Errors
    /// Returns error if tokenization or evaluation fails
    fn process_multimodal_input(
        &mut self,
        params: &LlamaCppOptions,
        add_bos: bool,
        formatted_chat_template: String,
    ) -> Result<(), Error> {
        let input_text = MtmdInputText {
            text: formatted_chat_template,
            add_special: add_bos,
            parse_special: true,
        };
        info!("MtmdInputText: {input_text:?}");

        let bitmap_refs: Vec<&MtmdBitmap> = self.bitmaps.iter().collect();
        if bitmap_refs.is_empty() {
            info!("No bitmaps provided, only tokenizing text");
        } else {
            info!("Tokenizing with {} bitmaps", bitmap_refs.len());
        }

        let chunks = self.mtmd_context.tokenize(input_text, &bitmap_refs)?;
        info!("Tokenization complete, {} chunks created", chunks.len());

        self.bitmaps.clear();

        self.n_past = chunks.eval_chunks(
            &self.mtmd_context,
            &self.base_context.context,
            0,
            0,
            params.n_batch as i32,
            true,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_2::model::params::LlamaModelParams;

    use super::*;

    #[test]
    #[ignore]
    fn test_simple() -> anyhow::Result<()> {
        let params = LlamaCppOptions::default();

        // Initialize backend
        let backend = LlamaBackend::init()?;

        // Load model
        let model_path = params.get_model_path()?;
        let model_params = LlamaModelParams::default();
        let model = Arc::new(LlamaModel::load_from_file(
            &backend,
            &model_path,
            &model_params,
        )?);

        // Load sampler
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(params.top_k),
            LlamaSampler::top_p(params.top_p, 0),
            LlamaSampler::temp(params.temperature),
            LlamaSampler::dist(params.get_seed()), // 随机种子，用于生成随机数, 应用于最后一个
        ]);

        // 上下文
        let context_history: Vec<LlamaChatMessage> = Vec::new();

        let mut mtmd_conntext = LlamaCppMtmdContext::new(model.clone(), &backend, &params)?;

        // Load media files
        // for image_path in &params.images {
        //     info!("Loading image: {image_path}");
        //     mtmd_conntext.load_media_file(image_path)?;
        // }
        // for audio_path in &params.audio {
        //     mtmd_conntext.load_media_file(audio_path)?;
        // }

        mtmd_conntext.load_data_buffer(&params.images[0])?;

        mtmd_conntext.eval_message(&params, true, &context_history)?;

        let results = mtmd_conntext.generate_response(&mut sampler, params.max_predict())?;

        println!("{results:?}");
        Ok(())
    }
}
