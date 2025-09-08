//! llama.cpp mtmd context

use std::{
    ffi::CString,
    io::{self, Write},
};

use llama_cpp_2::{
    context::LlamaContext,
    llama_batch::LlamaBatch,
    model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel, Special},
    mtmd::{MtmdBitmap, MtmdBitmapError, MtmdContext, MtmdContextParams, MtmdInputText},
    sampling::LlamaSampler,
};
use log::{error, info};

use crate::{error::Error, llama_cpp::LlamaCppOptions};

/// State of the MTMD CLI application.
#[allow(missing_debug_implementations)]
pub struct LlamaCppMtmdContext {
    /// The MTMD context for multimodal processing.
    pub mtmd_ctx: MtmdContext,
    /// The batch used for processing tokens.
    pub batch: LlamaBatch,
    /// The list of loaded bitmaps (images/audio).
    pub bitmaps: Vec<MtmdBitmap>,
    /// The number of past tokens processed.
    pub n_past: i32,
    /// The chat template used for formatting messages.
    pub chat_template: LlamaChatTemplate,
    /// The current chat messages history.
    pub chat: Vec<LlamaChatMessage>,
}

impl LlamaCppMtmdContext {
    /// Creates a new MTMD context
    ///
    /// # Errors
    pub fn new(model: &LlamaModel, params: &LlamaCppOptions) -> Result<Self, Error> {
        let use_gpu = { params.n_gpu_layers > 0 && !params.no_mmproj_offload };

        // Initialize MTMD context
        let mtmd_params = MtmdContextParams {
            use_gpu,
            print_timings: true,
            n_threads: params.n_threads,
            media_marker: CString::new(
                params
                    .media_marker
                    .as_ref()
                    .unwrap_or(&llama_cpp_2::mtmd::mtmd_default_marker().to_string())
                    .clone(),
            )?,
        };

        let mtmd_ctx =
            MtmdContext::init_from_file(&params.get_mmproj_path()?, model, &mtmd_params)?;

        let chat_template = model
            .chat_template(params.chat_template.as_deref())
            .map_err(|e| {
                error!("Failed to get chat template: {e}");
                e
            })?;

        let batch = LlamaBatch::new(params.n_ctx as usize, 1);

        Ok(Self {
            mtmd_ctx,
            batch,
            chat: Vec::new(),
            bitmaps: Vec::new(),
            n_past: 0,
            chat_template,
        })
    }

    /// Loads media (image or audio) from the specified file path
    /// # Errors
    pub fn load_media(&mut self, path: &str) -> Result<(), MtmdBitmapError> {
        let bitmap = MtmdBitmap::from_file(&self.mtmd_ctx, path)?;
        self.bitmaps.push(bitmap);
        Ok(())
    }

    /// Evaluates a chat message, tokenizing and processing it through the model
    /// # Errors
    pub fn eval_message(
        &mut self,
        model: &LlamaModel,
        context: &mut LlamaContext,
        msgs: Vec<LlamaChatMessage>,
        add_bos: bool,
        batch_size: i32,
    ) -> Result<(), Error> {
        self.chat.extend(msgs);

        // Format the message using chat template (simplified)
        let formatted_prompt = model.apply_chat_template(&self.chat_template, &self.chat, true)?;

        let input_text = MtmdInputText {
            text: formatted_prompt,
            add_special: add_bos,
            parse_special: true,
        };

        let bitmap_refs: Vec<&MtmdBitmap> = self.bitmaps.iter().collect();

        if bitmap_refs.is_empty() {
            info!("No bitmaps provided, only tokenizing text");
        } else {
            info!("Tokenizing with {} bitmaps", bitmap_refs.len());
        }

        // Tokenize the input
        let chunks = self.mtmd_ctx.tokenize(input_text, &bitmap_refs)?;

        info!("Tokenization complete, {} chunks created", chunks.len());

        // Clear bitmaps after tokenization
        self.bitmaps.clear();

        self.n_past = chunks.eval_chunks(&self.mtmd_ctx, context, 0, 0, batch_size, true)?;
        Ok(())
    }

    /// Generates a response by sampling tokens from the model
    /// # Errors
    pub fn generate_response(
        &mut self,
        model: &LlamaModel,
        context: &mut LlamaContext,
        sampler: &mut LlamaSampler,
        n_predict: i32,
    ) -> Result<(), Error> {
        let mut generated_tokens = Vec::new();
        let max_predict = if n_predict < 0 { i32::MAX } else { n_predict };

        for _i in 0..max_predict {
            // Sample next token
            let token = sampler.sample(context, -1);
            generated_tokens.push(token);
            sampler.accept(token);

            // Check for end of generation
            if model.is_eog_token(token) {
                break;
            }

            // Print token
            let piece = model.token_to_str(token, Special::Tokenize)?;
            info!("{piece}");
            io::stdout().flush()?;

            // Prepare next batch
            self.batch.clear();
            self.batch.add(token, self.n_past, &[0], true)?;
            self.n_past += 1;

            // Decode
            context.decode(&mut self.batch)?;
        }

        Ok(())
    }
}
