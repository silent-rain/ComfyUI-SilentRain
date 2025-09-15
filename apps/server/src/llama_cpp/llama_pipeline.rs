//! Unified pipeline for chat and vision processing

use std::sync::Arc;

use llama_cpp_2::model::LlamaChatMessage;

use crate::error::Error;
use crate::llama_cpp::LlamaCppOptions;
use super::{llama_core::LlamaCore, llama_base_context::LlamaBaseContext, llama_vision_extension::LlamaVisionExtension};

pub struct LlamaPipeline {
    core: LlamaCore,
    base_ctx: LlamaBaseContext,
    vision_ext: Option<LlamaVisionExtension>,
    history: Vec<LlamaChatMessage>,
}

impl LlamaPipeline {
    pub fn new(params: &LlamaCppOptions) -> Result<Self, Error> {
        let core = LlamaCore::new(params)?;
        let base_ctx = LlamaBaseContext::new(&core, params)?;
        let vision_ext = if params.enable_vision {
            Some(LlamaVisionExtension::new(&core.model, params)?)
        } else {
            None
        };
        Ok(Self {
            core,
            base_ctx,
            vision_ext,
            history: Vec::new(),
        })
    }

    pub fn generate_chat(&mut self, params: &LlamaCppOptions) -> Result<String, Error> {
        self.base_ctx.generate_response(params.n_predict)
    }

    pub fn generate_vision(&mut self, image: &[u8], params: &LlamaCppOptions) -> Result<String, Error> {
        if let Some(vision_ext) = &mut self.vision_ext {
            vision_ext.load_image(image)?;
            self.base_ctx.generate_response(params.n_predict)
        } else {
            Err(Error::VisionNotEnabled)
        }
    }
}