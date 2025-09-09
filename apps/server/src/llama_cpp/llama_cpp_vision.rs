//! llama.cpp vision

use std::num::NonZero;

use log::info;
use pyo3::pyclass;

use llama_cpp_2::{
    context::{
        params::{LlamaContextParams, LlamaPoolingType},
        LlamaContext,
    },
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaChatMessage, LlamaModel},
    sampling::LlamaSampler,
    send_logs_to_tracing, LogOptions,
};
use rand::TryRngCore;

use crate::{
    error::Error,
    llama_cpp::{LlamaCppMtmdContext, LlamaCppOptions},
    wrapper::comfyui::PromptServer,
};

#[pyclass(subclass)]
pub struct LlamaCppVision {}

impl PromptServer for LlamaCppVision {}

impl LlamaCppVision {
    pub fn generate(&mut self) -> Result<(), Error> {
        let params = LlamaCppOptions::default();

        // llama.cpp logs
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(params.verbose));

        // Initialize backend
        let backend = LlamaBackend::init()?;

        // Load model
        let model = self.load_model(&backend, &params)?;

        // Load context
        let mut context = self.load_context(&model, &backend, &params)?;

        // Load sampler
        let mut sampler = self.load_sampler(&params)?;

        // Create the MTMD context
        let mut ctx = LlamaCppMtmdContext::new(&model, &params)?;
        info!("Loading mtmd projection: {}", params.mmproj_path);

        info!("Model loaded successfully");

        // Add media marker if not present
        let mut user_prompt = params.user_prompt.clone();
        let default_marker = llama_cpp_2::mtmd::mtmd_default_marker().to_string();
        let media_marker = params.media_marker.as_ref().unwrap_or(&default_marker);
        if !user_prompt.contains(media_marker) {
            user_prompt.push_str(media_marker);
        }

        // Load media files
        for image_path in &params.images {
            info!("Loading image: {image_path}");
            ctx.load_media(image_path)?;
        }
        for audio_path in &params.audio {
            ctx.load_media(audio_path)?;
        }

        // Create user message
        let msgs = vec![
            LlamaChatMessage::new("system".to_string(), params.system_prompt)?,
            LlamaChatMessage::new("user".to_string(), user_prompt)?,
        ];

        info!("Evaluating message: {msgs:?}");

        // Evaluate the message (prefill)
        ctx.eval_message(&model, &mut context, msgs, true, params.n_batch as i32)?;

        // Generate response (decode)
        ctx.generate_response(&model, &mut context, &mut sampler, params.n_predict)?;

        Ok(())
    }

    /// Setup model parameters
    fn load_model(
        &self,
        backend: &LlamaBackend,
        params: &LlamaCppOptions,
    ) -> Result<LlamaModel, Error> {
        let model_path = params.get_model_path()?;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(params.n_gpu_layers); // Use n layers on GPU

        // for (k, v) in &key_value_overrides {
        //     let k = CString::new(k.as_bytes())?;
        //     model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        // }

        // Load model
        let model = LlamaModel::load_from_file(backend, &model_path, &model_params)?;

        info!("Loading model: {model_path:?}");

        Ok(model)
    }

    /// Setup context parameters
    fn load_context<'a>(
        &self,
        model: &'a LlamaModel,
        backend: &LlamaBackend,
        params: &LlamaCppOptions,
    ) -> Result<LlamaContext<'a>, Error> {
        let pooling_type = match params.pooling_type.as_str() {
            "None" => LlamaPoolingType::None,
            "Mean" => LlamaPoolingType::Mean,
            "Cls" => LlamaPoolingType::Cls,
            "Last" => LlamaPoolingType::Last,
            "Rank" => LlamaPoolingType::Rank,
            _ => LlamaPoolingType::Unspecified,
        };

        let context_params = LlamaContextParams::default()
            .with_n_threads(params.n_threads)
            .with_n_batch(params.n_threads_batch)
            .with_n_batch(params.n_batch)
            .with_n_ctx(NonZero::new(params.n_ctx))
            .with_embeddings(true)
            .with_pooling_type(pooling_type);

        // Create context
        let context = model.new_context(backend, context_params)?;

        Ok(context)
    }

    /// Setup sampler parameters
    fn load_sampler(&self, params: &LlamaCppOptions) -> Result<LlamaSampler, Error> {
        // 随机值
        let seed = if params.seed == -1 {
            // 随机值
            rand::rng().try_next_u32().unwrap_or(0)
        } else {
            params.seed as u32
        };

        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::greedy(),
            LlamaSampler::dist(seed),
            LlamaSampler::top_k(params.top_k),
            LlamaSampler::top_p(params.top_p, 0),
            LlamaSampler::temp(params.temperature),
        ]);
        Ok(sampler)
    }
}
