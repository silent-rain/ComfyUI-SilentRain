//! llama.cpp vision

use std::{num::NonZero, path::PathBuf};

use log::info;
use pyo3::pyclass;

use llama_cpp_2::{
    context::{
        params::{LlamaContextParams, LlamaPoolingType},
        LlamaContext,
    },
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaModel},
    sampling::LlamaSampler,
    send_logs_to_tracing, LogOptions,
};

use crate::{
    error::Error,
    llama_cpp::{LlamaCppMtmdContext, LlamaCppOptions},
    wrapper::{comfy::folder_paths::FolderPaths, comfyui::PromptServer},
};

#[pyclass(subclass)]
pub struct LlamaCppVision {}

impl PromptServer for LlamaCppVision {}

impl LlamaCppVision {
    pub fn generate(&mut self) -> Result<(), Error> {
        let params = LlamaCppOptions::default();

        // llama.cpp logs
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(params.verbose));

        // get model path
        let (model_path, mmproj_path) = self.get_model_path(&params)?;

        // Initialize backend
        let backend = LlamaBackend::init()?;

        // Load model
        let model = self.load_model(&backend, &params)?;
        info!("Loading model: {model_path:?}");

        // Load context
        let mut context = self.load_context(&model, &backend, &params)?;

        // Load sampler
        let mut sampler = self.load_sampler(&params)?;

        // Create the MTMD context
        let mut ctx = LlamaCppMtmdContext::new(&model, &params)?;
        info!("Loading mtmd projection: {}", params.mmproj_path);

        info!("Model loaded successfully");

        // run_single_turn(&mut ctx, &model, &mut context, &mut sampler, &params)?;

        Ok(())
    }

    /// get model path
    fn get_model_path(&self, params: &LlamaCppOptions) -> Result<(PathBuf, PathBuf), Error> {
        let base_models_dir = FolderPaths::default().model_path();
        let subfolder = "LLM";

        let model_path = base_models_dir
            .join(subfolder)
            .join(params.model_path.clone());
        let mmproj_path = base_models_dir
            .join(subfolder)
            .join(params.mmproj_path.clone());

        // Validate required parameters
        if !model_path.exists() {
            return Err(Error::InvalidPath(format!(
                "Model file not found: {}",
                model_path.to_string_lossy()
            )));
        }
        if !mmproj_path.exists() {
            return Err(Error::InvalidPath(format!(
                "Multimodal projection file not found: {}",
                mmproj_path.to_string_lossy()
            )));
        }

        info!("Loading model: {model_path:?}");

        Ok((model_path, mmproj_path))
    }

    /// Setup model parameters
    fn load_model(
        &self,
        backend: &LlamaBackend,
        params: &LlamaCppOptions,
    ) -> Result<LlamaModel, Error> {
        let model_params = LlamaModelParams::default().with_n_gpu_layers(params.n_gpu_layers); // Use n layers on GPU

        // for (k, v) in &key_value_overrides {
        //     let k = CString::new(k.as_bytes())?;
        //     model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        // }

        // Load model
        let model = LlamaModel::load_from_file(&backend, &params.model_path, &model_params)?;

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
        let context = model.new_context(&backend, context_params)?;

        Ok(context)
    }

    /// Setup sampler parameters
    fn load_sampler(&self, params: &LlamaCppOptions) -> Result<LlamaSampler, Error> {
        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::greedy(),
            LlamaSampler::dist(params.seed),
            LlamaSampler::top_k(params.top_k),
            LlamaSampler::top_p(params.top_p, 0),
            LlamaSampler::temp(params.temperature),
        ]);
        Ok(sampler)
    }
}
