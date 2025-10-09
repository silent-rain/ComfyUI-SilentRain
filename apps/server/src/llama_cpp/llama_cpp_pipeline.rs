//! llama.cpp pipeline

use std::sync::{Arc, RwLock};

use llama_cpp_2::{
    LogOptions,
    llama_backend::LlamaBackend,
    model::{LlamaChatMessage, LlamaModel, params::LlamaModelParams},
    sampling::LlamaSampler,
    send_logs_to_tracing,
};
use log::info;

use crate::{
    error::Error,
    llama_cpp::{
        LlamaCppBaseContext, LlamaCppMtmdContext, LlamaCppOptions, ModelManager, PromptMessageRole,
    },
};

pub struct LlamaCppPipeline {
    // 当前节点使用的模型缓存键
    context_history: Vec<LlamaChatMessage>, // 上下文缓存
    #[allow(unused)]
    backend: LlamaBackend,
    #[allow(unused)]
    model: Arc<LlamaModel>,
    base_context: LlamaCppBaseContext,
    mtmd_context: Option<Arc<RwLock<LlamaCppMtmdContext>>>,
    sampler: LlamaSampler,
}

unsafe impl Send for LlamaCppPipeline {}
unsafe impl Sync for LlamaCppPipeline {}

impl LlamaCppPipeline {
    pub fn new(params: &LlamaCppOptions) -> Result<Self, Error> {
        // llama.cpp logs
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(params.verbose));

        let model_manager = ModelManager::global();

        // Initialize backend
        let backend = LlamaBackend::init()?;

        // Load model
        let model = {
            let mut cache = model_manager
                .write()
                .map_err(|e| Error::LockError(e.to_string()))?;

            // 只包含影响模型加载的关键参数
            let model_hash = cache.cal_hash(&[
                params.model_path.clone(),
                params.n_gpu_layers.to_string(),
                params.main_gpu.to_string(),
                params.flash_attention.to_string(),
                params.n_ctx.to_string(),
            ]);

            cache.get_or_insert(&params.model_cache_key, &model_hash, || {
                Self::load_model(&backend, params)
            })?
        };

        // Load sampler
        let sampler = Self::load_sampler(params)?;

        let base_context = LlamaCppBaseContext::new(model.clone(), &backend, params)?;

        // Load MTMD context
        let mut mtmd_context = None;
        if !params.mmproj_path.is_empty() {
            mtmd_context = {
                let mut cache = model_manager
                    .write()
                    .map_err(|e| Error::LockError(e.to_string()))?;

                // 只包含影响模型加载的关键参数
                let model_hash = cache.cal_hash(&[
                    params.model_path.clone(),
                    params.mmproj_path.clone(),
                    params.media_marker.clone().unwrap_or_default(),
                    params.n_gpu_layers.to_string(),
                    params.n_threads.to_string(),
                    params.verbose.to_string(),
                ]);

                let mtmd_context =
                    cache.get_or_insert(&params.model_cache_key, &model_hash, || {
                        LlamaCppMtmdContext::new(model.clone(), &backend, params)
                            .map(|mtmd| Arc::new(RwLock::new(mtmd)))
                    })?;

                Some(mtmd_context)
            };
        }

        Ok(Self {
            context_history: Vec::new(),
            backend,
            model,
            base_context,
            mtmd_context,
            sampler,
        })
    }
}

impl LlamaCppPipeline {
    /// 生成聊天
    pub fn generate_chat(&mut self, params: &LlamaCppOptions) -> Result<String, Error> {
        self.base_context
            .eval_message(params, &self.context_history)?;

        let results = self
            .base_context
            .generate_response(&mut self.sampler, params.max_predict())?;

        // 添加助手回复到上下文历史
        if !results.is_empty() {
            let assistant_message =
                LlamaChatMessage::new(PromptMessageRole::Assistant.to_string(), results.clone())?;

            self.context_history.push(assistant_message);
        }

        if !params.keep_context {
            self.context_history.clear();
        }

        Ok(results)
    }

    // 视觉推理
    pub fn generate_vision(
        &mut self,
        params: &LlamaCppOptions,
        image: &[u8],
    ) -> Result<String, Error> {
        let binding = self
            .mtmd_context
            .clone()
            .ok_or_else(|| Error::OptionNone("MTMD context is not initialized".to_string()))?;
        let mut mtmd_conntext = binding
            .write()
            .map_err(|e| Error::LockError(e.to_string()))?;

        // Load media files
        // for image_path in &params.images {
        //     info!("Loading image: {image_path}");
        //     mtmd_conntext.load_media_file(image_path)?;
        // }
        // for audio_path in &params.audio {
        //     mtmd_conntext.load_media_file(audio_path)?;
        // }

        mtmd_conntext.load_data_buffer(image)?;

        mtmd_conntext.eval_message(params, true, &self.context_history)?;

        let results = mtmd_conntext.generate_response(&mut self.sampler, params.max_predict())?;

        // 添加助手回复到上下文历史
        if !results.is_empty() {
            let assistant_message =
                LlamaChatMessage::new(PromptMessageRole::Assistant.to_string(), results.clone())?;

            self.context_history.push(assistant_message);
        }

        if !params.keep_context {
            self.context_history.clear();
        }

        Ok(results)
    }
}

impl LlamaCppPipeline {
    /// Setup model parameters
    pub fn load_model(
        backend: &LlamaBackend,
        params: &LlamaCppOptions,
    ) -> Result<Arc<LlamaModel>, Error> {
        let model_path = params.get_model_path()?;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(params.n_gpu_layers); // Use n layers on GPU

        // for (k, v) in &key_value_overrides {
        //     let k = CString::new(k.as_bytes())?;
        //     model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        // }

        // Load model
        let model = LlamaModel::load_from_file(backend, &model_path, &model_params)?;

        info!("Loading model: {model_path:?}");

        Ok(Arc::new(model))
    }

    /// Setup sampler parameters
    pub fn load_sampler(params: &LlamaCppOptions) -> Result<LlamaSampler, Error> {
        /* penalties:
            减少重复性: penalties(64, 1.2, 0.0, 0.2)
            增加多样性: penalties(64, 1.1, 0.1, 0.0)
            默认平衡: penalties(64, 1.0, 0.0, 0.0)
        */

        let sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(params.top_k),
            LlamaSampler::top_p(params.top_p, 0),
            LlamaSampler::min_p(0.0, 0),
            LlamaSampler::temp(params.temperature),
            LlamaSampler::penalties(
                64,  // penalty_last_n: 考虑最近64个token的重复情况
                1.0, // penalty_repeat: 轻微惩罚重复token
                0.0, // penalty_freq: 惩罚重复的 token（正值减少重复）。-2.0 至 2.0
                0.0, // penalty_present: 惩罚新主题的出现（正值减少重复主题），-2.0 至 2.0
            ),
            // LlamaSampler::greedy(),   // 贪婪采样器，始终选择概率最高的 token, 应用于最后一个，该采样器会导致每次文本一样
            LlamaSampler::dist(params.get_seed()), // 随机种子，用于生成随机数, 应用于最后一个
        ]);
        Ok(sampler)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_simple() -> anyhow::Result<()> {
        let params = LlamaCppOptions::default();
        let mut pipeline = LlamaCppPipeline::new(&params)?;

        let results = pipeline.generate_chat(&params)?;
        println!("{results:?}");
        Ok(())
    }
}
