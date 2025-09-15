//! llama.cpp pipeline

use std::{
    hash::{DefaultHasher, Hash, Hasher},
    io::Write,
    num::NonZero,
    sync::{Arc, RwLock},
    time::Duration,
};

use llama_cpp_2::{
    context::{
        params::{LlamaContextParams, LlamaPoolingType},
        LlamaContext,
    },
    ggml_time_us,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{
        params::LlamaModelParams, AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, Special,
    },
    sampling::LlamaSampler,
    send_logs_to_tracing,
    token::LlamaToken,
    LogOptions,
};
use log::{error, info};

use crate::{
    error::Error,
    llama_cpp::{LlamaCppMtmdContext, LlamaCppOptions, PromptMessageRole},
};

pub struct LlamaCppPipeline {
    // 当前节点使用的模型缓存键
    context_history: Vec<LlamaChatMessage>, // 上下文缓存
    backend: LlamaBackend,
    model: Arc<LlamaModel>,
    context: Arc<RwLock<LlamaContext<'static>>>,
    mtmd_context: Option<LlamaCppMtmdContext>,
    sampler: LlamaSampler,
    batch: LlamaBatch,
}

unsafe impl Send for LlamaCppPipeline {}
unsafe impl Sync for LlamaCppPipeline {}

impl LlamaCppPipeline {
    pub fn new(params: &LlamaCppOptions) -> Result<Self, Error> {
        // llama.cpp logs
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(params.verbose));

        // Initialize backend
        let backend = LlamaBackend::init()?;

        // Load model
        let model = Arc::new(Self::load_model(&backend, params)?);

        // Load sampler
        let sampler = Self::load_sampler(params)?;

        // Load context
        let context = Arc::new(RwLock::new(Self::load_context(
            model.clone(),
            &backend,
            params,
        )?));

        // Create the MTMD context
        let mut mtmd_ctx = LlamaCppMtmdContext::new(&model, params)?;
        info!("Loading mtmd projection: {}", params.mmproj_path);

        info!("Model loaded successfully");

        // 创建批处理对象
        let batch = LlamaBatch::new(params.n_batch as usize, 1);

        Ok(Self {
            context_history: Vec::new(),
            backend,
            model,
            context,
            sampler,
            batch,
        })
    }
}

impl LlamaCppPipeline {
    /// 生成聊天
    pub fn generate_chat(&mut self, params: &LlamaCppOptions) -> Result<String, Error> {
        let t_main_start = ggml_time_us();
        // 获取当前批处理中的 token 数量，用作下一个 token 的位置索引
        let mut n_cur = self.batch.n_tokens();
        let mut n_decode = 0;
        let mut results = String::new();

        let mut context = self
            .context
            .write()
            .map_err(|e| Error::LockError(e.to_string()))?;

        // 循环生成 token，直到达到最大长度 max_predict
        while n_cur < params.max_predict() {
            // 采样下一个 token
            {
                // 使用采样器从上下文中采样下一个 token
                // batch.n_tokens() - 1 表示获取最后一个 token 的 logits
                let token = self.sampler.sample(&context, self.batch.n_tokens() - 1);

                // 接受采样的 token，更新采样器的内部状态
                self.sampler.accept(token);

                // 检查是否为结束标记 (End of Stream)
                if token == self.model.token_eos() {
                    println!();
                    break; // 如果是结束标记，则退出循环
                }

                // 将 token 转换为字符串
                let output_string = self
                    .model
                    .tokens_to_str(&[token], Special::Tokenize)
                    .unwrap_or("口".to_string());
                results.push_str(&output_string);

                if params.verbose {
                    // 打印解码后的字符串，不换行
                    print!("{output_string}");
                    // 刷新标准输出，确保文本立即显示
                    std::io::stdout().flush()?;
                }

                // 清空批处理对象，准备添加新的 token
                self.batch.clear();
                // 添加新生成的 token 到批处理中
                // 参数：token值, 位置索引, 序列ID数组, 是否需要计算 logits (true)
                self.batch.add(token, n_cur, &[0], true)?;
            }

            // 增加当前位置索引，为下一个 token 做准备
            n_cur += 1;

            // 解码新添加的 token，计算下一个 token 的概率分布
            context.decode(&mut self.batch).map_err(|e| {
                error!("failed to eval, {e}");
                e
            })?;

            n_decode += 1;
        }

        // 添加助手回复到上下文历史
        if !results.is_empty() {
            let assistant_message =
                LlamaChatMessage::new(PromptMessageRole::Assistant.to_string(), results.clone())?;

            self.context_history.push(assistant_message);
        }

        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
        info!(
            "decoded {} tokens in {:.2} s, speed {:.2} t/s\n",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32()
        );

        info!("{}", context.timings());

        Ok(results)
    }

    // // 视觉推理
    // pub fn generate_vision(&mut self, params: &LlamaCppOptions) -> Result<String, Error> {}
}

impl LlamaCppPipeline {
    /// Setup model parameters
    pub fn load_model(
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
    pub fn load_context(
        model: Arc<LlamaModel>,
        backend: &LlamaBackend,
        params: &LlamaCppOptions,
    ) -> Result<LlamaContext<'static>, Error> {
        let pooling_type = match params.pooling_type.as_str() {
            "None" => LlamaPoolingType::None,
            "Mean" => LlamaPoolingType::Mean,
            "Cls" => LlamaPoolingType::Cls,
            "Last" => LlamaPoolingType::Last,
            "Rank" => LlamaPoolingType::Rank,
            _ => {
                info!("Unknown pooling type: {}", params.pooling_type);
                LlamaPoolingType::Unspecified
            }
        };

        let context_params = LlamaContextParams::default()
            .with_n_threads(params.n_threads)
            .with_n_threads_batch(params.n_threads_batch)
            .with_n_batch(params.n_batch)
            .with_n_ctx(NonZero::new(params.n_ctx))
            .with_embeddings(true)
            .with_pooling_type(pooling_type);

        // Create context
        let context = model.new_context(backend, context_params)?;

        // 确保 context 的生命周期不依赖于 model
        let context = unsafe { std::mem::transmute(context) };

        Ok(context)
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

    /// update model parameters
    pub fn update_model(&mut self, params: &LlamaCppOptions) -> Result<(), Error> {
        // 缓存模型，不重新加载模型
        // if params.cache_model {
        //     return Ok(());
        // }
        let model = Arc::new(Self::load_model(&self.backend, params)?);
        // let context = Self::load_context(model.clone(), &self.backend, params)?;
        // let batch = LlamaBatch::new(params.n_batch as usize, 1);

        self.model = model;
        // self.context = Arc::new(RwLock::new(context));
        // self.batch = batch;
        Ok(())
    }

    pub fn unload_model(&mut self) -> Result<(), Error> {
        drop(self.model.clone());
        Ok(())
    }

    /// Update context parameters
    pub fn update_context(&mut self, params: &LlamaCppOptions) -> Result<(), Error> {
        let context = Self::load_context(self.model.clone(), &self.backend, params)?;

        self.context = Arc::new(RwLock::new(context));
        Ok(())
    }

    pub fn rest_batch(&mut self, params: &LlamaCppOptions) -> Result<(), Error> {
        let batch = LlamaBatch::new(params.n_batch as usize, 1);
        self.batch = batch;
        Ok(())
    }
}

impl LlamaCppPipeline {
    /// Create message
    pub fn create_message(
        &self,
        params: &LlamaCppOptions,
        context_history: Vec<LlamaChatMessage>,
    ) -> Result<Vec<LlamaChatMessage>, Error> {
        // Create message
        let mut msgs = vec![
            LlamaChatMessage::new(
                PromptMessageRole::System.to_string(),
                params.system_prompt.clone(),
            )?,
            LlamaChatMessage::new(
                PromptMessageRole::User.to_string(),
                params.user_prompt.clone(),
            )?,
        ];

        // Add history messages
        if params.keep_context {
            msgs.extend(context_history);
        }

        Ok(msgs)
    }

    /// Apply chat template
    pub fn apply_chat_template(
        &self,
        msgs: Vec<LlamaChatMessage>,
        params: &LlamaCppOptions,
    ) -> Result<Vec<LlamaToken>, Error> {
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

        // Format the message using chat template (simplified)
        let formatted_prompt = self
            .model
            .apply_chat_template(&chat_template, &msgs, true)
            .map_err(|e| {
                error!("Failed to apply chat template: {e}");
                e
            })?;

        // 将提示文本转换为 token 列表，AddBos::Always 表示始终在开头添加 BOS (Beginning of Sequence) token
        let tokens_list = self
            .model
            .str_to_token(&formatted_prompt, AddBos::Always)
            .map_err(|e| {
                error!("failed to tokenize {:?}, err: {}", formatted_prompt, e);
                e
            })?;

        // let tokens_list = self
        //     .context
        //     .model
        //     .str_to_token(&params.user_prompt, AddBos::Always)?;

        Ok(tokens_list)
    }

    /// Process user tokens
    pub fn process_user_tokens(&mut self, tokens_list: Vec<LlamaToken>) -> Result<(), Error> {
        // 计算 tokens_list 的最后一个索引
        let last_index: i32 = (tokens_list.len() - 1) as i32;

        // 遍历所有 token，i 是索引，token 是实际的 token 值
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode 只会为提示的最后一个 token 输出 logits（概率分布）
            // 判断当前 token 是否是最后一个
            let is_last = i == last_index;

            // 将 token 添加到批处理中
            // 参数：token值, 位置索引, 序列ID数组, 是否是最后一个token
            self.batch.add(token, i, &[0], is_last)?;
        }

        // 使用上下文解码批处理中的 token
        self.context
            .write()
            .map_err(|e| Error::LockError(e.to_string()))?
            .decode(&mut self.batch)
            .map_err(|e| {
                error!("llama_decode failed: {}", e);
                e
            })?;

        Ok(())
    }

    /// Verify tokens size
    pub fn validate_tokens_size(
        &self,
        tokens_size: usize,
        n_ctx: u32,
        n_predict: i32,
    ) -> Result<(), Error> {
        if n_predict < 0 {
            if tokens_size >= n_ctx as usize {
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
                "n_kv_req > n_ctx, the required kv cache size is not big enough
either reduce n_predict or increase n_ctx"
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

    /// Load user tokens
    pub fn load_user_tokens(&mut self, params: &LlamaCppOptions) -> Result<(), Error> {
        let msgs = self.create_message(&params, self.context_history.clone())?;
        let token_list = self.apply_chat_template(msgs, &params)?;

        self.validate_tokens_size(token_list.len(), params.n_ctx, params.n_predict)?;

        self.process_user_tokens(token_list)?;
        Ok(())
    }
}

impl LlamaCppPipeline {
    /// 计算模型缓存的哈希键
    /// 只包含影响模型加载的参数
    fn _compute_model_cache_key(&self, params: &LlamaCppOptions) -> String {
        let mut hasher = DefaultHasher::new();

        // 只包含影响模型加载的关键参数
        params.model_path.hash(&mut hasher);
        params.n_gpu_layers.hash(&mut hasher);
        params.main_gpu.hash(&mut hasher);
        params.flash_attention.hash(&mut hasher);
        params.n_ctx.hash(&mut hasher);

        format!("{:x}", hasher.finish())
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
        pipeline.update_model(&params)?;
        pipeline.update_context(&params)?;
        pipeline.rest_batch(&params)?;
        pipeline.load_user_tokens(&params)?;
        let results = pipeline.generate_chat(&params)?;
        println!("{results:?}");
        Ok(())
    }
}
