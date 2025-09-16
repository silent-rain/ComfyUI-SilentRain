//! llama.cpp context
//!
//!
//! pub fn chat_template(&self, name: Option<&str>) -> Result<LlamaChatTemplate, ChatTemplateError>
//!     - https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
//!     - https://github.com/ggml-org/llama.cpp/blob/master/src/llama-chat.cpp#L28
//!

use std::{io::Write, num::NonZero, sync::Arc, time::Duration};

use llama_cpp_2::{
    context::{
        params::{LlamaContextParams, LlamaPoolingType},
        LlamaContext,
    },
    ggml_time_us,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel, Special},
    sampling::LlamaSampler,
    token::LlamaToken,
};
use log::{error, info};

use crate::{
    error::Error,
    llama_cpp::{LlamaCppOptions, PromptMessageRole},
};

/// Llama Cpp Context
#[allow(missing_debug_implementations)]
pub struct LlamaCppBaseContext {
    model: Arc<LlamaModel>,
    /// The context for multimodal processing.
    pub context: LlamaContext<'static>,
    /// The batch used for processing tokens.
    batch: LlamaBatch,
    /// Enables verbose logging from llama.cpp.
    verbose: bool,
}

impl LlamaCppBaseContext {
    /// Creates a new context
    ///
    /// # Errors
    pub fn new(
        model: Arc<LlamaModel>,
        backend: &LlamaBackend,
        params: &LlamaCppOptions,
    ) -> Result<Self, Error> {
        let context = Self::load_context(model.clone(), backend, params)?;

        let batch = LlamaBatch::new(params.n_ctx as usize, 1);

        Ok(Self {
            model,
            context,
            batch,
            verbose: params.verbose,
        })
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
            "Unspecified" => LlamaPoolingType::Unspecified,
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
        let context = unsafe { std::mem::transmute::<LlamaContext<'_>, LlamaContext<'_>>(context) };

        Ok(context)
    }

    /// Evaluates a chat message, tokenizing and processing it through the model
    /// # Errors
    pub fn eval_message(
        &mut self,
        params: &LlamaCppOptions,
        context_history: &[LlamaChatMessage],
    ) -> Result<(), Error> {
        let msgs = self.create_message(params, context_history)?;

        let chat_template = self.chat_template(params)?;

        let token_list = self.apply_chat_template(msgs, chat_template)?;

        self.validate_tokens_size(token_list.len(), params.n_ctx, params.n_predict)?;

        self.rest_batch(params.n_batch as usize)?;

        self.process_user_tokens(token_list)?;

        Ok(())
    }

    /// Generates a response by sampling tokens from the model
    /// # Errors
    pub fn generate_response(
        &mut self,
        // context: &mut LlamaContext,
        sampler: &mut LlamaSampler,
        n_predict: i32,
    ) -> Result<String, Error> {
        // The `Decoder`
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        let t_main_start = ggml_time_us();
        // 获取当前批处理中的 token 数量，用作下一个 token 的位置索引
        let mut n_cur = self.batch.n_tokens();
        let mut n_decode = 0;
        let mut results = String::new();

        // 循环生成 token，直到达到最大长度 n_predict
        while n_cur < n_predict {
            // 采样下一个 token
            {
                // 使用采样器从上下文中采样下一个 token
                // batch.n_tokens() - 1 表示获取最后一个 token 的 logits
                let token = sampler.sample(&self.context, self.batch.n_tokens() - 1);

                // 接受采样的 token，更新采样器的内部状态
                sampler.accept(token);

                // 检查是否为结束标记 (End of Stream)
                if token == self.model.token_eos() {
                    println!();
                    break; // 如果是结束标记，则退出循环
                }

                // 将 token 转换为字符串
                // let output_string = self
                //     .model
                //     .tokens_to_str(&[token], Special::Tokenize)
                //     .unwrap_or("口".to_string());

                // let output_string = self.model.token_to_str(token, Special::Tokenize)?;
                let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
                // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                let mut output_string = String::with_capacity(32);
                let _decode_result =
                    decoder.decode_to_string(&output_bytes, &mut output_string, false);

                results.push_str(&output_string);

                if self.verbose {
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
            self.context.decode(&mut self.batch).map_err(|e| {
                error!("failed to eval, {e}");
                e
            })?;

            n_decode += 1;
        }

        let t_main_end = ggml_time_us();
        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
        info!(
            "decoded {} tokens in {:.2} s, speed {:.2} t/s\n",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32()
        );

        info!("{}", self.context.timings());
        Ok(results)
    }
}

impl LlamaCppBaseContext {
    pub fn rest_batch(&mut self, n_batch: usize) -> Result<(), Error> {
        let batch = LlamaBatch::new(n_batch, 1);
        self.batch = batch;
        Ok(())
    }
}

impl LlamaCppBaseContext {
    /// Create message
    pub fn create_message(
        &self,
        params: &LlamaCppOptions,
        context_history: &[LlamaChatMessage],
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
        msgs: Vec<LlamaChatMessage>,
        chat_template: LlamaChatTemplate,
    ) -> Result<Vec<LlamaToken>, Error> {
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

    /// Process user tokens
    fn process_user_tokens(&mut self, tokens_list: Vec<LlamaToken>) -> Result<(), Error> {
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
        self.context.decode(&mut self.batch).map_err(|e| {
            error!("llama_decode failed: {}", e);
            e
        })?;

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

        let mut base_conntext = LlamaCppBaseContext::new(model.clone(), &backend, &params)?;

        base_conntext.eval_message(&params, &context_history)?;

        let results = base_conntext.generate_response(&mut sampler, params.max_predict())?;

        println!("{results:?}");
        Ok(())
    }
}
