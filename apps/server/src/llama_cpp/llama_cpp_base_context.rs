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
    model::{LlamaChatMessage, LlamaModel, Special},
    sampling::LlamaSampler,
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
    pub batch: LlamaBatch,
    // /// The list of loaded bitmaps (images/audio).
    // pub bitmaps: Vec<MtmdBitmap>,
    // /// The number of past tokens processed.
    // pub n_past: i32,
    // /// The chat template used for formatting messages.
    // pub chat_template: LlamaChatTemplate,
    // /// The current chat messages history.
    // pub chat: Vec<LlamaChatMessage>,
    /// Enables verbose logging from llama.cpp.
    /// Useful for debugging and performance analysis.
    pub verbose: bool,
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
            // chat: Vec::new(),
            // bitmaps: Vec::new(),
            // n_past: 0,
            // chat_template,
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
                let output_string = self
                    .model
                    .tokens_to_str(&[token], Special::Tokenize)
                    .unwrap_or("口".to_string());
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

        // 添加助手回复到上下文历史
        // if !results.is_empty() {
        //     let assistant_message =
        //         LlamaChatMessage::new(PromptMessageRole::Assistant.to_string(), results.clone())?;

        //     self.context_history.push(assistant_message);
        // }

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
