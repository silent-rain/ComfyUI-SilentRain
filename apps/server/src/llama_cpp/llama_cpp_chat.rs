//! llama.cpp chat

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io::Write,
    num::NonZero,
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
    model::{params::LlamaModelParams, AddBos, LlamaChatMessage, LlamaModel, Special},
    sampling::LlamaSampler,
    send_logs_to_tracing,
    token::LlamaToken,
    LogOptions,
};
use log::{error, info};
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyErr, PyResult, Python,
};
use pythonize::depythonize;
use rand::TryRngCore;

use crate::{
    core::category::CATEGORY_LLAMA_CPP,
    error::Error,
    llama_cpp::{LlamaCppOptions, PromptMessageRole},
    wrapper::{
        comfy::folder_paths::FolderPaths,
        comfyui::{
            types::{
                NODE_BOOLEAN, NODE_IMAGE, NODE_INT, NODE_INT_MAX, NODE_LLAMA_CPP_OPTIONS,
                NODE_SEED_MAX, NODE_STRING,
            },
            PromptServer,
        },
    },
};

const SUBFOLDER: &str = "LLM";

#[pyclass(subclass)]
pub struct LlamaCppChat {
    // 当前节点使用的模型缓存键
    model_cache_key: String,
    // 上下文缓存
    context_history: Vec<LlamaChatMessage>,
    model: Option<LlamaModel>,
}

impl PromptServer for LlamaCppChat {}

#[pymethods]
impl LlamaCppChat {
    #[new]
    fn new() -> Self {
        Self {
            model_cache_key: String::new(),
            context_history: Vec::new(),
            model: None,
        }
    }

    #[classattr]
    #[pyo3(name = "EXPERIMENTAL")]
    fn experimental() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_IMAGE,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("captions",)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() {}

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LLAMA_CPP;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "llama.cpp chat"
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);

                let options = LlamaCppOptions::default();


                // 获取模型列表
                let model_list = FolderPaths::default().get_filename_list(SUBFOLDER);

                required.set_item(
                    "model_path",
                    (model_list.clone(), {
                        let params = PyDict::new(py);
                        // params.set_item("default", options.model_path)?;
                        params.set_item("tooltip", "model file path")?;
                        params
                    }),
                )?;


                required.set_item(
                    "system_prompt",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.system_prompt)?;
                        params.set_item("multiline", true)?;
                        params.set_item("tooltip", "The system prompt (or instruction) that guides the model's behavior. This is typically a high-level directiv")?;
                        params
                    }),
                )?;

                required.set_item(
                    "user_prompt",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.user_prompt)?;
                        params.set_item("multiline", true)?;
                        params.set_item("tooltip", "The user-provided input or query to the model. This is the dynamic part of the prompt that changes with each interaction. ")?;
                        params
                    }),
                )?;

                required.set_item(
                    "media_marker",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.media_marker)?;
                        params.set_item(
                            "tooltip",
                            "Media marker. If not provided, the default marker will be used.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_ctx",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.n_ctx)?;
                        params.set_item("min", 256)?;
                        params.set_item("max", NODE_INT_MAX)?;
                        params.set_item("step", 10)?;
                        params.set_item("tooltip", "Size of the prompt context window.")?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_predict",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.n_predict)?;
                        params.set_item("min", -1)?;
                        params.set_item("max", NODE_INT_MAX)?;
                        params.set_item("step", 10)?;
                        params.set_item(
                            "tooltip",
                            "Number of tokens to predict (-1 for unlimited).",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "seed",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.seed)?;
                        params.set_item("min", -1)?;
                        params.set_item("max", NODE_SEED_MAX)?;
                        params.set_item("step", 1)?;
                        params.set_item(
                            "tooltip",
                            "Seed for random number generation. Set to a fixed value for reproducible outputs.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "main_gpu",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.main_gpu)?;
                        params.set_item("min", 0)?;
                        params.set_item("step", 1)?;
                        params.set_item(
                            "tooltip",
                            "Index of the main GPU to use. Relevant for multi-GPU systems.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_gpu_layers",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.n_gpu_layers)?;
                        params.set_item("min", 0)?;
                        params.set_item("step", 1)?;
                        params.set_item(
                            "tooltip",
                            "Number of GPU layers to offload.",
                        )?;
                        params
                    }),
                )?;

                required.set_item(
                    "keep_context",
                    (NODE_BOOLEAN, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.keep_context)?;
                        params.set_item("tooltip", "Keep the context between requests")?;
                        params
                    }),
                )?;

                required
            })?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);
                optional.set_item(
                    "extra_options",
                    (NODE_LLAMA_CPP_OPTIONS, {
                        let params = PyDict::new(py);
                        params.set_item("tooltip", "llama.cpp extra options")?;
                        params
                    }),
                )?;

                optional
            })?;
            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute", signature = (model_path, system_prompt, user_prompt, media_marker, n_ctx, n_predict, seed, main_gpu, n_gpu_layers, keep_context, extra_options=None))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        model_path: String,
        system_prompt: String,
        user_prompt: String,
        media_marker: String,
        n_ctx: u32,
        n_predict: i32,
        seed: i32,
        main_gpu: i32,
        n_gpu_layers: u32,
        keep_context: bool,
        extra_options: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(String,)> {
        let params = self
            .options_parser(
                model_path,
                system_prompt,
                user_prompt,
                media_marker,
                n_ctx,
                n_predict,
                seed,
                main_gpu,
                n_gpu_layers,
                keep_context,
                extra_options,
            )
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("parameters error, {e}")))?;

        // 使用缓存生成结果
        let results = self.generate(&params);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("LlamaCppChat error, {e}");
                if let Err(e) = self.send_error(py, "LlamaCppChat".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LlamaCppChat {
    /// Parse the options from the parameters.
    ///
    /// images: [batch, height, width, channels]
    #[allow(clippy::too_many_arguments)]
    fn options_parser<'py>(
        &self,
        model_path: String,
        system_prompt: String,
        user_prompt: String,
        media_marker: String,
        n_ctx: u32,
        n_predict: i32,
        seed: i32,
        main_gpu: i32,
        n_gpu_layers: u32,
        keep_context: bool,
        extra_options: Option<Bound<'py, PyDict>>,
    ) -> Result<LlamaCppOptions, Error> {
        let mut options = LlamaCppOptions::default();

        if let Some(extra_options) = extra_options {
            options = depythonize(&extra_options)?;
        }

        options.model_path = model_path;
        options.system_prompt = system_prompt;
        options.user_prompt = user_prompt;
        options.media_marker = Some(media_marker);
        options.n_ctx = n_ctx;
        options.n_predict = n_predict;
        options.seed = seed;
        options.main_gpu = main_gpu;
        options.n_gpu_layers = n_gpu_layers;
        options.keep_context = keep_context;

        Ok(options)
    }

    /// 计算模型缓存的哈希键
    /// 只包含影响模型加载的参数
    fn compute_model_cache_key(&self, params: &LlamaCppOptions) -> String {
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

impl LlamaCppChat {
    /// 生成结果
    pub fn generate(&mut self, params: &LlamaCppOptions) -> Result<(String,), Error> {
        // llama.cpp logs
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(params.verbose));

        // Initialize backend
        let backend = LlamaBackend::init()?;

        // TODO 添加缓存 model
        // Load cache model
        // self.load_cache_model(&backend, params)?;
        // let model = self.model.as_ref().unwrap();

        // Load model
        let model = self.load_model(&backend, params)?;

        // Load context
        let mut context = self.load_context(&model, &backend, params)?;

        // Load sampler
        let mut sampler = self.load_sampler(params)?;

        // 创建批处理对象
        let mut batch = LlamaBatch::new(params.n_batch as usize, 1);

        // get tokens
        let tokens_list = self.create_message(&model, params)?;
        self.validate_tokens_size(tokens_list.len(), context.n_ctx(), params.n_predict)?;
        self.process_tokens(&mut context, &mut batch, tokens_list)?;

        let t_main_start = ggml_time_us();
        // 获取当前批处理中的 token 数量，用作下一个 token 的位置索引
        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;
        let mut results = String::new();

        // 循环生成 token，直到达到最大长度 max_predict
        while n_cur <= params.max_predict() {
            // 采样下一个 token
            {
                // 使用采样器从上下文中采样下一个 token
                // batch.n_tokens() - 1 表示获取最后一个 token 的 logits
                let token = sampler.sample(&context, batch.n_tokens() - 1);

                // 接受采样的 token，更新采样器的内部状态
                sampler.accept(token);

                // 检查是否为结束标记 (End of Stream)
                if token == model.token_eos() {
                    println!();
                    break; // 如果是结束标记，则退出循环
                }

                // 将 token 转换为字符串
                let output_string = model.tokens_to_str(&[token], Special::Tokenize)?;
                results.push_str(&output_string);

                if params.verbose {
                    // 打印解码后的字符串，不换行
                    print!("{output_string}");
                    // 刷新标准输出，确保文本立即显示
                    std::io::stdout().flush()?;
                }

                // 清空批处理对象，准备添加新的 token
                batch.clear();
                // 添加新生成的 token 到批处理中
                // 参数：token值, 位置索引, 序列ID数组, 是否需要计算 logits (true)
                batch.add(token, n_cur, &[0], true)?;
            }

            // 增加当前位置索引，为下一个 token 做准备
            n_cur += 1;

            // 解码新添加的 token，计算下一个 token 的概率分布
            context.decode(&mut batch).map_err(|e| {
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

        Ok((results,))
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

    // Load cache model
    fn load_cache_model(
        &mut self,
        backend: &LlamaBackend,
        params: &LlamaCppOptions,
    ) -> Result<(), Error> {
        // 计算模型缓存键
        let cache_key = self.compute_model_cache_key(params);

        if self.model_cache_key != cache_key {
            self.model_cache_key = cache_key;
            let model = self.load_model(backend, params)?;
            self.model = Some(model);
        }

        Ok(())
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
            .with_n_threads_batch(params.n_threads_batch)
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
            LlamaSampler::dist(seed), // 随机种子，用于生成随机数, 应用于最后一个
        ]);
        Ok(sampler)
    }

    /// Create message
    fn create_message(
        &mut self,
        model: &LlamaModel,
        params: &LlamaCppOptions,
    ) -> Result<Vec<LlamaToken>, Error> {
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
            for msg in &self.context_history {
                msgs.push(msg.clone());
            }
        } else {
            // 清空历史记录
            self.context_history.clear();
        }

        // 通过预定义模板进行格式化
        // let chat_template = model
        //     .chat_template(params.chat_template.as_deref())
        //     .map_err(|e| {
        //         error!("Failed to get chat template: {e}");
        //         e
        //     })?;

        // 通过自定义模板进行格式化
        // let chat_template = LlamaChatTemplate::new(
        //     params
        //         .chat_template
        //         .clone()
        //         .unwrap_or("default".to_string())
        //         .as_str(),
        // )?;

        // 默认模板
        let chat_template = model.chat_template(None).map_err(|e| {
            error!("Failed to get chat template: {e}");
            e
        })?;

        // Format the message using chat template (simplified)
        let formatted_prompt = model
            .apply_chat_template(&chat_template, &msgs, true)
            .map_err(|e| {
                error!("Failed to apply chat template: {e}");
                e
            })?;

        // 将提示文本转换为 token 列表，AddBos::Always 表示始终在开头添加 BOS (Beginning of Sequence) token
        let tokens_list = model
            .str_to_token(&formatted_prompt, AddBos::Always)
            .map_err(|e| {
                error!("failed to tokenize {:?}, err: {}", formatted_prompt, e);
                e
            })?;

        // let tokens_list = model.str_to_token(&params.user_prompt, AddBos::Always)?;

        Ok(tokens_list)
    }

    /// 处理 tokens
    fn process_tokens(
        &self,
        context: &mut LlamaContext<'_>,
        batch: &mut LlamaBatch,
        tokens_list: Vec<LlamaToken>,
    ) -> Result<(), Error> {
        // 计算 tokens_list 的最后一个索引
        let last_index: i32 = (tokens_list.len() - 1) as i32;

        // 遍历所有 token，i 是索引，token 是实际的 token 值
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode 只会为提示的最后一个 token 输出 logits（概率分布）
            // 判断当前 token 是否是最后一个
            let is_last = i == last_index;

            // 将 token 添加到批处理中
            // 参数：token值, 位置索引, 序列ID数组, 是否是最后一个token
            batch.add(token, i, &[0], is_last)?;
        }

        // 使用上下文解码批处理中的 token
        context.decode(batch).map_err(|e| {
            error!("llama_decode failed: {}", e);
            e
        })?;

        Ok(())
    }

    /// Verify if the token exceeds the maximum length
    fn validate_tokens_size(
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
}
