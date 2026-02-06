//! JoyCaption Predictor GGUF

use std::{
    ffi::CString,
    io::Cursor,
    num::NonZero,
    pin::pin,
    sync::{Arc, Mutex},
    time::Duration,
};

use base64::{Engine, engine::general_purpose};
use image::DynamicImage;
use lazy_static::lazy_static;
use llama_cpp_2::{
    LogOptions,
    context::{LlamaContext, params::LlamaContextParams},
    ggml_time_us,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{
        AddBos, LlamaChatMessage, LlamaModel,
        params::{LlamaModelParams, kv_overrides::ParamOverrideValue},
    },
    sampling::LlamaSampler,
    send_logs_to_tracing,
    token::LlamaToken,
};
use log::{error, info};
use serde_json::json;

use crate::{error::Error, wrapper::comfy::folder_paths::FolderPaths};

// 使用 lazy_static 或其他缓存机制
lazy_static! {
    static ref PREDICTOR: Arc<Mutex<Option<JoyCaptionPredictorGGUF>>> = Arc::new(Mutex::new(None));
}

/// JoyCaption 预测器
pub struct JoyCaptionPredictorGGUF {
    model: Arc<LlamaModel>,
    ctx: LlamaContext<'static>, // 使用Option包装，延迟初始化
}
// 确保线程安全
unsafe impl Send for JoyCaptionPredictorGGUF {}
unsafe impl Sync for JoyCaptionPredictorGGUF {}

impl JoyCaptionPredictorGGUF {
    /// 构造预测器
    #[allow(clippy::too_many_arguments)]
    pub fn new<'a>(
        model_name: &'a str,
        mmproj_name: &'a str,
        subfolder: Option<&'a str>,
        n_gpu_layers: u32,
        n_ctx: u32, // size of the prompt context (default: loaded from themodel)
        key_value_overrides: Vec<(String, ParamOverrideValue)>, // override some parameters of the model
        threads: Option<i32>, // number of threads to use during batch and prompt processing (default: use all available threads)
        threads_batch: Option<i32>, // size of the prompt context (default: loaded from themodel)"
    ) -> Result<Self, Error> {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(true));

        let base_models_dir = FolderPaths::default().model_path();
        let subfolder = subfolder.unwrap_or("llava_gguf");

        let model_path_full = base_models_dir.join(subfolder).join(model_name);
        let mmproj_path_full = base_models_dir.join(subfolder).join(mmproj_name);

        if !model_path_full.exists() {
            return Err(Error::InvalidPath(format!(
                "GGUF Model file not found: {}",
                model_path_full.to_string_lossy()
            )));
        }
        if !mmproj_path_full.exists() {
            return Err(Error::InvalidPath(format!(
                "GGUF Model file not found: {}",
                mmproj_path_full.to_string_lossy()
            )));
        }

        // init LLM
        let backend = LlamaBackend::init()?;

        // offload all layers to the gpu
        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let mut model_params = pin!(model_params);

        for (k, v) in &key_value_overrides {
            let k = CString::new(k.as_bytes())?;
            model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        }

        // load model
        let model = Arc::new(LlamaModel::load_from_file(
            &backend,
            &model_path_full,
            &model_params,
        )?);

        // initialize the context
        let mut ctx_params =
            LlamaContextParams::default().with_n_ctx(Some(NonZero::new(n_ctx).unwrap()));

        if let Some(threads) = threads {
            ctx_params = ctx_params.with_n_threads(threads);
        }
        if let Some(threads_batch) = threads_batch.or(threads) {
            ctx_params = ctx_params.with_n_threads_batch(threads_batch);
        }

        // 使用 unsafe 将 model 的生命周期延长到 'static
        // 这是安全的，因为 model 是 Arc 包装的，会一直存在
        let model_ref = unsafe { std::mem::transmute::<&LlamaModel, &'static LlamaModel>(&model) };

        // create the llama_context
        let ctx = model_ref.new_context(&backend, ctx_params)?;

        Ok(Self { model, ctx })
    }

    /// tokenize the prompt
    pub fn messages_tokens(
        &self,
        user_prompt: &str,
        max_length: i32,
    ) -> Result<Vec<LlamaToken>, Error> {
        if max_length <= 0 {
            return Err(Error::InvalidInput(
                "max_length must be positive".to_string(),
            ));
        }

        let tokens_list = self.ctx.model.str_to_token(user_prompt, AddBos::Always)?;

        let n_cxt = self.ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (max_length - tokens_list.len() as i32);

        info!("max_length = {max_length}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");

        // make sure the KV cache is big enough to hold all the prompt and generated tokens
        if n_kv_req > n_cxt {
            error!(
                "n_kv_req > n_ctx, the required kv cache size is not big enough
        either reduce n_len or increase n_ctx"
            )
        }

        if tokens_list.len() >= usize::try_from(max_length)? {
            error!("the prompt is too long, it has more tokens than max_length")
        }

        // print the prompt token-by-token
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        for token in &tokens_list {
            info!(
                "{}",
                self.model
                    .token_to_piece(*token, &mut decoder, true, None)?
            );
        }

        Ok(tokens_list)
    }

    /// generate eog tokens
    fn _eog_tokens(&self) -> Result<Vec<LlamaToken>, Error> {
        // Stop on newlines and conversation markers
        let end_marker: [&str; 4] = ["</s>", "Human:", "Assistant:", "\n\n"];
        let end_marker = end_marker.join("");

        let tokens_list = self.ctx.model.str_to_token(&end_marker, AddBos::Always)?;
        Ok(tokens_list)
    }

    /// 转换为 Base64 Data URL (自动检测格式)
    fn image_to_base64_url(
        image: &DynamicImage,
        format: Option<image::ImageFormat>,
    ) -> Result<String, Error> {
        // 确定保存格式 (默认PNG，如果是JPEG则用JPEG)
        let save_format = match format {
            Some(image::ImageFormat::Jpeg) => image::ImageFormat::Jpeg,
            _ => image::ImageFormat::Png,
        };

        // 将图像写入内存缓冲区 (使用Cursor包装Vec<u8>以满足Seek trait要求)
        let mut buffer = Cursor::new(Vec::new());
        image.write_to(&mut buffer, save_format)?;

        // Base64编码
        let img_base64 = general_purpose::STANDARD.encode(buffer.into_inner());

        // 生成Data URL
        let mime_type = match save_format {
            image::ImageFormat::Jpeg => "jpeg",
            _ => "png",
        };
        let image_url = format!("data:image/{};base64,{}", mime_type, img_base64);

        Ok(image_url)
    }

    /// generate image
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        image: &DynamicImage,
        system_prompt: &str,
        user_prompt: &str,
        n_tokens: i32,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        seed: u32,
    ) -> Result<String, Error> {
        // 转换为 Base64 Data URL
        let image_url = Self::image_to_base64_url(image, Some(image::ImageFormat::Png))?;

        // 构建对话
        let user_content = json!([
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "content": user_prompt.trim()},
        ])
        .to_string();

        // https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
        let chat_template = self.model.chat_template(Some("chatml"))?;
        let input_messages = self.model.apply_chat_template(
            &chat_template,
            &[
                LlamaChatMessage::new("system".to_string(), system_prompt.to_string())?,
                LlamaChatMessage::new("user".to_string(), user_content)?,
            ],
            true,
        )?;

        let tokens_list = self.messages_tokens(&input_messages, n_tokens)?;

        // create a llama_batch with size max_tokens
        // we use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(n_tokens as usize, 1);

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            // llama_decode will output logits only for the last token of the prompt
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }

        // llama decode
        self.ctx.decode(&mut batch)?;

        // main loop

        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;

        let t_main_start = ggml_time_us();

        // The `Decoder`
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::dist(seed),
            LlamaSampler::top_k(top_k),
            LlamaSampler::top_p(top_p, 0),
            LlamaSampler::temp(temperature),
            LlamaSampler::greedy(),
        ]);

        let mut results = String::new();
        while n_cur <= n_tokens {
            // sample the next token
            {
                let token = sampler.sample(&self.ctx, batch.n_tokens() - 1);

                sampler.accept(token);

                // is it an end of stream?
                if self.ctx.model.is_eog_token(token) {
                    break;
                }

                let piece = self.model.token_to_piece(token, &mut decoder, true, None)?;

                info!("{piece}");
                results.push_str(&piece);

                batch.clear();
                batch.add(token, n_cur, &[0], true)?;
            }

            n_cur += 1;

            self.ctx.decode(&mut batch)?;

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

        info!("timings: {}", self.ctx.timings());

        Ok(results)
    }
}
