//! llama.cpp Image Caption v2
//! 支持图像反推
use std::sync::Arc;

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods, PyType},
};
use pythonize::depythonize;
use tokio::sync::Semaphore;
use tracing::info;
use uuid::Uuid;

use llama_cpp_core::{
    ContexParams, PipelineConfig, chat_history,
    model::ModelConfig,
    request::{
        ChatMessagesBuilder, CreateChatCompletionRequestArgs, Metadata, Request, UserMessageBuilder,
    },
    response::response_extract_content,
    sampler::SamplerConfig,
    utils::image::Image,
};

use crate::{
    core::{
        category::CATEGORY_LLAMA_CPP,
        node_base::{InputSpec, InputType},
        utils::image::tensor_to_image_tensor_buffer,
    },
    error::Error,
    llama_cpp::{LlamaCppPromptHelperv2, llama_cpp_model_v2::LlamaCppModelParams},
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_LLAMA_CPP_MODEL_V2, NODE_LLAMA_CPP_OPTIONS_V2, NODE_STRING},
    },
};

/// LlamaCpp Image Caption v2
#[pyclass(subclass)]
pub struct LlamaCppImageCaptionv2 {}

impl PromptServer for LlamaCppImageCaptionv2 {}

#[pymethods]
impl LlamaCppImageCaptionv2 {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {})
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_STRING,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("response",)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LLAMA_CPP;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "llama.cpp image caption v2."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classattr]
    #[pyo3(name = "EXPERIMENTAL")]
    fn experimental() -> bool {
        true
    }

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        let sampler_config = SamplerConfig::default();
        let model_config = ModelConfig::default();
        let context_config = ContexParams::default();
        let session_id = Uuid::new_v4().to_string();

        InputSpec::new()
            .with_required(
                "model",
                InputType::custom(NODE_LLAMA_CPP_MODEL_V2)
                    .force_input(true)
                    .tooltip("model parameters"),
            )
            .with_optional(
                "options",
                InputType::custom(NODE_LLAMA_CPP_OPTIONS_V2)
                    .force_input(true)
                    .tooltip("options parameters"),
            )
            .with_optional(
                "images",
                InputType::image()
                    .force_input(true)
                    .tooltip("If images are not empty, image backpropagation will be performed; otherwise, text generation will occur"),
            )
            .with_required(
                "system_prompt",
                InputType::string()
                    .multiline(true)
                    .default("")
                    .tooltip("System prompt for the request"),
            )
            .with_required(
                "user_prompt",
                InputType::string()
                    .multiline(true)
                    .default("")
                    .tooltip("User prompt for the request"),
            )
            .with_required(
                "main_gpu",
                InputType::int()
                    .default(model_config.main_gpu)
                    .min(0)
                    .step_int(1)
                    .tooltip("Index of the main GPU to use. Relevant for multi-GPU systems"),
            )
            .with_required(
                "devices",
                InputType::string()
                    .default("")
                    .default(model_config.devices_str())
                    .tooltip("Device indices separated by commas (e.g. 0,1,2). Overrides main_gpu and enables multi-GPU"),
            )
            .with_required(
                "n_gpu_layers",
                InputType::int()
                    .default(model_config.n_gpu_layers)
                    .min(0)
                    .max_int(1000)
                    .step_int(1)
                    .tooltip("Number of GPU layers to offload. 0 = CPU only mode, >0 = GPU mode"),
            )
            .with_required(
                "n_ctx",
                InputType::int()
                    .default(context_config.n_ctx)
                    .min(512)
                    .step_int(1)
                    .tooltip("Size of the prompt context window. Defines the maximum context length the model can handle"),
            )
            .with_required(
                "n_predict",
                InputType::int()
                    .default(context_config.n_predict)
                    .min(0)
                    .step_int(1)
                    .tooltip("Number of tokens to predict (-1 for unlimited)"),
            )
            // ===== 采样参数 =====
            .with_required(
                "temperature",
                InputType::float()
                    .default(sampler_config.temperature)
                    .min_float(0.0)
                    .max_float(2.0)
                    .step_float(0.01)
                    .tooltip(
                        "Controls randomness. 0.0 means deterministic, higher means more random",
                    ),
            )
            .with_required(
                "top_k",
                InputType::int()
                    .default(sampler_config.top_k)
                    .min(0)
                    .step_int(1)
                    .tooltip("Controls diversity via top-k sampling. Higher values mean more diverse outputs"),
            )
            .with_required(
                "top_p",
                InputType::float()
                    .default(sampler_config.top_p)
                    .min_float(0.0)
                    .max_float(1.0)
                    .step_float(0.01)
                    .tooltip("Controls diversity via nucleus sampling. Lower values mean more focused outputs"),
            )
            .with_required(
                "min_p",
                InputType::float()
                    .default(sampler_config.min_p)
                    .min_float(0.0)
                    .max_float(1.0)
                    .step_float(0.01)
                    .tooltip("Min-p sampling threshold"),
            )
            .with_required(
                "seed",
                InputType::int()
                    .default(sampler_config.seed)
                    .min(0)
                    .step_int(1)
                    .tooltip("Seed for random number generation. 0 means random"),
            )
            // CONCURRENCY_LIMIT
            .with_required(
                "concurrency_limit",
                InputType::int()
                    .default(1)
                    .min(1)
                    .max(20)
                    .step_int(1)
                    .tooltip("Concurrency limit. Multiple images are simultaneously used for reverse inference, and context is not supported when performing reverse inference on multiple images."),
            )
            .with_required(
                "session_id",
                InputType::string()
                    .default(session_id)
                    .tooltip("Session ID for the request, Used for context isolation"),
            )
            .with_required(
                "keep_context",
                InputType::bool()
                    .default(false)
                    .tooltip("Maintain multiple rounds of conversation context. Context is not supported when concurrent limit is greater than 1"),
            )
            .build()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute", signature = (model, options, system_prompt, user_prompt, images, session_id, keep_context, **kwargs))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        model: Bound<'py, PyDict>,
        options: Option<Bound<'py, PyDict>>,
        system_prompt: String,
        user_prompt: String,
        images: Option<Bound<'py, PyAny>>,
        session_id: String,
        keep_context: bool,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(Vec<String>,)> {
        let (pipeline_config, requests, concurrency_limit, llama_cpp_model_params) = self
            .options_parser(
                model,
                options,
                system_prompt,
                user_prompt,
                images,
                session_id.clone(),
                kwargs,
            )
            .map_err(|e| {
                error!("LlamaCppImageCaptionv2 options parser: {e}");
                if let Err(e) =
                    self.send_error(py, "LlamaCppImageCaptionv2".to_string(), e.to_string())
                {
                    error!("send error failed: {e}");
                }
                PyErr::new::<PyRuntimeError, _>(e.to_string())
            })?;

        let total_tasks = requests.len();
        if total_tasks == 0 {
            return Ok((Vec::new(),));
        }

        // 获取 comfy.utils.ProgressBar
        let comfy = py.import("comfy")?;
        let utils = comfy.getattr("utils")?;
        // 创建 ProgressBar，总数为任务数
        let pbar = utils.call_method1("ProgressBar", (total_tasks,))?;

        // 创建进度通道
        let (progress_tx, progress_rx) = std::sync::mpsc::channel::<()>();

        // 将 Bound 转换为 PyObject，这样才能跨线程传递
        // Py<PyAny> 是 Send + Sync，可以安全地跨线程
        let pbar_py: Py<pyo3::PyAny> = pbar.unbind();
        let progress_handle = std::thread::spawn(move || {
            for _ in progress_rx {
                // 获取 GIL 并更新 ProgressBar
                Python::attach(|py| {
                    if let Err(e) = pbar_py.bind(py).call_method1("update", (1,)) {
                        error!("Failed to update progress bar: {e}");
                    }
                });
            }
        });

        let futures = self.generate(
            pipeline_config,
            requests,
            concurrency_limit,
            llama_cpp_model_params,
            progress_tx,
            session_id,
            keep_context,
        );

        // 使用 allow_threads 释放 GIL，然后在内部运行异步代码
        let result = py.detach(move || {
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                error!("Failed to create tokio runtime: {e}");
                PyErr::new::<PyRuntimeError, _>(format!("Failed to create tokio runtime: {e}"))
            })?;

            rt.block_on(async {
                let result: Result<_, Error> = futures.await;

                match result {
                    Ok(v) => Ok(v),
                    Err(e) => {
                        error!("LlamaCppImageCaptionv2 error: {e}");
                        Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
                    }
                }
            })
        });

        // 等待进度线程完成
        if let Err(e) = progress_handle.join() {
            error!("Progress thread panicked: {:?}", e);
        }

        result
    }
}

impl LlamaCppImageCaptionv2 {
    /// 解析参数
    #[allow(clippy::too_many_arguments)]
    fn options_parser<'py>(
        &self,
        model: Bound<'py, PyDict>,
        options: Option<Bound<'py, PyDict>>,
        system_prompt: String,
        user_prompt: String,
        images: Option<Bound<'py, PyAny>>,
        session_id: String,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<(PipelineConfig, Vec<Request>, u32, LlamaCppModelParams), Error> {
        let kwargs =
            kwargs.ok_or_else(|| Error::InvalidParameter("parameters is required".to_string()))?;

        if let Some(options) = options {
            kwargs.update(options.as_mapping()).map_err(|e| {
                error!("update kwargs error: {e}");
                Error::InvalidParameter(format!("update options kwargs error: {e}"))
            })?;
        }

        kwargs.update(model.as_mapping()).map_err(|e| {
            error!("update kwargs error: {e}");
            Error::InvalidParameter(format!("update model kwargs error: {e}"))
        })?;

        let concurrency_limit: u32 = kwargs
            .get_item("concurrency_limit")
            .ok()
            .flatten()
            .map(|v| v.extract().unwrap_or(1))
            .unwrap_or(1);

        let llama_cpp_model_params: LlamaCppModelParams = depythonize(&kwargs)?;
        let model_config: ModelConfig = depythonize(&kwargs)?;
        let sampler_config: SamplerConfig = depythonize(&kwargs)?;
        let mut context_config: ContexParams = depythonize(&kwargs)?;
        context_config = context_config.with_image_max_resolution(768); // 设置图片最大分辨率

        let pipeline_config = PipelineConfig {
            model: model_config,
            context: context_config,
            sampling: sampler_config,
        };

        // 图片资源处理
        let mut medias = Vec::new();
        if let Some(images) = images {
            info!("Start processing images ...");
            let image_raw_datas = tensor_to_image_tensor_buffer(&images)?;
            for (pixel_bytes, height, width, channels) in image_raw_datas {
                let media = Image::from_tensor(pixel_bytes, height as u32, width as u32, channels)?;
                medias.push(media.to_base64()?);
            }
            info!("Image processing completed ...");
        }

        let mut metadata = Metadata {
            session_id: Some(session_id),
            ..Default::default()
        };

        let mut requests = Vec::new();
        for media in &medias {
            if medias.len() > 1 {
                let session_id = Uuid::new_v4().to_string();
                metadata = Metadata {
                    session_id: Some(session_id),
                    ..Default::default()
                };
            }

            let request = CreateChatCompletionRequestArgs::default()
                .metadata(metadata.clone())
                .max_completion_tokens(2048u32)
                .model("Qwen3-VL-2B-Instruct")
                .messages(
                    ChatMessagesBuilder::new()
                        .system(&system_prompt)
                        .users(
                            UserMessageBuilder::new()
                                .text(&user_prompt)
                                .image_base64("image/png", media),
                        )
                        .build(),
                )
                .build()
                .map_err(|e| {
                    error!("CreateChatCompletionRequestArgs error: {e}");
                    Error::OpenAIError(e.to_string())
                })?;
            requests.push(request);
        }

        info!("pipeline_config: {pipeline_config:#?}");

        Ok((
            pipeline_config,
            requests,
            concurrency_limit,
            llama_cpp_model_params,
        ))
    }

    /// 生成聊天响应
    #[allow(clippy::too_many_arguments)]
    async fn generate(
        &self,
        pipeline_config: PipelineConfig,
        requests: Vec<Request>,
        concurrency_limit: u32,
        llama_cpp_model_params: LlamaCppModelParams,
        progress_tx: std::sync::mpsc::Sender<()>,
        session_id: String,
        keep_context: bool,
    ) -> Result<(Vec<String>,), Error> {
        // 移除上下文
        if !keep_context {
            let chat_history = chat_history();
            chat_history.remove(&session_id);
        }

        let semaphore = Arc::new(Semaphore::new(concurrency_limit as usize));
        let pipeline =
            LlamaCppPromptHelperv2::load_pipeline(pipeline_config, llama_cpp_model_params)?;

        // 生成所有并行任务
        let handles = requests
            .into_iter()
            .enumerate()
            .map(|(i, request)| {
                let semaphore = Arc::clone(&semaphore);
                let pipeline_clone = pipeline.clone();
                let progress_tx_clone = progress_tx.clone();

                tokio::spawn(async move {
                    let _permit = semaphore.acquire().await.map_err(|e| {
                        error!("获取Semaphore许可失败, err: {:#?}", e);
                        Error::AcquireError(e.to_string())
                    })?;

                    let output = pipeline_clone.generate(&request).await?;

                    // 任务完成，发送进度信号
                    if let Err(e) = progress_tx_clone.send(()) {
                        error!("发送进度信号失败: {e}");
                    }

                    info!("image {i} processing completed");
                    let content = response_extract_content(&output);
                    Ok::<_, Error>(content)
                })
            })
            .collect::<Vec<_>>();

        // 并行执行所有任务并收集结果
        let results = futures::future::try_join_all(handles)
            .await
            .map_err(|e| {
                error!("任务执行失败, err: {:#?}", e);
                Error::TaskJoinError(e.to_string())
            })?
            .into_iter()
            .collect::<Result<Vec<String>, Error>>()?;

        // 关闭通道
        drop(progress_tx);
        Ok((results,))
    }
}
