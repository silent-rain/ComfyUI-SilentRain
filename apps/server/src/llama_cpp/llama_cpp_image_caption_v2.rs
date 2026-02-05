//! llama.cpp Image Caption v2
//! 支持图像反推
use std::sync::Arc;

use futures::future::join_all;
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
    ContexParams, GenerateRequest, MediaData, Pipeline, PipelineConfig, model::ModelConfig,
    sampler::SamplerConfig,
};

use crate::{
    core::{
        category::CATEGORY_LLAMA_CPP,
        node_base::{InputSpec, InputType},
        utils::image::tensor_to_image_tensor_buffer,
    },
    error::Error,
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
        let generate_request = GenerateRequest::default();
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
                    .default(generate_request.system_prompt.unwrap_or_default())
                    .tooltip("System prompt for the request"),
            )
            .with_required(
                "user_prompt",
                InputType::string()
                    .multiline(true)
                    .default(generate_request.user_prompt)
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
                    .default(generate_request.session_id)
                    .tooltip("Session ID for the request, Used for context isolation"),
            )
            .with_required(
                "keep_context",
                InputType::bool()
                    .default(generate_request.keep_context)
                    .tooltip("Maintain multiple rounds of conversation context. Context is not supported when concurrent limit is greater than 1"),
            )
            .build()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute", signature = (model, options, images, **kwargs))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        model: Bound<'py, PyDict>,
        options: Option<Bound<'py, PyDict>>,
        images: Option<Bound<'py, PyAny>>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(Vec<String>,)> {
        let (pipeline_config, generate_request, medias, concurrency_limit) = self
            .options_parser(model, options, images, kwargs)
            .map_err(|e| {
                error!("LlamaCppImageCaptionv2 options parser: {e}");
                if let Err(e) =
                    self.send_error(py, "LlamaCppImageCaptionv2".to_string(), e.to_string())
                {
                    error!("send error failed: {e}");
                }
                PyErr::new::<PyRuntimeError, _>(e.to_string())
            })?;

        let futures = self.generate(pipeline_config, generate_request, medias, concurrency_limit);

        // 使用 allow_threads 释放 GIL，然后在内部运行异步代码
        py.detach(move || {
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
        })
    }
}

impl LlamaCppImageCaptionv2 {
    /// 解析参数
    fn options_parser<'py>(
        &self,
        model: Bound<'py, PyDict>,
        options: Option<Bound<'py, PyDict>>,
        images: Option<Bound<'py, PyAny>>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<(PipelineConfig, GenerateRequest, Vec<MediaData>, u32), Error> {
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

        let verbose: bool = kwargs
            .get_item("verbose")
            .ok()
            .flatten()
            .map(|v| v.extract().unwrap_or(false))
            .unwrap_or(false);

        let concurrency_limit: u32 = kwargs
            .get_item("concurrency_limit")
            .ok()
            .flatten()
            .map(|v| v.extract().unwrap_or(1))
            .unwrap_or(1);

        let model_config: ModelConfig = depythonize(&kwargs)?;
        let sampler_config: SamplerConfig = depythonize(&kwargs)?;
        let context_config: ContexParams = depythonize(&kwargs)?;
        let pipeline_config = PipelineConfig {
            model: model_config,
            context: context_config,
            sampling: sampler_config,
            verbose,
        };
        let generate_request: GenerateRequest = depythonize(&kwargs)?;

        // 图片资源处理
        let mut medias = Vec::new();
        if let Some(images) = images {
            info!("Start processing images ...");
            let image_raw_datas = tensor_to_image_tensor_buffer(&images)?;
            for (pixel_bytes, height, width, channels) in image_raw_datas {
                let media = generate_request.load_image_tensor_buffer(
                    pixel_bytes,
                    height as u32,
                    width as u32,
                    channels,
                )?;
                medias.push(media);
            }
            info!("Image processing completed ...");
        }

        Ok((pipeline_config, generate_request, medias, concurrency_limit))
    }

    /// 生成聊天响应
    async fn generate(
        &self,
        pipeline_config: PipelineConfig,
        generate_request: GenerateRequest,
        medias: Vec<MediaData>,
        concurrency_limit: u32,
    ) -> Result<(Vec<String>,), Error> {
        let semaphore = Arc::new(Semaphore::new(concurrency_limit as usize));
        let pipeline = Arc::new(Pipeline::try_new(pipeline_config)?);

        // 生成所有并发
        let futures = medias.into_iter().enumerate().map(|(i, media)| {
            let semaphore = Arc::clone(&semaphore);
            let pipeline_clone = pipeline.clone();

            {
                let generate_request_clone = generate_request.clone();
                async move {
                    let _permit = semaphore.acquire().await.map_err(|e| {
                        error!("获取Semaphore许可失败, err: {:#?}", e);
                        Error::AcquireError(e.to_string())
                    })?;

                    let generate_request_clone = generate_request_clone
                        .with_media(media)
                        .with_image_max_resolution(768);
                    let output = pipeline_clone.generate(&generate_request_clone).await?;

                    info!("iamge {i} prcessing completed");
                    Ok::<_, Error>(output.text)
                }
            }
        });

        // 并发执行所有 Future
        let results: Vec<String> = join_all(futures)
            .await
            .into_iter()
            .collect::<Result<Vec<String>, _>>()?
            .into_iter()
            .collect();

        Ok((results,))
    }
}
