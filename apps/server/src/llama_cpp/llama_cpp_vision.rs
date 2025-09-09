//! llama.cpp vision

use std::num::NonZero;

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
use log::{error, info};
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};
use pythonize::depythonize;
use rand::TryRngCore;

use crate::{
    core::category::CATEGORY_LLAMA_CPP,
    error::Error,
    llama_cpp::{LlamaCppMtmdContext, LlamaCppOptions},
    wrapper::comfyui::{
        types::{NODE_IMAGE, NODE_INT, NODE_LLAMA_CPP_OPTIONS, NODE_SEED_MAX, NODE_STRING},
        PromptServer,
    },
};

#[pyclass(subclass)]
pub struct LlamaCppVision {}

impl PromptServer for LlamaCppVision {}

#[pymethods]
impl LlamaCppVision {
    #[new]
    fn new() -> Self {
        Self {}
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
    fn return_types() -> (&'static str, &'static str) {
        (NODE_IMAGE, NODE_STRING)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("images", "captions")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() {}

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LLAMA_CPP;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "llama.cpp vision"
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

                required.set_item(
                    "images",
                    (NODE_IMAGE, {
                        let params = PyDict::new(py);
                        params.set_item("forceInput", true)?;
                        params.set_item("tooltip", "Path to image file(s)")?;
                        params
                    }),
                )?;

                required.set_item(
                    "model_path",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        // params.set_item("default", options.model_path)?;
                        params.set_item("tooltip", "model file path")?;
                        params
                    }),
                )?;

                required.set_item(
                    "mmproj_path",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.mmproj_path)?;
                        params.set_item("tooltip", "mmproj model file path")?;
                        params
                    }),
                )?;

                required.set_item(
                    "system_prompt",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.system_prompt)?;
                        params.set_item("tooltip", "The system prompt (or instruction) that guides the model's behavior. This is typically a high-level directiv")?;
                        params
                    }),
                )?;

                required.set_item(
                    "user_prompt",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.user_prompt)?;
                        params.set_item("tooltip", "The user-provided input or query to the model. This is the dynamic part of the prompt that changes with each interaction. ")?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_ctx",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.n_ctx)?;
                        params.set_item("min", 256)?;
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
                            "Seed for random number generation (default: 0). Set to a fixed value for reproducible outputs.",
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
                            "Number of GPU layers to offload (default: 0, CPU-only).",
                        )?;
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
                        params.set_item("forceInput", true)?;
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
    #[pyo3(name = "execute", signature = (images, model_path, mmproj_path, system_prompt, user_prompt, n_ctx, n_predict, main_gpu, n_gpu_layers, **kwargs))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        images: Bound<'py, PyAny>,
        model_path: String,
        mmproj_path: String,
        system_prompt: String,
        user_prompt: String,
        n_ctx: u32,
        n_predict: i32,
        main_gpu: i32,
        n_gpu_layers: u32,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyAny>, String)> {
        let params = self
            .options_parser(
                &images,
                model_path,
                mmproj_path,
                system_prompt,
                user_prompt,
                n_ctx,
                n_predict,
                main_gpu,
                n_gpu_layers,
                kwargs,
            )
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("parameters error, {e}")))?;

        let results = self.generate(images, &params);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("LlamaCppVision error, {e}");
                if let Err(e) = self.send_error(py, "LlamaCppVision".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LlamaCppVision {
    /// Parse the options from the parameters.
    ///
    /// images: [batch, height, width, channels]
    #[allow(clippy::too_many_arguments)]
    fn options_parser<'py>(
        &self,
        images: &Bound<'py, PyAny>,
        model_path: String,
        mmproj_path: String,
        system_prompt: String,
        user_prompt: String,
        n_ctx: u32,
        n_predict: i32,
        main_gpu: i32,
        n_gpu_layers: u32,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<LlamaCppOptions, Error> {
        let kwargs =
            kwargs.ok_or_else(|| Error::InvalidParameter("parameters is required".to_string()))?;
        let mut options: LlamaCppOptions = depythonize(&kwargs)?;

        options.model_path = model_path;
        options.mmproj_path = mmproj_path;
        options.system_prompt = system_prompt;
        options.user_prompt = user_prompt;
        options.n_ctx = n_ctx;
        options.n_predict = n_predict;
        options.main_gpu = main_gpu;
        options.n_gpu_layers = n_gpu_layers;

        let images_shape = images.call_method0("shape")?.extract::<usize>()?;
        if images_shape != 4 {
            return Err(Error::InvalidTensorShape(format!(
                "Expected [batch, height, width, channels] tensor, images shape: {images_shape}"
            )));
        }

        // image tensors -> Vec<tensor>
        // tensors.select(dim=0, index=0).numpy().tobytes()
        let mut image_vec = Vec::with_capacity(images_shape);
        for i in 0..images_shape {
            let image = images
                .call_method1("select", (0, i))?
                .call_method0("numpy")?
                .call_method0("tobytes")?
                .extract::<Vec<u8>>()?;
            image_vec.push(image);
        }
        options.images = image_vec;

        info!("options: {:?}", options);

        Ok(options)
    }
}

impl LlamaCppVision {
    pub fn generate<'py>(
        &mut self,
        images: Bound<'py, PyAny>,
        params: &LlamaCppOptions,
    ) -> Result<(Bound<'py, PyAny>, String), Error> {
        // llama.cpp logs
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(params.verbose));

        // Initialize backend
        let backend = LlamaBackend::init()?;

        // Load model
        let model = self.load_model(&backend, params)?;

        // Load context
        let mut context = self.load_context(&model, &backend, params)?;

        // Load sampler
        let mut sampler = self.load_sampler(params)?;

        // Create the MTMD context
        let mut ctx = LlamaCppMtmdContext::new(&model, params)?;
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
        // for image_path in &params.images {
        //     info!("Loading image: {image_path}");
        //     ctx.load_media_file(image_path)?;
        // }
        // for audio_path in &params.audio {
        //     ctx.load_media_file(audio_path)?;
        // }

        ctx.load_image(&params.images[0])?;

        // Create user message
        let msgs = vec![
            LlamaChatMessage::new("system".to_string(), params.system_prompt.clone())?,
            LlamaChatMessage::new("user".to_string(), user_prompt)?,
        ];

        info!("Evaluating message: {msgs:?}");

        // Evaluate the message (prefill)
        ctx.eval_message(&model, &mut context, msgs, true, params.n_batch as i32)?;

        // Generate response (decode)
        ctx.generate_response(&model, &mut context, &mut sampler, params.n_predict)?;

        Ok((images, String::new()))
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
