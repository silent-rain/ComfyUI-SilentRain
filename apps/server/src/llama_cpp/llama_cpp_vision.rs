//! llama.cpp vision

use std::sync::RwLock;

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use pythonize::depythonize;

use crate::{
    core::{category::CATEGORY_LLAMA_CPP, utils::image::tensor_to_raw_data},
    error::Error,
    llama_cpp::{LlamaCppOptions, LlamaCppPipeline, ModelManager},
    wrapper::{
        comfy::folder_paths::FolderPaths,
        comfyui::{
            PromptServer,
            types::{
                NODE_BOOLEAN, NODE_IMAGE, NODE_INT, NODE_INT_MAX, NODE_LLAMA_CPP_OPTIONS,
                NODE_SEED_MAX, NODE_STRING,
            },
        },
    },
};

static LLAMA_CPP_PIPELINE: RwLock<Option<LlamaCppPipeline>> = RwLock::new(None);

// 设置值
fn set_pipeline(pipeline: LlamaCppPipeline) -> Result<(), Error> {
    let mut w = LLAMA_CPP_PIPELINE
        .write()
        .map_err(|_e| Error::LockError("LLAMA_CPP_PIPELINE lock error".to_string()))?;
    *w = Some(pipeline);

    Ok(())
}

// 读取值
fn get_pipeline() -> Result<Option<LlamaCppPipeline>, Error> {
    let mut w = LLAMA_CPP_PIPELINE
        .write()
        .map_err(|_e| Error::LockError("LLAMA_CPP_PIPELINE lock error".to_string()))?;

    let pipeline = w.take();
    Ok(pipeline)
}

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


                // 获取模型列表
                let model_list = Self::get_model_list();
                let mmproj_model_list = Self::get_mmproj_model_list();

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
                    "mmproj_path",
                    (mmproj_model_list, {
                        let params = PyDict::new(py);
                        // params.set_item("default", options.mmproj_path)?;
                        params.set_item("tooltip", "mmproj model file path")?;
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
                    "n_ctx",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.n_ctx)?;
                        params.set_item("min", 0)?;
                        params.set_item("max", NODE_INT_MAX)?;
                        params.set_item("step", 1)?;
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
                        params.set_item("step", 1)?;
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
                        params.set_item("max", 10000)?;
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

                required.set_item(
                    "cache_model",
                    (NODE_BOOLEAN, {
                        let params = PyDict::new(py);
                        params.set_item("default", options.cache_model)?;
                        params.set_item("tooltip", "Whether to cache model between requests")?;
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
    #[pyo3(name = "execute", signature = (images, model_path, mmproj_path, system_prompt, user_prompt, n_ctx, n_predict, seed, main_gpu, n_gpu_layers, keep_context, cache_model, extra_options=None))]
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
        seed: i32,
        main_gpu: i32,
        n_gpu_layers: u32,
        keep_context: bool,
        cache_model: bool,
        extra_options: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyAny>, Vec<String>)> {
        let params = self
            .options_parser(
                &images,
                model_path,
                mmproj_path,
                system_prompt,
                user_prompt,
                n_ctx,
                n_predict,
                seed,
                main_gpu,
                n_gpu_layers,
                keep_context,
                cache_model,
                extra_options,
            )
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("parameters error, {e}")))?;

        let results = self.generate(py, images, &params);

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
        seed: i32,
        main_gpu: i32,
        n_gpu_layers: u32,
        keep_context: bool,
        cache_model: bool,
        extra_options: Option<Bound<'py, PyDict>>,
    ) -> Result<LlamaCppOptions, Error> {
        // 校验输入张量的形状
        let images_shape = images.getattr("shape")?.extract::<Vec<usize>>()?;
        if images_shape.len() != 4 {
            return Err(Error::InvalidTensorShape(format!(
                "Expected [batch, height, width, channels] tensor, images shape: {}",
                images_shape.len()
            )));
        }

        let mut options = LlamaCppOptions::default();

        if let Some(extra_options) = extra_options {
            options = depythonize(&extra_options)?;
        }

        options.model_path = model_path;
        options.mmproj_path = mmproj_path;
        options.system_prompt = system_prompt;
        options.user_prompt = user_prompt;
        options.n_ctx = n_ctx;
        options.n_predict = n_predict;
        options.seed = seed;
        options.main_gpu = main_gpu;
        options.n_gpu_layers = n_gpu_layers;
        options.keep_context = keep_context;
        options.cache_model = cache_model;
        options.model_cache_key = "model_llama_cpp_vision_cache_key".to_string();
        options.model_mtmd_context_cache_key =
            "model_mtmd_context_llama_cpp_vision_cache_key".to_string();

        Ok(options)
    }

    /// 获取模型列表
    fn get_model_list() -> Vec<String> {
        let llm_list = FolderPaths::default().get_filename_list("LLM");
        let text_encoders_list = FolderPaths::default().get_filename_list("text_encoders");

        [llm_list, text_encoders_list]
            .concat()
            .iter()
            .filter(|v| v.ends_with(".gguf"))
            .cloned()
            .collect::<Vec<String>>()
    }

    /// 获取mmproj模型列表
    fn get_mmproj_model_list() -> Vec<String> {
        let llm_list = FolderPaths::default().get_filename_list("LLM");
        let text_encoders_list = FolderPaths::default().get_filename_list("text_encoders");

        [llm_list, text_encoders_list]
            .concat()
            .iter()
            .filter(|v| v.ends_with(".gguf"))
            .filter(|v| v.contains("mmproj"))
            .cloned()
            .collect::<Vec<String>>()
    }
}

impl LlamaCppVision {
    pub fn generate<'py>(
        &mut self,
        py: Python<'py>,
        images: Bound<'py, PyAny>,
        params: &LlamaCppOptions,
    ) -> Result<(Bound<'py, PyAny>, Vec<String>), Error> {
        let image_raw_datas = tensor_to_raw_data(&images)?;

        // 进度条
        // from comfy.utils import ProgressBar
        // pbar = ProgressBar(len(pil_images))
        // pbar.update(1)
        let pbar = py
            .import("comfy")?
            .getattr("utils")?
            .call_method1("ProgressBar", (image_raw_datas.len(),))?;

        let pipeline = get_pipeline()?;

        let mut pipeline = match pipeline {
            Some(pipeline) => pipeline,
            None => LlamaCppPipeline::new(params)?,
        };

        if !params.cache_model {
            let mut model_manager = ModelManager::global()
                .write()
                .map_err(|e| Error::LockError(e.to_string()))?;
            model_manager.remove(&params.model_cache_key);
            model_manager.remove(&params.model_mtmd_context_cache_key);
        }

        let mut captions = Vec::new();

        for image_raw_data in image_raw_datas {
            let result = pipeline.generate_vision(params, &image_raw_data)?;
            captions.push(result);
            pbar.call_method1("update", (1,))?;
        }

        // 保存模型
        if params.cache_model {
            set_pipeline(pipeline)?;
        }

        Ok((images, captions))
    }
}
