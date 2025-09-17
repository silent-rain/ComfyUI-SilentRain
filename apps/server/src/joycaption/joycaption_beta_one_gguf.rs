//! JoyCaption Beta One GGUF
//!
//! rust 语言的实现
//! 原项目: https://github.com/fpgaminer/joycaption_comfyui
//! 引用: https://github.com/judian17/ComfyUI-joycaption-beta-one-GGUF

use std::sync::RwLock;

use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};

use crate::{
    core::{category::CATEGORY_JOY_CAPTION, utils::image::tensor_to_raw_data},
    error::Error,
    joycaption::joy_caption_ollama_prompter::{
        caption_length_choices, caption_length_map, caption_type_map, system_prompt, user_prompt,
    },
    llama_cpp::{LlamaCppOptions, LlamaCppPipeline},
    wrapper::{
        comfy::folder_paths::FolderPaths,
        comfyui::{
            types::{NODE_BOOLEAN, NODE_FLOAT, NODE_IMAGE, NODE_INT, NODE_SEED_MAX, NODE_STRING},
            PromptServer,
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

/// JoyCaption Beta One GGUF
#[pyclass(subclass)]
pub struct JoyCaptionnBetaOneGGUF {}

impl PromptServer for JoyCaptionnBetaOneGGUF {}

#[pymethods]
impl JoyCaptionnBetaOneGGUF {
    #[new]
    fn new() -> Self {
        Self {}
    }

    #[classattr]
    #[pyo3(name = "EXPERIMENTAL")]
    fn experimental() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str) {
        (NODE_STRING, NODE_STRING)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("query", "caption")
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
    const CATEGORY: &'static str = CATEGORY_JOY_CAPTION;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "JoyCaption GGUF, the open source caption generation model"
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

                required.set_item(
                    "image",
                    (NODE_IMAGE, {
                        let image = PyDict::new(py);
                        image.set_item("forceInput", true)?;
                        image
                    }),
                )?;

                // 获取模型列表
                let model_list = Self::get_model_list();
                let mmproj_model_list = Self::get_mmproj_model_list();

                required.set_item(
                    "gguf_model",
                    (model_list, {
                        let gguf_model = PyDict::new(py);
                        gguf_model.set_item("tooltip", "model file path")?;
                        gguf_model
                    }),
                )?;

                required.set_item(
                    "mmproj_file",
                    (mmproj_model_list, {
                        let mmproj_file = PyDict::new(py);
                        mmproj_file.set_item("tooltip", "mmproj model file path")?;
                        mmproj_file
                    }),
                )?;

                required.set_item(
                    "main_gpu",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 0)?;
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
                        let n_gpu_layers = PyDict::new(py);
                        n_gpu_layers.set_item("default", 0)?;
                        n_gpu_layers.set_item("min", 0)?;
                        n_gpu_layers.set_item("max", 1000)?;
                        n_gpu_layers.set_item("step", 1)?;
                        n_gpu_layers
                    }),
                )?;

                required.set_item(
                    "n_ctx",
                    (NODE_INT, {
                        let n_ctx = PyDict::new(py);
                        n_ctx.set_item("default", 2048)?;
                        n_ctx.set_item("min", 512)?;
                        n_ctx.set_item("max", 8192)?;
                        n_ctx.set_item("step", 1)?;
                        n_ctx
                    }),
                )?;

                let caption_type_map = caption_type_map();
                let caption_type_keys = caption_type_map.keys().cloned().collect::<Vec<_>>();
                required.set_item(
                    "caption_type",
                    (caption_type_keys, {
                        let caption_type = PyDict::new(py);
                        caption_type.set_item("default", "Descriptive (Casual)")?;
                        caption_type
                    }),
                )?;

                let caption_lengths = caption_length_choices();
                required.set_item(
                    "caption_length",
                    (caption_lengths, {
                        let caption_length = PyDict::new(py);
                        caption_length.set_item("default", "medium-length")?;
                        caption_length
                    }),
                )?;

                required.set_item(
                    "max_new_tokens",
                    (NODE_INT, {
                        let max_new_tokens = PyDict::new(py);
                        max_new_tokens.set_item("default", 512)?;
                        max_new_tokens.set_item("min", 0)?;
                        max_new_tokens.set_item("max", 4096)?;
                        max_new_tokens.set_item("step", 1)?;
                        max_new_tokens
                    }),
                )?;

                required.set_item(
                    "temperature",
                    (NODE_FLOAT, {
                        let temperature = PyDict::new(py);
                        temperature.set_item("default", 0.6)?;
                        temperature.set_item("min", 0.0)?;
                        temperature.set_item("max", 2.0)?;
                        temperature.set_item("step", 0.05)?;
                        temperature
                    }),
                )?;

                required.set_item(
                    "top_p",
                    (NODE_FLOAT, {
                        let top_p = PyDict::new(py);
                        top_p.set_item("default", 0.9)?;
                        top_p.set_item("min", 0.0)?;
                        top_p.set_item("max", 1.0)?;
                        top_p.set_item("step", 0.01)?;
                        top_p
                    }),
                )?;

                required.set_item(
                    "top_k",
                    (NODE_INT, {
                        let top_k = PyDict::new(py);
                        top_k.set_item("default", 40)?;
                        top_k.set_item("min", 0)?;
                        top_k.set_item("max", 100)?;
                        top_k.set_item("step", 1)?;
                        top_k
                    }),
                )?;

                required.set_item(
                    "seed",
                    (NODE_INT, {
                        let seed = PyDict::new(py);
                        seed.set_item("default", -1)?;
                        seed.set_item("min", -1)?;
                        seed.set_item("max", NODE_SEED_MAX)?;
                        seed.set_item("step", 1)?;
                        seed
                    }),
                )?;

                required.set_item(
                    "keep_context",
                    (NODE_BOOLEAN, {
                        let keep_context = PyDict::new(py);
                        keep_context.set_item("default", false)?;
                        keep_context.set_item("tooltip", "Keep the context between requests")?;
                        keep_context
                    }),
                )?;

                required.set_item(
                    "cache_model",
                    (NODE_BOOLEAN, {
                        let cache_model = PyDict::new(py);
                        cache_model.set_item("default", false)?;
                        cache_model
                            .set_item("tooltip", "Whether to cache model between requests")?;
                        cache_model
                    }),
                )?;

                required
            })?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);
                optional.set_item(
                    "extra_options",
                    (NODE_STRING, {
                        let extra_options = PyDict::new(py);
                        extra_options.set_item("forceInput", true)?;
                        extra_options
                    }),
                )?;

                optional
            })?;
            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        images: Bound<'py, PyAny>,
        gguf_model: &str,
        mmproj_file: &str,
        main_gpu: i32,
        n_gpu_layers: u32,
        n_ctx: u32,
        caption_type: &str,
        caption_length: &str,
        max_new_tokens: i32,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        seed: i32,
        keep_context: bool,
        cache_model: bool,
        extra_options: Option<Vec<String>>,
    ) -> PyResult<(Bound<'py, PyAny>, Vec<String>)> {
        let params = self
            .options_parser(
                &images,
                gguf_model,
                mmproj_file,
                main_gpu,
                n_gpu_layers,
                n_ctx,
                caption_type,
                caption_length,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                keep_context,
                cache_model,
                extra_options,
            )
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("parameters error, {e}")))?;

        let results = self.generate(images, &params, caption_length, caption_type);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("JoyCaptionnBetaOneGGUF error, {e}");
                if let Err(e) =
                    self.send_error(py, "JoyCaptionnBetaOneGGUF".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl JoyCaptionnBetaOneGGUF {
    /// Parse the options from the parameters.
    ///
    /// images: [batch, height, width, channels]
    #[allow(clippy::too_many_arguments)]
    fn options_parser<'py>(
        &self,
        images: &Bound<'py, PyAny>,
        gguf_model: &str,
        mmproj_file: &str,
        main_gpu: i32,
        n_gpu_layers: u32,
        n_ctx: u32,
        caption_type: &str,
        caption_length: &str,
        max_new_tokens: i32,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        seed: i32,
        keep_context: bool,
        cache_model: bool,
        extra_options: Option<Vec<String>>,
    ) -> Result<LlamaCppOptions, Error> {
        // 校验输入张量的形状
        let images_shape = images.getattr("shape")?.extract::<Vec<usize>>()?;
        if images_shape.len() != 4 {
            return Err(Error::InvalidTensorShape(format!(
                "Expected [batch, height, width, channels] tensor, images shape: {}",
                images_shape.len()
            )));
        }

        // 提示词处理
        let (system_prompt, user_prompt) = {
            let caption_length_map = caption_length_map();

            // system prompt
            let mut system_prompt = system_prompt(caption_type).to_string();
            // Add length enforcement to system prompt
            if caption_length.parse::<i32>().is_ok() {
                system_prompt +=
                    format!("\nIMPORTANT: Your response MUST NOT exceed {caption_length} words.")
                        .as_str();
            } else if let Some(length) = caption_length_map.get(caption_length) {
                system_prompt +=
                    format!("\nIMPORTANT: Keep your response approximately {length} words.")
                        .as_str();
            }

            // user prompt
            let user_prompt = user_prompt(caption_type, caption_length, extra_options);

            (system_prompt, user_prompt)
        };

        let options = LlamaCppOptions {
            model_path: gguf_model.to_string(),
            mmproj_path: mmproj_file.to_string(),
            system_prompt,
            user_prompt,
            n_ctx,
            n_predict: max_new_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            main_gpu,
            n_gpu_layers,
            keep_context,
            cache_model,
            ..Default::default()
        };

        Ok(options)
    }

    /// 获取模型列表
    fn get_model_list() -> Vec<String> {
        let llm_list = FolderPaths::default().get_filename_list("LLM");
        let text_encoders_list = FolderPaths::default().get_filename_list("llava_gguf");

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
        let text_encoders_list = FolderPaths::default().get_filename_list("llava_gguf");

        [llm_list, text_encoders_list]
            .concat()
            .iter()
            .filter(|v| v.ends_with(".gguf"))
            .filter(|v| v.contains("mmproj"))
            .cloned()
            .collect::<Vec<String>>()
    }
}

impl JoyCaptionnBetaOneGGUF {
    /// 推理
    #[allow(clippy::too_many_arguments)]
    fn generate<'py>(
        &self,
        images: Bound<'py, PyAny>,
        params: &LlamaCppOptions,
        caption_type: &str,
        caption_length: &str,
    ) -> Result<(Bound<'py, PyAny>, Vec<String>), Error> {
        let pipeline = get_pipeline()?;

        let mut pipeline = match pipeline {
            Some(pipeline) => pipeline,
            None => LlamaCppPipeline::new(params)?,
        };

        // 有缓存时，如果参数更新则重新加载模型
        if params.cache_model {
            pipeline.update_model(params)?;
            pipeline.update_mtmd_context(params)?;
        }

        let image_raw_datas = tensor_to_raw_data(&images)?;
        let mut captions = Vec::new();

        for image_raw_data in image_raw_datas {
            let response = pipeline.generate_vision(params, &image_raw_data)?;

            // response constraint length
            let response =
                Self::response_constraint_length(caption_length, caption_type, response.clone());

            captions.push(response);
        }

        // 保存模型
        if params.cache_model {
            set_pipeline(pipeline)?;
        }

        Ok((images, captions))
    }

    /// response constraint length
    fn response_constraint_length(
        caption_length: &str,
        caption_type: &str,
        caption: String,
    ) -> String {
        // Clean up the output
        let mut caption = caption
            .replace("ASSISTANT:", "")
            .replace("Human:", "")
            .trim()
            .to_string();

        if ["booru", "danbooru", "e621", "rule34"]
            .to_vec()
            .iter()
            .map(|v| v.to_lowercase())
            .any(|v| v.contains(caption_type))
        {
            // Keep only the comma-separated tags, remove any explanatory text
            let tags = caption
                .split(",")
                .collect::<Vec<_>>()
                .into_iter()
                .map(|v| v.trim())
                .collect::<Vec<_>>();

            caption = tags.join(", ");
        }

        let caption_length_map = caption_length_map();
        let target_length = if let Ok(length) = caption_length.parse::<usize>() {
            length
        } else if let Some(caption_length) = caption_length_map.get(caption_length) {
            caption_length.parse::<usize>().unwrap_or(100)
        } else {
            100 // default value, target length is medium-length
        };

        let words = caption
            .split_whitespace()
            .collect::<Vec<_>>()
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        if words.len() > target_length {
            caption = words[0..target_length].join(" ");
        }

        caption
    }
}
