//! JoyCaption Beta One GGUF
//!
//! rust 语言的实现
//! 原项目: https://github.com/fpgaminer/joycaption_comfyui
//! 引用: https://github.com/judian17/ComfyUI-joycaption-beta-one-GGUF

use candle_core::{Device, Tensor};
use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_JOY_CAPTION,
    error::Error,
    joycaption::joy_caption_ollama_prompter::{caption_length_choices, caption_type_map},
    wrapper::{
        comfyui::{
            types::{NODE_BOOLEAN, NODE_FLOAT, NODE_INT, NODE_SEED_MAX, NODE_STRING},
            PromptServer,
        },
        torch::tensor::TensorWrapper,
    },
};

/// JoyCaption Beta One GGUF
#[pyclass(subclass)]
pub struct JoyCaptionnBetaOneGGUF {
    device: Device,
}

impl PromptServer for JoyCaptionnBetaOneGGUF {}

#[pymethods]
impl JoyCaptionnBetaOneGGUF {
    #[new]
    fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
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
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);

                required.set_item(
                    "image",
                    ("IMAGE", {
                        let image = PyDict::new(py);
                        image.set_item("forceInput", true)?;
                        image
                    }),
                )?;

                required.set_item(
                    "gguf_model",
                    (NODE_STRING, {
                        let gguf_model = PyDict::new(py);
                        gguf_model.set_item("tooltip", "model file path")?;
                        gguf_model
                    }),
                )?;

                required.set_item(
                    "mmproj_file",
                    (NODE_STRING, {
                        let mmproj_file = PyDict::new(py);
                        mmproj_file.set_item("tooltip", "mmproj model file path")?;
                        mmproj_file
                    }),
                )?;

                required.set_item(
                    "n_gpu_layers",
                    (NODE_INT, {
                        let n_gpu_layers = PyDict::new(py);
                        n_gpu_layers.set_item("default", -1)?;
                        n_gpu_layers.set_item("min", -1)?;
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
                )?; // Seed input remains, but not used in model_key for now

                required.set_item(
                    "unload_after_generate",
                    (NODE_BOOLEAN, {
                        let unload_after_generate = PyDict::new(py);
                        unload_after_generate.set_item("default", false)?;
                        unload_after_generate
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
        image: Bound<'py, PyAny>,
        gguf_model: &str,
        mmproj_file: &str,
        n_gpu_layers: i32,
        n_ctx: i32,
        caption_type: &str,
        caption_length: &str,
        max_new_tokens: i32,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        seed: i32,
        unload_after_generate: bool,
        extra_options: Option<Vec<String>>,
    ) -> PyResult<(String, String)> {
        let image = TensorWrapper::<f32>::new(&image, &self.device)?.into_tensor();

        let results = self.generate(
            image,
            gguf_model,
            mmproj_file,
            n_gpu_layers,
            n_ctx,
            caption_type,
            caption_length,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            unload_after_generate,
            extra_options,
        );

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("JoyCaptionGGUF error, {e}");
                if let Err(e) = self.send_error(py, "JoyCaptionGGUF".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl JoyCaptionnBetaOneGGUF {
    /// 推理
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        image: Tensor,
        gguf_model: &str,
        mmproj_file: &str,
        n_gpu_layers: i32,
        n_ctx: i32,
        caption_type: &str,
        caption_length: &str,
        max_new_tokens: i32,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        seed: i32,
        unload_after_generate: bool,
        extra_options: Option<Vec<String>>,
    ) -> Result<(String, String), Error> {
        Ok(("".to_string(), "".to_string()))
    }
}
