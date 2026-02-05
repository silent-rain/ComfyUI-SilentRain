//! llama.cpp options
//!
//!
//! ## media_marker
//!     - chat_template.json
//!         - https://hf-mirror.com/mlabonne/gemma-3-4b-it-abliterated/blob/main/chat_template.json
//!         - image: <start_of_image>
//!     - gguf model
//!         - tokenizer.chat_template

use llama_cpp_core::{ContexParams, sampler::SamplerConfig, types::PoolingTypeMode};
use pyo3::{
    Bound, Py, PyResult, Python, pyclass, pymethods,
    types::{PyDict, PyType},
};
use serde::{Deserialize, Serialize};

use crate::{
    core::{
        category::CATEGORY_LLAMA_CPP,
        node_base::{InputSpec, InputType},
    },
    wrapper::comfyui::{PromptServer, types::NODE_LLAMA_CPP_OPTIONS_V2},
};

/// Options for the llama_cpp library
#[pyclass(subclass)]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LlamaCppOptionsv2 {}

impl PromptServer for LlamaCppOptionsv2 {}

#[pymethods]
impl LlamaCppOptionsv2 {
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
    fn return_types() -> (&'static str,) {
        (NODE_LLAMA_CPP_OPTIONS_V2,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("options",)
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
        "llama.cpp extra options"
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        let sampler_config = SamplerConfig::default();
        let context_config = ContexParams::default();

        InputSpec::new()
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
            // ===== 重复惩罚参数 =====
            .with_required(
                "penalty_last_n",
                InputType::int()
                    .default(sampler_config.penalty_last_n)
                    .min(0)
                    .step_int(1)
                    .tooltip("Size of the sliding window for repeat penalty"),
            )
            .with_required(
                "penalty_repeat",
                InputType::float()
                    .default(sampler_config.penalty_repeat)
                    .min_float(0.0)
                    .step_float(0.01)
                    .tooltip("Repeat penalty coefficient. Higher values enforce more diversity"),
            )
            .with_required(
                "penalty_freq",
                InputType::float()
                    .default(sampler_config.penalty_freq)
                    .step_float(0.01)
                    .tooltip("Frequency penalty coefficient. Penalizes tokens based on their frequency"),
            )
            .with_required(
                "penalty_present",
                InputType::float()
                    .default(sampler_config.penalty_present)
                    .step_float(0.01)
                    .tooltip("Presence penalty coefficient. Penalizes tokens already present in the context"),
            )
            // ===== 线程和批处理参数 =====
            .with_required(
                "n_threads",
                InputType::int()
                    .default(context_config.n_threads)
                    .min(0)
                    .step_int(1)
                    .tooltip("Number of threads to use during generation. 0 = auto"),
            )
            .with_required(
                "n_batch",
                InputType::int()
                    .default(context_config.n_batch)
                    .min(1)
                    .step_int(1)
                    .tooltip("Batch size for prompt processing. Larger values may improve throughput but increase memory usage"),
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
            // ===== 嵌入池化类型 =====
            .with_required(
                "pooling_type",
                InputType::list(vec![
                    PoolingTypeMode::None.to_string(),
                    PoolingTypeMode::Mean.to_string(),
                    PoolingTypeMode::Cls.to_string(),
                    PoolingTypeMode::Last.to_string(),
                    PoolingTypeMode::Rank.to_string(),
                    PoolingTypeMode::Unspecified.to_string(),
                ])
                .default(context_config.pooling_type.to_string())
                .tooltip("Pooling type for embeddings"),
            )
            // ===== 可选参数 =====
            .with_optional(
                "media_marker",
                InputType::string()
                    .tooltip("Media marker. If not provided, the default marker will be used"),
            )
            .with_optional(
                "chat_template",
                InputType::string()
                    .default(context_config.chat_template.unwrap_or_default())
                    .tooltip("Chat template to use, default template if not provided"),
            )
            .with_optional(
                "cmoe",
                InputType::bool()
                    .default(false)
                    .label_on("Yes")
                    .label_off("No")
                    .tooltip("Keep MoE layers on CPU"),
            )
            .with_required(
                "verbose",
                InputType::bool()
                    .default(context_config.verbose)
                    .tooltip("Enables verbose logging from llama.cpp"),
            )
            .build()
    }

    #[pyo3(name = "execute", signature = (**kwargs))]
    fn execute<'py>(
        &mut self,
        _py: Python<'py>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(Option<Bound<'py, PyDict>>,)> {
        Ok((kwargs,))
    }
}
