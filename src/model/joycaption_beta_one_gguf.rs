//! joycaption-beta-one-GGUF
//!
//! rust 语言的实现
//! 原项目: https://github.com/fpgaminer/joycaption_comfyui
//! 引用: https://github.com/judian17/ComfyUI-joycaption-beta-one-GGUF

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{ggml_time_us, send_logs_to_tracing, LogOptions};

fn mode() {}
