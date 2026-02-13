//! Rust 的全局缓存清理 v2
//!
//! v2 版本改进：
//! - 支持清理特定模型（通过 model_handle）
//! - 支持清理所有缓存
//! - 提供详细的清理报告
//! - 显示当前缓存统计信息

use std::sync::Arc;

use log::{error, info};
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyDict, PyType},
};

use llama_flow::{CacheManager, ChatHistoryManager, chat_history};

use crate::{
    core::{
        category::CATEGORY_LLAMA_CPP,
        node_base::{InputSpec, InputType},
    },
    error::Error,
    wrapper::comfyui::{PromptServer, types::NODE_STRING},
};

/// LlamaCpp Purge VRAM v2
///
/// # 输入参数
/// - `model_handle`: 可选，指定要清理的模型缓存 key
/// - `clear_all_models`: 是否清理所有模型缓存
/// - `clear_all_contexts`: 是否清理所有对话上下文
/// - `clear_chat_history`: 是否清理聊天历史
///
/// # 输出
/// - `report`: JSON 格式的清理报告
#[pyclass(subclass)]
pub struct LlamaCppPurgeVramv2 {
    cache: Arc<CacheManager>,
    history_cache: Arc<ChatHistoryManager>,
}

impl PromptServer for LlamaCppPurgeVramv2 {}

#[pymethods]
impl LlamaCppPurgeVramv2 {
    #[new]
    fn new() -> Self {
        Self {
            cache: CacheManager::global(),
            history_cache: chat_history(),
        }
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_NODE")]
    fn output_node() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_STRING,) // JSON 清理报告
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("report",)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LLAMA_CPP;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "llama.cpp VRAM cleanup v2 - Selective cache clearing with reports"
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
        InputSpec::new()
            .with_required(
                "trigger",
                InputType::any().tooltip("Any input to trigger cleanup"),
            )
            .with_required(
                "clear_models",
                InputType::bool().tooltip("Clear all model caches"),
            )
            .with_required(
                "clear_contexts",
                InputType::bool().tooltip("Clear all conversation contexts"),
            )
            .with_required("clear_all", InputType::bool().tooltip("Clear all caches"))
            .build()
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        trigger: Bound<'py, PyAny>,
        clear_models: bool,
        clear_contexts: bool,
        clear_all: bool,
    ) -> PyResult<(String,)> {
        let result = self.purge_vram(py, trigger, clear_models, clear_contexts, clear_all);

        match result {
            Ok(report) => Ok((report,)),
            Err(e) => {
                error!("LlamaCppPurgeVramv2 error: {e}");
                if let Err(e) =
                    self.send_error(py, "LlamaCppPurgeVramv2".to_string(), e.to_string())
                {
                    error!("send error failed: {e}");
                }
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LlamaCppPurgeVramv2 {
    /// 执行 VRAM 清理
    fn purge_vram<'py>(
        &self,
        _py: Python<'py>,
        _trigger: Bound<'py, PyAny>,
        clear_models: bool,
        clear_contexts: bool,
        clear_all: bool,
    ) -> Result<String, Error> {
        let mut report = PurgeReport::default();

        // 获取清理前的缓存状态
        let model_keys: Vec<String> = self.cache.get_keys()?;
        let context_keys: Vec<String> = self.history_cache.list_sessions();
        report.total_models = model_keys.len();
        report.total_contexts = context_keys.len();

        info!(
            "Starting VRAM cleanup v2, {} models, {} contexts",
            report.total_models, report.total_contexts,
        );

        // 清理所有模型
        if clear_models {
            self.cache.clear()?;
            let model_keys: Vec<String> = self.cache.get_keys()?;

            report.models_cleared = model_keys.len();
            info!("Cleared {} model caches", report.models_cleared);
        }

        // 清理所有上下文
        if clear_contexts {
            self.history_cache.clear();
            let context_keys: Vec<String> = self.history_cache.list_sessions();

            report.contexts_cleared = context_keys.len();
            info!("Cleared {} context caches", report.contexts_cleared);
        }

        // 清理所有
        if clear_all {
            self.cache.clear()?;
            self.history_cache.clear();

            let model_keys = self.cache.get_keys()?;
            let context_keys: Vec<String> = self.history_cache.list_sessions();

            let all_keys = context_keys.len() + model_keys.len();
            info!("Cleared {} caches", all_keys);
        }

        info!(
            "VRAM cleanup v2 complete: {} keys cleared",
            report.models_cleared + report.contexts_cleared
        );

        // 生成 JSON 报告
        let json_report = serde_json::to_string_pretty(&report).map_err(Error::SerdeJsonError)?;

        Ok(json_report)
    }
}

/// 清理报告
#[derive(Default, serde::Serialize)]
struct PurgeReport {
    pub total_contexts: usize,
    pub total_models: usize,
    pub models_cleared: usize,
    pub contexts_cleared: usize,
}
