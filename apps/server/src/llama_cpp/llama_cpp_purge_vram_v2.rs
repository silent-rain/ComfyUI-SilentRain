//! Rust 的全局缓存清理 v2
//!
//! v2 版本改进：
//! - 支持清理特定模型（通过 model_handle）
//! - 支持清理所有缓存
//! - 提供详细的清理报告
//! - 显示当前缓存统计信息

use log::{error, info};
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use std::collections::HashMap;

use llama_cpp_core::CacheManager;

use crate::{
    core::category::CATEGORY_LLAMA_CPP,
    error::Error,
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_BOOLEAN, NODE_STRING, any_type},
    },
};

/// LlamaCpp Purge VRAM v2
///
//! ### 版本差异
//! - **v1**: 清理所有缓存，无选择性
//! - **v2**:
//!   - 支持清理特定模型缓存
//!   - 支持清理对话上下文缓存
//!   - 提供详细的清理统计报告
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
    cache: std::sync::Arc<CacheManager>,
}

impl PromptServer for LlamaCppPurgeVramv2 {}

#[pymethods]
impl LlamaCppPurgeVramv2 {
    #[new]
    fn new() -> Self {
        Self {
            cache: CacheManager::global(),
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
        (NODE_STRING,)  // JSON 清理报告
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
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);

                // 触发信号
                required.set_item(
                    "trigger",
                    (any_type(py)?, {
                        let params = PyDict::new(py);
                        params.set_item("tooltip", "Any input to trigger cleanup")?;
                        params
                    }),
                )?;

                required.set_item(
                    "clear_all_models",
                    (NODE_BOOLEAN, {
                        let params = PyDict::new(py);
                        params.set_item("default", false)?;
                        params.set_item("tooltip", "Clear all model caches")?;
                        params
                    }),
                )?;

                required.set_item(
                    "clear_all_contexts",
                    (NODE_BOOLEAN, {
                        let params = PyDict::new(py);
                        params.set_item("default", false)?;
                        params.set_item("tooltip", "Clear all conversation contexts")?;
                        params
                    }),
                )?;

                required.set_item(
                    "clear_chat_history",
                    (NODE_BOOLEAN, {
                        let params = PyDict::new(py);
                        params.set_item("default", false)?;
                        params.set_item("tooltip", "Clear chat history")?;
                        params
                    }),
                )?;

                required
            })?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);

                // 可选的特定模型句柄
                optional.set_item(
                    "model_handle",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", "")?;
                        params.set_item("tooltip", "Specific model cache key to clear (empty = all)")?;
                        params
                    }),
                )?;

                optional
            })?;

            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute", signature = (trigger, clear_all_models, clear_all_contexts, clear_chat_history, model_handle=None))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        _trigger: Bound<'py, PyAny>,
        clear_all_models: bool,
        clear_all_contexts: bool,
        clear_chat_history: bool,
        model_handle: Option<String>,
    ) -> PyResult<(String,)> {
        let result = self.purge_vram(
            clear_all_models,
            clear_all_contexts,
            clear_chat_history,
            model_handle,
        );

        match result {
            Ok(report) => Ok((report,)),
            Err(e) => {
                error!("LlamaCppPurgeVramv2 error: {e}");
                if let Err(e) = self.send_error(py, "LlamaCppPurgeVramv2".to_string(), e.to_string()) {
                    error!("send error failed: {e}");
                }
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LlamaCppPurgeVramv2 {
    /// 执行 VRAM 清理
    fn purge_vram(
        &self,
        clear_all_models: bool,
        clear_all_contexts: bool,
        clear_chat_history: bool,
        model_handle: Option<String>,
    ) -> Result<String, Error> {
        let mut report = PurgeReport::default();

        // 获取清理前的缓存状态
        let keys_before = match self.cache.get_keys() {
            Ok(keys) => keys,
            Err(e) => {
                return Err(Error::LockError(e.to_string()));
            }
        };
        report.total_keys_before = keys_before.len();

        info!("Starting VRAM cleanup v2, {} keys in cache", keys_before.len());

        // 清理特定模型
        if let Some(handle) = model_handle {
            if !handle.is_empty() {
                info!("Clearing specific model: {}", handle);
                if let Ok(Some(_)) = self.cache.remove(&handle) {
                    report.models_cleared += 1;
                    report.specific_model_cleared = Some(handle);
                }
            }
        }

        // 清理所有模型
        if clear_all_models {
            let model_keys: Vec<String> = keys_before
                .iter()
                .filter(|k| k.starts_with("model_v2:") || k.starts_with("model:"))
                .cloned()
                .collect();

            for key in model_keys {
                if let Ok(Some(_)) = self.cache.remove(&key) {
                    report.models_cleared += 1;
                }
            }
            info!("Cleared {} model caches", report.models_cleared);
        }

        // 清理所有上下文
        if clear_all_contexts {
            let context_keys: Vec<String> = keys_before
                .iter()
                .filter(|k| k.starts_with("chat_ctx:") || k.starts_with("vision_ctx:"))
                .cloned()
                .collect();

            for key in context_keys {
                if let Ok(Some(_)) = self.cache.remove(&key) {
                    report.contexts_cleared += 1;
                }
            }
            info!("Cleared {} context caches", report.contexts_cleared);
        }

        // 清理聊天历史
        if clear_chat_history {
            let history_keys: Vec<String> = keys_before
                .iter()
                .filter(|k| k.starts_with("history_message_"))
                .cloned()
                .collect();

            for key in history_keys {
                if let Ok(Some(_)) = self.cache.remove(&key) {
                    report.chat_histories_cleared += 1;
                }
            }
            info!("Cleared {} chat histories", report.chat_histories_cleared);
        }

        // 获取清理后的缓存状态
        let keys_after = match self.cache.get_keys() {
            Ok(keys) => keys,
            Err(e) => {
                return Err(Error::LockError(e.to_string()));
            }
        };
        report.total_keys_after = keys_after.len();
        report.keys_cleared = report.total_keys_before.saturating_sub(report.total_keys_after);

        info!("VRAM cleanup v2 complete: {} keys cleared", report.keys_cleared);

        // 生成 JSON 报告
        let json_report = serde_json::to_string_pretty(&report)
            .map_err(|e| Error::SerdeJsonError(e))?;

        Ok(json_report)
    }
}

/// 清理报告
#[derive(Default, serde::Serialize)]
struct PurgeReport {
    pub total_keys_before: usize,
    pub total_keys_after: usize,
    pub keys_cleared: usize,
    pub models_cleared: usize,
    pub contexts_cleared: usize,
    pub chat_histories_cleared: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specific_model_cleared: Option<String>,
}
