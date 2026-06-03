//! ParamPort - 参数分发端点（多动态输出版）
//!
//! 上游必须连接一个 ParamHub.params（SR_PARAMS）输出。
//!
//! 节点形态：
//!   - 1 个 `params` 必选输入（SR_PARAMS）
//!   - 1 个 hidden widget `out_labels_json`：JSON 数组，由前端写入，
//!     索引对应输出端口位置，元素为 hub bundle 中的 label，例如：
//!     - `["MODEL","CLIP","VAE"]`
//!   - N 个预留 any 输出（与 ParamHub MAX_PARAM_SLOTS 对齐）：前端按需 hide/show
//!     未启用的输出位永远不会被下游连接，因此即使返回 None 也不会出错。
//!
//! 设计要点：
//!   - 通过预留固定数量输出位 + 前端动态 rename/hide 实现"动态输出"，
//!     这是 ComfyUI 社区惯用做法（参考 cg-use-everywhere、rgthree）
//!   - 跨电脑导入时，前端会在 onConfigure 时根据 widgets_values 还原
//!     输出端口结构与命名

use log::error;
use pyo3::{
    Bound, IntoPyObject, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyTuple, PyType},
};

use crate::{
    core::{
        category::CATEGORY_UTILS,
        node_base::{InputSpec, InputType},
    },
    error::Error,
    utils::param_hub::{MAX_PARAM_SLOTS, SR_PARAMS_TYPE},
    wrapper::comfyui::{PromptServer, types::any_type},
};

/// 输出端口数量上限（与 ParamHub 的输入槽数对齐）
pub const MAX_PARAM_PORT_OUTPUTS: usize = MAX_PARAM_SLOTS;

/// 参数分发端点
#[pyclass(subclass)]
pub struct ParamPort {}

impl PromptServer for ParamPort {}

#[pymethods]
impl ParamPort {
    #[new]
    fn new() -> Self {
        Self {}
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    /// 返回 N 个 any 类型输出（用 Python tuple 构造，避免在 Rust 写 32 元元组）
    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        let mut items: Vec<Bound<'_, PyAny>> = Vec::with_capacity(MAX_PARAM_PORT_OUTPUTS);
        for _ in 0..MAX_PARAM_PORT_OUTPUTS {
            items.push(any_type(py)?);
        }
        PyTuple::new(py, items)
    }

    /// 输出名 out_1..out_N（前端会改写为 hub label）
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        let names: Vec<String> = (1..=MAX_PARAM_PORT_OUTPUTS)
            .map(|i| format!("out_{i}"))
            .collect();
        PyTuple::new(py, names)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        let flags: Vec<bool> = vec![false; MAX_PARAM_PORT_OUTPUTS];
        PyTuple::new(py, flags)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_UTILS;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Fan-out a SR_PARAMS bundle from upstream Sr Param Hub into multiple typed outputs. The set of outputs is driven by the upstream Hub: each connected input on the Hub becomes an output here."
    }

    #[classattr]
    #[pyo3(name = "EXPERIMENTAL")]
    fn experimental() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        InputSpec::new()
            // 必须从 ParamHub 连过来
            .with_required(
                "params",
                InputType::custom(SR_PARAMS_TYPE)
                    .force_input(true)
                    .tooltip("Connect from Sr Param Hub.params"),
            )
            // 隐藏 widget：前端写入 JSON 数组（按输出端口索引顺序排列的 label 列表）
            .with_required(
                "out_labels_json",
                InputType::string().default("[]").multiline(true).tooltip(
                    "Managed by frontend; ordered labels mapping output port -> hub label",
                ),
            )
            .with_hidden("unique_id", InputType::custom("UNIQUE_ID"))
            .build()
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        params: Bound<'py, PyAny>,
        out_labels_json: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        match Self::distribute(py, params, out_labels_json) {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("ParamPort error, {e}");
                if let Err(se) = self.send_error(py, "ParamPort".to_string(), e.to_string()) {
                    error!("send error failed, {se}");
                }
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

/// 前端写入的端口映射 JSON 解析
fn parse_out_labels(raw: &str) -> Result<Vec<String>, Error> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_str::<Vec<String>>(trimmed)
        .map_err(|e| Error::InvalidInput(format!("invalid out_labels_json: {e}")))
}

impl ParamPort {
    /// 将 bundle 按 out_labels_json 拆分成 N 元 tuple
    fn distribute<'py>(
        py: Python<'py>,
        params: Bound<'py, PyAny>,
        out_labels_json: &str,
    ) -> Result<Bound<'py, PyTuple>, Error> {
        let labels = parse_out_labels(out_labels_json)?;

        // 上游 bundle 必为 dict（由 ParamHub 构造）
        let dict: Bound<'py, PyDict> = params.cast_into().map_err(|_| {
            Error::InvalidInput(
                "upstream `params` is not a SR_PARAMS dict (broken pipeline)".to_string(),
            )
        })?;

        let mut outputs: Vec<Bound<'py, PyAny>> = Vec::with_capacity(MAX_PARAM_PORT_OUTPUTS);
        for i in 0..MAX_PARAM_PORT_OUTPUTS {
            let label_opt = labels.get(i).map(String::as_str).unwrap_or("");
            if label_opt.is_empty() {
                outputs.push(
                    py.None()
                        .into_pyobject(py)
                        .map_err(|e| Error::TypeConversion(format!("None -> py: {e}")))?
                        .into_any(),
                );
                continue;
            }
            if dict.contains(label_opt).unwrap_or(false) {
                let v = dict.get_item(label_opt).map_err(|e| {
                    Error::InvalidInput(format!("failed to read parameter '{label_opt}': {e}"))
                })?;
                outputs.push(v);
            } else {
                outputs.push(
                    py.None()
                        .into_pyobject(py)
                        .map_err(|e| Error::TypeConversion(format!("None -> py: {e}")))?
                        .into_any(),
                );
            }
        }

        PyTuple::new(py, outputs).map_err(|e| Error::TypeConversion(format!("tuple: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyDict, PyDictMethods};

    #[test]
    fn test_parse_out_labels_empty() {
        assert!(parse_out_labels("").unwrap().is_empty());
        assert!(parse_out_labels("[]").unwrap().is_empty());
    }

    #[test]
    fn test_parse_out_labels_normal() {
        let v = parse_out_labels(r#"["MODEL","CLIP","VAE"]"#).unwrap();
        assert_eq!(v, vec!["MODEL", "CLIP", "VAE"]);
    }

    #[test]
    fn test_parse_out_labels_invalid() {
        assert!(parse_out_labels("not json").is_err());
    }

    #[test]
    fn test_distribute_picks_in_order() {
        Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("MODEL", "model_obj").unwrap();
            d.set_item("CLIP", 42i64).unwrap();
            d.set_item("VAE", "vae_obj").unwrap();
            let raw = r#"["VAE","CLIP","MODEL"]"#;
            let t = ParamPort::distribute(py, d.into_any(), raw).unwrap();
            assert_eq!(t.len().unwrap(), MAX_PARAM_PORT_OUTPUTS);
            let v0: String = t.get_item(0).unwrap().extract().unwrap();
            let v1: i64 = t.get_item(1).unwrap().extract().unwrap();
            let v2: String = t.get_item(2).unwrap().extract().unwrap();
            assert_eq!(v0, "vae_obj");
            assert_eq!(v1, 42);
            assert_eq!(v2, "model_obj");
            // 第 4 位之后为 None
            assert!(t.get_item(3).unwrap().is_none());
        });
    }

    #[test]
    fn test_distribute_missing_label_is_none() {
        Python::initialize();
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("MODEL", "m").unwrap();
            let raw = r#"["MODEL","CLIP"]"#;
            let t = ParamPort::distribute(py, d.into_any(), raw).unwrap();
            let v0: String = t.get_item(0).unwrap().extract().unwrap();
            assert_eq!(v0, "m");
            assert!(t.get_item(1).unwrap().is_none());
        });
    }
}
