//! ParamPort - 参数分发端点（多动态输出版）
use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyDict, PyTuple, PyType},
};

use crate::{
    core::{
        category::CATEGORY_UTILS,
        node_base::{InputSpec, InputType},
    },
    error::Error,
    utils::param_hub::{DEFAULT_PARAM_SLOTS, SR_PARAMS_TYPE},
    wrapper::comfyui::{PromptServer, types::any_type},
};

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
        let mut items: Vec<Bound<'_, PyAny>> = Vec::with_capacity(DEFAULT_PARAM_SLOTS);
        for _ in 0..DEFAULT_PARAM_SLOTS {
            items.push(any_type(py)?);
        }
        PyTuple::new(py, items)
    }

    /// 输出名 out_1..out_N（前端会改写为 hub label）
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        let names: Vec<String> = (1..=DEFAULT_PARAM_SLOTS)
            .map(|i| format!("out_{i}"))
            .collect();
        PyTuple::new(py, names)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list(py: Python<'_>) -> PyResult<Bound<'_, PyTuple>> {
        let flags: Vec<bool> = vec![false; DEFAULT_PARAM_SLOTS];
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
            .with_hidden("unique_id", InputType::custom("UNIQUE_ID"))
            .build()
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        params: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        match Self::distribute(py, params) {
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

impl ParamPort {
    /// 分布参数并返回处理结果
    fn distribute<'py>(
        py: Python<'py>,
        params: Bound<'py, PyDict>,
    ) -> Result<Bound<'py, PyTuple>, Error> {
        let dict: Bound<'py, PyDict> = params.cast_into().map_err(|_| {
            Error::InvalidInput(
                "upstream `params` is not a SR_PARAMS dict (broken pipeline)".to_string(),
            )
        })?;

        let mut outputs: Vec<Bound<'py, PyAny>> = Vec::new();
        for (_k, v) in dict {
            outputs.push(v);
        }

        PyTuple::new(py, outputs).map_err(|e| Error::TypeConversion(format!("tuple: {e}")))
    }
}
