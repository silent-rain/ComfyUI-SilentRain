//! ParamHub - 参数聚合中心（动态多输入版）

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyDict, PyType},
};

use crate::{
    core::{
        category::CATEGORY_UTILS,
        node_base::{InputSpec, InputType},
    },
    error::Error,
    wrapper::comfyui::PromptServer,
};

/// 自定义连线类型：Hub 与 Port 之间的传输管线
pub const SR_PARAMS_TYPE: &str = "SR_PARAMS";

/// 默认输入槽数量
pub const DEFAULT_PARAM_SLOTS: usize = 2;

/// 参数聚合中心
#[pyclass(subclass)]
pub struct ParamHub {}

impl PromptServer for ParamHub {}

#[pymethods]
impl ParamHub {
    #[new]
    fn new() -> Self {
        Self {}
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (SR_PARAMS_TYPE,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("params",)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() -> (&'static str,) {
        ("Connect to one or more Sr Param Port nodes to fan-out parameters",)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_UTILS;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Aggregate any number of typed inputs and forward them as a SR_PARAMS bundle to downstream Sr Param Port nodes."
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
        let mut spec = InputSpec::new().with_hidden("unique_id", InputType::custom("UNIQUE_ID"));

        // 预留 N 个通用动态 input 槽（type='*'，optional）
        // 前端按需 addInput / removeInput 控制可见性，未连线的槽 ComfyUI 不会传
        for i in 1..=DEFAULT_PARAM_SLOTS {
            spec = spec.with_optional(format!("param_{i}"), InputType::any());
        }

        spec.build()
    }

    #[pyo3(name = "execute", signature = (**kwargs))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        match Self::build_bundle(py, kwargs) {
            Ok(v) => Ok((v,)),
            Err(e) => {
                error!("ParamHub error, {e}");
                if let Err(se) = self.send_error(py, "ParamHub".to_string(), e.to_string()) {
                    error!("send error failed, {se}");
                }
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}
impl ParamHub {
    /// 构建参数包
    fn build_bundle<'py>(
        _py: Python<'py>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<Bound<'py, PyAny>, Error> {
        let kwargs =
            kwargs.ok_or_else(|| Error::InvalidParameter("parameters is required".to_string()))?;

        Ok(kwargs.into_any())
    }
}
