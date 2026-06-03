//! ParamHub - 参数聚合中心（动态多输入版）
//!
//! 节点形态：N 个 `*` 通用输入槽（前端按需 add/remove），1 个 `SR_PARAMS` 输出。
//! 节点把所有已连线 input 的实际对象按 label/name 收集成 dict，原样转发给下游 ParamPort。
//!
//! 设计要点：
//! - `INPUT_TYPES.optional` 预留 32 个 `param_1..=param_32` 通用槽（类型 `*`）。
//!   前端实际只展示有意义的槽，未连线的槽不会被 ComfyUI 调用，因此不会报错。
//! - 隐藏 widget `params_meta_json` 保存槽元信息（name, label, type 仅作显示/序列化用），
//!   随工作流一起保存，跨电脑导入仍可还原槽结构与重命名。
//! - `unique_id` hidden 输入用于前端定位节点。
//! - 后端纯函数：仅做 dict 组装，不持久化任何状态。

use log::error;
use pyo3::{
    Bound, IntoPyObject, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyDict, PyDictMethods, PyType},
};
use serde::{Deserialize, Serialize};

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

/// 动态输入槽数量上限（前端最多 add 到该数量）
pub const MAX_PARAM_SLOTS: usize = 32;

/// 单个槽的元信息（仅显示与持久化用，运行时实际类型由 link 决定）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotMeta {
    /// 后端 input 名（必须与 INPUT_TYPES.optional 中的 key 一致）：`param_1` 等
    pub name: String,
    /// 用户显示名 / 重命名（"VAE2"），同时也是导出 dict 的 key
    pub label: String,
    /// 当前连线类型（来自上游 output type），用于前端展示徽标；未连线时为 `*`
    #[serde(default = "default_slot_type")]
    pub r#type: String,
}

fn default_slot_type() -> String {
    "*".to_string()
}

/// ParamHub 持久化的元数据（widgets_values 中）
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParamHubMeta {
    #[serde(default = "default_version")]
    pub version: u32,
    /// 当前可见的槽列表（顺序与节点上 inputs 顺序一致）
    #[serde(default)]
    pub slots: Vec<SlotMeta>,
}

fn default_version() -> u32 {
    1
}

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
        let mut spec = InputSpec::new()
            // 隐藏 widget：前端写入槽元信息（name/label/type），跟随工作流序列化
            .with_required(
                "params_meta_json",
                InputType::string()
                    .default("{\"version\":1,\"slots\":[]}")
                    .multiline(true)
                    .tooltip("Managed by frontend; serialized slot meta"),
            )
            .with_hidden("unique_id", InputType::custom("UNIQUE_ID"));

        // 预留 N 个通用动态 input 槽（type='*'，optional）
        // 前端按需 addInput / removeInput 控制可见性，未连线的槽 ComfyUI 不会传
        for i in 1..=MAX_PARAM_SLOTS {
            spec = spec.with_optional(format!("param_{i}"), InputType::any());
        }

        spec.build()
    }

    #[pyo3(name = "execute", signature = (params_meta_json, **kwargs))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        params_meta_json: &str,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        match Self::build_bundle(py, params_meta_json, kwargs) {
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
    /// 用元信息中的 label 作为 dict key，把所有已连线槽汇成一个 PyDict 输出
    fn build_bundle<'py>(
        py: Python<'py>,
        params_meta_json: &str,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<Bound<'py, PyAny>, Error> {
        let meta = parse_meta(params_meta_json)?;
        let bundle = PyDict::new(py);

        if let Some(kwargs) = kwargs {
            for slot in &meta.slots {
                if !kwargs.contains(&slot.name).unwrap_or(false) {
                    continue;
                }
                let v = match kwargs.get_item(&slot.name) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                // 仅在 ComfyUI 实际传入了该 input 时（即有连线）才写入
                let key = if slot.label.is_empty() {
                    slot.name.clone()
                } else {
                    slot.label.clone()
                };
                bundle.set_item(key, v)?;
            }
        }

        // 同时把 meta 嵌进结果，便于下游 Port 在前端校验时无需反向遍历图
        // （也可不加；这里只放在 dict 一个特殊 key 下，不会与用户重命名冲突）
        let meta_obj = pythonize::pythonize(py, &meta)
            .map_err(|e| Error::TypeConversion(format!("meta -> py: {e}")))?;
        bundle.set_item("__sr_meta__", meta_obj)?;

        let any = bundle
            .into_pyobject(py)
            .map_err(|e| Error::TypeConversion(format!("dict -> py: {e}")))?
            .into_any();
        Ok(any)
    }
}

fn parse_meta(raw: &str) -> Result<ParamHubMeta, Error> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(ParamHubMeta::default());
    }
    serde_json::from_str(trimmed)
        .map_err(|e| Error::InvalidInput(format!("invalid params_meta_json: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_empty() {
        let m = parse_meta("").unwrap();
        assert!(m.slots.is_empty());
    }

    #[test]
    fn test_parse_full() {
        let raw = r#"{
            "version": 1,
            "slots": [
                {"name":"param_1","label":"VAE","type":"VAE"},
                {"name":"param_2","label":"CLIP2","type":"CLIP"}
            ]
        }"#;
        let m = parse_meta(raw).unwrap();
        assert_eq!(m.slots.len(), 2);
        assert_eq!(m.slots[0].label, "VAE");
        assert_eq!(m.slots[1].r#type, "CLIP");
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse_meta("not json").is_err());
    }
}
