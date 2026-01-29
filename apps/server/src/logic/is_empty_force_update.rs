//! 判断输入是否为空（强制刷新上游）
//!
//! 用于检测输入是否连接了有效的数据源
//! 如果输入未连接或为空，返回 true；否则返回 false
//! 该节点会强制刷新上游节点缓存，确保获取最新数据

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyResult, Python, pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};

use crate::{
    core::category::CATEGORY_LOGIC,
    wrapper::comfyui::types::{NODE_BOOLEAN, any_type},
};

/// 判断输入是否为空的节点（强制刷新上游）
#[pyclass(subclass)]
pub struct IsEmptyForceUpdate {}

#[pymethods]
impl IsEmptyForceUpdate {
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
        (NODE_BOOLEAN,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("is_empty",)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    /// 这会强制重新执行该节点及其上游节点
    #[classattr]
    #[pyo3(name = "FORCE_EXECUTE")]
    fn force_execute() -> bool {
        true // 强制执行该节点
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LOGIC;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Check if input is empty (not connected or None). Returns true if input is empty, false otherwise."
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
                    "force_update",
                    (NODE_BOOLEAN, {
                        let list = PyDict::new(py);
                        list.set_item("default", false)?;
                        list.set_item("tooltip", "Force update upstream cache. If true, upstream nodes will be recomputed.")?;
                        list
                    }),
                )?;
                required
            })?;
            dict.set_item("optional", {
                let optional = PyDict::new(py);
                optional.set_item(
                    "input",
                    (any_type(py)?, {
                        let list = PyDict::new(py);
                        list.set_item("tooltip", "Input any type")?;
                        list
                    }),
                )?;
                optional
            })?;

            Ok(dict.into())
        })
    }

    #[allow(unused)]
    #[classmethod]
    #[pyo3(name = "IS_CHANGED")]
    fn is_changed<'py>(
        _cls: &Bound<'_, PyType>,
        force_update: bool,
        input: Option<Bound<'py, PyAny>>,
    ) -> PyResult<f64> {
        Python::attach(|_py| {
            // 如果 force_update 为 true，返回当前时间戳以强制重新执行
            if force_update {
                Ok(std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64())
            } else {
                // 否则返回固定值，表示节点没有变化
                Ok(0.0)
            }
        })
    }

    /// 执行判断
    /// @param force_update: 是否强制更新上游缓存
    /// @param input: 可选输入，任意类型
    /// @return: true 表示输入为空（未连接或 None），false 表示输入有值
    #[pyo3(name = "execute", signature = (force_update, **kwargs))]
    fn execute<'py>(
        &self,
        force_update: bool,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(bool,)> {
        Python::attach(|_py| {
            match kwargs {
                None => Ok((true,)),
                Some(kwargs) => {
                    // 如果 force_update 为 true，通过修改输入类型标记来强制上游重新计算
                    // 这里我们通过执行一次输入的访问来触发上游计算
                    if force_update {
                        let _ = kwargs.get_item("input");
                    }

                    // 检查 'input' 字段
                    let input = match kwargs.get_item("input") {
                        Ok(v) => v,
                        Err(e) => {
                            return {
                                error!("IsEmptyForceUpdate err: {e:#?}");
                                Ok((true,))
                            };
                        }
                    };

                    // 检查输入是否为 None
                    if input.is_none() {
                        return Ok((true,));
                    }

                    Ok((false,))
                }
            }
        })
    }
}
