//! 判断输入是否为空
//!
//! 用于检测输入是否连接了有效的数据源
//! 如果输入未连接或为空，返回 true；否则返回 false

use log::error;
use pyo3::{
    Bound, Py, PyResult, Python, pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};

use crate::{
    core::category::CATEGORY_LOGIC,
    wrapper::comfyui::types::{NODE_BOOLEAN, any_type},
};

/// 判断输入是否为空的节点
#[pyclass(subclass)]
pub struct IsEmpty {}

#[pymethods]
impl IsEmpty {
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

    /// 执行判断
    /// @param input: 可选输入，任意类型
    /// @return: true 表示输入为空（未连接或 None），false 表示输入有值
    #[pyo3(name = "execute", signature = (**kwargs))]
    fn execute<'py>(&self, kwargs: Option<Bound<'py, PyDict>>) -> PyResult<(bool,)> {
        Python::attach(|_py| {
            match kwargs {
                None => Ok((true,)),
                Some(kwargs) => {
                    // 检查 'input' 字段
                    let input = match kwargs.get_item("input") {
                        Ok(v) => v,
                        Err(e) => {
                            return {
                                error!("IsEmpty err: {e:#?}");
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
