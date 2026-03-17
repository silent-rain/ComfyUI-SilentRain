//! 判断输入是否为空
//!
//! 用于检测输入是否连接了有效的数据源
//! 如果输入未连接或为空，返回 true；否则返回 false

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyResult, Python, pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods, PyType},
};

use crate::{
    core::category::CATEGORY_LOGIC,
    error::Error,
    wrapper::{
        comfyui::types::{NODE_BOOLEAN, any_type},
        python::isinstance_by_torch,
    },
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
    fn execute<'py>(
        &self,
        py: Python<'py>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(bool,)> {
        match kwargs {
            None => Ok((true,)),
            Some(kwargs) => {
                // 检查 'input' 字段，key 不存在（未连接）则视为空
                let input = match kwargs.get_item("input") {
                    Ok(v) => match v {
                        Some(value) => value,
                        None => return Ok((true,)), // 没有连接到节点
                    },
                    Err(e) => {
                        return {
                            error!("IsEmpty err: {e:#?}");
                            Ok((true,))
                        };
                    }
                };

                // 检查是否为 tensor，若 tensor 全为 0 则视为空遮罩
                match self.is_empty_tensor_mask(py, &input) {
                    Ok((true,)) => return Ok((true,)),
                    Err(e) => {
                        error!("IsEmpty check tensor mask err: {e:#?}");
                    }
                    _ => {}
                }

                Ok((false,))
            }
        }
    }
}
impl IsEmpty {
    /// 判断是否为 tensor，若 tensor 全为 0 则视为空遮罩
    fn is_empty_tensor_mask<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
    ) -> Result<(bool,), Error> {
        // 判断是否为 tensor
        if !isinstance_by_torch(py, input, "torch.Tensor")? {
            return Ok((false,));
        }

        // import torch
        let torch = match py.import("torch") {
            Ok(v) => v,
            Err(e) => {
                error!("IsEmpty import torch err: {e:#?}");
                return Ok((false,));
            }
        };

        /*
        if torch.all(mask == 0):
           return (True,)
        */

        // 检查 tensor 是否全为 0
        // mask == 0  →  布尔 tensor
        let mask_eq_zero = input.call_method1("__eq__", (0,))?;

        // torch.all(mask == 0)
        let all_zero = torch.call_method1("all", (mask_eq_zero,))?;
        // 提取 bool 值：torch.all() 返回 0 维 Tensor，需先调用 .item() 转为 Python 标量
        let all_zero_bool: bool = all_zero.call_method0("item")?.extract()?;

        Ok((all_zero_bool,))
    }
}
