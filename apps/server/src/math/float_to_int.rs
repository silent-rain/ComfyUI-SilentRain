//! 浮点数转整数

use log::error;
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use strum_macros::{Display, EnumString};

use crate::{
    core::category::CATEGORY_MATH,
    error::Error,
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_FLOAT, NODE_INT},
    },
};

/// 浮点数转整数的模式
///
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum FloatToIntMode {
    /// 直接截断（丢弃小数部分）
    #[strum(to_string = "Truncate")]
    Truncate,
    /// 向下取整（向负无穷方向取整）
    #[strum(to_string = "Floor")]
    Floor,
    /// 向上取整（向正无穷方向取整）
    #[strum(to_string = "Ceil")]
    Ceil,
    /// 四舍五入
    #[strum(to_string = "Round")]
    Round,
    /// 向零取整
    #[strum(to_string = "TruncateTowardsZero")]
    TruncateTowardsZero,
    /// 银行家舍入（向最近的偶数取整）
    #[strum(to_string = "BankersRounding")]
    BankersRounding,
}
/// 浮点数转整数
#[pyclass(subclass)]
pub struct FloatToInt {}

impl PromptServer for FloatToInt {}

#[pymethods]
impl FloatToInt {
    #[new]
    fn new() -> Self {
        Self {}
    }

    // 输入列表, 可选
    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    // 输出节点, 可选
    // #[classattr]
    // #[pyo3(name = "OUTPUT_NODE")]
    // fn output_node() -> bool {
    //     false
    // }

    // 过时标记, 可选
    // #[classattr]
    // #[pyo3(name = "DEPRECATED")]
    // fn deprecated() -> bool {
    //     false
    // }

    // 实验性的, 可选
    // #[classattr]
    // #[pyo3(name = "EXPERIMENTAL")]
    // fn experimental() -> bool {
    //     false
    // }

    // 返回参数类型
    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_INT,)
    }

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("int",)
    }

    // 返回参数提示
    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() -> (&'static str,) {
        ("",)
    }

    // 输出列表, 可选
    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    // 节点分类
    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_MATH;

    // 节点描述, 可选
    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Convert float to int"
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
                    "float",
                    (NODE_FLOAT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 0.0)?;
                        params.set_item("tooltip", "")?;
                        params
                    }),
                )?;

                required.set_item(
                    "mode",
                    (
                        vec![
                            FloatToIntMode::Truncate.to_string(),
                            FloatToIntMode::Floor.to_string(),
                            FloatToIntMode::Ceil.to_string(),
                            FloatToIntMode::Round.to_string(),
                            FloatToIntMode::TruncateTowardsZero.to_string(),
                            FloatToIntMode::BankersRounding.to_string(),
                        ],
                        {
                            let params = PyDict::new(py);
                            params.set_item("default", FloatToIntMode::Truncate.to_string())?;
                            params.set_item("tooltip", "convert float to int mode")?;
                            params
                        },
                    ),
                )?;

                required
            })?;

            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(&mut self, py: Python<'py>, float: f64, mode: &str) -> PyResult<(i64,)> {
        let results = self.float_to_int(float, mode);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("BatchRename error, {e}");
                if let Err(e) = self.send_error(py, "BatchRename".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl FloatToInt {
    /// 浮点数转整数
    fn float_to_int(&self, float: f64, mode: &str) -> Result<(i64,), Error> {
        // 解析枚举并执行模式匹配
        let mode = mode
            .parse::<FloatToIntMode>()
            .map_err(|e| Error::ParseEnumString(e.to_string()))?;

        let result = match mode {
            FloatToIntMode::Truncate => float.trunc() as i64,
            FloatToIntMode::Floor => float.floor() as i64,
            FloatToIntMode::Ceil => float.ceil() as i64,
            FloatToIntMode::Round => float.round() as i64,
            FloatToIntMode::TruncateTowardsZero => float.trunc() as i64,
            FloatToIntMode::BankersRounding => float.round() as i64,
        };

        Ok((result,))
    }
}
