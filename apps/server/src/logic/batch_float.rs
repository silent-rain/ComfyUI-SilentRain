//! 批处理浮点数

use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyErr, PyResult, Python,
};
use strum_macros::{Display, EnumString};

use crate::{
    core::{category::CATEGORY_LOGIC, utils::easing::Easing},
    error::Error,
    wrapper::comfyui::{
        types::{NODE_FLOAT, NODE_INT},
        PromptServer,
    },
};

/// 批处理浮点数模式
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum BatchFloatMode {
    /// 单值模式 - 处理单个浮点数值
    #[strum(to_string = "Single")]
    Single,

    /// 步进模式 - 处理多个浮点数值的序列
    #[strum(to_string = "Steps")]
    Steps,
}

/// 批处理浮点数
#[pyclass(subclass)]
pub struct BatchFloat {}

impl PromptServer for BatchFloat {}

#[pymethods]
impl BatchFloat {
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

    // 返回参数类型
    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str) {
        (NODE_FLOAT, NODE_INT)
    }

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("floats", "ints")
    }

    // 返回参数提示
    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() -> (&'static str, &'static str) {
        ("float values", "int values")
    }

    // 输出列表, 可选
    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (false, false)
    }

    // 节点分类
    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LOGIC;

    // 节点描述, 可选
    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Generate a batch of floating-point or integer values through interpolation."
    }

    // 调研方法函数名称
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
                    "mode",
                    (
                        vec![
                            BatchFloatMode::Single.to_string(),
                            BatchFloatMode::Steps.to_string(),
                        ],
                        {
                            let mode = PyDict::new(py);
                            mode.set_item("default", BatchFloatMode::Steps.to_string())?;
                            mode.set_item("tooltip", "Batch Mode")?;
                            mode
                        },
                    ),
                )?;

                required.set_item(
                    "count",
                    (NODE_INT, {
                        let count = PyDict::new(py);
                        count.set_item("default", 2)?;
                        count.set_item("tooltip", "")?;
                        count
                    }),
                )?;
                required.set_item(
                    "min",
                    (NODE_FLOAT, {
                        let min = PyDict::new(py);
                        min.set_item("default", 0.0)?;
                        min.set_item("tooltip", "")?;
                        min
                    }),
                )?;
                required.set_item(
                    "max",
                    (NODE_FLOAT, {
                        let max = PyDict::new(py);
                        max.set_item("default", 1.0)?;
                        max.set_item("tooltip", "")?;
                        max
                    }),
                )?;

                required.set_item(
                    "easing",
                    (
                        vec![
                            Easing::Linear.to_string(),
                            Easing::SineIn.to_string(),
                            Easing::SineOut.to_string(),
                            Easing::SineInOut.to_string(),
                            Easing::QuartIn.to_string(),
                            Easing::QuartOut.to_string(),
                            Easing::QuartInOut.to_string(),
                            Easing::CubicIn.to_string(),
                            Easing::CubicOut.to_string(),
                            Easing::CubicInOut.to_string(),
                            Easing::CircIn.to_string(),
                            Easing::CircOut.to_string(),
                            Easing::CircInOut.to_string(),
                            Easing::BackIn.to_string(),
                            Easing::BackOut.to_string(),
                            Easing::BackInOut.to_string(),
                            Easing::ElasticIn.to_string(),
                            Easing::ElasticOut.to_string(),
                            Easing::ElasticInOut.to_string(),
                            Easing::BounceIn.to_string(),
                            Easing::BounceOut.to_string(),
                            Easing::BounceInOut.to_string(),
                        ],
                        {
                            let easing = PyDict::new(py);
                            easing.set_item("default", Easing::Linear.to_string())?;
                            easing.set_item("tooltip", "")?;
                            easing
                        },
                    ),
                )?;

                required
            })?;

            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute")]
    fn execute(
        &mut self,
        py: Python,
        mode: String,
        count: usize,
        min: f64,
        max: f64,
        easing: String,
    ) -> PyResult<(Vec<f64>, Vec<i64>)> {
        let results = self.set_keyframes(mode, count, min, max, easing);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("BatchFloat error, {e}");
                if let Err(e) = self.send_error(py, "BatchFloat".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl BatchFloat {
    /// 设置序列
    ///
    /// # 参数
    /// - `mode`: 生成模式（单值或步进）
    /// - `count`: 生成值的数量
    /// - `min`: 最小值
    /// - `max`: 最大值
    /// - `easing`: 缓动函数类型
    ///
    /// # 错误
    /// 当步进模式但数量小于2时返回错误
    #[allow(clippy::too_many_arguments)]
    fn set_keyframes(
        &self,
        mode: String,
        count: usize,
        min: f64,
        max: f64,
        easing: String,
    ) -> Result<(Vec<f64>, Vec<i64>), Error> {
        // 解析枚举并执行模式匹配
        let mode = mode
            .parse::<BatchFloatMode>()
            .map_err(|e| Error::ParseEnumString(e.to_string()))?;

        // 解析枚举并执行模式匹配
        let easing = easing
            .parse::<Easing>()
            .map_err(|e| Error::ParseEnumString(e.to_string()))?;

        // 验证步进模式需要至少2个值
        if mode == BatchFloatMode::Steps && count < 2 {
            return Err(Error::InvalidParameter(
                "Steps mode requires at least a count of 2 values".to_string(),
            ));
        }

        let keyframes_f64 = match mode {
            // 单值模式：生成 count 个相同的 min 值
            BatchFloatMode::Single => vec![min; count],

            // 步进模式：生成缓动曲线上的值
            BatchFloatMode::Steps => {
                let mut values = Vec::with_capacity(count);

                for i in 0..count {
                    // 计算归一化的步进值 [0.0, 1.0]
                    let normalized_step = i as f64 / (count - 1) as f64;

                    // 应用缓动函数
                    let eased_step = easing.apply(normalized_step);

                    // 映射到 [min, max] 范围
                    let eased_value = min + (max - min) * eased_step;

                    values.push(eased_value);
                }

                values
            }
        };

        // 将浮点序列转换为整数
        let keyframes_i64 = keyframes_f64
            .iter()
            .map(|v| v.floor() as i64)
            .collect::<Vec<i64>>();

        Ok((keyframes_f64, keyframes_i64))
    }
}
