//! 洗牌任何列表

use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyAny, PyResult, Python,
};
use rand::{Rng, SeedableRng};

use crate::core::{
    category::CATEGORY_UTILS,
    types::{any_type, NODE_INT},
    PromptServer,
};

/// 洗牌任何列表
#[pyclass(subclass)]
pub struct ShuffleAnyList {}

impl PromptServer for ShuffleAnyList {}

#[pymethods]
impl ShuffleAnyList {
    #[new]
    fn new() -> Self {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_level(true)
            .with_file(true)
            .with_line_number(true)
            .try_init();
        Self {}
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types(py: Python) -> (Bound<'_, PyAny>, &'static str) {
        let any_type = any_type(py).unwrap();
        (any_type, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("out", "total")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (true, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_UTILS;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Using Fisher Yates algorithm to shuffle the order of elements in an array."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);
                required.set_item(
                    "any",
                    (any_type(py)?, {
                        let any = PyDict::new(py);
                        any.set_item("tooltip", "Input any list")?;
                        any
                    }),
                )?;
                required.set_item(
                    "seed",
                    (NODE_INT, {
                        let seed = PyDict::new(py);
                        seed.set_item("default", 1024)?;
                        seed.set_item("min", 0)?;
                        seed.set_item("max", 0xffffffffffffffffu64)?;
                        seed.set_item("step", 1)?;
                        seed
                    }),
                )?;
                required
            })?;
            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        any: Vec<Bound<'py, PyAny>>,
        seed: Vec<u64>,
    ) -> PyResult<(Vec<Bound<'py, PyAny>>, usize)> {
        let (results, total) = self.shuffle(any, seed[0]);
        Ok((results, total))
    }
}

impl ShuffleAnyList {
    /// 使用 Fisher-Yates 算法重新洗牌数组中的元素顺序
    /// 返回 (洗牌后的数组, 数组长度)
    fn shuffle<T>(&self, mut arr: Vec<T>, seed: u64) -> (Vec<T>, usize) {
        // 获取向量的长度
        let len = arr.len();

        // 使用给定的种子创建一个随机数生成器
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        // 实现 Fisher-Yates 洗牌算法[2,3,5](@ref)
        for i in (1..len).rev() {
            let j = rng.random_range(0..=i);
            arr.swap(i, j);
        }

        (arr, len)
    }
}
