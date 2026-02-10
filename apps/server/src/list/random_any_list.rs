//! 从任意列表随机一个元素

use pyo3::{
    Bound, Py, PyAny, PyResult, Python, pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use rand::{RngExt, SeedableRng};

use crate::{
    core::category::CATEGORY_LIST,
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_INT, NODE_SEED_MAX, any_type},
    },
};

/// 从任意列表随机一个元素
#[pyclass(subclass)]
pub struct RandomAnyList {}

impl PromptServer for RandomAnyList {}

#[pymethods]
impl RandomAnyList {
    #[new]
    fn new() -> Self {
        Self {}
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types(py: Python<'_>) -> (Bound<'_, PyAny>, &'static str) {
        let any_type = any_type(py).unwrap();
        (any_type, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("list", "total")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LIST;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Randomly select an element from any list."
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
                    "list",
                    (any_type(py)?, {
                        let list = PyDict::new(py);
                        list.set_item("tooltip", "Input any list")?;
                        list
                    }),
                )?;
                required.set_item(
                    "seed",
                    (NODE_INT, {
                        let seed = PyDict::new(py);
                        seed.set_item("default", 1024)?;
                        seed.set_item("min", 0)?;
                        seed.set_item("max", NODE_SEED_MAX)?;
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
        list: Vec<Bound<'py, PyAny>>,
        seed: Vec<u64>,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let (result, total) = self.pick_random(list, seed[0]);
        Ok((result, total))
    }
}

impl RandomAnyList {
    /// 从列表中随机选择一个元素
    /// 返回 (随机选择的元素, 列表元素总数)
    fn pick_random<T>(&self, mut list: Vec<T>, seed: u64) -> (T, usize) {
        let len = list.len();

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let index = rng.random_range(0..len);

        // swap_remove 是 O(1)，取出元素
        let element = list.swap_remove(index);

        (element, len)
    }
}
