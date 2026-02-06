//! ProgressBar 测试节点

use log::error;
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use std::thread;
use std::time::Duration;
use tracing::info;

use crate::{
    core::{
        category::CATEGORY_UTILS,
        node_base::{InputSpec, InputType},
    },
    error::Error,
    wrapper::comfyui::{PromptServer, types::NODE_INT},
};

/// ProgressBar 测试节点
#[pyclass(subclass)]
pub struct ProgressBarTest {}

impl PromptServer for ProgressBarTest {}

#[pymethods]
impl ProgressBarTest {
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
    #[classattr]
    #[pyo3(name = "OUTPUT_NODE")]
    fn output_node() -> bool {
        true
    }

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
        ("result",)
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
    const CATEGORY: &'static str = CATEGORY_UTILS;

    // 节点描述, 可选
    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Test ProgressBar functionality with a loop and sleep."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        InputSpec::new()
            .with_required(
                "n",
                InputType::int()
                    .default(10)
                    .min(1)
                    .max(100)
                    .step(1)
                    .tooltip("Number of iterations for the loop"),
            )
            .with_required(
                "sleep_ms",
                InputType::int()
                    .default(500)
                    .min(0)
                    .max(10000)
                    .step(100)
                    .tooltip("Sleep time in milliseconds for each iteration"),
            )
            .build()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute")]
    fn execute<'py>(&mut self, py: Python<'py>, n: usize, sleep_ms: u64) -> PyResult<(i64,)> {
        let results = self.run_progress_test(py, n, sleep_ms);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("ProgressBarTest error, {e}");
                if let Err(e) = self.send_error(py, "ProgressBarTest".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl ProgressBarTest {
    /// 执行 ProgressBar 测试
    fn run_progress_test(&self, py: Python<'_>, n: usize, sleep_ms: u64) -> Result<(i64,), Error> {
        // 获取 comfy.utils.ProgressBar
        let comfy = py.import("comfy")?;
        let utils = comfy.getattr("utils")?;

        // 创建 ProgressBar，总数为 n
        let pbar = utils.call_method1("ProgressBar", (n,))?;

        // 从 0 到 n 进行循环
        for i in 0..n {
            // 睡眠指定毫秒数
            thread::sleep(Duration::from_millis(sleep_ms));

            // 更新进度条，每次增加 1
            pbar.call_method1("update", (1,))?;
            info!("ProgressBarTest: updated progress bar to {}", i + 1);

            // 可选：释放 GIL 以允许其他 Python 线程运行
            py.check_signals()?;
        }

        Ok((n as i64,))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试 ProgressBar 测试节点
    /// ```shell
    /// LD_LIBRARY_PATH=~/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib \
    /// cargo test --package comfyui_silentrain --lib -- utils::progress_bar_test::tests::test_progress_bar --exact --show-output
    /// ```
    #[test]
    #[ignore]
    fn test_progress_bar() -> anyhow::Result<()> {
        Python::attach(|py| {
            let test = ProgressBarTest::new();
            let (result,) = test.run_progress_test(py, 5, 100)?;
            println!("Completed {} iterations", result);
            assert_eq!(result, 5);
            Ok(())
        })
    }
}
