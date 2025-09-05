//! 这是一个基础模板

use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyErr, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_UTILS,
    error::Error,
    wrapper::comfyui::{
        types::{NODE_INT, NODE_STRING},
        PromptServer,
    },
};

/// 基础模板
#[pyclass(subclass)]
pub struct Template {}

impl PromptServer for Template {}

#[pymethods]
impl Template {
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
        (NODE_STRING,)
    }

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("output",)
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
        "the description of the node."
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
                    "folder",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", "")?;
                        params.set_item("tooltip", "")?;
                        params
                    }),
                )?;

                required
            })?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);

                optional.set_item(
                    "start_index",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 0)?;
                        params.set_item("min", 0)?;
                        params.set_item("step", 0)?;
                        params
                    }),
                )?;

                optional
            })?;
            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        folder: &str,
        start_index: usize,
    ) -> PyResult<(String,)> {
        let results = self.todo(folder, start_index);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("Template error, {e}");
                if let Err(e) = self.send_error(py, "Template".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl Template {
    /// 待实现的逻辑
    fn todo(&self, folder: &str, _start_index: usize) -> Result<(String,), Error> {
        Ok((folder.to_string(),))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 测试模板
    /// ```shell
    /// LD_LIBRARY_PATH=~/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib \
    /// cargo test --package comfyui_silentrain --lib -- utils::template::tests::test_todo --exact --show-output
    /// ```
    #[test]
    // #[ignore]
    fn test_todo() -> anyhow::Result<()> {
        let tmp = Template::new();
        let (content,) = tmp.todo("x", 0)?;
        println!("content: {content}");
        assert!(content == "x");
        Ok(())
    }

    /// 测试模板
    /// `auto-initialize` feature is enabled.
    ///
    /// ```shell
    /// LD_LIBRARY_PATH=~/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib \
    /// cargo test --package comfyui_silentrain --lib -- utils::template::tests::test_todo_attach --exact --show-output
    /// ```
    #[test]
    #[ignore]
    fn test_todo_attach() -> anyhow::Result<()> {
        Python::attach(|_py| {
            let tmp = Template::new();
            let (content,) = tmp.todo("x", 0)?;
            println!("content: {content}");
            assert!(content == "x");
            Ok(())
        })
    }
}
