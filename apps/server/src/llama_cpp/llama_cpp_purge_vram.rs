//! Rust 的全局缓存清理
//!
//! 当前清理所有缓存，后期再进行细分

use std::collections::HashMap;

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};

use crate::{
    core::category::CATEGORY_LLAMA_CPP,
    error::Error,
    llama_cpp::ModelManager,
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_BOOLEAN, any_type},
    },
};

/// Rust 的全局缓存清理
#[pyclass(subclass)]
pub struct LlamaCppPurgeVram {}

impl PromptServer for LlamaCppPurgeVram {}

#[pymethods]
impl LlamaCppPurgeVram {
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

    // 过时标记, 可选
    // #[classattr]
    // #[pyo3(name = "DEPRECATED")]
    // fn deprecated() -> bool {
    //     false
    // }

    // 实验性的, 可选
    #[classattr]
    #[pyo3(name = "EXPERIMENTAL")]
    fn experimental() -> bool {
        true
    }

    // 返回参数类型
    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() {}

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() {}

    // 返回参数提示
    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() {}

    // 输出列表, 可选
    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() {}

    // 节点分类
    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LLAMA_CPP;

    // 节点描述, 可选
    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Rust's Global Cache Cleanup Node"
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
                    "any",
                    (any_type(py)?, {
                        let list = PyDict::new(py);
                        list.set_item("tooltip", "Input any type")?;
                        list
                    }),
                )?;

                required.set_item(
                    "model",
                    (NODE_BOOLEAN, {
                        let conditioning_shape = PyDict::new(py);
                        conditioning_shape.set_item("default", false)?;
                        conditioning_shape
                            .set_item("tooltip", "Whether to clean up the model cache.")?;
                        conditioning_shape
                    }),
                )?;

                required
            })?;

            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        any: Bound<'py, PyAny>,
        model: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let results = self.purge_vram(any, model);

        match results {
            Ok(_v) => self.node_result(py),
            Err(e) => {
                error!("LlamaCppPurgeVram error, {e}");
                if let Err(e) = self.send_error(py, "LlamaCppPurgeVram".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LlamaCppPurgeVram {
    /// 组合为前端需要的数据结构
    fn node_result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("result", HashMap::<String, String>::new())?;
        Ok(dict)
    }
}

impl LlamaCppPurgeVram {
    /// 清理全局缓存
    fn purge_vram<'py>(&self, _any: Bound<'py, PyAny>, _model: bool) -> Result<(), Error> {
        let mut cache = ModelManager::global()
            .write()
            .map_err(|e| Error::LockError(e.to_string()))?;

        // 清理所有缓存
        cache.clear();
        Ok(())
    }
}
