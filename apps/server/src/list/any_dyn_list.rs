//! 动态任意输入单输出节点

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods, PyType},
};

use crate::{
    core::category::CATEGORY_LIST,
    error::Error,
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_INT, any_type},
    },
};

/// 动态任意输入单输出节点
#[pyclass(subclass)]
pub struct AnyDynList {}

impl PromptServer for AnyDynList {}

#[pymethods]
impl AnyDynList {
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
    fn return_types(py: Python<'_>) -> (Bound<'_, PyAny>, &'static str) {
        let any_type = any_type(py).unwrap();
        (any_type, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("output", "total")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (true, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LIST;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "dynamic any input to list
        "
    }

    // 实验性的, 可选
    #[classattr]
    #[pyo3(name = "EXPERIMENTAL")]
    fn experimental() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        let max_any_num = 2;

        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", PyDict::new(py))?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);
                // 动态输入
                for i in 1..=max_any_num {
                    optional.set_item(
                        format!("any_{i}"),
                        (any_type(py)?, {
                            let params = PyDict::new(py);
                            params.set_item("forceInput", true)?;
                            params.set_item("tooltip", "Input any type")?;
                            params
                        }),
                    )?;
                }

                optional
            })?;

            dict.set_item("hidden", {
                let hidden = PyDict::new(py);
                hidden.set_item("prompt", "PROMPT")?;
                hidden.set_item("dynprompt", "DYNPROMPT")?;
                hidden.set_item("extra_pnginfo", "EXTRA_PNGINFO")?;
                // UNIQUE_ID is the unique identifier of the node, and matches the id property of the node on the client side.
                // It is commonly used in client-server communications (see messages).
                hidden.set_item("unique_id", "UNIQUE_ID")?;

                hidden
            })?;

            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute", signature = (**kwargs))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<(Vec<Bound<'py, PyAny>>, usize)> {
        let result = self.inputs_to_list(&kwargs);

        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("AnyDynList error, {e}");
                if let Err(e) = self.send_error(py, "AnyDynList".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }

    /// 返回需要评估的输入名称列表。
    ///
    /// 需要搭配 lazy 属性实用
    /// 如果有任何懒加载输入（lazy inputs）尚未被评估，当请求的字段值可用时，这个函数将被调用。
    /// 评估过的输入会作为参数传递给此函数，未评估的输入会传入值为 None。
    #[pyo3(name = "check_lazy_status", signature = (**kwargs))]
    fn check_lazy_status(&mut self, kwargs: Option<Bound<'_, PyDict>>) -> PyResult<Vec<String>> {
        if false {
            error!("check_lazy_status: {kwargs:?}");
        }

        // 打印 kwargs
        // let mut strs = Vec::new();
        // if let Some(kwargs) = kwargs {
        //     for (key, value) in kwargs.into_iter() {
        //         let key_str: String = key.extract()?;
        //         println!("Key: {}, Value: {:?}", key_str, value);
        //         strs.push(value.to_string());
        //     }
        // }

        // 返回需要评估的输入名称列表
        Ok(vec![])
    }
}

impl AnyDynList {
    /// 将所有的输入转换为列表
    fn inputs_to_list<'py>(
        &self,
        kwargs: &Option<Bound<'py, PyDict>>,
    ) -> Result<(Vec<Bound<'py, PyAny>>, usize), Error> {
        let mut list = Vec::new();

        if let Some(kwargs) = kwargs {
            // 指定获取key、value
            for (key, value) in kwargs.into_iter() {
                let key_str: String = key.extract()?;
                if key_str.starts_with("any_") {
                    list.push(value);
                }
            }
        }

        let total = list.len();

        Ok((list, total))
    }
}
