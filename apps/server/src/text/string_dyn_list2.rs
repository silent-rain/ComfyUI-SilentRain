//! 字符串动态列表

use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods, PyType},
    Bound, Py, PyErr, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_TEXT,
    error::Error,
    wrapper::comfyui::{
        types::{NODE_INT, NODE_STRING},
        PromptServer,
    },
};

const MAX_STRING_NUM: usize = 2;

/// 字符串动态列表
#[pyclass(subclass)]
pub struct StringDynList2 {}

impl PromptServer for StringDynList2 {}

#[pymethods]
impl StringDynList2 {
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
    fn return_types() -> (&'static str, &'static str, &'static str) {
        (NODE_STRING, NODE_STRING, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str, &'static str) {
        ("prompt_strings", "strings", "total")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool, bool) {
        (true, false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_TEXT;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Return a list of strings and the merged strings.
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
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);
                required.set_item(
                    "delimiter",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", ",")?;
                        params.set_item("tooltip", "String merge delimiter")?;
                        params
                    }),
                )?;
                required
            })?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);
                // 动态输入
                for i in 1..=MAX_STRING_NUM {
                    optional.set_item(
                        format!("string_{i}"),
                        (NODE_STRING, {
                            let params = PyDict::new(py);
                            params.set_item("forceInput", true)?;
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

    #[pyo3(name = "execute", signature = (delimiter, **kwargs))]
    fn execute(
        &mut self,
        py: Python,
        delimiter: String,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<(Vec<String>, Vec<String>, String, usize)> {
        let result = self.list_to_strings(delimiter, &kwargs);

        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("StringDynList2 error, {e}");
                if let Err(e) = self.send_error(py, "StringDynList2".to_string(), e.to_string()) {
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
    #[pyo3(name = "check_lazy_status", signature = (delimiter, **kwargs))]
    fn check_lazy_status(
        &mut self,
        delimiter: String,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<String>> {
        if false {
            error!("check_lazy_status: {delimiter:?} ==== {kwargs:?}");
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

impl StringDynList2 {
    /// 将字符串列表合并为字符串
    fn list_to_strings(
        &self,
        delimiter: String,
        kwargs: &Option<Bound<'_, PyDict>>,
    ) -> Result<(Vec<String>, Vec<String>, String, usize), Error> {
        let mut new_strings: Vec<String> = Vec::new();

        if let Some(kwargs) = kwargs {
            // 指定获取key、value
            for (key, value) in kwargs.into_iter() {
                let key_str: String = key.extract()?;
                if key_str.starts_with("string_") {
                    let value_str: String = value.extract()?;
                    new_strings.push(value_str);
                }
            }
        }

        let total = new_strings.len();
        let strings = new_strings.join(&delimiter);

        Ok((new_strings.clone(), new_strings, strings, total))
    }
}
