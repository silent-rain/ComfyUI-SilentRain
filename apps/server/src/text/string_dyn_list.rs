//! 字符串动态列表

use log::error;
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods, PyType},
};

use crate::{
    core::category::CATEGORY_TEXT,
    error::Error,
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_INT, NODE_STRING},
    },
};

/// 字符串动态列表
#[pyclass(subclass)]
pub struct StringDynList {}

impl PromptServer for StringDynList {}

#[pymethods]
impl StringDynList {
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
        let max_string_num = 20;

        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);

                required.set_item(
                    "string_num",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 2)?;
                        params.set_item("min", 2)?;
                        params.set_item("max", max_string_num)?;
                        params.set_item("step", 1)?;
                        params
                    }),
                )?;

                required.set_item(
                    "delimiter",
                    (NODE_STRING, {
                        let delimiter = PyDict::new(py);
                        delimiter.set_item("default", ",")?;
                        delimiter.set_item("tooltip", "String merge delimiter")?;
                        delimiter
                    }),
                )?;
                required
            })?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);
                // 动态输入
                for i in 1..=max_string_num {
                    optional.set_item(
                        format!("string_{i}"),
                        (NODE_STRING, {
                            let string = PyDict::new(py);
                            // string.set_item("forceInput", false)?;
                            string.set_item("default", "")?;
                            string
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

    #[pyo3(name = "execute", signature = (string_num, delimiter, **kwargs))]
    fn execute(
        &mut self,
        py: Python,
        string_num: usize,
        delimiter: String,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<(Vec<String>, String, usize)> {
        let result = self.list_to_strings(string_num, delimiter, &kwargs);

        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("StringDynList error, {e}");
                if let Err(e) = self.send_error(py, "StringDynList".to_string(), e.to_string()) {
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
    #[pyo3(name = "check_lazy_status", signature = (string_num, delimiter, **kwargs))]
    fn check_lazy_status(
        &mut self,
        string_num: usize,
        delimiter: String,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<String>> {
        if false {
            error!("check_lazy_status: {string_num:?} ==== {delimiter:?} ==== {kwargs:?}");
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

impl StringDynList {
    /// 将字符串列表合并为字符串
    fn list_to_strings(
        &self,
        _string_num: usize,
        delimiter: String,
        kwargs: &Option<Bound<'_, PyDict>>,
    ) -> Result<(Vec<String>, String, usize), Error> {
        let mut new_strings: Vec<String> = Vec::new();

        if let Some(kwargs) = kwargs {
            // 指定获取key、value
            for (key, value) in kwargs.into_iter() {
                let key_str: String = key.extract()?;
                if key_str.starts_with("string_") {
                    let value_str: String = value.extract()?;
                    if value_str.is_empty() {
                        continue;
                    }
                    new_strings.push(value_str);
                }
            }
        }

        let total = new_strings.len();
        let strings = new_strings.join(&delimiter);

        Ok((new_strings, strings, total))
    }
}
