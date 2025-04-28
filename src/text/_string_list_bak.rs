//! 字符串列表
//!
//! 尚未实现前端动态添加参数量的效果, 当前需要结合js进行动态处理。

use log::error;
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyErr, PyResult, Python,
};

use crate::{
    core::{
        category::CATEGORY_TEXT,
        types::{any_type, NODE_INT, NODE_STRING},
        PromptServer,
    },
    error::Error,
};

const MAX_STRING_NUM: i32 = 10;

/// 字符串列表
#[pyclass(subclass)]
pub struct StringList {}

impl PromptServer for StringList {}

#[pymethods]
impl StringList {
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
    fn return_types() -> (&'static str, &'static str, &'static str) {
        (NODE_STRING, NODE_STRING, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str, &'static str) {
        ("string_list", "strings", "total")
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
        "Return a list of strings and the merged strings."
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
                // required.set_item(
                //     "string_num",
                //     (NODE_INT, {
                //         let index = PyDict::new(py);
                //         index.set_item("default", 1)?;
                //         index.set_item("min", 1)?;
                //         index.set_item("max", MAX_STRING_NUM)?;
                //         index.set_item("step", 1)?;
                //         index.set_item("lazy", true)?;
                //         index
                //     }),
                // )?;
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
                optional.set_item(
                    "optional_string_list",
                    (any_type(py)?, {
                        let any = PyDict::new(py);
                        any.set_item("tooltip", "Input any list")?;
                        any
                    }),
                )?;

                // 动态输入
                for i in 1..=MAX_STRING_NUM {
                    optional.set_item(
                        format!("string_{}", i),
                        (NODE_STRING, {
                            let string = PyDict::new(py);
                            string.set_item("lazy", true)?;
                            string.set_item("forceInput", false)?;
                            string.set_item("default", "")?;
                            string
                        }),
                    )?;
                }

                optional
            })?;

            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute", signature = (delimiter, optional_string_list=[].to_vec(), **kwargs))]
    fn execute(
        &mut self,
        py: Python,
        delimiter: Vec<String>,
        optional_string_list: Vec<String>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<(Vec<String>, String, usize)> {
        let delimiter = delimiter[0].clone();
        let result = self.list_to_strings(delimiter, optional_string_list, kwargs);

        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("node execute failed, {e}");
                if let Err(e) = self.send_error(py, "SCAN_FILES_ERROR".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        e.to_string(),
                    ));
                };
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    e.to_string(),
                ))
            }
        }
    }

    // /// 返回需要评估的输入名称列表。
    // ///
    // /// 需要搭配 lazy 属性实用
    // /// 如果有任何懒加载输入（lazy inputs）尚未被评估，当请求的字段值可用时，这个函数将被调用。
    // /// 评估过的输入会作为参数传递给此函数，未评估的输入会传入值为 None。
    // #[pyo3(name = "check_lazy_status", signature = (**kwargs))]
    // fn check_lazy_status(
    //     &mut self,
    //     kwargs: Option<Bound<'_, PyDict>>,
    // ) -> PyResult<Vec<String>> {
    //     let mut strs = Vec::new();
    //     if let Some(kwargs) = kwargs {
    //         for (key, value) in kwargs.into_iter() {
    //             let key_str: String = key.extract()?;
    //             println!("Key: {}, Value: {:?}", key_str, value);
    //             strs.push(value.to_string());
    //         }
    //     }

    //     Ok(vec![])
    // }
}

impl StringList {
    /// 将字符串列表合并为字符串
    ///
    /// 输入为列表时: kwargs: Some({'string_1': ['0'], 'string_2': ['1']})  
    fn list_to_strings(
        &self,
        delimiter: String,
        optional_string_list: Vec<String>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> Result<(Vec<String>, String, usize), Error> {
        let mut new_strings: Vec<String> = Vec::new();
        if !optional_string_list.is_empty() {
            new_strings.extend(optional_string_list);
        }

        if let Some(kwargs) = kwargs {
            // 打印所有的 key、value
            // for (key, value) in kwargs.into_iter() {
            //     let key_str: String = key.extract()?;
            //     println!("Key: {}, Value: {:?}", key_str, value);
            // }

            // 指定获取key、value
            for i in 1..=MAX_STRING_NUM {
                let any_value = kwargs.get_item(format!("string_{}", i))?;
                // 多类型兼容处理
                let str_list: Vec<String> = match any_value.extract() {
                    Ok(list) => list,
                    Err(_) => vec![any_value.extract()?],
                };

                let str_list = str_list
                    .into_iter()
                    .filter(|v| !v.is_empty())
                    .collect::<Vec<String>>();
                if !str_list.is_empty() {
                    new_strings.extend(str_list);
                }
            }
        }

        let total = new_strings.len();
        let strings = new_strings.join(&delimiter);

        Ok((new_strings, strings, total))
    }
}
