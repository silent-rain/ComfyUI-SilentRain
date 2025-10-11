//! 字符串列表

use log::{error, info};
use pyo3::{
    Bound, IntoPyObject, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods, PyType},
};

use crate::{
    core::category::CATEGORY_TEXT,
    error::Error,
    wrapper::comfyui::{
        PromptServer,
        node_input::InputKwargs,
        types::{NODE_STRING, NODE_STRING_LIST},
    },
};

const MAX_STRING_NUM: usize = 5;

/// 字符串列表
#[pyclass(subclass)]
pub struct StringList {}

impl PromptServer for StringList {}

#[pymethods]
impl StringList {
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
        (NODE_STRING_LIST, NODE_STRING, NODE_STRING)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str, &'static str) {
        ("string_list", "prompt_strings", "strings")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool, bool) {
        (false, true, false)
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

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "notify";

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
                    (NODE_STRING_LIST, {
                        let any = PyDict::new(py);
                        any.set_item("tooltip", "Input any list")?;
                        any
                    }),
                )?;

                // 动态输入
                for i in 1..=MAX_STRING_NUM {
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

    #[pyo3(name = "execute", signature = (delimiter, optional_string_list=None, **kwargs))]
    fn execute(
        &mut self,
        py: Python,
        delimiter: String,
        optional_string_list: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<(Vec<String>, Vec<String>, String)> {
        let result = self.list_to_strings(delimiter, optional_string_list, &kwargs);

        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("StringList error, {e}");
                if let Err(e) = self.send_error(py, "StringList".to_string(), e.to_string()) {
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

    /// 执行逻辑并通知前端更新 UI
    #[pyo3(name = "notify", signature = (delimiter, optional_string_list=None, **kwargs))]
    fn notify<'py>(
        &mut self,
        py: Python<'py>,
        delimiter: String,
        optional_string_list: Option<Bound<'py, PyAny>>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self.list_to_strings(delimiter, optional_string_list, &kwargs);

        match result {
            Ok(v) => Ok(self.node_result(py, v)?),
            Err(e) => {
                error!("StringList error, {e}");
                if let Err(e) = self.send_error(py, "StringList".to_string(), e.to_string()) {
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
    #[pyo3(name = "check_lazy_status", signature = (delimiter, optional_string_list=None, **kwargs))]
    fn check_lazy_status(
        &mut self,
        delimiter: String,
        optional_string_list: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<String>> {
        if false {
            error!(
                "check_lazy_status: {delimiter:?} ==== {optional_string_list:?} ==== {kwargs:?}"
            );
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

impl StringList {
    /// 将字符串列表合并为字符串
    ///
    /// 输入为列表时: kwargs: Some({'string_1': ['0'], 'string_2': ['1']})  
    fn list_to_strings(
        &self,
        delimiter: String,
        optional_string_list: Option<Bound<'_, PyAny>>,
        kwargs: &Option<Bound<'_, PyDict>>,
    ) -> Result<(Vec<String>, Vec<String>, String), Error> {
        // 添加输入字符串个数的缓存
        if let Some(kwargs) = kwargs {
            let kwargs = InputKwargs::new(kwargs);
            let extra_pnginfo = kwargs.extra_pnginfo()?;
            let unique_id = kwargs.unique_id()?;

            info!("workflow_id: {:#?}", extra_pnginfo.workflow.id);
            info!("unique_id: {unique_id:#?}");

            // 打印实例地址的几种方式
            info!("对象地址: {:p}", std::ptr::addr_of!(self)); // 引用变量本身的地址
            info!("对象地址: {self:p}"); // 引用指向的对象的地址
            info!("对象地址: {:p}", self as *const _); // 引用指向的对象的地址
            info!("对象地址: 0x{:x}", self as *const _ as usize); // 引用指向的对象的地址
        }

        info!("delimiter: {delimiter:#?}, optional_string_list: {optional_string_list:#?}");

        let mut new_strings: Vec<String> = Vec::new();

        if let Some(optional_string_list) = optional_string_list {
            let string_list: Vec<String> = match optional_string_list.extract() {
                Ok(list) => list,
                Err(_) => vec![optional_string_list.extract()?],
            };
            if !string_list.is_empty() {
                new_strings.extend(string_list);
            }
        }

        if let Some(kwargs) = kwargs {
            // 指定获取key、value
            for i in 1..=MAX_STRING_NUM {
                let any_value = match kwargs.get_item(format!("string_{i}"))? {
                    Some(v) => v,
                    None => continue,
                };
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

        let strings = new_strings.join(&delimiter);

        Ok((new_strings.clone(), new_strings, strings))
    }
}

impl StringList {
    /// 组合为前端需要的数据结构
    fn node_result<'py, T>(&self, py: Python<'py>, result: T) -> PyResult<Bound<'py, PyDict>>
    where
        T: IntoPyObject<'py>,
    {
        let input_dict = PyDict::new(py);
        // input_dict.set_item("string_num", vec![5])?;

        let dict = PyDict::new(py);
        dict.set_item("ui", input_dict)?;
        dict.set_item("result", result)?;

        Ok(dict)
    }
}
