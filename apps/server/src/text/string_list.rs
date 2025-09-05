//! 字符串列表 - EXPERIMENTAL
//!
//! 尚未实现前端动态添加参数量的效果, 当前需要结合js进行动态处理。
//!
//! 使用：
//! - 先点击执行, 使参数生效
//! - 编辑-刷新节点定义
//! - 右键-刷新
//!
//! comfyui-kjnodes 实现方案:
//! ComfyUI/custom_nodes/comfyui-kjnodes/web/js/jsnodes.js

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use lazy_static::lazy_static;
use log::{error, info};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyDictMethods, PyType},
    Bound, IntoPyObject, Py, PyAny, PyErr, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_TEXT,
    error::Error,
    wrapper::comfyui::{
        node_input::InputKwargs,
        types::{any_type, NODE_INT, NODE_INT_MAX, NODE_STRING},
        PromptServer,
    },
};

static STRING_KEY: RwLock<String> = RwLock::new(String::new());

// 全局 HashMap 定义
lazy_static! {
    static ref STRING_NUM_MAP: Arc<RwLock<HashMap<String, u64>>> = {
        let mut map = HashMap::new();
        // 初始化默认值
        map.insert("default_key".to_string(), 5);
        Arc::new(RwLock::new(map))
    };
}

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
    fn return_types() -> (&'static str, &'static str, &'static str, &'static str) {
        (NODE_STRING, NODE_STRING, NODE_STRING, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str, &'static str, &'static str) {
        ("string_list", "prompt_strings", "strings", "total")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool, bool, bool) {
        (false, true, false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_TEXT;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Return a list of strings and the merged strings.
        use:
            - Click 'Execute' first to make the parameters take effect
            - Edit - Refresh Node Definition
            - Right click - Refresh
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
    const FUNCTION: &'static str = "notify";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);
                required.set_item(
                    "string_num",
                    (NODE_INT, {
                        let index = PyDict::new(py);
                        index.set_item("default", 5)?;
                        index.set_item("min", 2)?;
                        index.set_item("max", NODE_INT_MAX)?;
                        index.set_item("step", 1)?;
                        index.set_item("lazy", true)?;
                        index
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
                optional.set_item(
                    "optional_string_list",
                    (any_type(py)?, {
                        let any = PyDict::new(py);
                        any.set_item("tooltip", "Input any list")?;
                        any
                    }),
                )?;

                let string_key = Self::get_string_key().unwrap_or("default_key".to_string());
                let max_string_num = Self::get_num_map(&string_key)
                    .unwrap_or(Some(5))
                    .unwrap_or(5);

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

    #[pyo3(name = "execute", signature = (string_num, delimiter, optional_string_list=None, **kwargs))]
    fn execute(
        &mut self,
        py: Python,
        string_num: u64,
        delimiter: String,
        optional_string_list: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<(Vec<String>, Vec<String>, String, usize)> {
        let result = self.list_to_strings(string_num, delimiter, optional_string_list, &kwargs);

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
    #[pyo3(name = "notify", signature = (string_num, delimiter, optional_string_list=None, **kwargs))]
    fn notify<'py>(
        &mut self,
        py: Python<'py>,
        string_num: u64,
        delimiter: String,
        optional_string_list: Option<Bound<'py, PyAny>>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let result = self.list_to_strings(string_num, delimiter, optional_string_list, &kwargs);

        match result {
            Ok(v) => Ok(self.node_result(py, v)?),
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

    /// 返回需要评估的输入名称列表。
    ///
    /// 需要搭配 lazy 属性实用
    /// 如果有任何懒加载输入（lazy inputs）尚未被评估，当请求的字段值可用时，这个函数将被调用。
    /// 评估过的输入会作为参数传递给此函数，未评估的输入会传入值为 None。
    #[pyo3(name = "check_lazy_status", signature = (string_num, delimiter, optional_string_list=None, **kwargs))]
    fn check_lazy_status(
        &mut self,
        string_num: u64,
        delimiter: String,
        optional_string_list: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<String>> {
        if false {
            error!(
                "check_lazy_status: {string_num:?}  ==== {delimiter:?} ==== {optional_string_list:?} ==== {kwargs:?}"
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
        string_num: u64,
        delimiter: String,
        optional_string_list: Option<Bound<'_, PyAny>>,
        kwargs: &Option<Bound<'_, PyDict>>,
    ) -> Result<(Vec<String>, Vec<String>, String, usize), Error> {
        // 添加输入字符串个数的缓存
        if let Some(kwargs) = kwargs {
            let kwargs = InputKwargs::new(kwargs);
            let extra_pnginfo = kwargs.extra_pnginfo()?;
            let unique_id = kwargs.unique_id()?;

            info!("workflow_id: {:#?}", extra_pnginfo.workflow.id);
            info!("unique_id: {unique_id:#?}");

            let string_num_key = format!("{}_{}", extra_pnginfo.workflow.id, unique_id);
            Self::update_string_key(string_num_key.clone())?;
            Self::add_num_map(string_num_key, string_num)?;

            // 打印实例地址的几种方式
            info!("对象地址: {:p}", std::ptr::addr_of!(self)); // 引用变量本身的地址
            info!("对象地址: {self:p}"); // 引用指向的对象的地址
            info!("对象地址: {:p}", self as *const _); // 引用指向的对象的地址
            info!("对象地址: 0x{:x}", self as *const _ as usize); // 引用指向的对象的地址
        }

        error!(
            "string_num: {string_num:#?}, delimiter: {delimiter:#?}, optional_string_list: {optional_string_list:#?}"
        );

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
            for i in 1..=string_num {
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

        let total = new_strings.len();
        let strings = new_strings.join(&delimiter);

        Ok((new_strings.clone(), new_strings, strings, total))
    }
}

impl StringList {
    /// 获取全局的字符串Key
    fn get_string_key() -> Result<String, Error> {
        let key = STRING_KEY
            .read()
            .map_err(|e| Error::LockError(e.to_string()))?;
        Ok(key.clone())
    }

    /// 更新全局的字符串Key
    fn update_string_key(value: String) -> Result<(), Error> {
        let mut key = STRING_KEY
            .write()
            .map_err(|e| Error::LockError(e.to_string()))?;
        *key = value;
        Ok(())
    }

    /// 向全局 HashMap 添加键值对
    fn add_num_map(key: String, value: u64) -> Result<(), Error> {
        let mut map = STRING_NUM_MAP
            .write()
            .map_err(|e| Error::LockError(e.to_string()))?;
        map.insert(key, value);

        Ok(())
    }

    /// 从全局 HashMap 获取值
    fn get_num_map(key: &str) -> Result<Option<u64>, Error> {
        let map = STRING_NUM_MAP
            .read()
            .map_err(|e| Error::LockError(e.to_string()))?;
        Ok(map.get(key).cloned())
    }

    /// 打印全局 HashMap 内容
    fn _print_num_map() -> Result<(), Error> {
        let map = STRING_NUM_MAP
            .read()
            .map_err(|e| Error::LockError(e.to_string()))?;
        info!("全局 HashMap 内容:");
        for (key, value) in map.iter() {
            info!("{key}: {value}");
        }

        Ok(())
    }
}

impl StringList {
    /// 组合为前端需要的数据结构
    fn node_result<'py, T>(&self, py: Python<'py>, result: T) -> PyResult<Bound<'py, PyDict>>
    where
        T: IntoPyObject<'py>,
    {
        let input_dict = PyDict::new(py);
        input_dict.set_item("string_num", vec![5])?;

        let dict = PyDict::new(py);
        dict.set_item("ui", input_dict)?;
        dict.set_item("result", result)?;

        Ok(dict)
    }
}
