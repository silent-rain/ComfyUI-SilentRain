//! Qwen-Image扩散模型 TeaCache加速
//!
//! deep: pytorch
use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    ffi::c_str,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyModule, PyType},
};

use crate::{
    core::category::CATEGORY_MODEL,
    error::Error,
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_FLOAT, NODE_MODEL, NODE_STRING},
    },
};

/// Qwen-Image扩散模型 TeaCache加速
#[pyclass(subclass)]
pub struct TeaCacheQwenImage {}

impl PromptServer for TeaCacheQwenImage {}

#[pymethods]
impl TeaCacheQwenImage {
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

    // 实验性的, 可选
    #[classattr]
    #[pyo3(name = "EXPERIMENTAL")]
    fn experimental() -> bool {
        true
    }

    // 返回参数类型
    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_MODEL,)
    }

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("model",)
    }

    // 返回参数提示
    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() -> (&'static str,) {
        ("Qwen-Image 扩散模型 TeaCache加速",)
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
    const CATEGORY: &'static str = CATEGORY_MODEL;

    // 节点描述, 可选
    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Qwen-Image 扩散模型 TeaCache加速"
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

                required.set_item("model", (NODE_MODEL, { PyDict::new(py) }))?;

                required.set_item(
                    "model_type",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", "qwen_image")?;
                        params.set_item("tooltip", "Supported QwenImage diffusion model")?;
                        params
                    }),
                )?;

                required.set_item(
                    "rel_l1_thresh",
                    (NODE_FLOAT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 0.4)?;
                        params.set_item("min", 0.0)?;
                        params.set_item("max", 10.0)?;
                        params.set_item("step", 0.01)?;
                        params.set_item("tooltip", "How strongly to cache the output of diffusion model. This value must be non-negative.")?;
                        params
                    }),
                )?;

                required.set_item(
                    "start_percent",
                    (NODE_FLOAT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 0.0)?;
                        params.set_item("min", 0.0)?;
                        params.set_item("max", 1.0)?;
                        params.set_item("step", 0.01)?;
                        params.set_item("tooltip", "The start percentage of the steps that will apply TeaCache.")?;
                        params
                    }),
                )?;

                required.set_item(
                    "end_percent",
                    (NODE_FLOAT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 1.0)?;
                        params.set_item("min", 0.0)?;
                        params.set_item("max", 1.0)?;
                        params.set_item("step", 0.01)?;
                        params.set_item("tooltip", "The end percentage of the steps that will apply TeaCache.")?;
                        params
                    }),
                )?;

                required.set_item(
                    "cache_device",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", "cuda")?;
                        params.set_item("tooltip", "Device where the cache will reside.")?;
                        params
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
        model: Bound<'py, PyAny>,
        model_type: String,
        rel_l1_thresh: f64,
        start_percent: f64,
        end_percent: f64,
        cache_device: String,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        let results = self.load_unet(
            py,
            model,
            model_type,
            rel_l1_thresh,
            start_percent,
            end_percent,
            cache_device,
        );

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("TeaCacheQwenImage error, {e}");
                if let Err(e) = self.send_error(py, "TeaCacheQwenImage".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl TeaCacheQwenImage {
    /// 加载模型
    fn load_unet<'py>(
        &self,
        py: Python<'py>,
        model: Bound<'py, PyAny>,
        model_type: String,
        rel_l1_thresh: f64,
        start_percent: f64,
        end_percent: f64,
        cache_device: String,
    ) -> Result<(Bound<'py, PyAny>,), Error> {
        // 使用include_str!宏在编译时包含Python文件
        let python_code = c_str!(include_str!("py/teacache_qwen_image.py"));

        // 使用PyModule::from_code创建Python模块
        let module = PyModule::from_code(
            py,
            python_code,
            c"teacache_qwen_image.py",
            c"teacache_qwen_image",
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("创建Python模块失败: {}", e)))?;

        // 从模块中获取TeaCacheQwenImage类
        let teacache_class = module.getattr("TeaCacheQwenImage").map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("获取TeaCacheQwenImage类失败: {}", e))
        })?;

        // 调用Python类的构造函数
        let patched_model = teacache_class
            .call1((
                model,
                model_type,
                rel_l1_thresh,
                start_percent,
                end_percent,
                cache_device,
            ))
            .map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!("调用Python构造函数失败: {}", e))
            })?;

        // 获取patched_model的model属性
        let result_model = patched_model
            .getattr("model")
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("获取model属性失败: {}", e)))?;

        Ok((result_model,))
    }
}
