//! Flux.1-Dev 扩散模型 Block Swap Patch
//!
//! deep: pytorch
//!
//! 引用: https://github.com/MinusZoneAI/ComfyUI-FluxExt-MZ/blob/main/mz_fluxext_core.py
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
        types::{NODE_INT, NODE_MODEL},
    },
};

/// Flux.1-Dev 扩散模型 Block Swap Patch
#[pyclass(subclass)]
pub struct FluxBlockSwapPatch {}

impl PromptServer for FluxBlockSwapPatch {}

#[pymethods]
impl FluxBlockSwapPatch {
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
        ("Qwen-Image 扩散模型 Block Swap Patch",)
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
        "Qwen-Image 扩散模型 Block Swap Patch"
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
                    "double_blocks",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("min", 0)?;
                        params.set_item("max", 19)?;
                        params.set_item("default", 7)?;
                        params.set_item("tooltip", "double blocks 19")?;
                        params
                    }),
                )?;
                required.set_item(
                    "single_blocks",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("min", 0)?;
                        params.set_item("max", 38)?;
                        params.set_item("default", 7)?;
                        params.set_item("tooltip", "double blocks 38")?;
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
        double_blocks: usize,
        single_blocks: usize,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        let results = self.load_unet(py, model, double_blocks, single_blocks);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("FluxBlockSwapPatch error, {e}");
                if let Err(e) = self.send_error(py, "FluxBlockSwapPatch".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl FluxBlockSwapPatch {
    /// 加载模型
    fn load_unet<'py>(
        &self,
        py: Python<'py>,
        model: Bound<'py, PyAny>,
        double_blocks: usize,
        single_blocks: usize,
    ) -> Result<(Bound<'py, PyAny>,), Error> {
        // 使用include_str!宏在编译时包含Python文件
        let python_code = c_str!(include_str!("flux_block_swap_path.py"));

        // 使用PyModule::from_code创建Python模块
        let module = PyModule::from_code(
            py,
            python_code,
            c"flux_block_swap_path.py",
            c"flux_block_swap_path",
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("创建Python模块失败: {}", e)))?;

        // 从模块中获取FluxBlockSwapPatch类
        let qwen_class = module.getattr("FluxBlockSwapPatch").map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("获取FluxBlockSwapPatch类失败: {}", e))
        })?;

        // 调用Python类的构造函数
        let patched_model = qwen_class
            .call1((model, double_blocks, single_blocks))
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
