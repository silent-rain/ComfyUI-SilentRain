//! Nunchaku SDXL LoRA Loader
//!
//! GitHub:
//!     - https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/sdxl-turbo.py
//!     - https://github.com/nunchaku-tech/ComfyUI-nunchaku
//! Nunchaku lib:
//!     - https://huggingface.co/nunchaku-tech/nunchaku
//!     - diffusers
//! Model: https://huggingface.co/nunchaku-tech/nunchaku-sdxl
//!
//! deep: pytorch/nunchaku
//!

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
    wrapper::{
        comfy::folder_paths::FolderPaths,
        comfyui::{PromptServer, types::NODE_MODEL},
    },
};

/// Nunchaku SDXL LoRA Loader
#[pyclass(subclass)]
pub struct NunchakuSdxlLoraLoader {}

impl PromptServer for NunchakuSdxlLoraLoader {}

#[pymethods]
impl NunchakuSdxlLoraLoader {
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
        ("MODEL",)
    }

    // 返回参数提示
    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() -> (&'static str,) {
        ("The LoRA model is loaded.",)
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
        "LoRAs are used to modify the diffusion model, altering the way in which latents are denoised such as applying styles. You can link multiple LoRA nodes."
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

                // 获取LoRA列表
                let lora_list = Self::get_lora_list();

                required.set_item(
                    "model",
                    (
                        "MODEL",
                        {
                            let params = PyDict::new(py);
                            params.set_item("tooltip", "The diffusion model LoRA will be applied to. Make sure model is loaded by `Nunchaku SDXL UNet Loader`.")?;
                            params
                        },
                    ),
                )?;

                required.set_item(
                    "lora_name",
                    (lora_list.clone(), {
                        let params = PyDict::new(py);
                        params.set_item("tooltip", "The file name of the LoRA.")?;
                        params
                    }),
                )?;

                required.set_item(
                    "lora_strength",
                    (
                        "FLOAT",
                        {
                            let params = PyDict::new(py);
                            params.set_item("default", 1.0)?;
                            params.set_item("min", -100.0)?;
                            params.set_item("max", 100.0)?;
                            params.set_item("step", 0.01)?;
                            params.set_item("tooltip", "How strongly to modify the diffusion model. This value can be negative.")?;
                            params
                        },
                    ),
                )?;

                required
            })?;

            dict.set_item("optional", PyDict::new(py))?;
            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        model: &Bound<'py, PyAny>,
        lora_name: &str,
        lora_strength: f64,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        let results = self.load_lora(py, model, lora_name, lora_strength);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("NunchakuSdxlLoraLoader error, {e}");
                if let Err(e) =
                    self.send_error(py, "NunchakuSdxlLoraLoader".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl NunchakuSdxlLoraLoader {
    /// 获取LoRA列表
    fn get_lora_list() -> Vec<String> {
        let folder_paths = FolderPaths::default();
        let lora_list = folder_paths.get_filename_list("loras");

        lora_list
            .into_iter()
            .filter(|v| v.ends_with(".safetensors"))
            .collect::<Vec<String>>()
    }

    /// 加载LoRA
    fn load_lora<'py>(
        &mut self,
        py: Python<'py>,
        model: &Bound<'py, PyAny>,
        lora_name: &str,
        lora_strength: f64,
    ) -> Result<(Bound<'py, PyAny>,), Error> {
        // 创建临时模块结构
        let converter_code = c_str!(include_str!("py/nunchaku_sdxl_lora_converter.py"));
        let converter_module = PyModule::from_code(
            py,
            converter_code,
            c"nunchaku_sdxl_lora_converter.py",
            c"nunchaku_sdxl_lora_converter",
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("创建converter模块失败: {}", e)))?;

        let wrapper_code = c_str!(include_str!("py/comfy_sdxl_wrapper.py"));
        let wrapper_module = PyModule::from_code(
            py,
            wrapper_code,
            c"comfy_sdxl_wrapper.py",
            c"comfy_sdxl_wrapper",
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("创建wrapper模块失败: {}", e)))?;

        // 将wrapper模块添加到sys.modules中，使其可以被导入
        let sys = py
            .import("sys")
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("导入sys模块失败: {}", e)))?;
        let sys_modules = sys
            .getattr("modules")
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("获取sys.modules失败: {}", e)))?;

        // 将模块添加到sys.modules
        sys_modules
            .set_item("nunchaku_sdxl_lora_converter", &converter_module)
            .map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!(
                    "添加converter模块到sys.modules失败: {}",
                    e
                ))
            })?;
        sys_modules
            .set_item("comfy_sdxl_wrapper", &wrapper_module)
            .map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!("添加wrapper模块到sys.modules失败: {}", e))
            })?;

        // 现在加载nunchaku_sdxl_lora_loader.py，它应该能够导入comfy_sdxl_unet_wrapper模块
        let python_code = c_str!(include_str!("py/nunchaku_sdxl_lora_loader.py"));

        // 使用PyModule::from_code创建Python模块
        let module = PyModule::from_code(
            py,
            python_code,
            c"nunchaku_sdxl_lora_loader.py",
            c"nunchaku_sdxl_lora_loader",
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("创建Python模块失败: {}", e)))?;

        // 从模块中获取NunchakuSDXLLoraLoader类
        let loader_class = module.getattr("NunchakuSDXLLoraLoader").map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("获取NunchakuSDXLLoraLoader类失败: {}", e))
        })?;

        // 调用Python类的构造函数
        let loader_model = loader_class.call0().map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("调用Python构造函数失败: {}", e))
        })?;

        // 调用load_lora方法并获取返回的模型
        let model_result = loader_model
            .getattr("load_lora")
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("获取load_lora方法失败: {}", e)))?
            .call1((model, lora_name, lora_strength))
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("调用load_lora方法失败: {}", e)))?
            .get_item(0)?;

        Ok((model_result,))
    }
}
