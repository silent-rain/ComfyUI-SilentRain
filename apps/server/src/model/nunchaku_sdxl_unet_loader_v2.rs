//! Nunchaku SDXL UNet Loader
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
use strum_macros::{Display, EnumString};

use crate::{
    core::category::CATEGORY_MODEL,
    error::Error,
    wrapper::{
        comfy::folder_paths::FolderPaths,
        comfyui::{PromptServer, types::NODE_MODEL},
    },
};

/// 数据类型的选项
///
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum DTypeOption {
    /// bfloat16 数据类型
    #[strum(to_string = "bfloat16")]
    BFloat16,
    /// float16 数据类型
    #[strum(to_string = "float16")]
    Float16,
}

/// Nunchaku SDXL UNet Loader
#[pyclass(subclass)]
pub struct NunchakuSdxlUnetLoaderV2 {}

impl PromptServer for NunchakuSdxlUnetLoaderV2 {}

#[pymethods]
impl NunchakuSdxlUnetLoaderV2 {
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
        ("",)
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
        "Nunchaku SDXL UNet Loader"
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

                // 获取模型列表
                let model_list = Self::get_model_list();

                required.set_item(
                    "model_path",
                    (model_list.clone(), {
                        let params = PyDict::new(py);
                        params.set_item("tooltip", "The Nunchaku SDXL UNet model.")?;
                        params
                    }),
                )?;

                required.set_item(
                    "device_id",
                    (
                        "INT",
                        {
                            let params = PyDict::new(py);
                            params.set_item("default", 0)?;
                            params.set_item("min", 0)?;
                            params.set_item("step", 1)?;
                            params.set_item("display", "number")?;
                            params.set_item("lazy", true)?;
                            params.set_item("tooltip", "The GPU device ID to use for the model.")?;
                            params
                        },
                    ),
                )?;

                required.set_item(
                    "data_type",
                    (
                        vec![
                            DTypeOption::BFloat16.to_string(),
                            DTypeOption::Float16.to_string(),
                        ],
                        {
                            let params = PyDict::new(py);
                            params.set_item("default", DTypeOption::Float16.to_string())?;
                            params.set_item(
                                "tooltip",
                                "Specifies the model's data type. Default is `bfloat16` for compatible GPUs, otherwise `float16`.",
                            )?;
                            params
                        },
                    ),
                )?;

                required.set_item(
                    "cpu_offload",
                    (
                        vec![
                            "enable",
                            "disable",
                        ],
                        {
                            let params = PyDict::new(py);
                            params.set_item("default", "disable")?;
                            params.set_item(
                                "tooltip",
                                "Whether to enable CPU offload for the UNet model. Note: SDXL UNet does not support offload.",
                            )?;
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

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        model_path: &str,
        device_id: i32,
        data_type: &str,
        cpu_offload: &str,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        let results = self.load_model(py, model_path, device_id, data_type, cpu_offload);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("NunchakuSdxlUnetLoaderV2 error, {e}");
                if let Err(e) =
                    self.send_error(py, "NunchakuSdxlUnetLoaderV2".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl NunchakuSdxlUnetLoaderV2 {
    /// 获取模型列表
    fn get_model_list() -> Vec<String> {
        let folder_paths = FolderPaths::default();
        let unet_list = folder_paths.get_filename_list("unet");
        let diffusion_models_list = folder_paths.get_filename_list("diffusion_models");

        let mut model_list = [unet_list, diffusion_models_list]
            .concat()
            .into_iter()
            .filter(|v| v.ends_with(".safetensors"))
            .collect::<Vec<String>>();

        model_list.dedup();

        model_list
    }

    /// 加载模型
    fn load_model<'py>(
        &mut self,
        py: Python<'py>,
        model_path: &str,
        device_id: i32,
        data_type: &str,
        cpu_offload: &str,
    ) -> Result<(Bound<'py, PyAny>,), Error> {
        let g_model_path = FolderPaths::default().model_path();
        let mut model_full_path = model_path.to_string();
        for model_parent_name in ["unet", "diffusion_models"] {
            let model_path = g_model_path.join(model_parent_name).join(model_path);
            if model_path.exists() {
                model_full_path = model_path.to_string_lossy().to_string();
                break;
            }
        }

        // 使用include_str!宏在编译时包含Python文件
        let python_code = c_str!(include_str!("nunchaku_sdxl_unet_loader.py"));

        // 使用PyModule::from_code创建Python模块
        let module = PyModule::from_code(
            py,
            python_code,
            c"nunchaku_sdxl_unet_loader.py",
            c"nunchaku_sdxl_unet_loader",
        )
        .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("创建Python模块失败: {}", e)))?;

        // 从模块中获取NunchakuSDXLUNetLoader类
        let loader_class = module.getattr("NunchakuSDXLUNetLoader").map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("获取NunchakuSDXLUNetLoader类失败: {}", e))
        })?;

        // 调用Python类的构造函数
        let loader_model = loader_class.call0().map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("调用Python构造函数失败: {}", e))
        })?;

        // 调用load_model方法并获取返回的模型
        let load_model_result = loader_model
            .getattr("load_model")
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("获取load_model方法失败: {}", e)))?
            .call1((model_full_path, device_id, data_type, cpu_offload))
            .map_err(|e| {
                PyErr::new::<PyRuntimeError, _>(format!("调用load_model方法失败: {}", e))
            })?;

        // 从返回的元组中提取模型（第一个元素）
        let model = load_model_result
            .get_item(0)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(format!("提取模型失败: {}", e)))?;

        Ok((model,))
    }
}
