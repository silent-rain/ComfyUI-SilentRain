//! Nunchaku SDXL UNet Loader
//!
//! GitHub:
//!     - https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/sdxl-turbo.py
//!     - https://github.com/nunchaku-tech/ComfyUI-nunchaku
//! Nunchaku lib:
//!     - https://huggingface.co/nunchaku-tech/nunchaku
//!     - diffusers
//! Model: https://huggingface.co/nunchaku-tech/nunchaku-sdxl

use std::collections::HashMap;

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyString, PyType},
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
pub struct NunchakuSdxlUnetLoader {}

impl PromptServer for NunchakuSdxlUnetLoader {}

#[pymethods]
impl NunchakuSdxlUnetLoader {
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
                    "dtype_options",
                    (
                        vec![
                            DTypeOption::BFloat16.to_string(),
                            DTypeOption::Float16.to_string(),
                        ],
                        {
                            let mode = PyDict::new(py);
                            mode.set_item("default", DTypeOption::Float16.to_string())?;
                            mode.set_item(
                                "tooltip",
                                "Specifies the model's data type. Default is `bfloat16` for compatible GPUs, otherwise `float16`.",
                            )?;
                            mode
                        },
                    ),
                )?;

                required.set_item(
                    "cpu_offload",
                    (
                        vec!["enable", "disable"],
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
        dtype_options: &str,
        cpu_offload: &str,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        let results = self.load_model(py, model_path, device_id, dtype_options, cpu_offload);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("NunchakuSdxlUnetLoader error, {e}");
                if let Err(e) =
                    self.send_error(py, "NunchakuSdxlUnetLoader".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl NunchakuSdxlUnetLoader {
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
        dtype_options: &str,
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

        // 导入必要的Python模块
        let torch = py.import("torch")?;
        let nunchaku = py.import("nunchaku.models.unets.unet_sdxl")?;
        let comfy_model_patcher = py.import("comfy.model_patcher")?;
        let comfy_supported_models = py.import("comfy.supported_models")?;

        // 设置设备
        let device = if torch
            .getattr("cuda")?
            .getattr("is_available")?
            .call0()?
            .extract::<bool>()?
            && device_id
                < torch
                    .getattr("cuda")?
                    .getattr("device_count")?
                    .call0()?
                    .extract::<i32>()?
        {
            format!("cuda:{}", device_id)
        } else {
            "cpu".to_string()
        };

        // 获取设备对象
        let device_obj = if device.starts_with("cuda") {
            torch
                .getattr("cuda")?
                .getattr("device")?
                .call1((device_id,))?
        } else {
            torch.getattr("device")?.call1(("cpu",))?
        };

        // 确定数据类型
        let torch_dtype = if dtype_options == "bfloat16" {
            torch.getattr("bfloat16")?
        } else {
            torch.getattr("float16")?
        };

        // 加载Nunchaku SDXL UNet模型
        let unet = nunchaku
            .getattr("NunchakuSDXLUNet2DConditionModel")?
            .getattr("from_pretrained")?
            .call1((
                model_full_path.to_string(),
                HashMap::from([
                    ("device", device_obj.clone()),
                    ("torch_dtype", torch_dtype.clone()),
                    ("offload", PyString::new(py, cpu_offload).into_any()),
                ]),
            ))?;

        // 创建SDXL模型配置
        let unet_config = PyDict::new(py);
        unet_config.set_item("model_channels", 320)?;
        unet_config.set_item("use_linear_in_transformer", true)?;
        unet_config.set_item("transformer_depth", vec![0, 0, 2, 2, 10, 10])?;
        unet_config.set_item("context_dim", 2048)?;
        unet_config.set_item("adm_in_channels", 2816)?;
        unet_config.set_item("use_temporal_attention", false)?;
        let sdxl_class = comfy_supported_models.getattr("SDXL")?;
        let model_config = sdxl_class.call1((unet_config,))?;

        // 设置推理数据类型
        model_config
            .getattr("set_inference_dtype")?
            .call1((torch_dtype, py.None()))?;
        model_config.setattr("custom_operations", py.None())?;

        // 获取模型并设置UNet
        let model = model_config
            .getattr("get_model")?
            .call1((PyDict::new(py),))?;
        model.setattr("diffusion_model", unet)?;

        // 创建模型补丁器
        let model_patcher = comfy_model_patcher.getattr("ModelPatcher")?;
        let patched_model = model_patcher.call1((model, device_obj, device_id))?;

        Ok((patched_model,))
    }
}
