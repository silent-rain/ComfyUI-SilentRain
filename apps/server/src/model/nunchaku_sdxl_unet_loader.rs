//! Nunchaku SDXL UNet Loader
//!
//! GitHub:
//!     - https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/sdxl-turbo.py
//!     - https://github.com/nunchaku-tech/ComfyUI-nunchaku
//! Nunchaku lib:
//!     - https://huggingface.co/nunchaku-tech/nunchaku
//!     - diffusers
//! Model: https://huggingface.co/nunchaku-tech/nunchaku-sdxl

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
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
                    "dtype_options",
                    (
                        vec![
                            DTypeOption::Float16.to_string(),
                            DTypeOption::BFloat16.to_string(),
                        ],
                        {
                            let mode = PyDict::new(py);
                            mode.set_item("default", DTypeOption::Float16.to_string())?;
                            mode.set_item(
                                "tooltip",
                                "Specifies the model's data type. Default is `bfloat16`. For 20-series GPUs, which do not support `bfloat16`, use `float16` instead.",
                            )?;
                            mode
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
        dtype_options: &str,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        let results = self.load_model(py, model_path, dtype_options);

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
        &self,
        py: Python<'py>,
        model_path: &str,
        dtype_options: &str,
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

        error!("model_path2: {model_full_path}");

        let torch = py.import("torch")?;
        // from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
        let unet_sdxl = py.import("nunchaku.models.unets.unet_sdxl")?;

        // 创建模型实例
        let kwargs = PyDict::new(py);
        kwargs.set_item("torch_dtype", torch.getattr(dtype_options)?)?;
        // kwargs.set_item("device", "cpu")?;
        // kwargs.set_item("offload", false)?;

        let unet = unet_sdxl
            .getattr("NunchakuSDXLUNet2DConditionModel")?
            .call_method("from_pretrained", (model_full_path,), Some(&kwargs))?;

        // 移动到GPU
        // let unet = unet.call_method1("to", ("cuda",))?;

        // 移动到CPU
        // let unet = unet.call_method1("to", ("cpu",))?;

        Ok((unet,))
    }
}
