//! Flux Kontext Inpainting Conditioning
//!
//! 基于 Flux Kontext Inpainting Conditioning 的 Rust 实现。
//!
//! 引用: https://github.com/ZenAI-Vietnam/ComfyUI-Kontext-Inpainting
//!

use std::collections::HashMap;

use candle_core::{Device, IndexOp, Tensor};
use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyDict, PyDictMethods, PyList, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_CONDITIONING,
    error::Error,
    wrapper::{
        comfy::{
            node_helpers::{
                conditioning_set_values, conditionings_py2rs, conditionings_rs2py, latent_rs2py,
                Conditioning, ConditioningEtx,
            },
            vae::Vae,
        },
        comfyui::{
            types::{
                NODE_BOOLEAN, NODE_CONDITIONING, NODE_IMAGE, NODE_LATENT, NODE_MASK, NODE_VAE,
            },
            PromptServer,
        },
        torch::{
            nn::functional::{interpolate, InterpolationMode},
            tensor::TensorWrapper,
        },
    },
};

/// Flux Kontext Inpainting Conditioning
#[pyclass(subclass)]
pub struct FluxKontextInpaintingConditioning {
    device: Device,
}

impl PromptServer for FluxKontextInpaintingConditioning {}

#[pymethods]
impl FluxKontextInpaintingConditioning {
    #[new]
    fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    #[classattr]
    #[pyo3(name = "EXPERIMENTAL")]
    fn experimental() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str) {
        (NODE_CONDITIONING, NODE_LATENT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("conditioning", "latent")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_CONDITIONING;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Kontext Inpainting.
        Rust implementation based on Flux Kontext Inpainting Condition.
        "
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

                required.set_item(
                    "conditioning",
                    (NODE_CONDITIONING, {
                        let conditioning = PyDict::new(py);
                        conditioning
                    }),
                )?;
                required.set_item(
                    "vae",
                    (NODE_VAE, {
                        let vae = PyDict::new(py);
                        vae
                    }),
                )?;
                required.set_item(
                    "pixels",
                    (NODE_IMAGE, {
                        let pixels = PyDict::new(py);
                        pixels
                    }),
                )?;
                required.set_item(
                    "mask",
                    (NODE_MASK, {
                        let mask = PyDict::new(py);
                        mask
                    }),
                )?;
                required.set_item(
                    "noise_mask",
                    (NODE_BOOLEAN, {
                        let noise_mask = PyDict::new(py);
                        noise_mask.set_item("default", true)?;
                        noise_mask.set_item(
                            "tooltip",
                            "Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model."
                            )?;
                        noise_mask
                    }),
                )?;

                required
            })?;
            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        conditioning: Bound<'py, PyList>,
        vae: Bound<'py, PyAny>,
        pixels: Bound<'py, PyAny>,
        mask: Bound<'py, PyAny>,
        noise_mask: bool,
    ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyDict>)> {
        let results = self.encode(py, conditioning, vae, pixels, mask, noise_mask);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("FluxKontextInpaintingConditioning error, {e}");
                if let Err(e) = self.send_error(
                    py,
                    "FluxKontextInpaintingConditioning".to_string(),
                    e.to_string(),
                ) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl FluxKontextInpaintingConditioning {
    /// Encode the conditioning and pixels into a latent vector
    ///
    /// pixels: [B, H, W, C]
    /// mask: [B, H, W]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        conditioning: Bound<'py, PyList>,
        vae: Bound<'py, PyAny>,
        pixels: Bound<'py, PyAny>,
        mask: Bound<'py, PyAny>,
        noise_mask: bool,
    ) -> Result<(Bound<'py, PyList>, Bound<'py, PyDict>), Error> {
        // py params to rust
        let conditionings = conditionings_py2rs(conditioning, &self.device)?;
        let vae = Vae::new(vae)?;
        let pixels = TensorWrapper::<f32>::new(&pixels, &self.device)?.into_tensor();
        let mask = TensorWrapper::<f32>::new(&mask, &self.device)?.into_tensor();

        let x = pixels.dims()[1] / 8 * 8;
        let y = pixels.dims()[2] / 8 * 8;

        // 在第 1 维度上插入一个新的维度, [B, H, W] -> [B, 1, H, W]
        let mask = mask.unsqueeze(1)?;

        let mut mask = interpolate(
            py,
            &mask,
            Some((pixels.dims()[1], pixels.dims()[2])),
            None::<f32>,
            InterpolationMode::Bilinear,
            None,
            None,
            false,
        )?;

        let orig_pixels = pixels.clone();
        let mut pixels = orig_pixels.clone();
        if pixels.dims()[1] != x || pixels.dims()[2] != y {
            let x_offset = (pixels.dims()[1] % 8) / 2;
            let y_offset = (pixels.dims()[2] % 8) / 2;

            pixels = pixels
                .narrow(1, x_offset, x)? // 第1维: height [x_offset : x_offset+x]
                .narrow(2, y_offset, y)?; // 第2维: width [y_offset : y_offset+y]
            mask = mask
                .narrow(2, x_offset, x)? // 第2维: height [x_offset : x_offset+x]
                .narrow(3, y_offset, y)?; // 第3维: width [y_offset : y_offset+y]
        }

        // 计算反转并四舍五入的掩码（形状: [B, H, W]）
        let mut m = mask.round()?; // 四舍五入 [3]
        m = (1.0 - m)?; // 反转掩码
        m = m.squeeze(1)?; // 移除通道维度 [B, 1, H, W] -> [B, H, W]

        // 拆分通道（形状: [B, H, W, 1] * 3）
        let channels = pixels.chunk(3, 3)?;

        // 处理每个通道
        let processed = channels
            .into_iter()
            .map(|ch| {
                let ch = ch.squeeze(3)?; // 移除通道维度
                let ch = (ch - 0.5)?;
                let ch = ch.broadcast_mul(&m)?;
                (ch + 0.5)?.unsqueeze(3) // 恢复通道维度
            })
            .collect::<Result<Vec<_>, _>>()?;

        // 重组张量
        let pixels = Tensor::cat(&processed, 3)?; // 形状恢复为 [B,H,W,3]

        let concat_latent = vae.encode(py, &pixels)?;
        let orig_latent = vae.encode(py, &orig_pixels)?;

        let pixels_latent = self._encode_latent(py, &vae, orig_pixels)?;

        let c = conditioning_set_values(
            conditionings,
            HashMap::from([
                (
                    "concat_latent_image".to_string(),
                    ConditioningEtx::Tensor(concat_latent),
                ),
                (
                    "concat_mask".to_string(),
                    ConditioningEtx::Tensor(mask.clone()),
                ),
            ]),
            false,
        )?;

        let conditionings = self._concat_conditioning_latent(c, Some(pixels_latent))?;

        let mut out_latent = HashMap::new();
        out_latent.insert("samples".to_string(), orig_latent);
        if noise_mask {
            out_latent.insert("noise_mask".to_string(), mask);
        }

        let conditionings_py = conditionings_rs2py(py, conditionings)?;
        let out_latent_py = latent_rs2py(py, out_latent)?;

        Ok((conditionings_py, out_latent_py))
    }

    /// mask reshape
    /// python 的 reshape 函数的实现
    ///
    /// 使用 unsqueeze(1) 替代？
    ///
    /// [B, H, W] -> [B, 1, H, W]
    fn _reshape_mask(mask: &Tensor) -> Result<Tensor, Error> {
        let dims: &[usize] = mask.dims(); // 获取当前形状 [d0, d1, ..., dn]

        // 检查至少2维
        if dims.len() < 2 {
            return Err(Error::InvalidTensorShape(
                "Tensor must have ≥2 dimensions".to_string(),
            ));
        }

        // 计算新形状
        let total_elements: usize = dims.iter().product();
        let last_two_dims = dims[dims.len() - 2] * dims[dims.len() - 1];
        let inferred_dim = total_elements / last_two_dims; // 等效于Python的-1

        let new_shape = [
            inferred_dim,         // 自动计算的维度
            1,                    // 新增的维度
            dims[dims.len() - 2], // 倒数第2维
            dims[dims.len() - 1], // 最后1维
        ];

        Ok(mask.reshape(&new_shape)?)
    }

    /// Encode the pixels into a latent vector
    /// pixels[:, :, :, :3]  # shape: [B, H, W, 3]
    fn _encode_latent<'py>(
        &self,
        py: Python<'py>,
        vae: &Vae,
        pixels: Tensor,
    ) -> Result<HashMap<String, Tensor>, Error> {
        let rgb_pixels = pixels.i((.., .., .., ..3))?;
        let t = vae.encode(py, &rgb_pixels)?;
        Ok(HashMap::from([("samples".to_string(), t)]))
    }

    /// Concatenate the conditioning and latent vector
    fn _concat_conditioning_latent(
        &self,
        conditioning: Vec<Conditioning>,
        latent: Option<HashMap<String, Tensor>>,
    ) -> Result<Vec<Conditioning>, Error> {
        let mut conditioning = conditioning.clone();
        if let Some(latent) = latent {
            conditioning = conditioning_set_values(
                conditioning,
                HashMap::from([(
                    "reference_latents".to_string(),
                    ConditioningEtx::Tensors(vec![latent["samples"].clone()]),
                )]),
                true,
            )?;
        }

        Ok(conditioning)
    }
}
