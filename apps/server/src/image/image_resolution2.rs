//! 图片分辨率
//! - comfyui-easy-use 实现
//!
//! 依赖:
//! - python: torch
use std::collections::HashMap;

use candle_core::{Device, Tensor};
use pyo3::{
    prelude::*,
    types::{PyDict, PyType},
};
use strum_macros::{Display, EnumString};

use crate::{
    core::category::CATEGORY_IMAGE,
    error::Error,
    wrapper::{
        comfyui::types::{NODE_IMAGE, NODE_INT},
        torch::tensor::TensorWrapper,
    },
};

/// 图像缩放模式枚举
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum ResizeMode {
    /// 简单调整大小 (可能改变宽高比)
    #[strum(to_string = "Just Resize")]
    Resize,

    /// 裁剪并调整大小 (保持宽高比，裁剪超出部分)
    #[strum(to_string = "Crop and Resize")]
    InnerFit,

    /// 调整大小并填充 (保持宽高比，填充不足部分)
    #[strum(to_string = "Resize and Fill")]
    OuterFit,
}

// 图片分辨率
#[pyclass(subclass)]
pub struct ImageResolution2 {
    device: Device,
}

#[pymethods]
impl ImageResolution2 {
    #[new]
    fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    ) {
        (NODE_IMAGE, NODE_INT, NODE_INT, NODE_INT, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    ) {
        ("image", "resolution", "width", "height", "count")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool, bool, bool, bool) {
        (false, false, false, false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_IMAGE;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "This is a node for obtaining image resolution implemented by 'comfyui easy use'."
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
                    "image",
                    (NODE_IMAGE, {
                        let params = PyDict::new(py);
                        params
                    }),
                )?;
                required.set_item(
                    "resize_mode",
                    (
                        vec![
                            ResizeMode::Resize.to_string(),
                            ResizeMode::InnerFit.to_string(),
                            ResizeMode::OuterFit.to_string(),
                        ],
                        {
                            let mut encoding = HashMap::new();
                            encoding.insert("default", ResizeMode::Resize.to_string());
                            encoding
                        },
                    ),
                )?;
                required
            })?;
            Ok(dict.into())
        })
    }

    fn execute<'py>(
        &self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        resize_mode: String,
    ) -> PyResult<(Bound<'py, PyAny>, usize, usize, usize, usize)> {
        let image = TensorWrapper::<f32>::new(&image, &self.device)?.into_tensor();

        let (resolution, height, width, batch) = self.resolution(&image, resize_mode).unwrap();

        let tensor_wrapper: TensorWrapper<f32> = image.into();
        let py_tensor = tensor_wrapper.to_py_tensor(py)?;

        Ok((py_tensor, resolution, width, height, batch))
    }
}

impl ImageResolution2 {
    /// 获取图片分辨率
    fn resolution(
        &self,
        image: &Tensor,
        resize_mode: String,
    ) -> Result<(usize, usize, usize, usize), Error> {
        let (batch, raw_h, raw_w, _channels) = image.dims4()?;

        // 解析枚举并执行模式匹配
        let resize_mode = resize_mode
            .parse::<ResizeMode>()
            .map_err(|e| Error::ParseEnumString(e.to_string()))?;

        let (width, height) = (raw_w, raw_h);
        let (k0, k1) = (height as f32 / raw_h as f32, width as f32 / raw_w as f32);

        let resolution = match resize_mode {
            ResizeMode::OuterFit => f32::min(k0, k1) * f32::min(height as f32, width as f32),
            _ => f32::max(k0, k1) * f32::min(height as f32, width as f32),
        };

        // 8倍像素
        const PPU: f32 = 8.0;
        let resolution = ((resolution / PPU).round() * PPU) as usize;

        Ok((resolution, width, height, batch))
    }
}
