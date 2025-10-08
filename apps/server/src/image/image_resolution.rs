//! 图片分辨率
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
pub struct ImageResolution {
    device: Device,
}

#[pymethods]
impl ImageResolution {
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
        "Obtain image resolution."
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
                required.set_item("image", (NODE_IMAGE, { PyDict::new(py) }))?;
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

impl ImageResolution {
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

        // 分辨率计算逻辑
        let (width, height) = (raw_w, raw_h);

        let resolution =
            self.calculate_perfect_resolution(width, height, raw_w, raw_h, resize_mode)?;

        Ok((resolution, width, height, batch))
    }

    /// 核心计算逻辑（网页1的枚举模式匹配最佳实践）
    fn calculate_perfect_resolution(
        &self,
        raw_w: usize,
        raw_h: usize,
        target_w: usize,
        target_h: usize,
        mode: ResizeMode,
    ) -> Result<usize, Error> {
        // 各模式独立计算
        let estimation = match mode {
            ResizeMode::Resize => {
                // 仅拉伸, 采用最大缩放系数保证目标区域完全覆盖
                let scale_w = target_w as f32 / raw_w as f32;
                let scale_h = target_h as f32 / raw_h as f32;
                let scale = scale_w.max(scale_h);
                (raw_w as f32 * scale, raw_h as f32 * scale)
            }
            ResizeMode::InnerFit => {
                // 裁剪并拉伸, 保持宽高比的最小缩放策略，避免图像变形
                let scale_w = target_w as f32 / raw_w as f32;
                let scale_h = target_h as f32 / raw_h as f32;
                let scale = scale_w.min(scale_h);
                (raw_w as f32 * scale, raw_h as f32 * scale)
            }
            ResizeMode::OuterFit => {
                // 拉伸并填充, 智能填充算法，根据宽高比差异动态选择适配方向
                let aspect_ratio = raw_w as f32 / raw_h as f32;
                let target_ratio = target_w as f32 / target_h as f32;

                if aspect_ratio > target_ratio {
                    (target_w as f32, target_w as f32 / aspect_ratio)
                } else {
                    (target_h as f32 * aspect_ratio, target_h as f32)
                }
            }
        };

        // 像素完美对齐
        let (w, h) = self.pixel_perfect_adjustment(estimation.0 as usize, estimation.1 as usize)?;
        Ok(w * h) // 返回总像素量作为分辨率评估
    }

    /// 像素对齐优化
    fn pixel_perfect_adjustment(
        &self,
        width: usize,
        height: usize,
    ) -> Result<(usize, usize), Error> {
        const PPU: f32 = 8.0; // 像素单位

        Ok((
            ((width as f32 / PPU).round() * PPU) as usize,
            ((height as f32 / PPU).round() * PPU) as usize,
        ))
    }
}
