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
    core::{
        category::CATEGORY_IMAGE,
        tensor_wrapper::TensorWrapper,
        types::{NODE_IMAGE, NODE_INT},
    },
    error::Error,
};

/// 缩放模式枚举
#[pyclass]
#[derive(Debug, Clone, EnumString, Display)]
enum ResizeMode {
    #[strum(to_string = "Just Resize")]
    Resize,
    #[strum(to_string = "Crop and Resize")]
    InnerFit,
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
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);
                required.set_item(
                    "image",
                    ("IMAGE", {
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
        let image = TensorWrapper::new::<f32>(&image, &self.device)?.into_tensor();

        let (resolution, height, width, batch) = self.resolution(&image, resize_mode).unwrap();

        let tensor_wrapper: TensorWrapper = image.into();
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
        let (width, height) = (raw_w as f64, raw_h as f64);

        // let (k0, k1) = (height / raw_h as f64, width / raw_w as f64);
        // let estimation = match resize_mode {
        //     ResizeMode::OuterFit => f64::min(k0, k1) * f64::min(height, width),
        //     _ => f64::max(k0, k1) * f64::min(height, width),
        // };

        // // 最佳分辨率
        // let resolution = estimation.round() as usize;

        let resolution = self.calculate_perfect_resolution(
            width,
            height,
            raw_w as f64,
            raw_h as f64,
            resize_mode,
        )?;

        Ok((resolution, width as usize, height as usize, batch))
    }

    /// 核心计算逻辑（网页1的枚举模式匹配最佳实践）
    fn calculate_perfect_resolution(
        &self,
        raw_w: f64,
        raw_h: f64,
        target_w: f64,
        target_h: f64,
        mode: ResizeMode,
    ) -> Result<usize, Error> {
        // 各模式独立计算
        let estimation = match mode {
            ResizeMode::Resize => {
                // j仅拉伸, 采用最大缩放系数保证目标区域完全覆盖
                let scale_w = target_w / raw_w;
                let scale_h = target_h / raw_h;
                let scale = scale_w.max(scale_h);
                (raw_w * scale, raw_h * scale)
            }
            ResizeMode::InnerFit => {
                // 裁剪并拉伸, 保持宽高比的最小缩放策略，避免图像变形
                let scale_w = target_w / raw_w;
                let scale_h = target_h / raw_h;
                let scale = scale_w.min(scale_h);
                (raw_w * scale, raw_h * scale)
            }
            ResizeMode::OuterFit => {
                // 拉伸并填充, 智能填充算法，根据宽高比差异动态选择适配方向
                let aspect_ratio = raw_w / raw_h;
                let target_ratio = target_w / target_h;

                if aspect_ratio > target_ratio {
                    (target_w, target_w / aspect_ratio)
                } else {
                    (target_h * aspect_ratio, target_h)
                }
            }
        };

        // 像素完美对齐
        let (w, h) = self.pixel_perfect_adjustment(estimation.0, estimation.1)?;
        Ok((w * h) as usize) // 返回总像素量作为分辨率评估
    }

    /// 像素对齐优化
    fn pixel_perfect_adjustment(&self, width: f64, height: f64) -> Result<(u32, u32), Error> {
        const PPU: f64 = 8.0; // 像素单位
        let unit = 1.0 / PPU;

        Ok((
            (width / unit).round() as u32 * PPU as u32,
            (height / unit).round() as u32 * PPU as u32,
        ))
    }
}
