//! 图片预置分辨率
//! 提供常用的图像分辨率预设，支持快速选择标准尺寸
//!
//! 预置尺寸包括：
//! - 1:1 (1024x1024, 2048x2048, 4096x4096)
//! - 2:3 (848x1264, 1696x2528, 3392x5056)
//! - 3:2 (1264x848, 2528x1696, 5056x3392)
//! - 3:4 (896x1200, 1792x2400, 3584x4800)
//! - 4:3 (1200x896, 2400x1792, 4800x3584)
//! - 4:5 (928x1152, 1856x2304, 3712x4608)
//! - 5:4 (1152x928, 2304x1856, 4608x3712)
//! - 9:16 (768x1376, 1536x2752, 3072x5504)
//! - 16:9 (1376x768, 2752x1536, 5504x3072)
//! - 21:9 (1584x672, 3168x1344, 6336x2688)
//!
//! 依赖:
//! - python: torch

use candle_core::{Device, Result, Tensor};
use pyo3::{
    prelude::*,
    types::{PyDict, PyType},
};
use std::collections::HashMap;
use strum_macros::{Display, EnumString};

use crate::{
    core::category::CATEGORY_IMAGE,
    error::Error,
    wrapper::{
        comfyui::types::{NODE_IMAGE, NODE_INT, NODE_STRING},
        torch::{nn::functional::interpolation::Interpolation, tensor::TensorWrapper},
    },
};

/// 宽高比枚举
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum AspectRatio {
    /// 1:1 正方形
    #[strum(to_string = "1:1")]
    Ratio1_1,

    /// 2:3 竖版
    #[strum(to_string = "2:3")]
    Ratio2_3,

    /// 3:2 横版
    #[strum(to_string = "3:2")]
    Ratio3_2,

    /// 3:4 竖版
    #[strum(to_string = "3:4")]
    Ratio3_4,

    /// 4:3 横版
    #[strum(to_string = "4:3")]
    Ratio4_3,

    /// 4:5 竖版
    #[strum(to_string = "4:5")]
    Ratio4_5,

    /// 5:4 横版
    #[strum(to_string = "5:4")]
    Ratio5_4,

    /// 9:16 竖版手机
    #[strum(to_string = "9:16")]
    Ratio9_16,

    /// 16:9 横版宽屏
    #[strum(to_string = "16:9")]
    Ratio16_9,

    /// 21:9 超宽屏
    #[strum(to_string = "21:9")]
    Ratio21_9,
}

/// 分辨率级别枚举
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum ResolutionLevel {
    /// 1K 分辨率 (~1000 tokens)
    #[strum(to_string = "1K")]
    K1,

    /// 2K 分辨率 (~2000 tokens)
    #[strum(to_string = "2K")]
    K2,

    /// 4K 分辨率 (~4000 tokens)
    #[strum(to_string = "4K")]
    K4,
}

/// 图像缩放模式
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum ResizeMethod {
    /// 保持宽高比，填充空白
    #[strum(to_string = "Fit (Pad)")]
    Fit,

    /// 保持宽高比，裁剪多余
    #[strum(to_string = "Crop")]
    Crop,

    /// 拉伸到目标尺寸
    #[strum(to_string = "Stretch")]
    Stretch,
}

/// 图片预置分辨率节点
#[pyclass(subclass)]
pub struct ImagePresetResolution {
    device: Device,
}

#[pymethods]
impl ImagePresetResolution {
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
        (NODE_IMAGE, NODE_INT, NODE_INT, NODE_STRING, NODE_STRING)
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
        ("image", "width", "height", "preset_name", "resize_method")
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
        "Resize image to preset resolution with various resize methods. Includes 1K, 2K, and 4K resolutions for common aspect ratios."
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

                // 输入图像
                required.set_item("image", (NODE_IMAGE, { PyDict::new(py) }))?;

                // 宽高比选择
                required.set_item(
                    "aspect_ratio",
                    (
                        vec![
                            AspectRatio::Ratio1_1.to_string(),
                            AspectRatio::Ratio2_3.to_string(),
                            AspectRatio::Ratio3_2.to_string(),
                            AspectRatio::Ratio3_4.to_string(),
                            AspectRatio::Ratio4_3.to_string(),
                            AspectRatio::Ratio4_5.to_string(),
                            AspectRatio::Ratio5_4.to_string(),
                            AspectRatio::Ratio9_16.to_string(),
                            AspectRatio::Ratio16_9.to_string(),
                            AspectRatio::Ratio21_9.to_string(),
                        ],
                        {
                            let mut options = HashMap::new();
                            options.insert("default", AspectRatio::Ratio1_1.to_string());
                            options
                        },
                    ),
                )?;

                // 分辨率级别选择
                required.set_item(
                    "resolution_level",
                    (
                        vec![
                            ResolutionLevel::K1.to_string(),
                            ResolutionLevel::K2.to_string(),
                            ResolutionLevel::K4.to_string(),
                        ],
                        {
                            let mut options = HashMap::new();
                            options.insert("default", ResolutionLevel::K1.to_string());
                            options
                        },
                    ),
                )?;

                // 缩放方法
                required.set_item(
                    "resize_method",
                    (
                        vec![
                            ResizeMethod::Fit.to_string(),
                            ResizeMethod::Crop.to_string(),
                            ResizeMethod::Stretch.to_string(),
                        ],
                        {
                            let mut options = HashMap::new();
                            options.insert("default", ResizeMethod::Fit.to_string());
                            options
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
        aspect_ratio: String,
        resolution_level: String,
        resize_method: String,
    ) -> PyResult<(Bound<'py, PyAny>, usize, usize, String, String)> {
        // 转换输入图像为Tensor
        let input_tensor = TensorWrapper::<f32>::new(&image, &self.device)?.into_tensor();

        // 解析宽高比、分辨率级别和缩放方法
        let aspect_ratio = aspect_ratio
            .parse::<AspectRatio>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let resolution_level = resolution_level
            .parse::<ResolutionLevel>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let resize_method = resize_method
            .parse::<ResizeMethod>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // 获取目标尺寸
        let (target_width, target_height, preset_name) =
            self.get_target_size(aspect_ratio, resolution_level)?;

        // 执行图像缩放
        let resized_tensor =
            self.resize_image(&input_tensor, target_width, target_height, resize_method)?;

        // 转换回Python对象
        let tensor_wrapper: TensorWrapper<f32> = resized_tensor.into();
        let py_tensor = tensor_wrapper.to_py_tensor(py)?;

        Ok((
            py_tensor,
            target_width,
            target_height,
            preset_name,
            resize_method.to_string(),
        ))
    }
}

impl ImagePresetResolution {
    /// 获取目标尺寸
    fn get_target_size(
        &self,
        aspect_ratio: AspectRatio,
        resolution_level: ResolutionLevel,
    ) -> Result<(usize, usize, String), Error> {
        // 根据宽高比和分辨率级别计算尺寸
        let (width, height) = match aspect_ratio {
            AspectRatio::Ratio1_1 => match resolution_level {
                ResolutionLevel::K1 => (1024, 1024),
                ResolutionLevel::K2 => (2048, 2048),
                ResolutionLevel::K4 => (4096, 4096),
            },
            AspectRatio::Ratio2_3 => match resolution_level {
                ResolutionLevel::K1 => (848, 1264),
                ResolutionLevel::K2 => (1696, 2528),
                ResolutionLevel::K4 => (3392, 5056),
            },
            AspectRatio::Ratio3_2 => match resolution_level {
                ResolutionLevel::K1 => (1264, 848),
                ResolutionLevel::K2 => (2528, 1696),
                ResolutionLevel::K4 => (5056, 3392),
            },
            AspectRatio::Ratio3_4 => match resolution_level {
                ResolutionLevel::K1 => (896, 1200),
                ResolutionLevel::K2 => (1792, 2400),
                ResolutionLevel::K4 => (3584, 4800),
            },
            AspectRatio::Ratio4_3 => match resolution_level {
                ResolutionLevel::K1 => (1200, 896),
                ResolutionLevel::K2 => (2400, 1792),
                ResolutionLevel::K4 => (4800, 3584),
            },
            AspectRatio::Ratio4_5 => match resolution_level {
                ResolutionLevel::K1 => (928, 1152),
                ResolutionLevel::K2 => (1856, 2304),
                ResolutionLevel::K4 => (3712, 4608),
            },
            AspectRatio::Ratio5_4 => match resolution_level {
                ResolutionLevel::K1 => (1152, 928),
                ResolutionLevel::K2 => (2304, 1856),
                ResolutionLevel::K4 => (4608, 3712),
            },
            AspectRatio::Ratio9_16 => match resolution_level {
                ResolutionLevel::K1 => (768, 1376),
                ResolutionLevel::K2 => (1536, 2752),
                ResolutionLevel::K4 => (3072, 5504),
            },
            AspectRatio::Ratio16_9 => match resolution_level {
                ResolutionLevel::K1 => (1376, 768),
                ResolutionLevel::K2 => (2752, 1536),
                ResolutionLevel::K4 => (5504, 3072),
            },
            AspectRatio::Ratio21_9 => match resolution_level {
                ResolutionLevel::K1 => (1584, 672),
                ResolutionLevel::K2 => (3168, 1344),
                ResolutionLevel::K4 => (6336, 2688),
            },
        };

        // 确保尺寸是8的倍数（符合深度学习模型要求）
        let width = (width / 8) * 8;
        let height = (height / 8) * 8;

        let name = format!("{} - {}x{}", aspect_ratio, width, height);

        Ok((width, height, name))
    }

    /// 调整图像大小
    fn resize_image(
        &self,
        image: &Tensor,
        target_width: usize,
        target_height: usize,
        method: ResizeMethod,
    ) -> Result<Tensor, Error> {
        let (batch, orig_height, orig_width, channels) = image.dims4()?;

        match method {
            ResizeMethod::Stretch => {
                // 直接拉伸到目标尺寸
                self.interpolate_resize(py, image, target_width, target_height)
            }
            ResizeMethod::Fit => {
                // 保持宽高比，可能需要填充
                self.fit_resize(py, image, target_width, target_height)
            }
            ResizeMethod::Crop => {
                // 保持宽高比，可能需要裁剪
                self.crop_resize(py, image, target_width, target_height)
            }
        }
    }

    /// 插值缩放实现
    fn interpolate_resize<'py>(
        &self,
        py: Python<'py>,
        image: &Tensor,
        width: usize,
        height: usize,
    ) -> Result<Tensor, Error> {
        // 使用项目中现有的双线性插值进行缩放
        Interpolation::bilinear(py, image, width, height)
    }

    /// 适配缩放（保持比例，填充）
    fn fit_resize<'py>(
        &self,
        py: Python<'py>,
        image: &Tensor,
        width: usize,
        height: usize,
    ) -> Result<Tensor, Error> {
        let (_, orig_h, orig_w, _) = image.dims4()?;

        // 计算缩放比例
        let scale = (width as f32 / orig_w as f32).min(height as f32 / orig_h as f32);
        let new_w = (orig_w as f32 * scale) as usize;
        let new_h = (orig_h as f32 * scale) as usize;

        // 先缩放到合适大小
        let resized = self.interpolate_resize(py, image, new_w, new_h)?;

        // 然后填充到目标尺寸
        self.pad_to_size(&resized, width, height)
    }

    /// 裁剪缩放（保持比例，裁剪）
    fn crop_resize<'py>(
        &self,
        py: Python<'py>,
        image: &Tensor,
        width: usize,
        height: usize,
    ) -> Result<Tensor, Error> {
        let (_, orig_h, orig_w, _) = image.dims4()?;

        // 计算缩放比例
        let scale = (width as f32 / orig_w as f32).max(height as f32 / orig_h as f32);
        let new_w = (orig_w as f32 * scale) as usize;
        let new_h = (orig_h as f32 * scale) as usize;

        // 先缩放到合适大小
        let resized = self.interpolate_resize(py, image, new_w, new_h)?;

        // 然后中心裁剪到目标尺寸
        self.center_crop(&resized, width, height)
    }

    /// 填充到指定尺寸
    fn pad_to_size(&self, image: &Tensor, width: usize, height: usize) -> Result<Tensor, Error> {
        let (batch, cur_h, cur_w, channels) = image.dims4()?;

        // 计算填充量
        let pad_h = height.saturating_sub(cur_h);
        let pad_w = width.saturating_sub(cur_w);

        let pad_top = pad_h / 2;
        let pad_bottom = pad_h - pad_top;
        let pad_left = pad_w / 2;
        let pad_right = pad_w - pad_left;

        // 使用常数值填充（中性灰或黑色）
        let padding_value = 0.0f32;
        let padding_tensor = Tensor::full(
            (batch, height, width, channels),
            padding_value,
            image.device(),
        )?;

        // 将原始图像居中放置在填充后的画布上
        let start_h = pad_top;
        let start_w = pad_left;

        // 创建填充后的张量
        let mut padded = padding_tensor;

        // 在中心位置放置原始图像
        padded = padded.slice_assign(
            &[
                0..batch,
                start_h..start_h + cur_h,
                start_w..start_w + cur_w,
                0..channels,
            ],
            image,
        )?;

        Ok(padded)
    }

    /// 中心裁剪
    fn center_crop(&self, image: &Tensor, width: usize, height: usize) -> Result<Tensor, Error> {
        let (_, cur_h, cur_w, _) = image.dims4()?;

        // 计算裁剪起始位置
        let start_h = (cur_h.saturating_sub(height)) / 2;
        let start_w = (cur_w.saturating_sub(width)) / 2;

        // 裁剪到目标尺寸
        let cropped = image
            .narrow(1, start_h, height)?
            .narrow(2, start_w, width)?;

        Ok(cropped)
    }
}
