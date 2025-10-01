//! 列表转批次

use candle_core::{Device, IndexOp, Tensor};
use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyList, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};
use strum_macros::{Display, EnumString};

use crate::{
    core::category::CATEGORY_LIST,
    error::Error,
    wrapper::{
        comfyui::{types::any_type, PromptServer},
        python::isinstance,
        torch::{nn::functional::interpolation::Interpolation, tensor::TensorWrapper},
    },
};

/// 图像调整方法枚举
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum ResizeMethod {
    /// 拉伸图像以完全匹配目标尺寸 (可能改变宽高比)
    #[strum(to_string = "stretch")]
    Stretch,

    /// 保持宽高比调整大小 (可能不会完全填充目标尺寸)
    #[strum(to_string = "keep proportion")]
    KeepProportion,

    /// 填充或裁剪以完全匹配目标尺寸 (保持宽高比)
    #[strum(to_string = "fill / crop")]
    FillOrCrop,

    /// 保持原始宽高比并填充边缘 (保持宽高比)
    #[strum(to_string = "pad")]
    Pad,
}

/// 填充或裁剪
#[derive(Debug, Default)]
pub struct Padding {
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
}

/// 偏移
#[derive(Debug, Default)]
pub struct Offset {
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
}

/// 列表转批次
#[pyclass(subclass)]
pub struct ListToBatch {
    device: Device,
}

impl PromptServer for ListToBatch {}

#[pymethods]
impl ListToBatch {
    #[new]
    fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types(py: Python<'_>) -> (Bound<'_, PyAny>,) {
        let any_type = any_type(py).unwrap();
        (any_type,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("batch",)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LIST;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Convert any list into batches."
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
                    "list",
                    (any_type(py)?, {
                        let list = PyDict::new(py);
                        list.set_item("tooltip", "Input any list")?;
                        list
                    }),
                )?;
                required
            })?;
            Ok(dict.into())
        })
    }

    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        list: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        // 判断列表是否为空
        if list.is_empty() {
            // 空列表
            let empty = PyList::new(py, list)?.into_any();
            return Ok((empty,));
        }
        // 判断 list 中是否为图片
        let item = &list[0];
        // 图片
        if isinstance(py, item, "torch.Tensor")? {
            // 将列表转换为图片批次
            let images = self
                .image_list_to_batch(py, &list)
                .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
            return Ok((images,));
        }

        let list = PyList::new(py, list)?.into_any();
        Ok((list,))
    }
}

impl ListToBatch {
    /// 将列表转换为图片批次
    pub fn image_list_to_batch<'py>(
        &self,
        py: Python<'py>,
        images: &Vec<Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, Error> {
        // 1. 将Python对象转换为Tensor列表
        let image_tensors: Vec<Tensor> = images
            .iter()
            .map(|image| {
                TensorWrapper::new::<f32>(image, &self.device).map(|wrapper| wrapper.into_tensor())
            })
            .collect::<Result<Vec<_>, _>>()?;

        // 2. 检查是否有至少一个张量
        if image_tensors.is_empty() {
            return Err(Error::TensorErr(candle_core::Error::msg(
                "Cannot create batch from empty list",
            )));
        }

        // 3. 获取第一个张量的形状作为参考 [1,H,W,C]
        let first_image_tensor = &image_tensors[0];
        let first_image_shape = first_image_tensor.dims();
        let (batch, height, width) = (
            first_image_shape[0],
            first_image_shape[1],
            first_image_shape[2],
        );
        if first_image_shape.len() != 4 || batch != 1 {
            return Err(Error::TensorErr(candle_core::Error::msg(
                "Expected tensors with shape [1, H, W, C]",
            )));
        }

        // 4. 调整后续张量形状以匹配第一个张量
        for image_tensor in image_tensors[1..].iter() {
            let image_tensor_shape = image_tensor.dims();

            // 检查高度和宽度是否匹配
            if image_tensor_shape[1..2] != first_image_shape[1..2] {
                let tensor = image_tensor.i(0)?;
                error!("=============: {:#?}", tensor.dims());
            }

            // if tensor_shape[2] != first_shape[2] || tensor_shape[1] != first_shape[1] {
            //     // 需要上采样 (模拟Python中的common_upscale)
            //     *tensor = self.upscale_tensor(
            //         tensor,
            //         first_shape[2], // target width
            //         first_shape[1], // target height
            //         "bilinear",
            //         "center",
            //     )?;

            //     // 确保通道维度在正确位置
            //     if tensor_shape[3] != first_shape[3] {
            //         *tensor = tensor.permute([0, 3, 1, 2])?; // [1,H,W,C] -> [1,C,H,W]
            //         *tensor = tensor.permute([0, 2, 3, 1])?; // [1,C,H,W] -> [1,H,W,C]
            //     }
            // }
        }

        // 5. 在批次维度上拼接张量
        let image_batch = Tensor::cat(&image_tensors, 0)?;

        // 6. 转换回Python对象
        let py_batch = TensorWrapper::from_tensor(image_batch).to_py_tensor(py)?;
        Ok(py_batch)
    }

    /// 张量缩放
    ///
    /// 输入张量形状 [1,H,W,C]
    #[allow(clippy::too_many_arguments)]
    fn upscale_tensor<'py>(
        &self,
        py: Python<'py>,
        image: &Tensor,
        width: usize,
        height: usize,
        interpolation: &str,
        method: ResizeMethod,
        condition: &str,
        multiple_of: usize,
    ) -> Result<Tensor, Error> {
        // 获取当前张量形状 [1,H,W,C]
        let dims = image.dims();
        if dims.len() != 4 || dims[0] != 1 {
            return Err(Error::TensorErr(candle_core::Error::msg(
                "Expected tensor with shape [1, H, W, C]",
            )));
        }

        // 乘以系数
        let (width, height) = if multiple_of > 1 {
            (
                width - (width % multiple_of),
                height - (height % multiple_of),
            )
        } else {
            (width, height)
        };

        // 图片缩放
        let (width, height, offset, pad) = Self::image_resize(image, width, height, method)?;

        let (_, oh, ow, _) = image.dims4()?;
        let should_resize = match condition {
            "always" => true,
            "downscale if bigger" => oh > height || ow > width,
            "upscale if smaller" => oh < height || ow < width,
            "bigger area" => oh * ow > height * width,
            "smaller area" => oh * ow < height * width,
            _ => false,
        };
        if should_resize {
            // 转换为 [1,C,H,W] 格式以进行上采样
            let mut tensor = image.permute([0, 3, 1, 2])?; // [1,H,W,C] -> [1,C,H,W]

            // 根据插值方法选择上采样方式
            // 解析枚举并执行模式匹配
            let interpolation = interpolation
                .parse::<Interpolation>()
                .map_err(|e| Error::ParseEnumString(e.to_string()))?;

            tensor = match interpolation {
                Interpolation::Nearest => Interpolation::nearest(&tensor, height, width)?,
                Interpolation::Bilinear => Interpolation::bilinear(py, &tensor, height, width)?,
                Interpolation::Bicubic => Interpolation::bicubic(py, &tensor, height, width)?,
                Interpolation::Area => Interpolation::area(py, &tensor, height, width)?,
                Interpolation::NearestExact => {
                    Interpolation::nearest_exact(py, &tensor, height, width)?
                }
                Interpolation::Lanczos => Interpolation::lanczos(&tensor, height, width)?,
            };
            if method == ResizeMethod::Pad
                && (pad.left > 0 || pad.right > 0 || pad.top > 0 || pad.bottom > 0)
            {
                todo!();
            }
        }

        // 4. 转换回 [1,H,W,C] 格式
        // let resized = resized.permute([0, 2, 3, 1])?; // [1,C,H,W] -> [1,H,W,C]

        Ok(resized)
    }

    fn image_resize(
        image: &Tensor,
        width: usize,
        height: usize,
        method: ResizeMethod,
    ) -> Result<(usize, usize, Offset, Padding), Error> {
        const MAX_RESOLUTION: usize = 16384;
        let mut offset = Offset::default();
        let mut pad = Padding::default();
        let (_, oh, ow, _) = image.dims4()?;

        let (width, height) = match method {
            ResizeMethod::KeepProportion | ResizeMethod::Pad => {
                // 处理宽度或高度为0的情况
                let (width, height) = if width == 0 && oh < height {
                    (MAX_RESOLUTION, height)
                } else if width == 0 && oh >= height {
                    (ow, height)
                } else if height == 0 && ow < width {
                    (width, MAX_RESOLUTION)
                } else if height == 0 && ow >= width {
                    (width, oh)
                } else {
                    (width, height)
                };

                let ratio = (width as f32 / ow as f32).min(height as f32 / oh as f32);
                let new_width = (ow as f32 * ratio).round() as usize;
                let new_height = (oh as f32 * ratio).round() as usize;

                if method == ResizeMethod::Pad {
                    pad.left = (width - new_width) / 2;
                    pad.right = width - new_width - pad.left;
                    pad.top = (height - new_height) / 2;
                    pad.bottom = height - new_height - pad.top;
                }

                (new_width, new_height)
            }
            ResizeMethod::FillOrCrop => {
                let width = if width > 0 { width } else { ow };
                let height = if height > 0 { height } else { oh };

                let ratio = (width as f32 / ow as f32).max(height as f32 / oh as f32);
                let new_width = (ow as f32 * ratio).round() as usize;
                let new_height = (oh as f32 * ratio).round() as usize;

                let x = ((new_width - width) / 2) as i32;
                let y = ((new_height - height) / 2) as i32;
                offset.x2 = offset.x1 + width;
                offset.y2 = offset.y1 + height;

                if offset.x2 > new_width {
                    offset.x1 -= offset.x2 - new_width;
                }
                if x < 0 {
                    offset.x1 = 0;
                }
                if offset.y2 > new_height {
                    offset.y1 -= offset.y2 - new_height;
                }
                if y < 0 {
                    offset.y1 = 0;
                }

                (new_width, new_height)
            }
            ResizeMethod::Stretch => {
                let width = if width > 0 { width } else { ow };
                let height = if height > 0 { height } else { oh };

                (width, height)
            }
        };

        Ok((width, height, offset, pad))
    }
}
