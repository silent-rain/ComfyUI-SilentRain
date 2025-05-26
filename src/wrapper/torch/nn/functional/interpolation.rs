//! 图像插值方法枚举
//!

use candle_core::{DType, Device, Tensor};
use image::{imageops::FilterType, DynamicImage, ImageBuffer, Rgb};
use pyo3::{pyclass, types::PyAnyMethods, IntoPyObject, Python};
use strum_macros::{Display, EnumString};

use crate::{error::Error, wrapper::torch::tensor::TensorWrapper};

/// 图像插值方法枚举
///
/// 定义了在图像缩放、旋转等变换时使用的像素插值算法
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
pub enum Interpolation {
    /// 最近邻插值法
    ///
    /// - 速度最快但质量最低
    /// - 直接取最近的像素值，会产生锯齿
    /// - 适用于像素艺术或需要保留锐利边缘的情况
    #[strum(to_string = "nearest")]
    Nearest,

    /// 双线性插值法
    ///
    /// - 中等速度和质量
    /// - 对周围4个像素进行加权平均
    /// - 适用于大多数常规缩放场景
    #[strum(to_string = "bilinear")]
    Bilinear,

    /// 双三次插值法
    ///
    /// - 较高质量但速度较慢
    /// - 对周围16个像素进行三次多项式插值
    /// - 产生更平滑的边缘，适合照片放大
    #[strum(to_string = "bicubic")]
    Bicubic,

    /// 区域插值法
    ///
    /// - 通过像素区域关系进行插值
    /// - 缩小图像时效果较好，能保留更多细节
    /// - 放大时会产生块状效果
    #[strum(to_string = "area")]
    Area,

    /// 精确最近邻插值法
    ///
    /// - 改进版的最近邻插值
    /// - 更精确的像素位置计算
    /// - 适用于需要精确像素位置的科学计算
    #[strum(to_string = "nearest-exact")]
    NearestExact,

    /// Lanczos插值法
    ///
    /// - 最高质量但计算量最大
    /// - 使用Lanczos窗口函数进行插值
    /// - 能保留锐利边缘同时减少锯齿
    /// - 适合高质量图像处理
    #[strum(to_string = "lanczos")]
    Lanczos,
}

impl Interpolation {
    /// 最近邻插值法
    pub fn nearest(tensor: &Tensor, width: usize, height: usize) -> Result<Tensor, Error> {
        let result = tensor.upsample_nearest2d(height, width)?;
        Ok(result)
    }

    /// 双线性插值法
    pub fn bilinear<'py>(
        py: Python<'py>,
        tensor: &Tensor,
        width: usize,
        height: usize,
    ) -> Result<Tensor, Error> {
        let outputs = py_interpolate(
            py,
            tensor,
            Some((width, height)),
            None::<f32>,
            InterpolationMode::Bilinear,
            None,
            None,
            false,
        )?;

        Ok(outputs)
    }

    /// 双三次插值法
    pub fn bicubic<'py>(
        py: Python<'py>,
        tensor: &Tensor,
        width: usize,
        height: usize,
    ) -> Result<Tensor, Error> {
        let outputs = py_interpolate(
            py,
            tensor,
            Some((width, height)),
            None::<f32>,
            InterpolationMode::Bicubic,
            None,
            None,
            false,
        )?;

        Ok(outputs)
    }

    /// 区域插值法
    pub fn area<'py>(
        py: Python<'py>,
        tensor: &Tensor,
        width: usize,
        height: usize,
    ) -> Result<Tensor, Error> {
        let outputs = py_interpolate(
            py,
            tensor,
            Some((width, height)),
            None::<f32>,
            InterpolationMode::Area,
            None,
            None,
            false,
        )?;

        Ok(outputs)
    }

    /// 精确最近邻插值法
    pub fn nearest_exact<'py>(
        py: Python<'py>,
        tensor: &Tensor,
        width: usize,
        height: usize,
    ) -> Result<Tensor, Error> {
        let outputs = py_interpolate(
            py,
            tensor,
            Some((width, height)),
            None::<f32>,
            InterpolationMode::NearestExact,
            None,
            None,
            false,
        )?;

        Ok(outputs)
    }

    /// 批处理 Lanczos 插值
    ///
    /// (输入输出均为 [N,H,W,C])
    pub fn lanczos(tensor: &Tensor, width: usize, height: usize) -> Result<Tensor, Error> {
        let batch_size = tensor.dims()[0];
        let device = tensor.device();

        // 并行处理每个样本
        let results: Vec<Tensor> = (0..batch_size)
            .map(|i| {
                let img = Self::tensor_slice_to_image(tensor, i)?;
                let resized = img.resize(
                    width as u32,
                    height as u32,
                    image::imageops::FilterType::Lanczos3,
                );
                Self::image_to_tensor_slice(resized, device)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // 重新堆叠为批处理形式 [N,H,W,C]
        Ok(Tensor::stack(&results, 0)?)
    }

    /// 使用Lanczos插值调整张量大小
    ///
    /// 输入张量形状 [1, H, W, C]
    pub fn lanczos2(tensor: &Tensor, width: usize, height: usize) -> Result<Tensor, Error> {
        // 1. 验证输入张量形状 [1, H, W, C]
        let dims = tensor.dims();
        if dims.len() != 4 || dims[0] != 1 {
            return Err(Error::TensorErr(candle_core::Error::msg(
                "Expected tensor with shape [1, H, W, C]",
            )));
        }

        // 2. 获取设备信息
        let device = tensor.device();
        let dtype = tensor.dtype();

        // 3. 将张量转换为CPU上的f32类型
        let tensor = tensor.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
        let (_, orig_height, orig_width, channels) = tensor.dims4()?;

        // 4. 转换为图像格式
        let img_buf = tensor.squeeze(0)?; // [H, W, C]
        let img_array: Vec<f32> = img_buf.flatten_all()?.to_vec1()?;

        // 5. 创建图像缓冲区
        let mut img_buf = ImageBuffer::new(orig_width as u32, orig_height as u32);
        for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
            let base = ((y * orig_width as u32 + x) * channels as u32) as usize;
            let r = img_array[base];
            let g = if channels > 1 { img_array[base + 1] } else { r };
            let b = if channels > 1 { img_array[base + 2] } else { r };
            let a = if channels > 3 {
                img_array[base + 3]
            } else {
                1.0
            };

            *pixel = image::Rgba([
                (r * 255.0).clamp(0.0, 255.0) as u8,
                (g * 255.0).clamp(0.0, 255.0) as u8,
                (b * 255.0).clamp(0.0, 255.0) as u8,
                (a * 255.0).clamp(0.0, 255.0) as u8,
            ]);
        }

        // 6. 使用Lanczos重采样
        let dyn_img = DynamicImage::ImageRgba8(img_buf);
        let resized = dyn_img.resize_exact(width as u32, height as u32, FilterType::Lanczos3);

        // 7. 转换回张量
        let resized_buf = resized.to_rgba8();
        let mut resized_data = Vec::with_capacity(width * height * channels);

        for pixel in resized_buf.pixels() {
            resized_data.push(pixel[0] as f32 / 255.0);
            if channels > 1 {
                resized_data.push(pixel[1] as f32 / 255.0);
            }
            if channels > 2 {
                resized_data.push(pixel[2] as f32 / 255.0);
            }
            if channels > 3 {
                resized_data.push(pixel[3] as f32 / 255.0);
            }
        }

        // 8. 创建并返回张量
        let result = Tensor::from_vec(resized_data, (1, height, width, channels), &Device::Cpu)?
            .to_device(device)?
            .to_dtype(dtype)?;

        Ok(result)
    }
}

impl Interpolation {
    /// 将 [N,H,W,C] 张量的单个样本转为图像
    fn tensor_slice_to_image(tensor: &Tensor, idx: usize) -> Result<DynamicImage, Error> {
        let dims = tensor.dims();
        let (_, h, w, c) = (dims[0], dims[1], dims[2], dims[3]);

        // 提取单个样本 [H,W,C]
        let slice = tensor.narrow(0, idx, 1)?.squeeze(0)?;

        // 获取数据并确保在 [0,1] 范围
        let data = slice.flatten_all()?.to_vec1::<f32>()?;

        match c {
            1 => Ok(ImageBuffer::from_fn(w as u32, h as u32, |x, y| {
                let v = data[y as usize * w + x as usize].clamp(0.0, 1.0);
                image::Luma([(v * 255.0) as u8])
            })
            .into()),
            3 => Ok(ImageBuffer::from_fn(w as u32, h as u32, |x, y| {
                let base = (y as usize * w + x as usize) * 3;
                Rgb([
                    (data[base].clamp(0.0, 1.0) * 255.0) as u8,
                    (data[base + 1].clamp(0.0, 1.0) * 255.0) as u8,
                    (data[base + 2].clamp(0.0, 1.0) * 255.0) as u8,
                ])
            })
            .into()),
            _ => Err(Error::TensorErr(candle_core::Error::msg(format!(
                "Unsupported channels: {c}"
            )))),
        }
    }

    /// 将图像转回张量切片 [H,W,C]
    fn image_to_tensor_slice(img: DynamicImage, device: &Device) -> Result<Tensor, Error> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let rgb = img.to_rgb8();

        let data: Vec<f32> = rgb
            .pixels()
            .flat_map(|p| [p[0], p[1], p[2]].map(|v| v as f32 / 255.0))
            .collect();

        Ok(Tensor::from_vec(data, (h, w, 3), device)?)
    }
}

/// 表示插值算法的枚举
///
/// mode: nearest|linear|bilinear|bicubic|trilinear|area|nearest-exact
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
pub enum InterpolationMode {
    #[strum(to_string = "nearest")]
    Nearest,
    #[strum(to_string = "linear")]
    Linear,
    #[strum(to_string = "bilinear")]
    Bilinear,
    #[strum(to_string = "bicubic")]
    Bicubic,
    #[strum(to_string = "trilinear")]
    Trilinear,
    #[strum(to_string = "area")]
    Area,
    #[strum(to_string = "nearest-exact")]
    NearestExact,
}

/// Down/up samples the input.
///
/// torch.nn.functional.interpolate 的封装
///
/// Args:
/// size: optional, usize or (usize,) or (usize, usize) or (usize, usize, usize)
/// scale_factor: optional, f32 or (f32,)
/// mode: InterpolationMode
/// align_corners: optional, bool
/// recompute_scale_factor: optional, bool
/// antialias: bool
/// device: &Device
#[allow(clippy::too_many_arguments)]
pub fn py_interpolate<'py, SIZE, SF>(
    py: Python<'py>,
    input: &Tensor,
    size: Option<SIZE>,
    scale_factor: Option<SF>,
    mode: InterpolationMode,
    align_corners: Option<bool>,
    recompute_scale_factor: Option<bool>,
    antialias: bool,
) -> Result<Tensor, Error>
where
    SIZE: IntoPyObject<'py>,
    SF: IntoPyObject<'py>,
{
    let device = input.device();

    let torch_module = py.import("torch")?;
    let functional = torch_module.getattr("nn")?.getattr("functional")?;

    let py_input = TensorWrapper::from_tensor(input.clone()).to_py_tensor(py)?;
    let mode = mode.to_string();

    // 准备参数
    let args = (
        py_input,
        size,
        scale_factor,
        mode,
        align_corners,
        recompute_scale_factor,
        antialias,
    );

    // 调用 PyTorch 的 interpolate
    let py_any = functional.call_method("interpolate", args, None)?;
    let result = TensorWrapper::new::<f32>(&py_any, device)?.into_tensor();
    Ok(result)
}
