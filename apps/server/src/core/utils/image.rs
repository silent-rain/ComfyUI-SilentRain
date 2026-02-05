//! image 与 tensor 相互转换
//!
use std::io::Cursor;

use base64::{Engine, engine::general_purpose};
use candle_core::{DType, Device, Tensor};
use image::{
    DynamicImage, GenericImageView, GrayImage, ImageBuffer, ImageFormat, Rgb, RgbImage, RgbaImage,
};
use log::error;
use pyo3::{Bound, PyAny, types::PyAnyMethods};

use crate::error::Error;

/// 将张量转换为图像
///
/// samples: NHWC
pub fn tensor_to_image(samples: &Tensor) -> Result<Vec<DynamicImage>, Error> {
    // NHWC -> NCWH
    let samples = samples.permute((0, 3, 2, 1))?;

    let (batch, _, width, height) = samples.dims4()?;

    // NCWH -> Vec<CWH>
    let mut samples = samples.chunk(batch, 0)?;

    let mut images = Vec::with_capacity(batch);
    for sample in samples.iter_mut() {
        // 维度重排 CWH -> HWC
        let tensor = sample.to_device(&Device::Cpu)?.permute((2, 1, 0))?;

        // 转换为数组视图
        let array = tensor.to_vec1::<f32>()?;

        // 数值处理 (缩放 + clip + 类型转换)
        let data: Vec<u8> = array
            .iter()
            .map(|&x| (255.0 * x).clamp(0.0, 255.0) as u8)
            .collect();

        // 创建图像
        let img_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(width as u32, height as u32, data).ok_or(Error::ImageBuffer)?;

        let image = DynamicImage::ImageRgb8(img_buffer);
        images.push(image);
    }

    Ok(images)
}

/// 将张量转换为图像
///
/// tensor: BHWC
pub fn tensor_to_images2(tensor: &Tensor) -> Result<Vec<DynamicImage>, Error> {
    let (batch, _, _, _) = tensor.dims4()?;

    // BHWC -> Vec<HWC>
    let mut tensors = tensor.chunk(batch, 0)?;

    let mut images = Vec::with_capacity(batch);
    for tensor in tensors.iter_mut() {
        let img = tensor_to_image2(tensor)?;
        images.push(img);
    }

    Ok(images)
}

/// 将张量转换为图像
///
/// tensor: HWC/1HWC
pub fn tensor_to_image2(tensor: &Tensor) -> Result<DynamicImage, Error> {
    // 检查张量形状
    let (height, width, channels) = match tensor.dims() {
        [h, w, c] => (*h, *w, *c),
        [b, h, w, c] if *b == 1 => (*h, *w, *c),
        _ => {
            return Err(Error::InvalidTensorShape(format!(
                "Invalid tensor shape: {:#?}",
                tensor.dims()
            )));
        }
    };

    // 转换为f32并缩放 - 使用运算符重载
    let tensor = (tensor.to_dtype(DType::F32)? * 255.0)?;
    // 或者使用显式创建标量张量的方法：
    // let scalar = Tensor::new(255.0, tensor.device())?;
    // let tensor = tensor.to_dtype(DType::F32)?.mul(&scalar)?;

    // 裁剪到0-255范围
    let tensor = tensor.clamp(0.0, 255.0)?;

    // 转换为u8
    let tensor = tensor.to_dtype(DType::U8)?;

    // //  加维度重排 CHW -> HWC
    // let tensor = tensor.permute([1, 2, 0])?;

    // 转换为缓存
    let buffer = tensor.contiguous()?.flatten_all()?.to_vec1()?;

    // 转换为图像
    match channels {
        1 => {
            // 灰度图像
            let img = GrayImage::from_raw(width as u32, height as u32, buffer)
                .ok_or(Error::ImageBuffer)?;
            Ok(DynamicImage::ImageLuma8(img))
        }
        3 => {
            // RGB图像
            let img = RgbImage::from_raw(width as u32, height as u32, buffer)
                .ok_or(Error::ImageBuffer)?;
            Ok(DynamicImage::ImageRgb8(img))
        }
        4 => {
            // RGBA图像
            let img = RgbaImage::from_raw(width as u32, height as u32, buffer)
                .ok_or(Error::ImageBuffer)?;
            Ok(DynamicImage::ImageRgba8(img))
        }
        _ => Err(Error::UnsupportedNumberOfChannels(channels as u32)),
    }
}

/// 将图像转换为张量
///
/// output: HWC
pub fn image_to_tensor(image: &DynamicImage, device: &Device) -> Result<Tensor, Error> {
    let (width, height) = image.dimensions();

    let img_buffer = image.to_rgb32f().into_raw();
    // HWC
    let tensor = Tensor::from_vec(img_buffer, (height as usize, width as usize, 3), device)?;

    Ok(tensor)
}

/// 从图像中获取mask图像
pub fn image_mask(image: &DynamicImage) -> Result<DynamicImage, Error> {
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();

    // 提取alpha通道
    let alpha_data: Vec<u8> = rgba
        .pixels()
        .filter(|p| p.0[3] == 255)
        .map(|v| v.0[3])
        .collect();

    // 转换为灰度图像
    let img_buffer = ImageBuffer::from_raw(width, height, alpha_data).ok_or(Error::ImageBuffer)?;

    // 转换为DynamicImage然后再转换为Luma8
    let gray_image = DynamicImage::ImageRgba8(img_buffer);

    Ok(gray_image)
}

/// 从图像提取alpha通道并转换成Mask张量
///
/// output: [1, H, W]
pub fn image_mask_to_tensor(image: &DynamicImage, device: &Device) -> Result<Tensor, Error> {
    // 转换为RGBA格式获取alpha通道
    let rgba = image.to_rgba8();
    let (width, height) = rgba.dimensions();

    // 检查是否有透明像素
    let mask = if rgba.pixels().all(|p| p.0[3] == 255) {
        // 所有像素都是不透明的，没有alpha通道
        // 没有透明像素，直接返回全零张量
        // [1, H, W]
        Tensor::zeros((1, height as usize, width as usize), DType::F32, device)?
    } else {
        // 提取alpha通道
        let mask_data: Vec<f32> = rgba
            .pixels() // 每个像素4字节 (RGBA)
            .map(|p| p[3] as f32 / 255.0) // 提取并归一化alpha值
            .collect();

        // 转换为tensor [H, W]
        let tensor = Tensor::from_vec(mask_data, (height as usize, width as usize), device)?
            .to_dtype(DType::F32)?;
        // [H, W] -> [1, H, W]
        tensor.unsqueeze(0)?
    };

    // 反转 mask (1 - alpha)
    let mask = (1.0 - mask)?;

    Ok(mask)
}

/// 将未知格式的图片张量转换为 `data:image/...;base64,...` 格式的 URL
///
/// # 参数
/// - `images`: 形状为 `[batch, height, width, channels]` 的 PyTorch 张量
/// - `preferred_format`: 优先尝试的格式（如 `"png"` 或 `"jpeg"`），失败时自动回退
///
/// # 返回
/// - `Vec<String>`: 每个元素是一个 `data:image/...` 格式的 URL
pub fn tensor_to_images_data_url<'py>(
    images: &Bound<'py, PyAny>,
    preferred_format: Option<&str>,
) -> Result<Vec<String>, Error> {
    // 校验输入张量的形状
    let images_shape = images.getattr("shape")?.extract::<Vec<usize>>()?;
    if images_shape.len() != 4 {
        return Err(Error::InvalidTensorShape(format!(
            "Expected [batch, height, width, channels] tensor, images shape: {}",
            images_shape.len()
        )));
    }
    let batch = images_shape[0];
    let height = images_shape[1];
    let width = images_shape[2];
    let channels = images_shape[3];

    // 优先尝试的格式（默认 PNG）
    let (mut mime_type, mut image_format) = match preferred_format {
        Some("jpeg") | Some("jpg") => ("image/jpeg", ImageFormat::Jpeg),
        _ => ("image/png", ImageFormat::Png), // 默认 PNG
    };

    let mut data_urls = Vec::new();

    // 遍历每张图片
    for i in 0..batch {
        // 提取张量数据并转换为 NumPy 数组
        let numpy_array = images
            .call_method1("select", (0, i))?
            .call_method0("numpy")?
            .call_method1("__mul__", (255,))?
            .call_method1("astype", ("uint8",))?
            .call_method0("tobytes")?;

        // 转换为字节数组
        let pixel_bytes = numpy_array.extract::<Vec<u8>>()?;

        // 根据通道数创建图像
        let dynamic_img = match channels {
            3 => {
                let img = RgbImage::from_raw(width as u32, height as u32, pixel_bytes)
                    .ok_or_else(|| Error::ImageBuffer)?;
                DynamicImage::ImageRgb8(img)
            }
            4 => {
                let img = RgbaImage::from_raw(width as u32, height as u32, pixel_bytes)
                    .ok_or_else(|| Error::ImageBuffer)?;
                DynamicImage::ImageRgba8(img)
            }
            _ => {
                error!(
                    "Unsupported number of channels: {}. Expected 3 or 4.",
                    channels
                );
                return Err(Error::ImageBuffer);
            }
        };

        // 将图像编码为指定格式的字节流
        // 将图像写入内存缓冲区 (使用Cursor包装Vec<u8>以满足Seek trait要求)
        // 尝试编码为优先格式，失败时回退到 PNG
        let mut buffer = Cursor::new(Vec::new());
        let encode_result = dynamic_img.write_to(&mut buffer, image_format);
        if encode_result.is_err() {
            // 回退到 PNG
            mime_type = "image/png";
            image_format = ImageFormat::Png;
            buffer = Cursor::new(Vec::new());
            dynamic_img.write_to(&mut buffer, image_format)?;
        }

        // 将字节流编码为 Base64
        let encoded = general_purpose::STANDARD.encode(buffer.into_inner());

        // 拼接为 data URL
        let data_url = format!("data:{};base64,{}", mime_type, encoded);
        data_urls.push(data_url);
    }

    Ok(data_urls)
}

/// 将 Python 图片张量转换为图片 RAW 数据
///
/// images: [batch, height, width, channels]
pub fn tensor_to_raw_data<'py>(images: &Bound<'py, PyAny>) -> Result<Vec<Vec<u8>>, Error> {
    // 校验形状
    let images_shape = images.getattr("shape")?.extract::<Vec<usize>>()?;
    if images_shape.len() != 4 {
        return Err(Error::InvalidTensorShape(format!(
            "Expected [batch, height, width, channels] tensor, images shape: {}",
            images_shape.len()
        )));
    }
    let batch = images_shape[0];
    let height = images_shape[1];
    let width = images_shape[2];
    let channels = images_shape[3];

    let mut raw_data_list = Vec::new();

    // 遍历 images 对象中的每个图像
    for i in 0..batch {
        let numpy_array = images
            .call_method1("select", (0, i))?
            .call_method0("numpy")?
            .call_method1("__mul__", (255,))?
            .call_method1("astype", ("uint8",))?
            .call_method0("tobytes")?;

        // 转换为字节数组
        let pixel_bytes = numpy_array.extract::<Vec<u8>>()?;

        // 根据通道数创建图像
        let dynamic_img = match channels {
            3 => {
                let img = RgbImage::from_raw(width as u32, height as u32, pixel_bytes)
                    .ok_or_else(|| Error::ImageBuffer)?;
                DynamicImage::ImageRgb8(img)
            }
            4 => {
                let img = RgbaImage::from_raw(width as u32, height as u32, pixel_bytes)
                    .ok_or_else(|| Error::ImageBuffer)?;
                DynamicImage::ImageRgba8(img)
            }
            _ => {
                error!(
                    "Unsupported number of channels: {}. Expected 3 or 4.",
                    channels
                );
                return Err(Error::ImageBuffer);
            }
        };

        // 编码为PNG格式
        let mut png_data = Vec::new();
        let mut cursor = Cursor::new(&mut png_data);
        dynamic_img.write_to(&mut cursor, ImageFormat::Png)?;

        raw_data_list.push(png_data);
    }

    Ok(raw_data_list)
}

/// images: [batch, height, width, channels]
#[allow(clippy::type_complexity)]
pub fn tensor_to_image_tensor_buffer<'py>(
    images: &Bound<'py, PyAny>,
) -> Result<Vec<(Vec<u8>, usize, usize, usize)>, Error> {
    // 校验形状
    let images_shape = images.getattr("shape")?.extract::<Vec<usize>>()?;
    if images_shape.len() != 4 {
        return Err(Error::InvalidTensorShape(format!(
            "Expected [batch, height, width, channels] tensor, images shape: {}",
            images_shape.len()
        )));
    }
    let batch = images_shape[0];
    let height = images_shape[1];
    let width = images_shape[2];
    let channels = images_shape[3];

    let mut raw_data_list = Vec::new();

    // 遍历 images 对象中的每个图像
    for i in 0..batch {
        let numpy_array = images
            .call_method1("select", (0, i))?
            .call_method0("numpy")?
            .call_method1("__mul__", (255,))?
            .call_method1("astype", ("uint8",))?
            .call_method0("tobytes")?;

        // 转换为字节数组
        let pixel_bytes = numpy_array.extract::<Vec<u8>>()?;

        raw_data_list.push((pixel_bytes, height, width, channels));
    }

    Ok(raw_data_list)
}
