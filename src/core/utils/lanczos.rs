//! comfy.utils.lanczos rust 实现
//! TODO 待验证
//!
use candle_core::{Device, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Rgb};

use crate::error::Error;

/// 批处理 Lanczos 插值
///
/// samples: NHWC
pub fn lanczos(samples: &Tensor, width: usize, height: usize) -> Result<Tensor, Error> {
    let device = samples.device();

    // 转换为图像
    let images: Vec<DynamicImage> = tensor_to_image(samples)?;

    // 使用Lanczos重采样缩放
    let resized: Vec<DynamicImage> = images
        .into_iter()
        .map(|img| img.resize(width as u32, height as u32, FilterType::Lanczos3))
        .collect();

    // 转换回张量
    let tensors = resized
        .into_iter()
        .map(|img| image_to_tensor(&img, device))
        .collect::<Result<Vec<_>, _>>()?;

    // 堆叠张量
    Ok(Tensor::stack(&tensors, 0)?)
}

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

/// 将图像转换为张量
///
/// output: HWC
pub fn image_to_tensor(image: &DynamicImage, device: &Device) -> Result<Tensor, Error> {
    let (width, height) = image.dimensions();
    let rgb = image.to_rgb8();

    // 预分配内存
    let mut data = vec![0.0f32; (width * height * 3) as usize];

    // 并行处理像素
    rgb.pixels().enumerate().for_each(|(i, p)| {
        let base = i * 3;
        data[base] = p[0] as f32 / 255.0;
        data[base + 1] = p[1] as f32 / 255.0;
        data[base + 2] = p[2] as f32 / 255.0;
    });

    // CHW
    let tensor = Tensor::from_vec(data, (3, height as usize, width as usize), device)?;

    // 维度重排 CHW -> HWC
    let tensor = tensor.permute((1, 2, 0))?;
    Ok(tensor)
}
