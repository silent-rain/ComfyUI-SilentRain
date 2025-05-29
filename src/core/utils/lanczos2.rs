//! lanczos 插值
//! TODO 待验证

use candle_core::{DType, Device, Tensor};
use image::{imageops::FilterType, DynamicImage, ImageBuffer, Rgb};

use crate::error::Error;

/// 批处理 Lanczos 插值
///
/// (输入输出均为 [N,H,W,C])
pub fn lanczos(tensor: &Tensor, width: usize, height: usize) -> Result<Tensor, Error> {
    let batch_size = tensor.dims()[0];
    let device = tensor.device();

    // 并行处理每个样本
    let results: Vec<Tensor> = (0..batch_size)
        .map(|i| {
            let img = tensor_slice_to_image(tensor, i)?;
            let resized = img.resize(
                width as u32,
                height as u32,
                image::imageops::FilterType::Lanczos3,
            );
            image_to_tensor_slice(resized, device)
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
