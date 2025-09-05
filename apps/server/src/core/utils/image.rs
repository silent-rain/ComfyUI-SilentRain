//! image 与 tensor 相互转换
//!
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Rgb, RgbImage, RgbaImage};

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
            )))
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
