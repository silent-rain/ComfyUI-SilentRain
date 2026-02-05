//! 图像处理工具
use std::io::Cursor;

use image::{DynamicImage, ImageFormat, RgbImage, RgbaImage, imageops::FilterType};
use tracing::{error, info};

use crate::error::Error;

/// 图像处理工具
#[derive(Debug, Default, Clone)]
pub struct Image {
    img: DynamicImage,
}

impl Image {
    /// 从文件加载图像
    pub fn from_file(path: &str) -> Result<Self, Error> {
        let img = image::open(path).map_err(|e| {
            error!("Failed to open image: {}", e);
            e
        })?;

        Ok(Image { img })
    }

    /// 从tensor加载图像
    pub fn from_tensor(
        data: Vec<u8>,
        height: u32,
        width: u32,
        channels: usize,
    ) -> Result<Self, Error> {
        // 根据通道数创建图像
        let img = match channels {
            3 => {
                info!("Creating RGB image from {} bytes", data.len());
                let img = RgbImage::from_raw(width, height, data).ok_or_else(|| {
                    error!("Failed to create RGB image from raw data");
                    Error::ImageBuffer
                })?;
                DynamicImage::ImageRgb8(img)
            }
            4 => {
                info!("Creating RGBA image from {} bytes", data.len());
                let img = RgbaImage::from_raw(width, height, data).ok_or_else(|| {
                    error!("Failed to create RGBA image from raw data");
                    Error::ImageBuffer
                })?;
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

        info!(
            "Image created successfully: {}x{}",
            img.width(),
            img.height()
        );
        Ok(Image { img })
    }

    /// 图片缩放
    pub fn resize(&self, width: u32, height: u32) -> Result<Self, Error> {
        let img = self.img.resize(width, height, FilterType::Lanczos3);
        Ok(Image { img })
    }

    /// 按比例缩放
    pub fn resize_to_scale(&self, scale: f32) -> Result<Self, Error> {
        let new_width = (self.width() as f32 * scale) as u32;
        let new_height = (self.height() as f32 * scale) as u32;
        self.resize(new_width, new_height)
    }

    /// 按指定分辨率缩放
    ///
    /// 将图像按指定宽度缩放，高度按比例自适应
    pub fn resize_to_resolution(&self, resolution: u32) -> Result<Self, Error> {
        let scale = resolution as f32 / self.width() as f32;
        self.resize_to_scale(scale)
    }

    /// 按最长边缩放
    pub fn resize_to_longest(&self, length: u32) -> Result<Self, Error> {
        let scale = length as f32 / self.longest() as f32;
        self.resize_to_scale(scale)
    }

    /// 获取最长边
    pub fn longest(&self) -> u32 {
        self.width().max(self.height())
    }

    /// Jpeg 图片压缩
    pub fn compress_jpeg(&self) -> Result<Self, Error> {
        // 格式化为JPEG格式以压缩图片
        let img = self.img.clone().into_rgb8();
        let mut buffer = Vec::new();
        img.write_to(&mut Cursor::new(&mut buffer), ImageFormat::Jpeg)?;
        let compressed_img = image::load_from_memory(&buffer)?;

        Ok(Image {
            img: compressed_img,
        })
    }

    /// 图像转换为 buffer
    pub fn to_vec(&self) -> Result<Vec<u8>, Error> {
        let mut buffer = Vec::new();
        self.img
            .write_to(&mut Cursor::new(&mut buffer), ImageFormat::Png)?;

        Ok(buffer)
    }

    /// 保存图像
    pub fn save(&self, path: &str) -> Result<(), Error> {
        self.img.save(path)?;
        Ok(())
    }

    /// 宽度
    pub fn width(&self) -> u32 {
        self.img.width()
    }

    /// 高度
    pub fn height(&self) -> u32 {
        self.img.height()
    }

    /// 获取图像
    pub fn to_image(&self) -> DynamicImage {
        self.img.clone()
    }
}
