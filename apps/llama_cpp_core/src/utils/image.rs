//! 图像处理工具
use std::io::Cursor;

use base64::{Engine, engine::general_purpose};
use image::{DynamicImage, ImageFormat, RgbImage, RgbaImage, imageops::FilterType};
use tracing::{error, info};

use crate::{error::Error, message_plugins::ImageSource};

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

    /// 根据配置的最大分辨率缩放图像
    ///
    /// 如果图像最长边超过配置的最大分辨率，则按比例缩放
    /// 配置值小于 64 时会自动调整为 64
    ///
    /// # Arguments
    /// * `max_resolution` - 配置的最大分辨率
    ///
    /// # Returns
    /// - `Ok(Image)` - 缩放后的图像（如果需要缩放）或原图像（如果不需要）
    ///
    /// # Example
    /// ```
    /// let img = Image::from_bytes(&data)?;
    /// let resized = img.resize_with_max_resolution(768)?;
    /// ```
    pub fn resize_with_max_resolution(&self, max_resolution: u32) -> Result<Self, Error> {
        // 确保最小分辨率为 64
        let effective_max = if max_resolution < 64 {
            64
        } else {
            max_resolution
        };

        // 获取图像最长边
        let longest = self.longest();

        // 如果图像尺寸在限制范围内，直接返回原图
        if longest <= effective_max {
            return Ok(self.clone());
        }

        // 按比例缩放
        info!(
            "Resizing image from {}x{} to fit max resolution {}",
            self.width(),
            self.height(),
            effective_max
        );
        self.resize_to_longest(effective_max)
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

    /// 从二进制数据加载图像
    pub fn from_bytes(data: &[u8]) -> Result<Self, Error> {
        let img = image::load_from_memory(data).map_err(|e| {
            error!("Failed to load image from bytes: {}", e);
            e
        })?;
        Ok(Image { img })
    }

    /// 转换为 base64 编码字符串
    pub fn to_base64(&self) -> Result<String, Error> {
        let buffer = self.to_vec()?;
        Ok(general_purpose::STANDARD.encode(&buffer))
    }

    /// 转换为特定格式的 base64 编码字符串
    ///
    /// # Arguments
    /// * `format` - 图像格式，如 "png", "jpeg"
    pub fn to_base64_with_format(&self, format: &str) -> Result<String, Error> {
        let mut buffer = Vec::new();
        let image_format = match format.to_lowercase().as_str() {
            "jpeg" | "jpg" => ImageFormat::Jpeg,
            "png" => ImageFormat::Png,
            "webp" => ImageFormat::WebP,
            "gif" => ImageFormat::Gif,
            _ => ImageFormat::Png,
        };

        self.img
            .write_to(&mut Cursor::new(&mut buffer), image_format)?;

        Ok(general_purpose::STANDARD.encode(&buffer))
    }
}

/// 解码图像来源列表
///
/// 将所有 ImageSource 解码为二进制数据列表
///
/// # Arguments
/// * `image_sources` - 图像来源列表
///
/// # Returns
/// - `Vec<Vec<u8>>` - 解码后的图像数据列表
pub async fn decode_image_sources(image_sources: &[ImageSource]) -> Result<Vec<Vec<u8>>, Error> {
    let mut decoded_images = Vec::new();

    for image_source in image_sources {
        match image_source {
            ImageSource::Url(url) => {
                info!("Loading media from URL: {}", url);
                let data = download_image(url).await?;
                decoded_images.push(data);
            }
            ImageSource::Base64(media_type, base64_data) => {
                info!("Loading media from base64 string (type: {})", media_type);
                let data = general_purpose::STANDARD.decode(base64_data)?;
                decoded_images.push(data);
            }
        }
    }

    Ok(decoded_images)
}

/// 从 URL 下载图片
///
/// # Arguments
/// * `url` - 图片 URL
///
/// # Returns
/// - `Vec<u8>` - 下载的图片数据
async fn download_image(url: &str) -> Result<Vec<u8>, Error> {
    info!("Downloading image from: {}", url);

    let response = reqwest::get(url)
        .await
        .map_err(|e| Error::HttpRequest(format!("Failed to send request: {}", e)))?;

    if !response.status().is_success() {
        return Err(Error::HttpRequest(format!(
            "HTTP error: {} for URL: {}",
            response.status(),
            url
        )));
    }

    let data = response.bytes().await.map_err(|e| Error::ImageDownload {
        url: url.to_string(),
        message: format!("Failed to download image: {}", e),
    })?;

    info!("Successfully downloaded {} bytes from URL", data.len());
    Ok(data.to_vec())
}
