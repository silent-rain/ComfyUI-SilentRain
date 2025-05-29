//! 图像插值方法
//!
//! Down/up samples the input.
//!
//! Tensor interpolated to either the given :attr:`size` or the given
//! :attr:`scale_factor`
//!
//! The algorithm used for interpolation is determined by :attr:`mode`.
//!
//! Currently temporal, spatial and volumetric sampling are supported, i.e.
//! expected inputs are 3-D, 4-D or 5-D in shape.
//!
//! The input dimensions are interpreted in the form:
//! `mini-batch x channels x [optional depth] x [optional height] x width`.
//!
//! The modes available for resizing are: `nearest`, `linear` (3D-only),
//! `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`, `nearest-exact`
//!

use candle_core::Tensor;
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
        let outputs = interpolate(
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
        let outputs = interpolate(
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
        let outputs = interpolate(
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
        let outputs = interpolate(
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
#[allow(clippy::too_many_arguments)]
pub fn interpolate<'py, SIZE, SF>(
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

    let input = TensorWrapper::from_tensor(input.clone()).to_py_tensor(py)?;
    let mode = mode.to_string();

    // 准备参数
    let args = (
        input,
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
