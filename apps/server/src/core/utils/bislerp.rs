//! comfy.utils.bislerp rust 实现
//! TODO 待验证
use candle_core::{DType, Device, Tensor, D};
use pyo3::Python;

use crate::{
    error::Error,
    wrapper::torch::nn::functional::{interpolate, InterpolationMode},
};

/// Spherical linear interpolation (slerp) between two batches of vectors
fn slerp(b1: &Tensor, b2: &Tensor, r: &Tensor) -> Result<Tensor, Error> {
    let c = b1.dim(D::Minus1)?;

    // Calculate norms
    let b1_norms = b1.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let b2_norms = b2.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;

    // Normalize
    let b1_normalized = b1.broadcast_div(&b1_norms)?;
    let b2_normalized = b2.broadcast_div(&b2_norms)?;

    // Handle zero norms
    let b1_normalized = b1_normalized.where_cond(
        &b1_norms.broadcast_eq(&Tensor::zeros_like(&b1_norms)?)?,
        &Tensor::zeros_like(&b1_normalized)?,
    )?;
    let b2_normalized = b2_normalized.where_cond(
        &b2_norms.broadcast_eq(&Tensor::zeros_like(&b2_norms)?)?,
        &Tensor::zeros_like(&b2_normalized)?,
    )?;

    // Calculate dot product
    let dot = (b1_normalized.broadcast_mul(&b2_normalized)?).sum(D::Minus1)?;

    // 最基础的实现方式
    // let omega = dot.acos()?;
    let omega_values: Vec<f32> = dot
        .to_vec1::<f32>()? // 转换为Vec
        .into_iter()
        .map(|x| x.clamp(-1.0, 1.0).acos())
        .collect();
    let omega = Tensor::from_vec(omega_values, dot.shape(), dot.device())?;

    let so = omega.sin()?;

    // Main slerp calculation
    let r_squeezed = r.squeeze(D::Minus1)?;
    let term1 = ((Tensor::ones_like(&r_squeezed)? - &r_squeezed)?.broadcast_mul(&omega)?).sin()?;
    let term1 = term1.broadcast_div(&so)?.unsqueeze(D::Minus1)?;
    let term1 = term1.broadcast_mul(&b1_normalized)?;

    let term2 = (r_squeezed.broadcast_mul(&omega)?).sin()?;
    let term2 = term2.broadcast_div(&so)?.unsqueeze(D::Minus1)?;
    let term2 = term2.broadcast_mul(&b2_normalized)?;

    let mut res = (term1 + term2)?;
    let scale =
        (b1_norms.broadcast_mul(&(Tensor::ones_like(r)? - r)?)? + b2_norms.broadcast_mul(r)?)?;
    res = res.broadcast_mul(&scale.broadcast_left(c)?)?;

    // Handle edge cases
    let close = dot.gt(&(Tensor::ones_like(&dot)? - 1e-5)?)?;
    res = res.where_cond(&close.unsqueeze(D::Minus1)?, b1)?;

    let opposite = dot.lt(1e-5 - 1.0)?;
    let linear = (b1.broadcast_mul(&(Tensor::ones_like(r)? - r)?)? + b2.broadcast_mul(r)?)?;
    res = res.where_cond(&opposite.unsqueeze(D::Minus1)?, &linear)?;

    Ok(res)
}

/// Generate bilinear interpolation data
fn generate_bilinear_data<'py>(
    py: Python<'py>,
    length_old: usize,
    length_new: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor), Error> {
    // Create coordinates
    let coords_1 = Tensor::arange(0, length_old as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((1, 1, 1, length_old))?;

    // Interpolate coordinates
    let coords_1 = interpolate(
        py,
        &coords_1,
        Some((1, length_new)),
        None::<f32>,
        InterpolationMode::Bilinear,
        None,
        None,
        false,
    )?;

    let ratios = (&coords_1 - &coords_1.floor()?)?;
    let coords_1 = coords_1.to_dtype(DType::U32)?;

    let mut coords_2 = (Tensor::arange(0, length_old as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((1, 1, 1, length_old))?
        + 1.0)?;

    // Handle edge case
    let last_col = (coords_2.narrow(D::Minus1, length_old - 1, 1)? - 1.0)?;
    let prefix = coords_2.narrow(D::Minus1, 0, length_old - 1)?;
    coords_2 = Tensor::cat(&[&prefix, &last_col], D::Minus1)?;

    let coords_2 = interpolate(
        py,
        &coords_2,
        Some((1, length_new)),
        None::<f32>,
        InterpolationMode::Bilinear,
        None,
        None,
        false,
    )?;
    let coords_2 = coords_2.to_dtype(DType::U32)?;

    Ok((ratios, coords_1, coords_2))
}

/// Bilateral interpolation (bislerp) for tensor upscaling
pub fn bislerp<'py>(
    py: Python<'py>,
    samples: &Tensor,
    width: usize,
    height: usize,
) -> Result<Tensor, Error> {
    let orig_dtype = samples.dtype();
    let samples = samples.to_dtype(DType::F32)?;
    let (n, c, h, w) = samples.dims4()?;
    let (h_new, w_new) = (height, width);

    // Linear interpolation along width
    let (ratios, coords_1, coords_2) = generate_bilinear_data(py, w, w_new, samples.device())?;
    let coords_1 = coords_1.broadcast_as((n, c, h, w_new))?;
    let coords_2 = coords_2.broadcast_as((n, c, h, w_new))?;
    let ratios = ratios.broadcast_as((n, 1, h, w_new))?;

    let pass_1 = samples
        .gather(&coords_1, D::Minus1)?
        .permute([0, 3, 2, 1])? // 等价于 movedim(1, -1)
        .reshape((n * h * w, c))?;
    let pass_2 = samples
        .gather(&coords_2, D::Minus1)?
        .permute([0, 3, 2, 1])?
        .reshape((n * h * w, c))?;
    let ratios = ratios.permute([0, 3, 2, 1])?.reshape((n * h * w, 1))?;

    let mut result = slerp(&pass_1, &pass_2, &ratios)?;
    result = result.reshape((n, h, w_new, c))?.permute([0, 3, 2, 1])?;

    // Linear interpolation along height
    let (ratios, coords_1, coords_2) = generate_bilinear_data(py, h, h_new, samples.device())?;
    let coords_1 = coords_1
        .reshape((1, 1, h_new, 1))?
        .broadcast_as((n, c, h_new, w_new))?;
    let coords_2 = coords_2
        .reshape((1, 1, h_new, 1))?
        .broadcast_as((n, c, h_new, w_new))?;
    let ratios = ratios
        .reshape((1, 1, h_new, 1))?
        .broadcast_as((n, 1, h_new, w_new))?;

    let pass_1 = result
        .gather(&coords_1, D::Minus2)?
        .permute([0, 3, 2, 1])?
        .reshape((n * h * w, c))?;
    let pass_2 = result
        .gather(&coords_2, D::Minus2)?
        .permute([0, 3, 2, 1])?
        .reshape((n * h * w, c))?;
    let ratios = ratios.permute([0, 3, 2, 1])?.reshape((n * h * w, 1))?;

    result = slerp(&pass_1, &pass_2, &ratios)?;
    result = result
        .reshape((n, h_new, w_new, c))?
        .permute([0, 3, 2, 1])?;

    Ok(result.to_dtype(orig_dtype)?)
}
