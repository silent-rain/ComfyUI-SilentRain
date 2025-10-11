//! comfy.utils.common_upscale rust 实现

use candle_core::{D, Tensor};
use pyo3::Python;

use crate::{
    error::Error,
    wrapper::torch::nn::functional::{InterpolationMode, interpolate},
};

use super::{bislerp::bislerp, lanczos::lanczos};

/// Upscales samples with optional center cropping and various upscaling methods
pub fn common_upscale<'py>(
    py: Python<'py>,
    samples: &Tensor,
    width: usize,
    height: usize,
    upscale_method: &str,
    crop: &str,
) -> Result<Tensor, Error> {
    let orig_shape = samples.dims();

    // Handle higher dimensional tensors
    let samples = if orig_shape.len() > 4 {
        let mut samples = samples.reshape((
            orig_shape[0],
            orig_shape[1],
            orig_shape[2..orig_shape.len() - 2]
                .iter()
                .product::<usize>(), // Replace -1 with actual calculation
            orig_shape[orig_shape.len() - 2],
            orig_shape[orig_shape.len() - 1],
        ))?;

        // 动态生成permute顺序
        let mut permute_order: Vec<usize> = (0..samples.dims().len()).collect();
        permute_order.swap(1, 2); // 相当于 movedim(2, 1)
        samples = samples.permute(&*permute_order)?;

        // 计算中间维度乘积
        let mid_dims = orig_shape[2..orig_shape.len() - 2]
            .iter()
            .product::<usize>();
        samples = samples.reshape((
            orig_shape[0] * mid_dims, // 替换原来的-1
            orig_shape[1],
            orig_shape[orig_shape.len() - 2],
            orig_shape[orig_shape.len() - 1],
        ))?;

        samples
    } else {
        samples.clone()
    };

    // Apply cropping if needed
    let s = if crop == "center" {
        let old_width = samples.dim(D::Minus1)?;
        let old_height = samples.dim(D::Minus2)?;
        let old_aspect = old_width as f32 / old_height as f32;
        let new_aspect = width as f32 / height as f32;

        let (x, y) = if old_aspect > new_aspect {
            let x = ((old_width as f32 - old_width as f32 * (new_aspect / old_aspect)) / 2.0)
                .round() as usize;
            (x, 0)
        } else if old_aspect < new_aspect {
            let y = ((old_height as f32 - old_height as f32 * (old_aspect / new_aspect)) / 2.0)
                .round() as usize;
            (0, y)
        } else {
            (0, 0)
        };

        samples
            .narrow(D::Minus2, y, old_height - y * 2)?
            .narrow(D::Minus1, x, old_width - x * 2)?
    } else {
        samples
    };

    // Apply upscaling
    let out = match upscale_method {
        "bislerp" => bislerp(py, &s, width, height)?,
        "lanczos" => lanczos(&s, width, height)?,
        _ => {
            // For other methods, use candle's interpolate
            // 解析枚举并执行模式匹配
            let upscale_method = upscale_method
                .parse::<InterpolationMode>()
                .map_err(|e| Error::ParseEnumString(e.to_string()))?;
            interpolate(
                py,
                &s,
                Some((height, width)),
                None::<f32>,
                upscale_method,
                None,
                None,
                false,
            )?
        }
    };

    // Restore original shape if needed
    if orig_shape.len() == 4 {
        return Ok(out);
    }

    let mut new_shape = orig_shape[..orig_shape.len() - 2].to_vec();
    new_shape.push(height);
    new_shape.push(width);

    out.reshape(new_shape)
        .and_then(|t| {
            // 动态生成permute顺序
            let mut permute_order: Vec<usize> = (0..t.dims().len()).collect();
            permute_order.swap(1, 2); // 相当于 movedim(2, 1)
            t.permute(&*permute_order)
        })
        .map_err(Error::from)
}
