//! comfy.utils

use candle_core::Tensor;
use pyo3::{Python, types::PyAnyMethods};

use crate::{error::Error, wrapper::torch::tensor::TensorWrapper};

pub fn common_upscale<'py>(
    py: Python<'py>,
    samples: &Tensor,
    width: usize,
    height: usize,
    upscale_method: &str, // bislerp | lanczos | torch.nn.functional.interpolate.(mode)
    crop: &str,           // "" | center
) -> Result<Tensor, Error> {
    let device = samples.device();

    let torch_module = py.import("comfy")?;
    let utils = torch_module.getattr("utils")?;

    let samples = TensorWrapper::<f32>::from_tensor(samples.clone()).to_py_tensor(py)?;

    // 准备参数
    let args = (samples, width, height, upscale_method, crop);

    let py_any = utils.call_method("interpolate", args, None)?;
    let result = TensorWrapper::<f32>::new(&py_any, device)?.into_tensor();
    Ok(result)
}

pub fn lanczos<'py>(
    py: Python<'py>,
    samples: &Tensor,
    width: usize,
    height: usize,
) -> Result<Tensor, Error> {
    let device = samples.device();

    let torch_module = py.import("comfy")?;
    let utils = torch_module.getattr("utils")?;

    let samples = TensorWrapper::<f32>::from_tensor(samples.clone()).to_py_tensor(py)?;

    // 准备参数
    let args = (samples, width, height);

    let py_any = utils.call_method("lanczos", args, None)?;
    let result = TensorWrapper::<f32>::new(&py_any, device)?.into_tensor();
    Ok(result)
}
