//! Pads tensor.

use candle_core::Tensor;
use pyo3::{pyclass, types::PyAnyMethods, Python};
use strum_macros::{Display, EnumString};

use crate::{error::Error, wrapper::torch::tensor::TensorWrapper};

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
pub enum PadMode {
    #[strum(to_string = "constant")]
    Constant,
    #[strum(to_string = "reflect")]
    Reflect,
    #[strum(to_string = "replicate")]
    Replicate,
    #[strum(to_string = "circular")]
    Circular,
}

pub fn pad<'py>(
    py: Python<'py>,
    input: &Tensor,
    pad: &[usize],
    mode: PadMode,
    value: Option<f32>,
) -> Result<Tensor, Error> {
    let device = input.device();

    let torch_module = py.import("torch")?;
    let functional = torch_module.getattr("nn")?.getattr("functional")?;

    let input = TensorWrapper::from_tensor(input.clone()).to_py_tensor(py)?;
    let mode = mode.to_string();

    // 准备参数
    let args = (input, pad, mode, value);

    // 调用 PyTorch 的 pad
    let py_any = functional.call_method("pad", args, None)?;
    let result = TensorWrapper::new::<f32>(&py_any, device)?.into_tensor();
    Ok(result)
}
