//! VAE Object for comfyui
//!

use candle_core::Tensor;
use pyo3::{types::PyAnyMethods, Bound, PyAny, Python};

use crate::{error::Error, wrapper::torch::tensor::TensorWrapper};

/// VAE
#[derive(Debug)]
pub struct Vae<'py> {
    vae: Bound<'py, PyAny>,
}

impl<'py> Vae<'py> {
    pub fn new(vae: Bound<'py, PyAny>) -> Result<Self, Error> {
        Ok(Self { vae })
    }

    /// Encode
    ///
    pub fn encode(&self, py: Python<'py>, pixel_samples: &Tensor) -> Result<Tensor, Error> {
        let device = pixel_samples.device();

        let pixel_samples =
            TensorWrapper::<f32>::from_tensor(pixel_samples.clone()).to_py_tensor(py)?;

        // 准备参数
        let args = (pixel_samples,);

        let py_any = self.vae.call_method("encode", args, None)?;
        let result = TensorWrapper::<f32>::new(&py_any, device)?.into_tensor();
        Ok(result)
    }
}
