//! 图片简易分辨率
//! 依赖:
//! - python: torch

use candle_core::{Device, Tensor};
use pyo3::{
    prelude::*,
    types::{PyDict, PyType},
};

use crate::{
    core::category::CATEGORY_IMAGE,
    error::Error,
    wrapper::{
        comfyui::types::{NODE_IMAGE, NODE_INT},
        torch::tensor::TensorWrapper,
    },
};

// 图片分辨率
#[pyclass(subclass)]
pub struct ImageSimpleResolution {
    device: Device,
}

#[pymethods]
impl ImageSimpleResolution {
    #[new]
    fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    ) {
        (NODE_IMAGE, NODE_INT, NODE_INT, NODE_INT, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    ) {
        ("image", "resolution", "width", "height", "count")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool, bool, bool, bool) {
        (false, false, false, false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_IMAGE;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Obtain a simple image resolution that is a multiple of 8. Take the shortest edge as the resolution."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);
                required.set_item("image", (NODE_IMAGE, { PyDict::new(py) }))?;
                required
            })?;
            Ok(dict.into())
        })
    }

    fn execute<'py>(
        &self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, usize, usize, usize, usize)> {
        let image = TensorWrapper::<f32>::new(&image, &self.device)?.into_tensor();

        let (resolution, height, width, batch) = self.resolution(&image).unwrap();

        let tensor_wrapper: TensorWrapper<f32> = image.into();
        let py_tensor = tensor_wrapper.to_py_tensor(py)?;

        Ok((py_tensor, resolution, width, height, batch))
    }
}

impl ImageSimpleResolution {
    /// 获取图片分辨率
    fn resolution(&self, image: &Tensor) -> Result<(usize, usize, usize, usize), Error> {
        let (batch, height, width, _channels) = image.dims4()?; // NHWC格式

        // 分辨率计算逻辑
        // 对齐至8的倍数（四舍五入）
        let aligned_height = ((height as f32) / 8.0).round() as usize * 8;
        let aligned_width = ((width as f32) / 8.0).round() as usize * 8;

        // 计算最佳分辨率
        let resolution = usize::min(aligned_height, aligned_width);

        Ok((resolution, aligned_width, aligned_height, batch))
    }
}
