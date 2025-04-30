//! 图片分辨率
use candle_core::{DType, Device, Tensor};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList, PyType},
};

use crate::{
    core::{
        category::CATEGORY_IMAGE,
        types::{NODE_IMAGE, NODE_INT},
    },
    error::Error,
};

/// 缩放模式枚举
#[derive(Debug, Clone)]
enum ResizeMode {
    Resize,
    InnerFit,
    OuterFit,
}

// 图片分辨率
#[pyclass(subclass)]
pub struct ImageResolution {
    device: Device,
}

#[pymethods]
impl ImageResolution {
    #[new]
    fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    #[classmethod]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let mut required = PyDict::new(py);
                required.set_item(
                    "image",
                    ("IMAGE", {
                        let params = PyDict::new(py);
                        params
                    }),
                )?;
                required.set_item(
                    "resize_mode",
                    vec![
                        "",
                        // ResizeMode::Resize,
                        // ResizeMode::InnerFit,
                        // ResizeMode::OuterFit,
                    ],
                )?;
                required
            })?;
            Ok(dict.into())
        })
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str, &'static str, &'static str) {
        (NODE_IMAGE, NODE_INT, NODE_INT, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str, &'static str, &'static str) {
        ("image", "resolution", "width", "height")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_IMAGE;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Obtain image resolution."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    fn execute<'py>(
        &self,
        image: Bound<'py, PyAny>,
        resize_mode: String,
    ) -> PyResult<(usize, usize, usize, usize)> {
        let result = self.resolution(image, resize_mode).unwrap();

        Ok((0, 0, 0, 0))
    }
}

impl ImageResolution {
    /// 获取图片分辨率
    fn resolution<'py>(
        &self,
        image: Bound<'py, PyAny>,
        resize_mode: String,
    ) -> Result<(Tensor, usize, usize, usize), Error> {
        let data = image.extract::<Vec<f32>>()?; // 假设接收float32数组

        // Tensor::from_slice(data, shape.as_slice(), &Device::Cpu)
        //     .map_err(|e| PyErr::new::<pyo3::exceptions::PyTypeError, _>(e.to_string()))

        // let tensor = image.extract::<Tensor>()?;
        // let image = unsafe { image.downcast_unchecked::<Tensor>() };

        let image = Tensor::new(data, &Device::Cpu)?;
        // tensor.shape();

        // 张量形状解析
        let (b, raw_h, raw_w, c) = image.dims4()?;

        // 分辨率计算逻辑
        let (width, height) = (raw_w as f64, raw_h as f64);
        let (k0, k1) = (height / raw_h as f64, width / raw_w as f64);

        // let estimation = match resize_mode.parse::<ResizeMode>()? {
        //     ResizeMode::OuterFit => f64::min(k0, k1) * f64::min(height, width),
        //     _ => f64::max(k0, k1) * f64::min(height, width),
        // };

        // let result = estimation.round() as i64;
        // let text = format!(
        //     "Width: {}\nHeight: {}\nPixelPerfect: {}",
        //     width, height, result
        // );

        Ok((image, 0, 0, 0))
    }
}
