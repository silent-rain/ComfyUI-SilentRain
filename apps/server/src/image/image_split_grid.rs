//! 图像拆分网格

use candle_core::{Device, Tensor};
use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_IMAGE,
    error::Error,
    wrapper::{
        comfyui::{
            types::{NODE_IMAGE, NODE_INT},
            PromptServer,
        },
        torch::tensor::TensorWrapper,
    },
};

/// 图像拆分网格
#[pyclass(subclass)]
pub struct ImageSplitGrid {
    device: Device,
}

impl PromptServer for ImageSplitGrid {}

#[pymethods]
impl ImageSplitGrid {
    #[new]
    fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    // 输入列表, 可选
    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    // 输出节点, 可选
    // #[classattr]
    // #[pyo3(name = "OUTPUT_NODE")]
    // fn output_node() -> bool {
    //     false
    // }

    // 过时标记, 可选
    // #[classattr]
    // #[pyo3(name = "DEPRECATED")]
    // fn deprecated() -> bool {
    //     false
    // }

    // 实验性的, 可选
    // #[classattr]
    // #[pyo3(name = "EXPERIMENTAL")]
    // fn experimental() -> bool {
    //     false
    // }

    // 返回参数类型
    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_IMAGE,)
    }

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("images",)
    }

    // 返回参数提示
    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() -> (&'static str,) {
        ("",)
    }

    // 输出列表, 可选
    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    // 节点分类
    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_IMAGE;

    // 节点描述, 可选
    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Split an image into a grid."
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

                required.set_item(
                    "image",
                    (NODE_IMAGE, {
                        let params = PyDict::new(py);
                        params
                    }),
                )?;

                required.set_item(
                    "row",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 1)?;
                        params.set_item("min", 1)?;
                        params.set_item("step", 1)?;
                        params
                    }),
                )?;
                required.set_item(
                    "column",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 1)?;
                        params.set_item("min", 1)?;
                        params.set_item("step", 1)?;
                        params
                    }),
                )?;

                required
            })?;

            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        row: usize,
        column: usize,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        let results = self.image_to_grid(py, image, row, column);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("ImageSplitGrid error, {e}");
                if let Err(e) = self.send_error(py, "ImageSplitGrid".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl ImageSplitGrid {
    /// 将图片拆分成网格
    ///
    /// image shape: [batch, height, width, channels]
    fn image_to_grid<'py>(
        &self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        row: usize,
        column: usize,
    ) -> Result<(Bound<'py, PyAny>,), Error> {
        let image = TensorWrapper::<f32>::new(&image, &self.device)?.into_tensor();

        // 获取图像尺寸
        let (_batch, height, width, _channels) = image.dims4().map_err(|e| {
            Error::InvalidTensorShape(format!(
                "Expected a 4D tensor, but got a {}D tensor, err: {:?}",
                image.dims().len(),
                e
            ))
        })?;

        // 计算子图尺寸
        let sub_width = width / column;
        let sub_height = height / row;

        let mut new_images = Vec::new();

        // 遍历网格
        for i in 0..row {
            for j in 0..column {
                // 计算裁剪位置
                let x = j * sub_width;
                let y = i * sub_height;

                // 裁剪图像
                let cropped = self.crop(&image, sub_width, sub_height, x, y)?;
                new_images.push(cropped);
            }
        }

        // 在第0维度上连接所有图像
        let image_batch = Tensor::cat(&new_images, 0)?;

        // 转换回PyAny
        let py_tensor = TensorWrapper::<f32>::from_tensor(image_batch).to_py_tensor(py)?;

        Ok((py_tensor,))
    }

    /// 裁剪图像
    fn crop(
        &self,
        image: &Tensor,
        sub_width: usize,
        sub_height: usize,
        x: usize,
        y: usize,
    ) -> Result<Tensor, Error> {
        // 确保x和y不超出图像边界
        let (_batch, raw_height, raw_width, _channels) = image.dims4()?;
        let x = x.min(raw_width - 1);
        let y = y.min(raw_height - 1);

        // 裁剪图像
        let cropped = image.narrow(1, y, sub_height)?.narrow(2, x, sub_width)?;

        Ok(cropped)
    }
}
