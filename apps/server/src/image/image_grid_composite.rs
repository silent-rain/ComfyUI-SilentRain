//! 图像网格合并

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

/// 图像网格合并
#[pyclass(subclass)]
pub struct ImageGridComposite {
    device: Device,
}

impl PromptServer for ImageGridComposite {}

#[pymethods]
impl ImageGridComposite {
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
        ("image",)
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
        "Merge the image grid into one image."
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
                    "images",
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
        images: Bound<'py, PyAny>,
        row: usize,
        column: usize,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        let results = self.image_to_grid(py, images, row, column);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("ImageGridComposite error, {e}");
                if let Err(e) = self.send_error(py, "ImageGridComposite".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl ImageGridComposite {
    /// 将图像网格合并图像
    ///
    /// image shape: [batch, height, width, channels]
    fn image_to_grid<'py>(
        &self,
        py: Python<'py>,
        images: Bound<'py, PyAny>,
        row: usize,
        column: usize,
    ) -> Result<(Bound<'py, PyAny>,), Error> {
        let images = TensorWrapper::<f32>::new(&images, &self.device)?.into_tensor();

        // 获取图像尺寸
        let (batch, _height, _width, _channels) = images.dims4().map_err(|e| {
            Error::InvalidTensorShape(format!(
                "Expected a 4D tensor, but got a {}D tensor, err: {:?}",
                images.dims().len(),
                e
            ))
        })?;

        // 检查输入的行列数是否合理
        if row * column > batch {
            return Err(Error::InvalidParameter(format!(
                "Row * Column ({}) exceeds the number of images ({})",
                row * column,
                batch
            )));
        }

        // 将batch转换成列表
        let images = images.chunk(batch, 0)?;

        // 创建一个空的网格图像
        let mut grid_rows = Vec::with_capacity(row);

        // 按行合并图像
        for r in 0..row {
            let mut row_images = Vec::with_capacity(column);

            // 获取当前行的所有图像并水平合并
            for c in 0..column {
                let idx = r * column + c;
                // 获取单个图像
                if let Some(image) = images.get(idx) {
                    row_images.push(image);
                }
            }

            // 水平合并当前行的图像 (dim=2 对应width维度)
            let row_tensor = Tensor::cat(&row_images, 2)?;
            grid_rows.push(row_tensor);
        }

        // 垂直合并所有行 (dim=1 对应height维度)
        let grid_image = Tensor::cat(&grid_rows, 1)?;

        // 转换回PyAny
        let py_tensor = TensorWrapper::<f32>::from_tensor(grid_image).to_py_tensor(py)?;

        Ok((py_tensor,))
    }
}
