//! 批次转列表
//!
//! 依赖:
//! - python: torch

use candle_core::{Device, IndexOp};
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyList, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_LIST,
    error::Error,
    wrapper::{
        comfyui::{types::any_type, PromptServer},
        python::isinstance,
        torch::tensor::TensorWrapper,
    },
};

/// 批次转列表
#[pyclass(subclass)]
pub struct BatchToList {
    device: Device,
}

impl PromptServer for BatchToList {}

#[pymethods]
impl BatchToList {
    #[new]
    fn new() -> Self {
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_level(true)
            .with_file(true)
            .with_line_number(true)
            .try_init();
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
    fn return_types(py: Python) -> (Bound<'_, PyAny>,) {
        let any_type = any_type(py).unwrap();
        (any_type,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("list",)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (true,)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LIST;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Convert any batch into a list."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);
                required.set_item(
                    "batch",
                    (any_type(py)?, {
                        let batch = PyDict::new(py);
                        batch.set_item("tooltip", "Input any batch")?;
                        batch
                    }),
                )?;
                required
            })?;
            Ok(dict.into())
        })
    }

    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        batch: Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        // 图片
        if isinstance(py, &batch, "torch.Tensor")? {
            let image_list = self
                .image_batch_to_list(py, &batch)
                .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
            return Ok((image_list,));
        }

        // 将batch（原列表数据）数据直接输出为list
        let list = batch;
        Ok((list,))
    }
}

impl BatchToList {
    pub fn image_batch_to_list<'py>(
        &self,
        py: Python<'py>,
        images: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, Error> {
        // 1. 将Python对象转换为Candle张量
        let images_tensor: candle_core::Tensor =
            TensorWrapper::<f32>::new(images, &self.device)?.into_tensor();

        // 2. 检查张量维度，确保是批量数据 (B, C, H, W)
        if images_tensor.dims().len() != 4 {
            return Err(Error::TensorErr(candle_core::Error::msg(
                "Expected 4D tensor (batch, channels, height, width)",
            )));
        }

        // 3. 获取批量大小
        let batch_size = images_tensor.dim(0)?;

        // 4. 准备返回的结果列表
        let mut results = Vec::with_capacity(batch_size);

        // 5. 处理批量中的每个图像
        for i in 0..batch_size {
            // 5.1 提取单张图像
            let single_image = images_tensor.i(i)?;

            // 5.2 添加批次维度 [H, W, C] -> [1, H, W, C]
            let single_image = single_image.unsqueeze(0)?;

            // 5.23 将张量转换回Python对象
            let tensor_wrapper: TensorWrapper<f32> = single_image.into();
            let py_tensor = tensor_wrapper.to_py_tensor(py)?;

            // 5.4 添加到结果列表
            results.push(py_tensor);
        }

        let results = PyList::new(py, &results)?.into_any();

        Ok(results)
    }
}
