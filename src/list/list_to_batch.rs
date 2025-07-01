//! 列表转批次

use candle_core::{Device, IndexOp, Tensor};
use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyList, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};
use strum_macros::{Display, EnumString};

use crate::{
    core::category::CATEGORY_LIST,
    error::Error,
    wrapper::{
        comfy::utils::common_upscale,
        comfyui::{types::any_type, PromptServer},
        python::isinstance,
        torch::tensor::TensorWrapper,
    },
};

/// 图像调整方法枚举
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum ResizeMethod {
    /// 拉伸图像以完全匹配目标尺寸 (可能改变宽高比)
    #[strum(to_string = "stretch")]
    Stretch,

    /// 保持宽高比调整大小 (可能不会完全填充目标尺寸)
    #[strum(to_string = "keep proportion")]
    KeepProportion,

    /// 填充或裁剪以完全匹配目标尺寸 (保持宽高比)
    #[strum(to_string = "fill / crop")]
    FillOrCrop,

    /// 保持原始宽高比并填充边缘 (保持宽高比)
    #[strum(to_string = "pad")]
    Pad,
}

/// 列表转批次
#[pyclass(subclass)]
pub struct ListToBatch {
    device: Device,
}

impl PromptServer for ListToBatch {}

#[pymethods]
impl ListToBatch {
    #[new]
    fn new() -> Self {
        Self {
            device: Device::Cpu,
        }
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types(py: Python<'_>) -> (Bound<'_, PyAny>,) {
        let any_type = any_type(py).unwrap();
        (any_type,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("batch",)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LIST;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Convert any list into batches."
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
                    "list",
                    (any_type(py)?, {
                        let list = PyDict::new(py);
                        list.set_item("tooltip", "Input any list")?;
                        list
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
        list: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        // 判断列表是否为空
        if list.is_empty() {
            // 空列表
            let empty = PyList::new(py, list)?.into_any();
            return Ok((empty,));
        }
        // 判断 list 中是否为图片
        let item = &list[0];
        // 图片
        if isinstance(py, item, "torch.Tensor")? {
            // 将列表转换为图片批次
            let images = self
                .image_list_to_batch(py, &list)
                .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
            return Ok((images,));
        }

        let list = PyList::new(py, list)?.into_any();
        Ok((list,))
    }
}

impl ListToBatch {
    /// 将列表转换为图片批次
    pub fn image_list_to_batch<'py>(
        &self,
        py: Python<'py>,
        images: &Vec<Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, Error> {
        // 将Python对象转换为Tensor列表
        let image_tensors: Vec<Tensor> = images
            .iter()
            .map(|image| {
                TensorWrapper::<f32>::new(image, &self.device).map(|wrapper| wrapper.into_tensor())
            })
            .collect::<Result<Vec<_>, _>>()?;

        // 检查是否有至少一个张量
        if image_tensors.is_empty() {
            return Err(Error::TensorErr(candle_core::Error::msg(
                "Cannot create batch from empty list",
            )));
        }

        // 3. 获取第一个张量的形状作为参考 [1,H,W,C]
        let first_image_tensor = &image_tensors[0];
        let first_image_shape = first_image_tensor.dims();
        let (batch, height, width) = (
            first_image_shape[0],
            first_image_shape[1],
            first_image_shape[2],
        );
        if first_image_shape.len() != 4 || batch != 1 {
            return Err(Error::TensorErr(candle_core::Error::msg(
                "Expected tensors with shape [1, H, W, C]",
            )));
        }

        // 4. 调整后续张量形状以匹配第一个张量
        let mut images = vec![first_image_tensor.clone()];
        for image_tensor in image_tensors[1..].iter() {
            let image_tensor_shape = image_tensor.dims();

            // 检查高度和宽度是否匹配
            if image_tensor_shape[1..2] == first_image_shape[1..2] {
                images.push(image_tensor.clone());
                continue;
            }

            let tensor = image_tensor.i(0)?;

            let image_tensor = common_upscale(
                py,
                &tensor.permute((0, 3, 2, 1))?,
                width,
                height,
                "bilinear",
                "center",
            )?
            .permute((0, 3, 2, 1))?;
            images.push(image_tensor);
        }

        // 5. 在批次维度上拼接张量
        let image_batch = Tensor::cat(&images, 0)?;
        error!("image_batch: {:#?}", image_batch.dims());

        // 6. 转换回Python对象
        let py_batch = TensorWrapper::<f32>::from_tensor(image_batch).to_py_tensor(py)?;
        Ok(py_batch)
    }
}
