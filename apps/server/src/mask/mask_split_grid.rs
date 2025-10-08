//! 遮罩拆分网格

use candle_core::Device;
use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};

use crate::{
    core::category::CATEGORY_MASK,
    error::Error,
    wrapper::{
        comfyui::{
            PromptServer,
            types::{NODE_INT, NODE_MASK},
        },
        torch::tensor::TensorWrapper,
    },
};

/// 遮罩拆分网格
#[pyclass(subclass)]
pub struct MaskSplitGrid {
    device: Device,
}

impl PromptServer for MaskSplitGrid {}

#[pymethods]
impl MaskSplitGrid {
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
        (NODE_MASK,)
    }

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("masks",)
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
    const CATEGORY: &'static str = CATEGORY_MASK;

    // 节点描述, 可选
    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Split an mask into a grid."
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

                required.set_item("mask", (NODE_MASK, { PyDict::new(py) }))?;

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
        mask: Bound<'py, PyAny>,
        row: usize,
        column: usize,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        let results = self.mask_to_grid(py, mask, row, column);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("MaskSplitGrid error, {e}");
                if let Err(e) = self.send_error(py, "MaskSplitGrid".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl MaskSplitGrid {
    /// 将遮罩拆分成网格
    ///
    /// mask shape: [batch, height, width]
    fn mask_to_grid<'py>(
        &self,
        py: Python<'py>,
        mask: Bound<'py, PyAny>,
        row: usize,
        column: usize,
    ) -> Result<(Bound<'py, PyAny>,), Error> {
        let mask = TensorWrapper::<f32>::new(&mask, &self.device)?.into_tensor();

        // 获取遮罩尺寸
        let (_batch, height, width) = mask.dims3().map_err(|e| {
            Error::InvalidTensorShape(format!(
                "Expected a 3D tensor, but got a {}D tensor, err: {:?}",
                mask.dims().len(),
                e
            ))
        })?;

        // 计算子图尺寸
        let sub_width = width / column;
        let sub_height = height / row;

        let mut new_masks = Vec::new();

        // 遍历网格
        for i in 0..row {
            for j in 0..column {
                // 计算裁剪位置
                let x = j * sub_width;
                let y = i * sub_height;

                // 裁剪遮罩
                let cropped = self.crop(&mask, sub_width, sub_height, x, y)?;
                new_masks.push(cropped);
            }
        }

        // 在第0维度上连接所有遮罩
        let mask_batch = candle_core::Tensor::cat(&new_masks, 0)?;

        // 转换回PyAny
        let py_tensor = TensorWrapper::<f32>::from_tensor(mask_batch).to_py_tensor(py)?;

        Ok((py_tensor,))
    }

    /// 裁剪遮罩
    fn crop(
        &self,
        mask: &candle_core::Tensor,
        sub_width: usize,
        sub_height: usize,
        x: usize,
        y: usize,
    ) -> Result<candle_core::Tensor, Error> {
        // 确保x和y不超出遮罩边界
        let (_batch, raw_height, raw_width) = mask.dims3()?;
        let x = x.min(raw_width - 1);
        let y = y.min(raw_height - 1);

        // 裁剪遮罩
        let cropped = mask.narrow(1, y, sub_height)?.narrow(2, x, sub_width)?;

        Ok(cropped)
    }
}
