//! 遮罩网格合并

use candle_core::{Device, Tensor};
use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_MASK,
    error::Error,
    wrapper::{
        comfyui::{
            types::{NODE_INT, NODE_MASK},
            PromptServer,
        },
        torch::tensor::TensorWrapper,
    },
};

/// 遮罩网格合并
#[pyclass(subclass)]
pub struct MaskGridComposite {
    device: Device,
}

impl PromptServer for MaskGridComposite {}

#[pymethods]
impl MaskGridComposite {
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
        ("mask",)
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
        "Merge the mask grid into one mask."
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
                    "masks",
                    (NODE_MASK, {
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
        masks: Bound<'py, PyAny>,
        row: usize,
        column: usize,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        let results = self.mask_to_grid(py, masks, row, column);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("MaskGridComposite error, {e}");
                if let Err(e) = self.send_error(py, "MaskGridComposite".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl MaskGridComposite {
    /// 将遮罩网格合并遮罩
    ///
    /// mask shape: [batch, height, width, channels]
    fn mask_to_grid<'py>(
        &self,
        py: Python<'py>,
        masks: Bound<'py, PyAny>,
        row: usize,
        column: usize,
    ) -> Result<(Bound<'py, PyAny>,), Error> {
        let masks = TensorWrapper::<f32>::new(&masks, &self.device)?.into_tensor();

        // 获取遮罩尺寸
        let (batch, _height, _width) = masks.dims3().map_err(|e| {
            Error::InvalidTensorShape(format!(
                "Expected a 3D tensor, but got a {}D tensor, err: {:?}",
                masks.dims().len(),
                e
            ))
        })?;

        // 检查输入的行列数是否合理
        if row * column > batch {
            return Err(Error::InvalidParameter(format!(
                "Row * Column ({}) exceeds the number of masks ({})",
                row * column,
                batch
            )));
        }

        // 将batch转换成列表
        let masks = masks.chunk(batch, 0)?;

        // 创建一个空的网格遮罩
        let mut grid_rows = Vec::with_capacity(row);

        // 按行合并遮罩
        for r in 0..row {
            let mut row_masks = Vec::with_capacity(column);

            // 获取当前行的所有遮罩并水平合并
            for c in 0..column {
                let idx = r * column + c;
                if let Some(mask) = masks.get(idx) {
                    // 获取单个遮罩
                    row_masks.push(mask);
                }
            }

            // 水平合并当前行的遮罩 (dim=2 对应width维度)
            let row_tensor = Tensor::cat(&row_masks, 2)?;
            grid_rows.push(row_tensor);
        }

        // 垂直合并所有行 (dim=1 对应height维度)
        let grid_mask = Tensor::cat(&grid_rows, 1)?;

        // 转换回PyAny
        let py_tensor = TensorWrapper::<f32>::from_tensor(grid_mask).to_py_tensor(py)?;

        Ok((py_tensor,))
    }
}
