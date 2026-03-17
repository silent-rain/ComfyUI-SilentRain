//! 按较短边缩放遮罩
//!
//! 节点：ResizeMaskByShorterEdge
//!
//! 将遮罩按较短边等比缩放到指定尺寸。
//! 遮罩格式：[B, H, W]（BHW）
//!
//! 遮罩缩放：调用 torch.nn.functional.interpolate（双线性插值）

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};

use crate::{
    core::{
        category::CATEGORY_IMAGE,
        node_base::{InputSpec, InputType},
    },
    error::Error,
    wrapper::comfyui::{PromptServer, types::NODE_MASK},
};

/// 按较短边缩放遮罩
#[pyclass(subclass)]
pub struct ResizeMaskByShorterEdge {}

impl Default for ResizeMaskByShorterEdge {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptServer for ResizeMaskByShorterEdge {}

#[pymethods]
impl ResizeMaskByShorterEdge {
    #[new]
    pub fn new() -> Self {
        Self {}
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_MASK,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("mask",)
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
        "Resize mask so that the shorter edge matches the specified length, \
        preserving aspect ratio. Mask format: [B,H,W]."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        InputSpec::new()
            .with_required("mask", InputType::mask())
            .with_required(
                "shorter_edge",
                InputType::int()
                    .default(768)
                    .min(64)
                    .max(8192)
                    .step(8)
                    .tooltip("Target length for the shorter edge"),
            )
            .build()
    }

    #[pyo3(name = "execute", signature = (mask, shorter_edge))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        mask: Bound<'py, PyAny>,
        shorter_edge: i64,
    ) -> PyResult<(Bound<'py, PyAny>,)> {
        let result = self.run(py, mask, shorter_edge);

        match result {
            Ok(v) => Ok((v,)),
            Err(e) => {
                error!("ResizeMaskByShorterEdge error: {e}");
                if let Err(send_err) =
                    self.send_error(py, "ResizeMaskByShorterEdge".to_string(), e.to_string())
                {
                    error!("send error failed: {send_err}");
                }
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl ResizeMaskByShorterEdge {
    /// 按短边计算目标尺寸，返回 (new_height, new_width)
    ///
    /// 短边缩放到 shorter_edge，长边按比例计算，并对齐到 8 的倍数。
    fn calc_target_size(height: i64, width: i64, shorter_edge: i64) -> (i64, i64) {
        if height <= width {
            // height 是短边
            let scale = shorter_edge as f64 / height as f64;
            let new_h = shorter_edge;
            let new_w = ((width as f64 * scale).round() as i64 / 8) * 8;
            (new_h, new_w)
        } else {
            // width 是短边
            let scale = shorter_edge as f64 / width as f64;
            let new_h = ((height as f64 * scale).round() as i64 / 8) * 8;
            let new_w = shorter_edge;
            (new_h, new_w)
        }
    }

    /// 通过 torch.nn.functional.interpolate 缩放遮罩
    ///
    /// 输入：[B, H, W]（BHW）
    /// 输出：[B, new_H, new_W]（BHW）
    fn resize_mask<'py>(
        py: Python<'py>,
        mask: &Bound<'py, PyAny>,
        new_height: i64,
        new_width: i64,
    ) -> Result<Bound<'py, PyAny>, Error> {
        let torch = py.import("torch")?;
        let f = torch.getattr("nn")?.getattr("functional")?;

        // BHW -> B1HW（添加通道维度）
        let mask_4d = mask.call_method1("unsqueeze", (1,))?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("size", (new_height, new_width))?;
        kwargs.set_item("mode", "bilinear")?;
        kwargs.set_item("align_corners", false)?;
        let resized = f.call_method("interpolate", (&mask_4d,), Some(&kwargs))?;

        // B1HW -> BHW（移除通道维度）
        let result = resized.call_method1("squeeze", (1,))?;
        Ok(result)
    }

    /// 主执行逻辑
    pub fn run<'py>(
        &self,
        py: Python<'py>,
        mask: Bound<'py, PyAny>,
        shorter_edge: i64,
    ) -> Result<Bound<'py, PyAny>, Error> {
        // 读取原始尺寸：mask 形状为 [B, H, W]
        let shape = mask.getattr("shape")?;
        let orig_height: i64 = shape.get_item(1)?.extract()?;
        let orig_width: i64 = shape.get_item(2)?.extract()?;

        // 按短边计算目标尺寸
        let (new_height, new_width) = Self::calc_target_size(orig_height, orig_width, shorter_edge);

        // 缩放遮罩
        let resized_mask = Self::resize_mask(py, &mask, new_height, new_width)?;

        Ok(resized_mask)
    }
}
