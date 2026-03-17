//! 按较长边缩放图像与遮罩
//!
//! 节点：ResizeImageMaskByLongerEdge
//!
//! 将图像（和可选的遮罩）按较长边等比缩放到指定尺寸。
//! 图像格式：[B, H, W, C]（BHWC）
//! 遮罩格式：[B, H, W]（BHW）
//!
//! 图像缩放：调用 comfy_extras.nodes_dataset.ResizeImagesByLongerEdgeNode
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
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_IMAGE, NODE_MASK},
    },
};

/// 按较长边缩放图像与遮罩
#[pyclass(subclass)]
pub struct ResizeImageMaskByLongerEdge {}

impl Default for ResizeImageMaskByLongerEdge {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptServer for ResizeImageMaskByLongerEdge {}

#[pymethods]
impl ResizeImageMaskByLongerEdge {
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
    fn return_types() -> (&'static str, &'static str) {
        (NODE_IMAGE, NODE_MASK)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("image", "mask")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_IMAGE;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Resize image (and optional mask) so that the longer edge matches the specified length, \
        preserving aspect ratio. Image format: [B,H,W,C]; Mask format: [B,H,W]."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        InputSpec::new()
            .with_required("image", InputType::image())
            .with_required(
                "longer_edge",
                InputType::int()
                    .default(1024)
                    .min(64)
                    .max(8192)
                    .step(8)
                    .tooltip("Target length for the longer edge"),
            )
            .with_optional("mask", InputType::mask())
            .build()
    }

    #[pyo3(name = "execute", signature = (image, longer_edge, mask=None))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        longer_edge: i64,
        mask: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let result = self.run(py, image, longer_edge, mask);

        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("ResizeImageMaskByLongerEdge error: {e}");
                if let Err(send_err) =
                    self.send_error(py, "ResizeImageMaskByLongerEdge".to_string(), e.to_string())
                {
                    error!("send error failed: {send_err}");
                }
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl ResizeImageMaskByLongerEdge {
    /// 通过 Python 调用 ResizeImagesByLongerEdgeNode 缩放图像
    ///
    /// 输入/输出：[B, H, W, C]（BHWC）
    fn resize_image<'py>(
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
        longer_edge: i64,
    ) -> Result<Bound<'py, PyAny>, Error> {
        let module = py.import("comfy_extras.nodes_dataset")?;
        let node_cls = module.getattr("ResizeImagesByLongerEdgeNode")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("images", image)?;
        kwargs.set_item("longer_edge", longer_edge)?;
        let node_output = node_cls.call_method("execute", (), Some(&kwargs))?;
        // execute 返回 (tensor,)，取第一个元素
        let result = node_output.get_item(0)?;
        Ok(result)
    }

    /// 通过 Python 调用 torch.nn.functional.interpolate 缩放遮罩
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

        // 调用 interpolate
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
        image: Bound<'py, PyAny>,
        longer_edge: i64,
        mask: Option<Bound<'py, PyAny>>,
    ) -> Result<(Bound<'py, PyAny>, Bound<'py, PyAny>), Error> {
        // 缩放图像
        let resized_image = Self::resize_image(py, &image, longer_edge)?;

        // 获取缩放后的尺寸：resized_image 形状为 [B, H, W, C]
        let shape = resized_image.getattr("shape")?;
        let new_height: i64 = shape.get_item(1)?.extract()?;
        let new_width: i64 = shape.get_item(2)?.extract()?;

        // 缩放遮罩（如果提供）
        let resized_mask = match mask {
            Some(m) => Self::resize_mask(py, &m, new_height, new_width)?,
            None => {
                // 未提供遮罩时，生成全零遮罩 [B, new_H, new_W]
                let torch = py.import("torch")?;
                let batch: i64 = resized_image.getattr("shape")?.get_item(0)?.extract()?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("dtype", torch.getattr("float32")?)?;

                torch.call_method("zeros", ((batch, new_height, new_width),), Some(&kwargs))?
            }
        };

        Ok((resized_image, resized_mask))
    }
}
