//! Reference Latent Combo
//!
//! 组合节点：将多张参考图像通过 ResizeImagesByLongerEdge + VAEEncode + ReferenceLatent
//! 进行链式拼接，生成带有参考 latent 的 conditioning 输出。
//!
//! 节点流程：
//!   main_image → ResizeImagesByLongerEdge(main_long_edge) → VAEEncode → latent
//!   → ReferenceLatent(positive/negative)
//!   ref_imageN → ResizeImagesByLongerEdge(ref_long_edge 或 main_long_edge) → VAEEncode
//!   → latent → ReferenceLatent(positive/negative)

use log::error;
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyList, PyType},
};

use crate::{
    core::{
        category::CATEGORY_CONDITIONING,
        node_base::{InputSpec, InputType},
    },
    error::Error,
    image::ResizeImageMaskByLongerEdge,
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_CONDITIONING, NODE_INT, NODE_LATENT, NODE_MASK},
    },
};
const DEFAULT_REF_LONG_EDGE: i64 = 512;

/// Reference Latent Combo 组合节点
#[pyclass(subclass)]
pub struct ReferenceLatentCombo {}

impl PromptServer for ReferenceLatentCombo {}

#[pymethods]
impl ReferenceLatentCombo {
    #[new]
    fn new() -> Self {
        Self {}
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
        &'static str,
    ) {
        (
            NODE_CONDITIONING,
            NODE_CONDITIONING,
            NODE_LATENT,
            NODE_MASK,
            NODE_INT,
            NODE_INT,
        )
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    ) {
        (
            "positive",
            "negative",
            "main_latent",
            "main_mask",
            "width",
            "height",
        )
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool, bool, bool, bool, bool) {
        (false, false, false, false, false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_CONDITIONING;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Combo node: chains multiple reference images through ResizeImagesByLongerEdge + VAEEncode + ReferenceLatent. \
        ref_long_index specifies which ref images (1-based, comma-separated) use ref_long_edge; others use main_long_edge."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        InputSpec::new()
            .with_required("vae", InputType::vae())
            .with_required("positive", InputType::conditioning())
            .with_required("negative", InputType::conditioning())
            .with_required("main_image", InputType::image())
            .with_optional("main_mask", InputType::mask())
            .with_required(
                "main_long_edge",
                InputType::int()
                    .default(1024)
                    .min(64)
                    .max(8192)
                    .step(8)
                    .tooltip("Longer edge size for main image resize"),
            )
            .with_required(
                "ref_long_index",
                InputType::string()
                    .default("1,2,3,4,5")
                    .tooltip("Comma-separated 1-based indices of ref images that use ref_long_edge (e.g. '1,2,3,4'). Others use main_long_edge."),
            )
            .with_required(
                "ref_long_edge",
                InputType::int()
                    .default(1024)
                    .min(64)
                    .max(8192)
                    .step(8)
                    .tooltip("Longer edge size for ref images listed in ref_long_index"),
            )
            .with_optional("ref_image1", InputType::image())
            .with_optional("ref_image2", InputType::image())
            .with_optional("ref_image3", InputType::image())
            .with_optional("ref_image4", InputType::image())
            .with_optional("ref_image5", InputType::image())
            .build()
    }

    #[pyo3(name = "execute", signature = (positive, negative, vae, main_image, main_long_edge, ref_long_index, ref_long_edge, main_mask=None, ref_image1=None, ref_image2=None, ref_image3=None, ref_image4=None, ref_image5=None))]
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        positive: Bound<'py, PyAny>,
        negative: Bound<'py, PyAny>,
        vae: Bound<'py, PyAny>,
        main_image: Bound<'py, PyAny>,
        main_long_edge: i64,
        ref_long_index: String,
        ref_long_edge: i64,
        main_mask: Option<Bound<'py, PyAny>>,
        ref_image1: Option<Bound<'py, PyAny>>,
        ref_image2: Option<Bound<'py, PyAny>>,
        ref_image3: Option<Bound<'py, PyAny>>,
        ref_image4: Option<Bound<'py, PyAny>>,
        ref_image5: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        Bound<'py, PyAny>,
        i64,
        i64,
    )> {
        let result = self.run(
            py,
            positive,
            negative,
            vae,
            main_image,
            main_long_edge,
            ref_long_index,
            ref_long_edge,
            main_mask,
            [ref_image1, ref_image2, ref_image3, ref_image4, ref_image5],
        );

        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("ReferenceLatentCombo error: {e}");
                if let Err(send_err) =
                    self.send_error(py, "ReferenceLatentCombo".to_string(), e.to_string())
                {
                    error!("send error failed: {send_err}");
                }
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl ReferenceLatentCombo {
    /// 解析 ref_long_index 字符串，返回使用 ref_long_edge 的 1-based 索引集合
    fn parse_ref_long_index(ref_long_index: &str) -> Vec<usize> {
        ref_long_index
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect()
    }

    /// 直接调用 ResizeImageMaskByLongerEdge Rust 实现缩放图像和遮罩
    /// 返回 (resized_image, resized_mask_or_zeros)
    fn resize_image_mask<'py>(
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
        longer_edge: i64,
        mask: Option<&Bound<'py, PyAny>>,
    ) -> Result<(Bound<'py, PyAny>, Bound<'py, PyAny>), Error> {
        let node = ResizeImageMaskByLongerEdge::new();
        let (resized_image, resized_mask) =
            node.run(py, image.clone(), longer_edge, mask.cloned())?;
        Ok((resized_image, resized_mask))
    }

    /// 通过 Python 调用 ResizeImagesByLongerEdge.execute(images, longer_edge)
    /// 返回缩放后的图像 tensor
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
        // NodeOutput 是一个包含结果的对象，取第一个输出
        let result = node_output.get_item(0)?;
        Ok(result)
    }

    /// 获取图像的宽高（从 tensor shape 直接读取）
    /// 返回 (width, height)
    fn get_image_size<'py>(image: &Bound<'py, PyAny>) -> Result<(i64, i64), Error> {
        // image 形状为 [B, H, W, C]
        let shape = image.getattr("shape")?;
        let height: i64 = shape.get_item(1)?.extract()?;
        let width: i64 = shape.get_item(2)?.extract()?;
        Ok((width, height))
    }

    /// 通过 Python 调用 VAEEncode.encode(vae, pixels)
    /// 返回 latent dict {"samples": tensor}
    fn vae_encode<'py>(
        py: Python<'py>,
        vae: &Bound<'py, PyAny>,
        pixels: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, Error> {
        let module = py.import("nodes")?;
        let node_cls = module.getattr("VAEEncode")?;
        let node_inst = node_cls.call0()?;
        let result = node_inst.call_method1("encode", (vae, pixels))?;
        // encode 返回 ({"samples": tensor},)，取第一个元素
        let latent = result.get_item(0)?;
        Ok(latent)
    }

    /// 通过 Python 调用 node_helpers.conditioning_set_values 实现 ReferenceLatent 逻辑
    /// 将 latent["samples"] 追加到 conditioning 的 reference_latents 中
    fn apply_reference_latent<'py>(
        py: Python<'py>,
        conditioning: &Bound<'py, PyAny>,
        latent: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, Error> {
        let node_helpers = py.import("node_helpers")?;
        let samples = latent.get_item("samples")?;
        let values = PyDict::new(py);
        let ref_list = PyList::new(py, [&samples])?;
        values.set_item("reference_latents", ref_list)?;
        let kw = PyDict::new(py);
        kw.set_item("append", true)?;
        let result = node_helpers.call_method(
            "conditioning_set_values",
            (conditioning, values),
            Some(&kw),
        )?;
        Ok(result)
    }

    /// 主执行逻辑
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &self,
        py: Python<'py>,
        positive: Bound<'py, PyAny>,
        negative: Bound<'py, PyAny>,
        vae: Bound<'py, PyAny>,
        main_image: Bound<'py, PyAny>,
        main_long_edge: i64,
        ref_long_index: String,
        ref_long_edge: i64,
        main_mask: Option<Bound<'py, PyAny>>,
        ref_images: [Option<Bound<'py, PyAny>>; 5],
    ) -> Result<
        (
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            Bound<'py, PyAny>,
            i64,
            i64,
        ),
        Error,
    > {
        // 解析哪些 ref 图像使用 ref_long_edge
        let use_ref_edge_indices = Self::parse_ref_long_index(&ref_long_index);

        // 1. 处理 main_image：使用 ResizeImageMaskByLongerEdge 缩放图像和遮罩
        let (main_image_resized, main_mask_resized) =
            Self::resize_image_mask(py, &main_image, main_long_edge, main_mask.as_ref())?;
        let main_latent = Self::vae_encode(py, &vae, &main_image_resized)?;

        // 3. 将 main latent 应用到 positive 和 negative
        let mut pos = positive;
        let mut neg = negative;
        pos = Self::apply_reference_latent(py, &pos, &main_latent)?;
        neg = Self::apply_reference_latent(py, &neg, &main_latent)?;

        // 4. 处理每张 ref 图像
        for (i, ref_image_opt) in ref_images.into_iter().enumerate() {
            let ref_idx = i + 1; // 1-based
            let Some(ref_image) = ref_image_opt else {
                continue; // 未连接则跳过
            };

            // 判断使用哪个 long_edge
            let edge = if use_ref_edge_indices.contains(&ref_idx) {
                ref_long_edge
            } else {
                DEFAULT_REF_LONG_EDGE
            };

            let ref_resized = Self::resize_image(py, &ref_image, edge)?;
            let ref_latent = Self::vae_encode(py, &vae, &ref_resized)?;

            pos = Self::apply_reference_latent(py, &pos, &ref_latent)?;
            neg = Self::apply_reference_latent(py, &neg, &ref_latent)?;
        }

        // 2. 获取缩放后图像的宽高
        let (width, height) = Self::get_image_size(&main_image_resized)?;

        Ok((pos, neg, main_latent, main_mask_resized, width, height))
    }
}
