use pyo3::{
    prelude::*,
    types::{PyBool, PyDict, PyTuple, PyType},
};
use tracing::info;

use comfyui_v3::{
    comfy_node,
    error::{Error, Result},
    node::ComfyNode,
    node::PromptServer,
    schema::{NodeOutput, NodeSchema, hidden::Hidden, input::ImageInput, output::Output},
};

use crate::core::category::Category;

/// A ComfyUI v3 image node — converts an image to grayscale.
///
/// This is an example of using the `#[comfy_node]` macro for pure-Rust
/// node development.
#[comfy_node]
#[pyclass(subclass)]
#[derive(Default)]
pub struct ExampleNodeMacros {}

// Note: PromptServer is implemented for GrayscaleImage (macro adds #[pyclass] to it)
impl PromptServer for ExampleNodeMacros {}

impl ComfyNode for ExampleNodeMacros {
    fn new() -> Self {
        Self {}
    }

    fn define_schema() -> Result<NodeSchema> {
        Ok(NodeSchema::new("ExampleNodeMacros")
            .with_display_name("SR Example Node Macros")
            .with_category(Category::Example)
            .with_description("An example node that demonstrates the use of node macros.")
            .with_deprecated(false)
            .with_experimental(true)
            .with_input_list(false)
            .with_output_node(false)
            .with_inputs([ImageInput::new("image").into()])
            .with_outputs([Output::image("image_out")
                .display_name("image")
                .tooltip("Grayscale image.")])
            .with_hidden([Hidden::UNIQUE_ID, Hidden::EXTRA_PNGINFO]))
    }

    fn execute<'py>(
        _py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<NodeOutput> {
        info!("GrayscaleImage::execute");

        let kwargs = kwargs.ok_or_else(|| Error::Execution("kwargs is None".to_string()))?;

        let image: Py<PyAny> = kwargs
            .get_item("image")
            .ok()
            .flatten()
            .ok_or_else(|| Error::Execution("missing input 'image'".to_string()))?
            .into();

        // ------------------------------------------------------------------
        // TEMPLATE: 这里可以接入实际的图像处理逻辑
        // 例如使用 numpy 或其他图像处理库
        // ------------------------------------------------------------------
        // 目前作为示例，直接将输入原样返回。
        let ret = NodeOutput::new().add_arg(image);

        Ok(ret)
    }
}
