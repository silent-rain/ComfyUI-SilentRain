use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    pymethods,
    types::{PyDict, PyTuple, PyType},
};
use tracing::{error, info};

use comfyui_v3::{
    node::PromptServer,
    schema::{
        NodeOutput, NodeSchema,
        hidden::Hidden,
        input::{BoolInput, ImageInput},
        output::Output,
    },
};

/// An example ComfyUI v3 image node — inverts an image.
///
/// Mirrors the functionality of the `example_node_v3.py` image processing
/// example but demonstrates how a real image-manipulation node is structured
/// in a production crate.
#[pyclass(subclass)]
#[derive(Default)]
pub struct InvertImage {}

impl PromptServer for InvertImage {}

#[pymethods]
impl InvertImage {
    #[new]
    fn new() -> Self {
        Self {}
    }

    /// Define the node's schema (metadata, inputs, outputs).
    #[classmethod]
    fn define_schema<'py>(
        _cls: Bound<'py, PyType>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        NodeSchema::new("InvertImage")
            .with_display_name("SR Invert Image")
            .with_category("SilentRain/image")
            .with_description("Invert the colours of an image.")
            .with_deprecated(false)
            .with_experimental(false)
            .with_input_list(false)
            .with_output_node(false)
            .with_inputs([
                ImageInput::new("image").into(),
                BoolInput::new("include_alpha")
                    .default_value(false)
                    .tooltip("Also invert the alpha channel.")
                    .into(),
            ])
            .with_outputs([Output::image("image_out")
                .display_name("image")
                .is_output_list(false)
                .tooltip("Inverted image.")])
            .with_hidden([Hidden::UNIQUE_ID, Hidden::EXTRA_PNGINFO])
            .into_py_schema(py)
    }

    /// Execute the node logic.
    #[classmethod]
    #[pyo3(name = "execute", signature = (*args, **kwargs))]
    fn execute_py<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        info!("InvertImage::execute_py args={args}, kwargs={kwargs:?}");

        let result = match InvertImage::new().execute_rs(py, args, kwargs) {
            Ok(result) => result,
            Err(e) => {
                error!("Error executing InvertImage:\n{e:#?}");
                if let Err(e) = Self::send_error(py, "InvertImage error".to_string(), e.to_string())
                {
                    error!("send_error failed, {e:#?}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                }
                return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
            }
        };

        Ok(result)
    }
}

impl InvertImage {
    pub fn execute_rs<'py>(
        &self,
        py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> anyhow::Result<Bound<'py, PyAny>> {
        let kwargs = kwargs.ok_or_else(|| anyhow::anyhow!("kwargs is None"))?;

        let image: Py<PyAny> = kwargs
            .get_item("image")
            .ok()
            .flatten()
            .ok_or_else(|| anyhow::anyhow!("missing input 'image'"))?
            .into();

        let _include_alpha: bool = kwargs
            .get_item("include_alpha")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or(false);

        // ------------------------------------------------------------------
        // TEMPLATE: 这里可以接入实际的图像处理逻辑（例如 numpy 操作）
        // ------------------------------------------------------------------
        // 目前作为示例，直接将输入原样返回。
        let ret = NodeOutput::new().add_arg(image).to_py_obj(py)?;

        Ok(ret)
    }
}
