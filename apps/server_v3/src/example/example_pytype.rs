use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    pymethods,
    types::{PyBool, PyDict, PyTuple, PyType},
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

use crate::core::category::Category;

/// An example ComfyUI v3 image node — inverts an image.
///
/// Mirrors the functionality of the `example_node_v3.py` image processing
/// example but demonstrates how a real image-manipulation node is structured
/// in a production crate.
#[pyclass(subclass)]
#[derive(Default)]
pub struct ExamplePytype {}

impl PromptServer for ExamplePytype {}

#[pymethods]
impl ExamplePytype {
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
        NodeSchema::new("ExamplePytype")
            .with_display_name("SR Example Pytype")
            .with_category(Category::Example)
            .with_description("An example node that demonstrates the use of Pytypes.")
            .with_deprecated(false)
            .with_experimental(true)
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
    ///
    /// This method is called by ComfyUI to execute the node.
    /// The method signature matches io.ComfyNode.execute() expectation.
    #[classmethod]
    #[pyo3(name = "execute")]
    fn execute<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        image: Py<PyAny>,
        include_alpha: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        info!(
            "InvertImage::execute_py called with include_alpha={}",
            include_alpha
        );

        let result = match ExamplePytype::new().execute_rs(py, image, include_alpha) {
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

    /// Optional: Validate inputs before execution.
    ///
    /// This method is called by ComfyUI to validate inputs.
    /// Return true if inputs are valid, or an error message if not.
    #[classmethod]
    #[pyo3(name = "validate_inputs", signature = (*_args, **_kwargs))]
    fn validate_inputs<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Default implementation: always valid
        // Ok(PyBool::new(py, true).into_bound().into_any())
        // true.into_bound_py_any(py)
        Ok(PyBool::new(py, true).as_any().clone())
    }

    /// Optional: Control when the node is re-executed.
    ///
    /// This method returns a value that will be compared to the one returned
    /// the last time the node was executed. If it is different, the node will
    /// be executed again.
    #[classmethod]
    #[pyo3(name = "fingerprint_inputs", signature = (*_args, **_kwargs))]
    fn fingerprint_inputs<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Default implementation: return empty string (always re-execute if inputs change)
        Ok("".into_pyobject(py).unwrap().into_any())
    }

    /// Check lazy inputs — return names of inputs that still need evaluation.
    #[classmethod]
    #[pyo3(name = "check_lazy_status", signature = (*_args, **_kwargs))]
    fn check_lazy_status<'py>(
        _cls: &Bound<'_, PyType>,
        _py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<String>> {
        // No lazy inputs needed for this simple node.
        Ok(vec![])
    }
}

impl ExamplePytype {
    pub fn execute_rs<'py>(
        &self,
        py: Python<'py>,
        image: Py<PyAny>,
        include_alpha: bool,
    ) -> anyhow::Result<Bound<'py, PyAny>> {
        info!(
            "InvertImage::execute_rs called with include_alpha={}",
            include_alpha
        );

        // ------------------------------------------------------------------
        // TODO: 这里可以接入实际的图像处理逻辑（例如 numpy 操作）
        // ------------------------------------------------------------------
        // 目前作为示例，直接将输入原样返回。
        // 实际实现时，应该使用 numpy 或 torch 进行图像反转操作：
        // let np = py.import("numpy")?;
        // let inverted = np.call_method1("invert", (image,))?;

        let ret = NodeOutput::new().add_arg(image).to_py_obj(py)?;

        Ok(ret)
    }
}
