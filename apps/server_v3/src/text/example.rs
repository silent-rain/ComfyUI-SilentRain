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
        input::{BoolInput, StringInput},
        output::Output,
    },
};

/// An example ComfyUI v3 text node — echoes input text.
///
/// Demonstrates how a text-processing / conditioning node is structured.
#[pyclass(subclass)]
#[derive(Default)]
pub struct TextEcho;

impl PromptServer for TextEcho {}

#[pymethods]
impl TextEcho {
    /// Define the node's schema.
    #[classmethod]
    fn define_schema<'py>(
        _cls: Bound<'py, PyType>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        NodeSchema::new("TextEcho")
            .with_display_name("SR Text Echo")
            .with_category("SilentRain/text")
            .with_description("Echo the input text unchanged.")
            .with_deprecated(false)
            .with_experimental(false)
            .with_input_list(false)
            .with_output_node(false)
            .with_inputs([
                StringInput::new("text")
                    .with_default("Hello SilentRain!")
                    .with_lazy(true)
                    .with_tooltip("Input text string.")
                    .into(),
                BoolInput::new("print_to_console")
                    .with_default(false)
                    .with_tooltip("Print the text to the Rust console.")
                    .into(),
            ])
            .with_outputs([Output::string("text_out")
                .with_display_name("text")
                .with_is_output_list(false)
                .with_tooltip("Same text that was input.")])
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
        info!("TextEcho::execute_py args={args}, kwargs={kwargs:?}");

        let result = match TextEcho.execute_rs(py, args, kwargs) {
            Ok(result) => result,
            Err(e) => {
                error!("Error executing TextEcho:\n{e:#?}");
                if let Err(e) = Self::send_error(py, "TextEcho error".to_string(), e.to_string()) {
                    error!("send_error failed, {e:#?}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                }
                return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
            }
        };

        Ok(result)
    }

    /// Check lazy inputs — return names of inputs that still need evaluation.
    #[classmethod]
    #[pyo3(signature = (*_args, **_kwargs))]
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

impl TextEcho {
    pub fn execute_rs<'py>(
        &self,
        py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> anyhow::Result<Bound<'py, PyAny>> {
        let kwargs = kwargs.ok_or_else(|| anyhow::anyhow!("kwargs is None"))?;

        let text: String = kwargs
            .get_item("text")
            .ok()
            .flatten()
            .ok_or_else(|| anyhow::anyhow!("missing input 'text'"))?
            .extract()
            .map_err(|e| anyhow::anyhow!("'text' type mismatch: {e}"))?;

        let print_to_console: bool = kwargs
            .get_item("print_to_console")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or(false);

        if print_to_console {
            println!("[TextEcho] {text}");
        }

        let ret = NodeOutput::new().add_arg_from(py, text)?.to_py_obj(py)?;

        Ok(ret)
    }
}
