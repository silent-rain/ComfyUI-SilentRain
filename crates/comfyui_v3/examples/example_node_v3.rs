//! ComfyUI v3 Node Example — Rust Implementation
//!
//! This example demonstrates how to define a ComfyUI v3 node entirely in Rust
//! using the `comfyui_v3` crate. It mirrors the functionality of the Python
//! example (`example_node_v3.py`) but leverages Rust's type safety and
//! zero-cost abstractions.
//!
//! # Usage
//!
//! ```rust,no_run
//! use comfyui_v3::prelude::*;
//! use pyo3::prelude::*;
//!
//! Python::with_gil(|py| {
//!     let ext = ExtensionBuilder::new("ExampleExtension")
//!         .with_node::<ExampleNode>()
//!         .build(py)
//!         .unwrap();
//! })
//! ```

use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use comfyui_v3::{
    node::{ComfyNode, ExtensionBuilder},
    schema::{
        NodeSchema,
        hidden::Hidden,
        input::{ComboInput, FloatInput, ImageInput, IntInput, StringInput},
        output::Output,
    },
};

// ---------------------------------------------------------------------------
// ExampleNode
// ---------------------------------------------------------------------------

/// An example ComfyUI v3 node implemented in Rust.
///
/// This node inverts an image and optionally prints input values to the screen.
/// It mirrors the Python `Example` node from `example_node_v3.py`.
#[derive(Default)]
pub struct ExampleNode;

impl ComfyNode for ExampleNode {
    /// Define the node's schema (metadata, inputs, outputs).
    ///
    /// Called once at registration time. Returns a Python object compatible
    /// with ComfyUI v3's `NodeOptions` — in this example we build a
    /// [`NodeSchema`] on the Rust side and convert it to Python via
    /// [`NodeSchema::into_py_schema`].
    fn define_schema<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        NodeSchema::new("Example") // 节点ID
            .with_display_name("Example Node") // 节点显示名称
            .with_category("Example") // 节点分类
            .with_description("This is an example node.") // 节点描述, 可选
            .with_deprecated(false) // 过时标记, 可选
            .with_experimental(true) // 实验性的, 可选
            .with_input_list(false) // 输入是否为列表, 可选
            .with_output_node(false) // 是否为输出节点, 可选
            .with_inputs([
                ImageInput::new("image").into(),
                IntInput::new("int_field")
                    .with_min(0)
                    .with_max(4096)
                    .with_step(64)
                    .with_lazy(true)
                    .into(),
                FloatInput::new("float_field")
                    .with_default(1.0)
                    .with_min(0.0)
                    .with_max(10.0)
                    .with_step(0.01)
                    .with_round(0.001)
                    .with_lazy(true)
                    .into(),
                ComboInput::new("print_to_screen", ["enable", "disable"]).into(),
                StringInput::new("string_field")
                    .with_default("Hello world!")
                    .with_lazy(true)
                    .into(),
            ])
            .with_outputs([Output::image("imageout")])
            .with_hidden([Hidden::UNIQUE_ID, Hidden::EXTRA_PNGINFO])
            .into_py_schema(py)
    }

    /// Execute the node logic.
    ///
    /// Receives `*args` and `**kwargs` exactly as Python would call it.
    /// For this example we pull inputs by name from kwargs (ComfyUI v3
    /// convention) and return a dict with the output value(s).
    fn execute<'py>(
        &self,
        py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        kwargs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let image: Py<PyAny> = kwargs.get_item("image")?.unwrap().unbind();
        let print_to_screen: String = kwargs.get_item("print_to_screen")?.unwrap().extract()?;

        if print_to_screen == "enable" {
            let string_field: String = kwargs.get_item("string_field")?.unwrap().extract()?;
            let int_field: i64 = kwargs.get_item("int_field")?.unwrap().extract()?;
            let float_field: f64 = kwargs.get_item("float_field")?.unwrap().extract()?;

            println!(
                "Your input contains:\n\
                 string_field aka input text: {}\n\
                 int_field: {}\n\
                 float_field: {}\n",
                string_field, int_field, float_field
            );
        }

        // Invert the image: image = 1.0 - image
        let one = py.eval(c"1.0", None, None)?.unbind();
        let inverted = one.bind(py).call_method1("__sub__", (image,))?;

        // Build return dict { "imageout": inverted }
        let ret = PyDict::new(py);
        ret.set_item("imageout", inverted)?;

        Ok(ret.into_any())
    }

    /// Optional: determine which lazy inputs still need evaluation.
    fn check_lazy_status<'py>(
        _py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        kwargs: &Bound<'py, PyDict>,
    ) -> PyResult<Vec<String>> {
        let print_to_screen: String = kwargs.get_item("print_to_screen")?.unwrap().extract()?;

        if print_to_screen == "enable" {
            Ok(vec![
                "int_field".to_string(),
                "float_field".to_string(),
                "string_field".to_string(),
            ])
        } else {
            Ok(Vec::new())
        }
    }

    /// Optional: return a fingerprint string for cache invalidation.
    fn fingerprint_inputs<'py>(
        _py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: &Bound<'py, PyDict>,
    ) -> PyResult<Option<String>> {
        Ok(None)
    }
}

// ---------------------------------------------------------------------------
// Extension Registration
// ---------------------------------------------------------------------------

/// Build the example extension with our node.
///
/// This function acquires the GIL and creates a Python `io.ComfyNode`
/// subclass backed by [`ExampleNode`].
///
/// # Panics
/// Panics if called outside the Python interpreter (use [`build_extension_py`]
/// when exposing to Python).
pub fn build_extension() -> comfyui_v3::node::ComfyExtension {
    Python::try_attach(|py| {
        ExtensionBuilder::new("ExampleExtension")
            .with_node::<ExampleNode>()
            .build(py)
    })
}

// ---------------------------------------------------------------------------
// pyo3 bindings (optional — only needed if compiling as cdylib)
// ---------------------------------------------------------------------------

/// Python module for this example.
///
/// When compiled as a `cdylib`, this module can be imported from Python:
///
/// ```python
/// import example_node_v3
/// ext = example_node_v3.build_extension()
/// ```
#[pymodule]
#[pyo3(name = "example_node_v3")]
fn init_example_node_v3(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(build_extension_py, m)?)?;
    Ok(())
}

/// Python-facing wrapper for [`build_extension`].
#[pyfunction]
fn build_extension_py() -> comfyui_v3::node::ComfyExtension {
    build_extension()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_definition() {
        pyo3::prepare_freethreaded_python();
        Python::try_attach(|py| {
            let schema = ExampleNode::define_schema(py).unwrap();
            // schema is a Python object; verify it has expected attributes
            let node_id: String = schema.getattr("node_id").unwrap().extract().unwrap();
            assert_eq!(node_id, "Example");
        });
    }

    #[test]
    fn test_extension_builder() {
        let ext = build_extension();
        assert_eq!(ext.name, "ExampleExtension");
        assert_eq!(ext.node_classes().len(), 1);
    }
}

// ---------------------------------------------------------------------------
// main (required for example binary)
// ---------------------------------------------------------------------------

fn main() {
    println!("ComfyUI v3 Example Node");
    println!("=======================");
    println!();
    println!("This example demonstrates how to define a ComfyUI v3 node in Rust.");
    println!();
    println!("To use this node, compile it as a cdylib and import from Python:");
    println!();
    println!("    from example_node_v3 import build_extension");
    println!("    ext = build_extension()");
    println!();
}
