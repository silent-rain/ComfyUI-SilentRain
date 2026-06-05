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
//!         .build(py)?;
//! })
//! ```

use anyhow::Context;
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    pymethods,
    types::{PyDict, PyTuple, PyType},
};
use tracing::{error, info};

use comfyui_v3::{
    node::{ExtensionBuilder, PromptServer, extension::pytype_wrapper},
    schema::{
        NodeOutput, NodeSchema,
        hidden::Hidden,
        input::{BoolInput, ComboInput, FloatInput, ImageInput, IntInput, StringInput},
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
#[pyclass(subclass)]
#[derive(Default)]
pub struct ExampleNode;

impl PromptServer for ExampleNode {}

#[pymethods]
impl ExampleNode {
    /// Define the node's schema (metadata, inputs, outputs).
    ///
    /// Called once at registration time. Returns a Python object compatible
    /// with ComfyUI v3's `NodeOptions` — in this example we build a
    /// [`NodeSchema`] on the Rust side and convert it to Python via
    /// [`NodeSchema::into_py_schema`].
    #[classmethod]
    fn define_schema<'py>(
        _cls: Bound<'py, PyType>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
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
                    .min(0)
                    .max(4096)
                    .step(64)
                    .lazy(true)
                    .tooltip("int tips.")
                    .into(),
                FloatInput::new("float_field")
                    .default_value(1.0)
                    .min(0.0)
                    .max(10.0)
                    .step(0.01)
                    .round(0.001)
                    .lazy(true)
                    .into(),
                StringInput::new("string_field")
                    .default_value("Hello world!")
                    .lazy(true)
                    .into(),
                BoolInput::new("bool_field").default_value(true).into(),
                ComboInput::new("combo_field", ["enable", "disable"]).into(),
            ])
            .with_outputs([
                Output::image("imageout")
                    .display_name("image out")
                    .is_output_list(false)
                    .tooltip("image tips."),
                Output::int("int_out")
                    .display_name("int out")
                    .is_output_list(false)
                    .tooltip("int tips."),
                Output::float("float_out")
                    .display_name("float out")
                    .is_output_list(false)
                    .tooltip("float tips."),
                Output::string("string_out")
                    .display_name("string out")
                    .is_output_list(false)
                    .tooltip("string tips."),
                Output::boolean("bool_out")
                    .display_name("bool out")
                    .is_output_list(false)
                    .tooltip("bool tips."),
                Output::combo("combo_out", ["enable", "disable"])
                    .display_name("combo out")
                    .is_output_list(false)
                    .tooltip("combo tips."),
            ])
            .with_hidden([Hidden::UNIQUE_ID, Hidden::EXTRA_PNGINFO])
            .into_py_schema(py)
    }

    /// Execute the node logic.
    ///
    /// Receives `*args` and `**kwargs` exactly as Python would call it.
    /// For this example we pull inputs by name from kwargs (ComfyUI v3
    /// convention) and return a dict with the output value(s).
    #[classmethod]
    #[pyo3(name = "execute", signature = (*args, **kwargs))]
    fn execute_py<'py>(
        _cls: &Bound<'_, PyType>,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        info!("execute_py, args: {args}, kwargs: {kwargs:?}");

        let result = match ExampleNode.execute_rs(py, args, kwargs) {
            Ok(result) => result,
            Err(e) => {
                error!("Error executing node:\n{e:#?}");
                if let Err(e) =
                    Self::send_error(py, "Error executing node".to_string(), e.to_string())
                {
                    error!("send error failed, {e:#?}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
            }
        };

        Ok(result)
    }

    /// Optional: determine which lazy inputs still need evaluation.
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs))]
    fn check_lazy_status<'py>(
        _cls: &Bound<'_, PyType>,
        _py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Vec<String>> {
        info!("fingerprint_inputs, args: {args}, kwargs: {kwargs:?}");

        let kwargs = kwargs.ok_or_else(|| PyErr::new::<PyRuntimeError, _>("kwargs is None"))?;

        let combo_field: String = kwargs
            .get_item("combo_field")?
            .ok_or_else(|| PyErr::new::<PyRuntimeError, _>("combo_field is None"))?
            .extract()?;

        if combo_field == "enable" {
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
    #[classmethod]
    #[pyo3(signature = (*args, **kwargs))]
    fn fingerprint_inputs<'py>(
        _cls: &Bound<'py, PyType>,
        _py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Option<String>> {
        println!("fingerprint_inputs, args: {args}, kwargs: {kwargs:?}");
        Ok(None)
    }
}

impl ExampleNode {
    pub fn execute_rs<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> anyhow::Result<Bound<'py, PyAny>> {
        println!("execute_rs, args: {args}, kwargs: {kwargs:?}");

        let kwargs = kwargs.context("kwargs is None")?;

        let image: Py<PyAny> = kwargs
            .get_item("image")
            .context("failed to get 'image' from kwargs")?
            .context("missing required input 'image'")?
            .into();

        let int_field: i64 = kwargs
            .get_item("int_field")
            .context("failed to get 'int_field' from kwargs")?
            .context("missing required input 'int_field'")?
            .extract()
            .context("'int_field' type mismatch, expected i64")?;

        let float_field: f64 = kwargs
            .get_item("float_field")
            .context("failed to get 'float_field' from kwargs")?
            .context("missing required input 'float_field'")?
            .extract()
            .context("'float_field' type mismatch, expected f64")?;

        let string_field: String = kwargs
            .get_item("string_field")
            .context("failed to get 'string_field' from kwargs")?
            .context("missing required input 'string_field'")?
            .extract()
            .context("'string_field' type mismatch, expected String")?;

        let bool_field: bool = kwargs
            .get_item("bool_field")
            .context("failed to get 'bool_field' from kwargs")?
            .context("missing required input 'bool_field'")?
            .extract()
            .context("'bool_field' type mismatch, expected bool")?;

        let combo_field: String = kwargs
            .get_item("combo_field")
            .context("failed to get 'combo_field' from kwargs")?
            .context("missing required input 'combo_field'")?
            .extract()
            .context("'combo_field' type mismatch, expected String")?;

        // Example: return multiple values using add_arg_from
        // This method accepts any type that implements IntoPy<PyObject>
        let ret = NodeOutput::new()
            .add_arg(image)
            .add_arg_from(py, int_field)?
            .add_arg_from(py, float_field)?
            .add_arg_from(py, string_field)?
            .add_arg_from(py, bool_field)?
            .add_arg_from(py, combo_field)?
            .to_py_obj(py)?;

        Ok(ret)
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
#[pyfunction]
pub fn build_extension<'py>(py: Python<'py>) -> PyResult<Py<PyAny>> {
    ExtensionBuilder::new()
        .add_node(pytype_wrapper::<ExampleNode>(py))
        .add_node_wrapper::<ExampleNode>(py)
        .build(py)
}

/// ComfyUI entrypoint function.
///
/// ```python
/// def comfy_entrypoint():
///     return SilentRainV3Extension()
/// ```
#[pyfunction]
pub fn comfy_entrypoint<'py>(py: Python<'py>) -> PyResult<Py<PyAny>> {
    build_extension(py)
}

// ---------------------------------------------------------------------------
// pyo3 bindings (optional — only needed if compiling as cdylib)
// ---------------------------------------------------------------------------

/// Python module for this example.
///
/// When compiled as a `cdylib`, this module can be imported from Python:
///
/// ```python
/// import example_node
/// ext = example_node.build_extension()
/// ```
#[pymodule]
#[pyo3(name = "example_node")]
mod extension_module {
    use pyo3::{Bound, PyResult, types::PyModule};

    use comfyui_v3::core::logger::init_log;

    #[pymodule_export]
    use super::comfy_entrypoint;

    #[pymodule_export]
    use super::build_extension;

    // 模块初始化时运行代码
    #[pymodule_init]
    fn init(_m: &Bound<'_, PyModule>) -> PyResult<()> {
        // Initialize tracing
        init_log();

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyList;

    use super::*;

    const TEST_COMFYUI_PATH: &str = "/home/one/code/ComfyUI";

    // cargo test -p comfyui_v3 --example example_node_v3 -- tests::test_schema_definition --nocapture
    #[test]
    fn test_schema_definition() -> anyhow::Result<()> {
        Python::attach(|py| -> PyResult<()> {
            // 添加模块搜索路径
            let sys = py.import("sys")?;
            let binding = sys.getattr("path")?;
            let path = binding.cast::<PyList>()?;
            path.insert(0, TEST_COMFYUI_PATH)?; // 或者使用 append

            // 测试直接在 Rust 中调用类方法
            let class = py.get_type::<ExampleNode>();

            let schema = ExampleNode::define_schema(class, py)?;

            println!("=== {:?}", schema);

            // schema is a Python object; verify it has expected attributes
            let node_id: String = schema.getattr("node_id")?.extract()?;
            assert_eq!(node_id, "Example");
            Ok(())
        })?;

        Ok(())
    }

    // cargo test -p comfyui_v3 --example example_node_v3 -- tests::test_io_node_output --nocapture
    #[test]
    fn test_io_node_output() -> anyhow::Result<()> {
        Python::attach(|py| -> PyResult<()> {
            // 添加模块搜索路径
            let sys = py.import("sys")?;
            let binding = sys.getattr("path")?;
            let path = binding.cast::<PyList>()?;
            path.insert(0, TEST_COMFYUI_PATH)?; // 或者使用 append

            let io = py.import("comfy_api.latest")?.getattr("io")?;
            let cls = io.getattr("NodeOutput")?;

            // Build args tuple
            let py_args = PyTuple::new(py, [1])?;

            // Build kwargs dict
            let kwargs = PyDict::new(py);

            println!("args: {}, kwargs: {:?}", py_args, kwargs);

            let result = if kwargs.is_empty() {
                cls.call0()?
            } else {
                cls.call(py_args, Some(&kwargs))?
            };

            println!("=== {:?}", result.getattr("result")?);

            Ok(())
        })?;

        Ok(())
    }

    // cargo test -p comfyui_v3 --example example_node_v3 -- tests::test_execute --nocapture
    #[test]
    fn test_execute() -> anyhow::Result<()> {
        let _ = tracing_subscriber::fmt()
            .with_ansi(true)
            .with_max_level(tracing::Level::DEBUG)
            .with_level(true)
            .with_file(true)
            .with_line_number(true)
            .with_target(false)
            .try_init();

        Python::attach(|py| -> PyResult<()> {
            // 添加模块搜索路径
            let sys = py.import("sys")?;
            let binding = sys.getattr("path")?;
            let path = binding.cast::<PyList>()?;
            path.insert(0, TEST_COMFYUI_PATH)?; // 或者使用 append

            // 测试直接在 Rust 中调用类方法
            let class = py.get_type::<ExampleNode>();

            let args = PyTuple::new(py, Vec::<i32>::new())?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("image", "x")?;
            kwargs.set_item("int_field", 1)?;
            kwargs.set_item("float_field", 1.0)?;
            kwargs.set_item("string_field", "Hello world!")?;
            kwargs.set_item("bool_field", true)?;
            kwargs.set_item("combo_field", "enable")?;

            let result = ExampleNode::execute_py(&class, py, &args, Some(kwargs))?;

            // Debug: print result info
            println!("result type: {}", result.get_type().repr()?);
            println!("result repr: {}", result.repr()?);
            println!("result is callable: {}", result.is_callable());
            println!("result is None: {}", result.is_none());

            println!("============================");

            // Debug: print all attributes of result
            let binding = result.getattr("result")?;
            let result = binding.cast::<PyTuple>()?;
            println!("result: {:?}", result);

            Ok(())
        })?;

        Ok(())
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
