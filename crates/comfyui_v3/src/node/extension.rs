use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple, PyType};

/// Python-facing helper class for ComfyUI v3 extensions.
///
/// This struct provides the Rust implementation of extension functionality.
/// To properly integrate with ComfyUI's extension system, Python code should
/// create a subclass of `comfy_api.latest.ComfyExtension` that delegates
/// to this Rust helper.
#[pyclass(name = "ComfyExtensionHelper", module = "comfyui_v3")]
#[derive(Debug, Default)]
pub struct ComfyExtension {
    /// Registered node classes (Python `type[io.ComfyNode]`).
    nodes: Vec<Py<PyType>>,
}

#[pymethods]
impl ComfyExtension {
    /// Constructor exposed to Python.
    #[new]
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Return the list of node classes exposed by this extension.
    ///
    /// This method returns a list of Python `type` objects representing
    /// the nodes registered with this extension.
    pub fn get_node_list<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let mut out = Vec::with_capacity(self.nodes.len());
        for cls in &self.nodes {
            out.push(cls.bind(py).clone().into_any());
        }
        Ok(out)
    }

    /// Get the number of registered nodes (helper for debugging).
    #[getter]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
}

// ---------------------------------------------------------------------------
// ExtensionBuilder — fluent API for constructing a ComfyExtension
// ---------------------------------------------------------------------------

/// Builder for [`ComfyExtension`].
///
/// Provides a chainable API to register Rust [`ComfyNode`] implementations
/// as Python-backed `io.ComfyNode` subclasses.
pub struct ExtensionBuilder {
    nodes: Vec<Py<PyType>>,
}

impl ExtensionBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Register a single Rust node type.
    ///
    /// `node_class` should be a Python `type` object (a class) that
    /// subclasses `io.ComfyNode`.
    pub fn with_node(mut self, node_class: Py<PyType>) -> Self {
        self.nodes.push(node_class);
        self
    }

    /// Register multiple node types.
    pub fn with_nodes(mut self, nodes: Vec<Py<PyType>>) -> Self {
        self.nodes = nodes;
        self
    }

    /// Consume the builder and produce a [`ComfyExtension`].
    pub fn build(self) -> ComfyExtension {
        ComfyExtension { nodes: self.nodes }
    }
}

// ---------------------------------------------------------------------------
// Python Module Functions
// ---------------------------------------------------------------------------

/// Create a Python class that inherits from `comfy_api.latest.ComfyExtension`
/// and wraps a Rust [`ComfyExtension`] helper.
///
/// This function can be used in `comfy_entrypoint()` to easily create
/// a properly inherited extension.
///
/// The dynamically created class will:
/// - Inherit from `comfy_api.latest.ComfyExtension` (passes isinstance checks)
/// - Properly call `super().__init__()` in its `__init__`
/// - Expose `async get_node_list` that delegates to the Rust helper
///
/// # Python Usage
///
/// ```python
/// import my_rust_module
///
/// def comfy_entrypoint():
///     helper = my_rust_module.ComfyExtensionHelper()
///     # Add nodes to helper...
///     return my_rust_module.create_extension_subclass(helper)
/// ```
#[pyfunction]
pub fn create_extension_subclass<'py>(
    py: Python<'py>,
    helper: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // Import the Python base class: comfy_api.latest.ComfyExtension
    let comfy_api = py.import("comfy_api.latest")?;
    let base_class = comfy_api.getattr("ComfyExtension")?;

    // Get Python's built-in `type` function
    let type_fn = py.eval(&CString::new("type")?, None, None)?;

    // Create class dict with methods
    let class_dict = PyDict::new(py);

    // Store helper as a class attribute for access in methods
    class_dict.set_item("__rust_helper", helper)?;

    // __init__ method — calls super().__init__() and stores helper
    let init_code = py.eval(
        &CString::new(
            r#"
def __init__(self_, *args, **kwargs):
    super(type(self_), self_).__init__(*args, **kwargs)
"#,
        )?,
        None,
        None,
    )?;
    class_dict.set_item("__init__", init_code)?;

    // get_node_list method — async, delegates to the Rust helper
    let get_node_list_code = py.eval(
        &CString::new(
            r#"
async def get_node_list(self_):
    return self_.__rust_helper.get_node_list()
"#,
        )?,
        None,
        None,
    )?;
    class_dict.set_item("get_node_list", get_node_list_code)?;

    // Create the class using type(name, bases, dict)
    let name = PyString::new(py, "RustComfyExtension");
    let bases = PyTuple::new(py, &[base_class])?;
    let inherited_class = type_fn.call((name, bases, class_dict), None)?;

    // Instantiate and return
    inherited_class.call((), None)
}

/// Helper to get the `comfy_api.latest.ComfyExtension` base class.
///
/// This is useful if you want to manually create a subclass in Python.
#[pyfunction]
pub fn get_comfy_extension_base<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let comfy_api = py.import("comfy_api.latest")?;
    comfy_api.getattr("ComfyExtension")
}
