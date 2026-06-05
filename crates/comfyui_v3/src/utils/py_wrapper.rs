//! Python wrapper utilities for ComfyUI v3 nodes.
//!
//! This module provides helper functions to wrap Rust node types into
//! Python-backed `io.ComfyNode` subclasses and create ComfyUI extension wrappers.

use pyo3::{
    Py, PyTypeInfo, Python,
    ffi::c_str,
    prelude::*,
    types::{PyAny, PyType},
};

/// A utility function to get the Python type wrapper for a Rust type.
pub fn pytype_wrapper<'py, T: PyTypeInfo>(py: Python<'py>) -> Py<PyType> {
    py.get_type::<T>().into()
}

/// Create a subclass that inherits from both the Rust class and `io.ComfyNode`.
///
/// This ensures the class has all required methods and attributes from ComfyNode.
/// The original class name, qualname, and module are preserved for compatibility.
///
/// # Arguments
///
/// * `py` - Python GIL guard
/// * `rust_class` - The Rust node class (a Python type object)
///
/// # Returns
///
/// Returns a new Python class that is a subclass of both `rust_class` and `io.ComfyNode`.
///
/// # Example
///
/// ```python
/// # In Python (called from Rust)
/// class Subclass(rust_class, io.ComfyNode):
///     pass
/// Subclass.__name__ = rust_class.__name__
/// Subclass.__qualname__ = rust_class.__qualname__
/// Subclass.__module__ = rust_class.__module__
/// ```
pub fn create_comfy_node_subclass<'py>(
    py: Python<'py>,
    rust_class: Py<PyType>,
) -> PyResult<Bound<'py, PyAny>> {
    let python_code = c_str!(
        "
def make_comfy_node_subclass(rust_class):
    '''
    Create a subclass that inherits from both rust_class and io.ComfyNode.
    This ensures the class has all required methods and attributes from ComfyNode.
    '''
    class Subclass(rust_class, io.ComfyNode):
        pass
    # Preserve the original class name and module for compatibility
    Subclass.__name__ = rust_class.__name__
    Subclass.__qualname__ = rust_class.__qualname__
    Subclass.__module__ = rust_class.__module__
    return Subclass
        "
    );

    let module = PyModule::from_code(
        py,
        python_code,
        c"comfy_node_subclass.py",
        c"comfy_node_subclass",
    )?;

    let make_subclass = module.getattr("make_comfy_node_subclass")?;
    make_subclass.call1((rust_class,))
}

/// Create a `ComfyExtensionWrapper` instance that wraps Rust nodes.
///
/// This function creates a Python class that inherits from `ComfyExtension`
/// and returns an instance of it with the given nodes registered.
///
/// # Arguments
///
/// * `py` - Python GIL guard
/// * `nodes` - A list of Python type objects (nodes) to register
///
/// # Returns
///
/// Returns a `ComfyExtensionWrapper` instance with the nodes registered.
///
/// # Example
///
/// ```python
/// # In Python (called from Rust)
/// class ComfyExtensionWrapper(ComfyExtension):
///     nodes: list[type[io.ComfyNode]] = []
///
///     def __init__(self, nodes: list[type[io.ComfyNode]] = []):
///         super().__init__()
///         self.nodes = nodes
///
///     @override
///     async def get_node_list(self) -> list[type[io.ComfyNode]]:
///         return self.nodes
/// ```
pub fn create_extension_wrapper<'py>(
    py: Python<'py>,
    nodes: Vec<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    let python_code = c_str!(
        "
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

class ComfyExtensionWrapper(ComfyExtension):
    nodes: list[type[io.ComfyNode]] = []

    def __init__(self, nodes: list[type[io.ComfyNode]] = []):
        super().__init__()
        self.nodes = nodes

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return self.nodes
        "
    );

    let module = PyModule::from_code(
        py,
        python_code,
        c"comfy_extension_wrapper.py",
        c"comfy_extension_wrapper",
    )?;

    let py_class = module.getattr("ComfyExtensionWrapper")?;
    let py_class_obj = py_class.call1((nodes,))?;

    Ok(py_class_obj.into())
}
