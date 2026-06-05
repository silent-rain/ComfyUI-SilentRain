use pyo3::{PyTypeInfo, prelude::*, types::PyType};

use crate::utils::py_wrapper::{create_comfy_node_subclass, create_extension_wrapper};

/// Builder for [`ComfyExtension`].
///
/// Provides a chainable API to register Rust [`ComfyNode`] implementations
/// as Python-backed `io.ComfyNode` subclasses.
#[derive(Debug, Default)]
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
    pub fn add_node(mut self, node: Py<PyType>) -> Self {
        self.nodes.push(node);
        self
    }

    /// Register multiple node types.
    pub fn add_nodes(mut self, nodes: Vec<Py<PyType>>) -> Self {
        self.nodes.extend(nodes);
        self
    }

    /// Register multiple node types.
    pub fn with_nodes(mut self, nodes: Vec<Py<PyType>>) -> Self {
        self.nodes = nodes;
        self
    }

    /// A utility function to get the Python type wrapper for a Rust type.
    pub fn add_node_wrapper<'py, T: PyTypeInfo>(mut self, py: Python<'py>) -> Self {
        let node: Py<PyType> = py.get_type::<T>().into();
        self.nodes.push(node);
        self
    }

    /// Get the number of registered nodes (helper for debugging).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Consume the builder and produce a [`ComfyExtensionWrapper`].
    /// The wrapper for [`comby_api.latest import.ComfyExtension`]
    pub fn build<'py>(self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        // Wrap each node as a ComfyNode subclass
        let mut comfy_nodes: Vec<Py<PyAny>> = Vec::new();
        for node in self.nodes {
            let subclass = create_comfy_node_subclass(py, node)?;
            comfy_nodes.push(subclass.into());
        }

        // Create the extension wrapper with all wrapped nodes
        create_extension_wrapper(py, comfy_nodes)
    }
}
