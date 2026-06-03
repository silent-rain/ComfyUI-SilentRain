use pyo3::{PyTypeInfo, ffi::c_str, prelude::*, types::PyType};

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
    pub fn build<'py>(self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let python_code = c_str!(
            "
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

        // 使用PyModule::from_code创建Python模块
        let module = PyModule::from_code(
            py,
            python_code,
            c"comfy_extension_wrapper.py",
            c"comfy_extension_wrapper",
        )?;

        // 从模块中获取类
        let py_class = module.getattr("ComfyExtensionWrapper")?;

        // 调用Python类的构造函数
        let py_class_obj = py_class.call1((self.nodes,))?;

        Ok(py_class_obj.into())
    }
}

/// A utility function to get the Python type wrapper for a Rust type.
pub fn pytype_wrapper<'py, T: PyTypeInfo>(py: Python<'py>) -> Py<PyType> {
    py.get_type::<T>().into()
}
