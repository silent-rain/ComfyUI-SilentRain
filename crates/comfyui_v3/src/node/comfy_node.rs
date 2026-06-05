//! Pure Rust trait for ComfyUI v3 node development.
//!
//! This trait is designed to be used with the `#[comfy_node]` proc-macro
//! from the `comfy_node_macros` crate.
//!
//! # Example
//!
//! ```ignore
//! use comfyui_v3::node::ComfyNode;
//! use comfyui_v3::schema::{NodeSchema, NodeOutput};
//! use comfyui_v3::error::Result;
//!
//! struct MyNode;
//!
//! impl ComfyNode for MyNode {
//!     fn define_schema() -> Result<NodeSchema> {
//!         Ok(NodeSchema::new("MyNode")
//!             .with_display_name("My Node")
//!             .with_category("image"))
//!     }
//!
//!     fn execute(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> Result<NodeOutput> {
//!         // implementation
//!     }
//! }
//! ```
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use crate::error::Result;
use crate::schema::{NodeOutput, NodeSchema};

/// Pure Rust trait for ComfyUI v3 nodes.
///
/// Implement this trait for your node struct, then apply `#[comfy_node]`
/// attribute macro to automatically generate the Python interop layer.
///
/// The trait methods return Rust types (`NodeSchema`, `NodeOutput`) instead
/// of raw Python objects, making the implementation more idiomatic and safe.
///
/// # Example
///
/// ```ignore
/// #[comfy_node]
/// struct InvertImage;
///
/// impl ComfyNode for InvertImage {
///     fn define_schema() -> NodeSchema {
///         NodeSchema::new("InvertImage")
///             .with_display_name("Invert Image")
///             .with_category("image")
///     }
///
///     fn execute(py: Python<'_>, kwargs: Option<&Bound<'_, PyDict>>) -> NodeOutput {
///         // implementation
///     }
/// }
/// ```
pub trait ComfyNode: Send + Sync + Default {
    /// Create a new instance of the node.
    ///
    /// This is required for Python instantiation via `#[new]`.
    /// The `#[comfy_node]` macro will automatically generate the
    /// Python constructor that calls this method.
    fn new() -> Self
    where
        Self: Sized,
    {
        Self::default()
    }

    /// Define the node's schema (metadata, inputs, outputs).
    ///
    /// This should return a [`NodeSchema`] value that describes the node.
    /// The `#[comfy_node]` macro will automatically convert this to a Python
    /// `io.Schema` object.
    fn define_schema() -> Result<NodeSchema>;

    /// Execute the node logic.
    ///
    /// - `py`: The Python GIL token.
    /// - `args`: Positional arguments (usually empty for v3 nodes).
    /// - `kwargs`: Keyword arguments containing the input values.
    ///
    /// Returns a [`NodeOutput`] value that will be converted to a Python object
    /// by the `#[comfy_node]` macro.
    fn execute<'py>(
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<NodeOutput>;

    /// Optional: validate node inputs before execution.
    ///
    /// Equivalent to V1's `VALIDATE_INPUTS`.
    ///
    /// Return `Ok(())` to indicate validation success.
    /// Return `Err(Error::Validation(msg))` to indicate validation failure.
    ///
    /// The default implementation returns `Ok(())` (no validation needed).
    fn validate_inputs<'py>(
        _py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<()> {
        Ok(())
    }

    /// Optional: return a fingerprint string for cache invalidation.
    ///
    /// When the returned value differs from the previous execution fingerprint,
    /// ComfyUI will re-run the node even if none of the input connections
    /// changed.  Useful for nodes like `LoadImage` that depend on external
    /// files.
    ///
    /// The default implementation returns `None` (fallback to normal
    /// input-change detection).
    fn fingerprint_inputs<'py>(
        _py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<Option<String>> {
        Ok(None)
    }

    /// Optional: determine which lazy inputs still need evaluation.
    ///
    /// Return a list of input **names** that must be evaluated before
    /// `execute` can proceed.
    ///
    /// The default implementation returns an empty list (no lazy inputs needed).
    fn check_lazy_status<'py>(
        _py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<Vec<String>> {
        Ok(Vec::new())
    }
}
