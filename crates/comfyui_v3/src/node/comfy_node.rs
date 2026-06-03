use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

/// Trait for ComfyUI v3 nodes.
///
/// Every node must implement this trait so that it can be registered with a
/// `ComfyExtension` and executed by the ComfyUI engine.
///
/// The trait methods map 1-to-1 to the Python `io.ComfyNode` classmethods:
///
/// | Python                    | Rust (`ComfyNode`)      |
/// |---------------------------|-------------------------|
/// | `define_schema`           | `define_schema`         |
/// | `execute`                 | `execute`               |
/// | `check_lazy_status`       | `check_lazy_status`     |
/// | `fingerprint_inputs`      | `fingerprint_inputs`    |
///
/// # Design Philosophy
///
/// The trait signatures intentionally use raw Python types (`Bound<'py, PyAny>`,
/// `Bound<'py, PyTuple>`, `Bound<'py, PyDict>`) rather than Rust wrappers:
///
/// - **Return types**: `define_schema` and `execute` return Python objects directly,
///   so that ComfyUI v3 sees the same types as pure-Python nodes.
/// - **Execute arguments**: Instead of a single `PyDict`, `execute` receives
///   `*args` and `**kwargs` exactly as Python would call it.  This allows node
///   implementations to declare strongly-typed parameter lists and let pyo3's
///   `#[pymethods]` machinery do the unpacking.
///
/// # Safety
/// Implementors must be `Send + Sync` because node instances may be shared
/// across async task boundaries by the ComfyUI runtime.
pub trait ComfyNode: Send + Sync {
    /// Define the node's schema (metadata, inputs, outputs).
    ///
    /// Called once at registration time. Must return a Python object that is
    /// compatible with ComfyUI v3's `NodeOptions` — e.g. a `NodeOptions`
    /// subclass instance or a plain dict with the expected keys.
    fn define_schema<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;

    /// Execute the node logic.
    ///
    /// `args` and `kwargs` mirror the Python call `self.execute(*args, **kwargs)`.
    /// ComfyUI v3 passes evaluated inputs as positional / keyword arguments
    /// according to the node's INPUT_TYPES definition.
    ///
    /// The return value is a Python object — typically a `dict` mapping output
    /// names to their values, or a `tuple` for multi-output nodes.
    fn execute<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: &Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyAny>>;

    /// Optional: determine which lazy inputs still need evaluation.
    ///
    /// Same calling convention as `execute`.  Return a list of input **names**
    /// that must be evaluated before `execute` can proceed.
    ///
    /// The default implementation returns an empty list (no lazy inputs needed).
    fn check_lazy_status<'py>(
        _py: Python<'py>,
        _args: &Bound<'py, PyTuple>,
        _kwargs: &Bound<'py, PyDict>,
    ) -> PyResult<Vec<String>> {
        Ok(Vec::new())
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
        _kwargs: &Bound<'py, PyDict>,
    ) -> PyResult<Option<String>> {
        Ok(None)
    }
}
