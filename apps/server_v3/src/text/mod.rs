/// Text processing nodes for SilentRain v3.
use pyo3::{
    prelude::*,
    types::{PyModule, PyModuleMethods, PyType},
};

pub mod example;
pub use example::TextEcho;

/// Create the `text` Python submodule.
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let sub = PyModule::new(py, "text")?;
    sub.add_class::<example::TextEcho>()?;
    Ok(sub)
}

/// Collect all text nodes as Python types.
pub fn node_register(py: Python<'_>) -> PyResult<Vec<Py<PyType>>> {
    Ok(vec![py.get_type::<TextEcho>().into()])
}
