/// Image processing nodes for SilentRain v3.
use pyo3::{
    prelude::*,
    types::{PyModule, PyModuleMethods, PyType},
};

pub mod example;
pub use example::InvertImage;

/// Create the `image` Python submodule.
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let sub = PyModule::new(py, "image")?;
    sub.add_class::<example::InvertImage>()?;
    Ok(sub)
}

/// Collect all image nodes as Python types.
pub fn node_register(py: Python<'_>) -> PyResult<Vec<Py<PyType>>> {
    Ok(vec![py.get_type::<InvertImage>().into()])
}
