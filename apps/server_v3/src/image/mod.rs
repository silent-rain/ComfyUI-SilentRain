/// Image processing nodes for SilentRain v3.
use pyo3::{prelude::*, types::PyType};

mod invert_image;
pub use invert_image::InvertImage;

#[pymodule]
pub mod image {
    pub use super::*;

    #[pymodule_export]
    pub use InvertImage;
}

/// Collect all image nodes as Python types.
pub fn node_register(py: Python<'_>) -> PyResult<Vec<Py<PyType>>> {
    Ok(vec![py.get_type::<InvertImage>().into()])
}
