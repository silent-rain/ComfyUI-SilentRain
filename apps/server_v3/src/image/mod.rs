/// Image processing nodes for SilentRain v3.
use pyo3::{prelude::*, types::PyType};

pub mod example;
pub mod example_node_macros;

pub use example::InvertImage;
pub use example_node_macros::ExampleNodeMacros;

#[pymodule]
pub mod image {
    pub use super::*;

    #[pymodule_export]
    pub use InvertImage;

    #[pymodule_export]
    pub use ExampleNodeMacros;
}

/// Collect all image nodes as Python types.
pub fn node_register(py: Python<'_>) -> PyResult<Vec<Py<PyType>>> {
    Ok(vec![
        py.get_type::<InvertImage>().into(),
        py.get_type::<ExampleNodeMacros>().into(),
    ])
}
