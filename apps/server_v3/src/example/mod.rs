/// Image processing nodes for SilentRain v3.
use pyo3::{prelude::*, types::PyType};

mod example_node_macros;
pub use example_node_macros::ExampleNodeMacros;

mod example_pytype;
pub use example_pytype::ExamplePytype;

#[pymodule]
pub mod image {
    pub use super::*;

    #[pymodule_export]
    pub use ExamplePytype;

    #[pymodule_export]
    pub use ExampleNodeMacros;
}

/// Collect all image nodes as Python types.
pub fn node_register(py: Python<'_>) -> PyResult<Vec<Py<PyType>>> {
    Ok(vec![
        py.get_type::<ExamplePytype>().into(),
        py.get_type::<ExampleNodeMacros>().into(),
    ])
}
