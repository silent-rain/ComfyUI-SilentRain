use pyo3::{prelude::*, types::PyString};

#[pyclass(name = "AlwaysEqualProxy2")]
pub struct AlwaysEqualProxy2 {
    inner: Py<PyString>,
}

#[pymethods]
impl AlwaysEqualProxy2 {
    #[new]
    pub fn new(py: Python<'_>, s: &str) -> PyResult<Self> {
        Ok(Self {
            inner: PyString::new(py, s).into(),
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __eq__(&self, _other: &Bound<'_, PyAny>) -> bool {
        true
    }

    fn __ne__(&self, _other: &Bound<'_, PyAny>) -> bool {
        false
    }
}
