use pyo3::PyErr;
use thiserror::Error;

// Re-export for convenience
pub use pyo3;

/// Convenience type alias for Result with our Error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in the comfyui_v3 crate.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Schema error: {0}")]
    Schema(String),

    #[error("Node execution error: {0}")]
    Execution(String),

    #[error("Input validation error: {0}")]
    Validation(String),

    #[error("Python interop error: {0}")]
    PythonInterop(String),

    #[error("Extension error: {0}")]
    Extension(String),

    #[error(transparent)]
    PyO3(#[from] pyo3::PyErr),

    /// 透传 anyhow 错误链（用于应用层上下文包装）
    #[error(transparent)]
    Anyhow(#[from] anyhow::Error),
}

/// Allow automatic conversion from our Error to PyErr.
/// This enables using ? operator in functions that return PyResult.
impl From<Error> for PyErr {
    fn from(err: Error) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}
