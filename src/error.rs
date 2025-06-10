//! 错误处理
use std::io;

#[allow(unused)]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error, {0}")]
    Io(io::Error),
    #[error("invalid directory, {0}")]
    InvalidDirectory(String),
    #[error("encode error, {0}")]
    Encode(String),
    #[error("decode error, {0}")]
    Decode(String),
    #[error("type not supported")]
    TypeNotSupported,
    #[error("type conversion failed, {0}")]
    TypeConversion(String),
    #[error("type downcast failed, {0}")]
    DowncastFailed(String),

    #[error("the input list is empty")]
    InputListEmpty,
    #[error("index out of range, {0}")]
    IndexOutOfRange(String),
    #[error("error in obtaining list items at specified index")]
    GetListIndex,

    #[error("py error, {0}")]
    PyErr(#[from] pyo3::PyErr),
    #[error("pythonize error, {0}")]
    PythonizeError(#[from] pythonize::PythonizeError),
    #[error("py missing kwargs, {0}")]
    PyMissingKwargs(String),

    #[error("tensor error, {0}")]
    TensorErr(#[from] candle_core::Error),
    #[error("numpy error, {0}")]
    NotContiguousError(#[from] numpy::NotContiguousError),
    #[error("strum error, {0}")]
    ParseEnumString(String),

    #[error("creating image buffer error")]
    ImageBuffer,
    #[error("image error, {0}")]
    ImageError(#[from] image::ImageError),
    #[error("file path not exist, {0}")]
    FilePathNotExist(String),
    #[error("invalid parameter, {0}")]
    InvalidParameter(String),
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}
