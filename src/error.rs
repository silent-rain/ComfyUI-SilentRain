//! 错误处理

#[allow(unused)]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    // 标准库错误处理
    #[error("io error, {0}")]
    Io(std::io::Error),
    #[error("parse int error, {0}")]
    ParseIntError(#[from] std::num::ParseIntError),
    #[error("system time error, {0}")]
    SystemTimeError(#[from] std::time::SystemTimeError),
    // std::sync::poison::rwlock
    #[error("lock error, {0}")]
    LockError(String),
    // std::sync::once_lock::OnceLock
    #[error("once lock error, {0}")]
    OnceLock(String),
    #[error("option none, {0}")]
    OptionNone(String),

    #[error("serde json error, {0}")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("regex error, {0}")]
    RegexError(#[from] regex::Error),

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

    #[error("the list is empty")]
    ListEmpty,
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
    #[error("py downcast error, {0}")]
    PyDowncastError(String),

    #[error("tensor error, {0}")]
    TensorErr(#[from] candle_core::Error),
    #[error("invalid tensor shape, {0}")]
    InvalidTensorShape(String),
    #[error("numpy error, {0}")]
    NotContiguousError(#[from] numpy::NotContiguousError),
    #[error("strum error, {0}")]
    ParseEnumString(String),

    #[error("image error, {0}")]
    ImageError(#[from] image::ImageError),
    #[error("creating image buffer error")]
    ImageBuffer,
    #[error("unsupported number of channels: {0}")]
    UnsupportedNumberOfChannels(u32),
    #[error("png encoding error, {0}")]
    PngEncodingError(#[from] png::EncodingError),

    #[error("invalid directory, {0}")]
    InvalidDirectory(String),
    #[error("invalid parameter, {0}")]
    InvalidParameter(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
