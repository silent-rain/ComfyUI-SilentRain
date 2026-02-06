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
    #[error("ffi error, {0}")]
    FfiNulError(#[from] std::ffi::NulError),
    #[error("num error, {0}")]
    TryFromIntError(#[from] std::num::TryFromIntError),
    // std::sync::poison::rwlock
    #[error("lock error, {0}")]
    LockError(String),
    // std::sync::once_lock::OnceLock
    #[error("once lock error, {0}")]
    OnceLock(String),
    #[error("option none, {0}")]
    OptionNone(String),
    #[error("strum error, {0}")]
    ParseEnumString(String),

    #[error("serde json error, {0}")]
    SerdeJsonError(#[from] serde_json::Error),
    #[error("regex error, {0}")]
    RegexError(#[from] regex::Error),
    #[error("llama_cpp_2 error, {0}")]
    LlamaCppError(#[from] llama_cpp_2::LlamaCppError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaCppModelLoadError(#[from] llama_cpp_2::LlamaModelLoadError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaCppContextLoadError(#[from] llama_cpp_2::LlamaContextLoadError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaCppStringToTokenError(#[from] llama_cpp_2::StringToTokenError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaCppBatchAddError(#[from] llama_cpp_2::llama_batch::BatchAddError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaCppDecodeError(#[from] llama_cpp_2::DecodeError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaCppTokenToStringError(#[from] llama_cpp_2::TokenToStringError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaCppNewLlamaChatMessageError(#[from] llama_cpp_2::NewLlamaChatMessageError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaCppChatTemplateError(#[from] llama_cpp_2::ChatTemplateError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaCppApplyChatTemplateError(#[from] llama_cpp_2::ApplyChatTemplateError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaMtmdInitError(#[from] llama_cpp_2::mtmd::MtmdInitError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaMtmdTokenizeError(#[from] llama_cpp_2::mtmd::MtmdTokenizeError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaMtmdEvalError(#[from] llama_cpp_2::mtmd::MtmdEvalError),
    #[error("llama_cpp_2 error, {0}")]
    LlamaMtmdBitmapError(#[from] llama_cpp_2::mtmd::MtmdBitmapError),

    #[error(transparent)]
    LlamaCppCoreError(#[from] llama_cpp_core::error::Error),

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
    #[error("no match found")]
    NoMatchFound,

    #[error("py error, {0}")]
    PyErr(#[from] pyo3::PyErr),
    #[error("pythonize error, {0}")]
    PythonizeError(#[from] pythonize::PythonizeError),
    #[error("py missing kwargs, {0}")]
    PyMissingKwargs(String),
    #[error("py downcast error, {0}")]
    PyDowncastError(String),
    #[error("py cast error, {0}")]
    Pyo3CastError(String),

    #[error("tensor error, {0}")]
    TensorErr(#[from] candle_core::Error),
    #[error("invalid tensor shape, {0}")]
    InvalidTensorShape(String),
    #[error("numpy error, {0}")]
    NotContiguousError(#[from] numpy::NotContiguousError),

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

    #[error("file not found, {0}")]
    FileNotFound(String),
    #[error("invalid path, {0}")]
    InvalidPath(String),
    #[error("invalid input, {0}")]
    InvalidInput(String),

    #[error("Model is not initialized, {0}")]
    ModelNotInitialized(String),
    #[error("Model mtmd context is not initialized, {0}")]
    ModelMtmdContextNotInitialized(String),

    #[error("Failed to obtain semaphore license, {0}")]
    AcquireError(String),
    #[error("task join error, {0}")]
    TaskJoinError(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
