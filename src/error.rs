//! 错误处理
use std::io;

#[allow(unused)]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error, {0}")]
    Io(io::Error),
    #[error("invalid directory, {0}")]
    InvalidDirectory(String),
    #[error("encoding error, {0}")]
    EncodingError(String),
    #[error("decode error, {0}")]
    DecodeError(String),
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}
