//! 错误类型定义

use image::ImageError;
use llama_cpp_2::{
    ApplyChatTemplateError, ChatTemplateError, DecodeError, LlamaContextLoadError, LlamaCppError,
    LlamaModelLoadError, NewLlamaChatMessageError, StringToTokenError, TokenToStringError,
    llama_batch::BatchAddError,
    mtmd::{MtmdBitmapError, MtmdEvalError, MtmdInitError, MtmdTokenizeError},
};
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// 推理引擎错误类型
#[derive(Debug, Error)]
pub enum Error {
    // ==================== 通用错误 ====================
    #[error("Unknown error")]
    Unknown,

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Lock error: {0}")]
    LockError(String),

    #[error("Stream error: {0}")]
    Stream(String),

    // ==================== 后端错误 ====================
    #[error(transparent)]
    LlamaCppError(#[from] LlamaCppError),
    #[error(transparent)]
    LlamaContextLoadError(#[from] LlamaContextLoadError),
    #[error(transparent)]
    LlamaModelLoadError(#[from] LlamaModelLoadError),
    #[error(transparent)]
    ApplyChatTemplateError(#[from] ApplyChatTemplateError),
    #[error(transparent)]
    StringToTokenError(#[from] StringToTokenError),
    #[error(transparent)]
    DecodeError(#[from] DecodeError),
    #[error(transparent)]
    NewLlamaChatMessageError(#[from] NewLlamaChatMessageError),
    #[error(transparent)]
    BatchAddError(#[from] BatchAddError),
    #[error(transparent)]
    TokenToStringError(#[from] TokenToStringError),
    #[error(transparent)]
    ChatTemplateError(#[from] ChatTemplateError),
    #[error(transparent)]
    MtmdInitError(#[from] MtmdInitError),
    #[error(transparent)]
    MtmdEvalError(#[from] MtmdEvalError),
    #[error(transparent)]
    MtmdTokenizeError(#[from] MtmdTokenizeError),
    #[error(transparent)]
    MtmdBitmapError(#[from] MtmdBitmapError),
    #[error(transparent)]
    OpenAIError(#[from] async_openai::error::OpenAIError),

    // #[error(transparent)]
    // NulError(#[from] std::ffi::NulError),

    // ==================== 模型错误 ====================
    #[error("Model load failed: {path}")]
    ModelLoad { path: String },

    #[error("Model not found: {path}")]
    ModelNotFound { path: String },

    #[error("Invalid model type for operation")]
    InvalidModelType,

    #[error("Model already loaded: {path}")]
    ModelAlreadyLoaded { path: String },

    // ==================== 推理错误 ====================
    #[error("Generation failed: {reason}")]
    Generation { reason: String },

    #[error("Generation timeout after {duration_ms}ms")]
    GenerationTimeout { duration_ms: u64 },

    #[error("Context limit exceeded")]
    ContextLimitExceeded,

    // ==================== 输入错误 ====================
    #[error("Invalid input for {field}: {message}")]
    InvalidInput { field: String, message: String },

    #[error("Missing required input: {field}")]
    MissingInput { field: String },

    #[error("Unsupported media type: {media_type}")]
    UnsupportedMedia { media_type: String },

    #[error("MMProj path required for multimodal input")]
    MissingMmprojPath,

    #[error("Image buffer error")]
    ImageBuffer,

    #[error(transparent)]
    ImageError(#[from] ImageError),
    #[error(transparent)]
    Base64decodeError(#[from] base64::DecodeError),

    // ==================== 配置错误 ====================
    #[error("Invalid configuration: {message}")]
    Config { message: String },

    #[error("Invalid configuration value for {field}")]
    ConfigInvalid { field: String },

    // ==================== 缓存错误 ====================
    #[error("Cache error: {message}")]
    Cache { message: String },

    #[error("Cache entry not found: {key}")]
    CacheNotFound { key: String },

    #[error("Cache limit exceeded")]
    CacheLimitExceeded,

    #[error("Cache not initialized: {0}")]
    CacheNotInitialized(String),

    // ==================== IO和序列化错误 ====================
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    // ==================== 网络错误 ====================
    #[error("HTTP request failed: {0}")]
    HttpRequest(String),

    #[error("Failed to download image from URL: {url}, err: {message}")]
    ImageDownload { url: String, message: String },

    // ==================== 外部错误 ====================
    #[error("Unsupported feature: {feature}")]
    UnsupportedFeature { feature: String },
    #[error("Failed to create tokio runtime: {0}")]
    RuntimeError(String),
    #[error("Failed to obtain semaphore license, {0}")]
    AcquireError(String),
    #[error("task join error, {0}")]
    TaskJoinError(String),
}
