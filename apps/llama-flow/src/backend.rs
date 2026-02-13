//! Backend implementation for the LLaMA language model

use llama_cpp_2::{LlamaCppError, llama_backend::LlamaBackend};
use tracing::error;

use crate::error::Error;

pub struct Backend {}

impl Backend {
    /// Initialize backend
    pub fn init_backend() -> Result<LlamaBackend, Error> {
        let backend = match LlamaBackend::init() {
            Ok(backend) => backend,
            Err(LlamaCppError::BackendAlreadyInitialized) => LlamaBackend {},
            Err(e) => {
                error!("Failed to initialize backend: {}", e);
                return Err(Error::LlamaCppError(e));
            }
        };
        Ok(backend)
    }
}
