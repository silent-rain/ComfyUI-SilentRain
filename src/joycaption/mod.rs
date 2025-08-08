//! JoyCaption
//!
//!

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

mod joy_caption_extra_options;
pub use joy_caption_extra_options::JoyCaptionExtraOptions;
mod joy_caption_ollama_prompter;
pub use joy_caption_ollama_prompter::JoyCaptionOllamaPrompter;

mod joycaption_beta_one_gguf;
pub use joycaption_beta_one_gguf::JoyCaptionnBetaOneGGUF;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "joycaption")?;
    submodule.add_class::<JoyCaptionExtraOptions>()?;
    submodule.add_class::<JoyCaptionOllamaPrompter>()?;
    submodule.add_class::<JoyCaptionnBetaOneGGUF>()?;
    Ok(submodule)
}
