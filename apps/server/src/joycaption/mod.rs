//! JoyCaption

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

use crate::core::node::NodeRegister;

mod joy_caption_extra_options;
pub use joy_caption_extra_options::JoyCaptionExtraOptions;
mod joy_caption_ollama_prompter;
pub use joy_caption_ollama_prompter::JoyCaptionOllamaPrompter;

mod joycaption_predictor;
pub use joycaption_predictor::JoyCaptionPredictorGGUF;

mod joycaption_beta_one_gguf;
pub use joycaption_beta_one_gguf::JoyCaptionBetaOneGGUF;

mod joycaption_beta_one_custom_gguf;
pub use joycaption_beta_one_custom_gguf::JoyCaptionBetaOneCustomGGUF;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "joycaption")?;
    submodule.add_class::<JoyCaptionExtraOptions>()?;
    submodule.add_class::<JoyCaptionOllamaPrompter>()?;
    submodule.add_class::<JoyCaptionBetaOneGGUF>()?;
    submodule.add_class::<JoyCaptionBetaOneCustomGGUF>()?;
    Ok(submodule)
}

/// JoyCaption node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        NodeRegister(
            "JoyCaptionExtraOptions",
            py.get_type::<JoyCaptionExtraOptions>(),
            "Sr JoyCaption Extra Options",
        ),
        NodeRegister(
            "JoyCaptionOllamaPrompter",
            py.get_type::<JoyCaptionOllamaPrompter>(),
            "Sr JoyCaption Ollama Prompter",
        ),
        NodeRegister(
            "JoyCaptionBetaOneGGUF",
            py.get_type::<JoyCaptionBetaOneGGUF>(),
            "Sr JoyCaption Beta One GGUF",
        ),
        NodeRegister(
            "JoyCaptionBetaOneCustomGGUF",
            py.get_type::<JoyCaptionBetaOneCustomGGUF>(),
            "Sr JoyCaption Beta One Custom GGUF",
        ),
    ];
    Ok(nodes)
}
