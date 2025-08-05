//! 文本相关的节点

mod text_box;
pub use text_box::TextBox;

mod text_to_list;
pub use text_to_list::TextToList;

mod string_list_to_sting;
pub use string_list_to_sting::StringListToSting;

mod string_list;
pub use string_list::StringList;

mod save_text;
pub use save_text::{SaveText, SaveTextMode};

mod regular_string;
pub use regular_string::RegularString;

mod load_kontext_presets;
pub use load_kontext_presets::LoadKontextPresets;

mod load_kontext_presets_assistant;
pub use load_kontext_presets_assistant::LoadKontextPresetsAssistant;

mod load_kontext_bfl_presets_assistant;
pub use load_kontext_bfl_presets_assistant::LoadKontextBflPresetsAssistant;

mod load_wan_presets;
pub use load_wan_presets::LoadWanPresets;

mod wan22_official_prompt_selector;
pub use wan22_official_prompt_selector::Wan22OfficialPromptSelector;

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

/// 文本模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "text")?;
    submodule.add_class::<TextToList>()?;
    Ok(submodule)
}
