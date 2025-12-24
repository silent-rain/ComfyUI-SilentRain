//! 文本相关的节点
use pyo3::{
    Bound, PyResult, Python,
    types::{PyModule, PyModuleMethods},
};

use crate::core::node::NodeRegister;

mod string_constant;
pub use string_constant::StringConstant;

mod text_box;
pub use text_box::TextBox;

mod text_to_list;
pub use text_to_list::TextToList;

mod string_list_to_sting;
pub use string_list_to_sting::StringListToSting;

mod string_list;
pub use string_list::StringList;

mod string_dyn_list;
pub use string_dyn_list::StringDynList;

mod string_dyn_list2;
pub use string_dyn_list2::StringDynList2;

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

mod remove_empty_lines;
pub use remove_empty_lines::RemoveEmptyLines;

/// 文本模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "text")?;
    submodule.add_class::<StringConstant>()?;
    submodule.add_class::<TextBox>()?;
    submodule.add_class::<TextToList>()?;
    submodule.add_class::<StringListToSting>()?;
    submodule.add_class::<StringList>()?;
    submodule.add_class::<StringDynList>()?;
    submodule.add_class::<StringDynList2>()?;
    submodule.add_class::<SaveText>()?;
    submodule.add_class::<RegularString>()?;
    submodule.add_class::<LoadKontextPresets>()?;
    submodule.add_class::<LoadKontextPresetsAssistant>()?;
    submodule.add_class::<LoadKontextBflPresetsAssistant>()?;
    submodule.add_class::<LoadWanPresets>()?;
    submodule.add_class::<Wan22OfficialPromptSelector>()?;
    submodule.add_class::<RemoveEmptyLines>()?;
    Ok(submodule)
}

/// Text node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        NodeRegister(
            "StringConstant",
            py.get_type::<StringConstant>(),
            "Sr String Constant",
        ),
        NodeRegister("TextBox", py.get_type::<TextBox>(), "Sr Text Box"),
        NodeRegister("TextToList", py.get_type::<TextToList>(), "Sr Text To List"),
        NodeRegister(
            "StringListToSting",
            py.get_type::<StringListToSting>(),
            "Sr String List To Sting",
        ),
        NodeRegister("StringList", py.get_type::<StringList>(), "Sr String List"),
        NodeRegister(
            "StringDynList",
            py.get_type::<StringDynList>(),
            "Sr String Dyn List",
        ),
        NodeRegister(
            "StringDynList2",
            py.get_type::<StringDynList2>(),
            "Sr String Dyn List 2",
        ),
        NodeRegister("SaveText", py.get_type::<SaveText>(), "Sr Save Text"),
        NodeRegister(
            "RegularString",
            py.get_type::<RegularString>(),
            "Sr Regular String",
        ),
        NodeRegister(
            "LoadKontextPresets",
            py.get_type::<LoadKontextPresets>(),
            "Sr Load Kontext Presets",
        ),
        NodeRegister(
            "LoadKontextPresetsAssistant",
            py.get_type::<LoadKontextPresetsAssistant>(),
            "Sr Load Kontext Presets Assistant",
        ),
        NodeRegister(
            "LoadKontextBflPresetsAssistant",
            py.get_type::<LoadKontextBflPresetsAssistant>(),
            "Sr Load Kontext Bfl Presets Assistant",
        ),
        NodeRegister(
            "LoadWanPresets",
            py.get_type::<LoadWanPresets>(),
            "Sr Load Wan Presets",
        ),
        NodeRegister(
            "Wan22OfficialPromptSelector",
            py.get_type::<Wan22OfficialPromptSelector>(),
            "Sr Wan2.2 Official Prompt Selector",
        ),
        NodeRegister(
            "RemoveEmptyLines",
            py.get_type::<RemoveEmptyLines>(),
            "Sr Remove Empty Lines",
        ),
    ];
    Ok(nodes)
}
