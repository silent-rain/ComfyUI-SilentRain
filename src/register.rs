//! 节点注册

use pyo3::{PyResult, Python};

use crate::{
    conditioning::{ConditioningConsoleDebug, FluxKontextInpaintingConditioning},
    core::node::NodeRegister,
    image::{
        ImageResolution, ImageResolution2, ImageSimpleResolution, LoadImagesFromFolder,
        SaveImageText, SaveImages,
    },
    joycaption::{
        JoyCaptionExtraOptions, JoyCaptionOllamaPrompter, JoyCaptionnBetaOneCustomGGUF,
        JoyCaptionnBetaOneGGUF,
    },
    list::{
        BatchToList, IndexFromAnyList, ListBridge, ListCount, ListToBatch, MergeMultiList,
        RandomAnyList, ShuffleAnyList,
    },
    logic::BatchFloat,
    text::{
        LoadKontextBflPresetsAssistant, LoadKontextPresets, LoadKontextPresetsAssistant,
        LoadWanPresets, RegularString, SaveText, StringList, StringListToSting, TextBox,
        TextToList, Wan22OfficialPromptSelector,
    },
    utils::{BatchRename, BridgeAnything, ConsoleDebug, FileScanner, WorkflowInfo},
};

pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        // utils
        NodeRegister(
            "FileScanner",
            py.get_type::<FileScanner>(),
            "Sr File Scanner",
        ),
        NodeRegister(
            "BridgeAnything",
            py.get_type::<BridgeAnything>(),
            "Sr Bridge Anything",
        ),
        NodeRegister(
            "WorkflowInfo",
            py.get_type::<WorkflowInfo>(),
            "Sr Workflow Info",
        ),
        NodeRegister(
            "BatchRename",
            py.get_type::<BatchRename>(),
            "Sr Batch Rename",
        ),
        NodeRegister(
            "ConsoleDebug",
            py.get_type::<ConsoleDebug>(),
            "Sr Console Debug",
        ),
        // text
        NodeRegister("TextBox", py.get_type::<TextBox>(), "Sr Text Box"),
        NodeRegister("TextToList", py.get_type::<TextToList>(), "Sr Text To List"),
        NodeRegister(
            "StringListToSting",
            py.get_type::<StringListToSting>(),
            "Sr String List To Sting",
        ),
        NodeRegister("StringList", py.get_type::<StringList>(), "Sr String List"),
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
        // list
        NodeRegister(
            "IndexFromAnyList",
            py.get_type::<IndexFromAnyList>(),
            "Sr Index From Any List",
        ),
        NodeRegister("ListCount", py.get_type::<ListCount>(), "Sr List Count"),
        NodeRegister(
            "ShuffleAnyList",
            py.get_type::<ShuffleAnyList>(),
            "Sr Shuffle Any List",
        ),
        NodeRegister(
            "RandomAnyList",
            py.get_type::<RandomAnyList>(),
            "Sr Random Any List",
        ),
        NodeRegister("ListBridge", py.get_type::<ListBridge>(), "Sr List Bridge"),
        NodeRegister(
            "ListToBatch",
            py.get_type::<ListToBatch>(),
            "Sr List To Batch",
        ),
        NodeRegister(
            "BatchToList",
            py.get_type::<BatchToList>(),
            "Sr Batch To List",
        ),
        NodeRegister(
            "MergeMultiList",
            py.get_type::<MergeMultiList>(),
            "Sr Merge Multi List",
        ),
        // logic
        NodeRegister("BatchFloat", py.get_type::<BatchFloat>(), "Sr Batch Float"),
        // image
        NodeRegister(
            "ImageSimpleResolution",
            py.get_type::<ImageSimpleResolution>(),
            "Sr Image Simple Resolution",
        ),
        NodeRegister(
            "ImageResolution",
            py.get_type::<ImageResolution>(),
            "Sr Image Resolution",
        ),
        NodeRegister(
            "ImageResolution2",
            py.get_type::<ImageResolution2>(),
            "Sr Image Resolution2",
        ),
        NodeRegister(
            "LoadImagesFromFolder",
            py.get_type::<LoadImagesFromFolder>(),
            "Sr Load Images From Folder",
        ),
        NodeRegister("SaveImages", py.get_type::<SaveImages>(), "Sr Save Images"),
        NodeRegister(
            "SaveImageText",
            py.get_type::<SaveImageText>(),
            "Sr Save Image Text",
        ),
        // conditioning
        NodeRegister(
            "FluxKontextInpaintingConditioning",
            py.get_type::<FluxKontextInpaintingConditioning>(),
            "Sr Flux Kontext Inpainting Conditioning",
        ),
        // Conditioning
        NodeRegister(
            "ConditioningConsoleDebug",
            py.get_type::<ConditioningConsoleDebug>(),
            "Sr Conditioning Console Debug",
        ),
        // Model
        // JoyCaption
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
            "JoyCaptionnBetaOneGGUF",
            py.get_type::<JoyCaptionnBetaOneGGUF>(),
            "Sr JoyCaptionn Beta One GGUF",
        ),
        NodeRegister(
            "JoyCaptionnBetaOneCustomGGUF",
            py.get_type::<JoyCaptionnBetaOneCustomGGUF>(),
            "Sr JoyCaptionn Beta One Custom GGUF",
        ),
    ];
    Ok(nodes)
}
