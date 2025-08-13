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

/// 节点注册
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let mut nodes: Vec<NodeRegister> = Vec::new();
    nodes.extend(utils_register(py)?);
    nodes.extend(text_register(py)?);
    nodes.extend(list_register(py)?);
    nodes.extend(logic_register(py)?);
    nodes.extend(image_register(py)?);
    nodes.extend(conditioning_register(py)?);
    nodes.extend(joy_caption_register(py)?);
    Ok(nodes)
}

/// Utils node register
fn utils_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        //
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
    ];
    Ok(nodes)
}

/// Text node register
fn text_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
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
    ];
    Ok(nodes)
}

/// List node register
fn list_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
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
    ];
    Ok(nodes)
}

/// Logic node register
fn logic_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![NodeRegister(
        "BatchFloat",
        py.get_type::<BatchFloat>(),
        "Sr Batch Float",
    )];
    Ok(nodes)
}

/// Image node register
fn image_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
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
    ];
    Ok(nodes)
}

/// Conditioning node register
fn conditioning_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        NodeRegister(
            "FluxKontextInpaintingConditioning",
            py.get_type::<FluxKontextInpaintingConditioning>(),
            "Sr Flux Kontext Inpainting Conditioning",
        ),
        NodeRegister(
            "ConditioningConsoleDebug",
            py.get_type::<ConditioningConsoleDebug>(),
            "Sr Conditioning Console Debug",
        ),
    ];
    Ok(nodes)
}

/// JoyCaption node register
fn joy_caption_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
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
