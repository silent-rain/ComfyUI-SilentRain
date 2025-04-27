//! 节点注册

use pyo3::{PyResult, Python};

use crate::{
    core::node::NodeRegister,
    list::{IndexAnything, ListCount, RandomAnyList, ShuffleAnyList},
    text::{StringListToSting, TextBox, TextToList},
    utils::FileScanner,
};

pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        // utils
        NodeRegister(
            "FileScanner",
            py.get_type::<FileScanner>(),
            "Sr File Scanner",
        ),
        // text
        NodeRegister("TextBox", py.get_type::<TextBox>(), "Sr Text Box"),
        NodeRegister("TextToList", py.get_type::<TextToList>(), "Sr Text To List"),
        NodeRegister(
            "StringListToSting",
            py.get_type::<StringListToSting>(),
            "Sr String List To Sting",
        ),
        // list
        NodeRegister(
            "IndexAnything",
            py.get_type::<IndexAnything>(),
            "Sr Index Anything",
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
        // logic
    ];
    Ok(nodes)
}
