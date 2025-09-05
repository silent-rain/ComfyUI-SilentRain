//! 列表节点
use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

use crate::core::node::NodeRegister;

mod index_from_any_list;
pub use index_from_any_list::IndexFromAnyList;

mod list_count;
pub use list_count::ListCount;

mod shuffle_any_list;
pub use shuffle_any_list::ShuffleAnyList;

mod random_any_list;
pub use random_any_list::RandomAnyList;

mod list_bridge;
pub use list_bridge::ListBridge;

mod list_to_batch;
pub use list_to_batch::ListToBatch;

mod batch_to_list;
pub use batch_to_list::BatchToList;

mod merge_multi_list;
pub use merge_multi_list::MergeMultiList;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "list")?;
    submodule.add_class::<IndexFromAnyList>()?;
    submodule.add_class::<ListCount>()?;
    submodule.add_class::<ShuffleAnyList>()?;
    submodule.add_class::<RandomAnyList>()?;
    submodule.add_class::<ListBridge>()?;
    submodule.add_class::<ListToBatch>()?;
    submodule.add_class::<BatchToList>()?;
    submodule.add_class::<MergeMultiList>()?;
    Ok(submodule)
}

/// List node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
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
