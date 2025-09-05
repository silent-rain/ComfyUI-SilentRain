//! 列表节点

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

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "list")?;
    submodule.add_class::<IndexFromAnyList>()?;
    Ok(submodule)
}
