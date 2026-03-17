//! 遮罩

use pyo3::{
    Bound, PyResult, Python,
    types::{PyModule, PyModuleMethods},
};

use crate::core::node::NodeRegister;

mod mask_split_grid;
pub use mask_split_grid::MaskSplitGrid;

mod mask_grid_composite;
pub use mask_grid_composite::MaskGridComposite;

mod resize_mask_by_longer_edge;
pub use resize_mask_by_longer_edge::ResizeMaskByLongerEdge;

mod resize_mask_by_shorter_edge;
pub use resize_mask_by_shorter_edge::ResizeMaskByShorterEdge;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "math")?;
    submodule.add_class::<MaskSplitGrid>()?;
    submodule.add_class::<MaskGridComposite>()?;
    submodule.add_class::<ResizeMaskByLongerEdge>()?;
    submodule.add_class::<ResizeMaskByShorterEdge>()?;
    Ok(submodule)
}

/// mask node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        NodeRegister(
            "MaskSplitGrid",
            py.get_type::<MaskSplitGrid>(),
            "Sr Mask Split Grid",
        ),
        NodeRegister(
            "MaskGridComposite",
            py.get_type::<MaskGridComposite>(),
            "Sr Mask Grid Composite",
        ),
        NodeRegister(
            "ResizeMaskByLongerEdge",
            py.get_type::<ResizeMaskByLongerEdge>(),
            "Sr Resize Mask By Longer Edge",
        ),
        NodeRegister(
            "ResizeMaskByShorterEdge",
            py.get_type::<ResizeMaskByShorterEdge>(),
            "Sr Resize Mask By Shorter Edge",
        ),
    ];
    Ok(nodes)
}
