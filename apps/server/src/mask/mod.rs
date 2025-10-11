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

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "math")?;
    submodule.add_class::<MaskSplitGrid>()?;
    submodule.add_class::<MaskGridComposite>()?;
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
    ];
    Ok(nodes)
}
