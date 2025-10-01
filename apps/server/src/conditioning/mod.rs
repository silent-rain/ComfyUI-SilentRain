//! Conditioning

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

use crate::core::node::NodeRegister;

mod flux_kontext_inpainting_conditioning;
pub use flux_kontext_inpainting_conditioning::FluxKontextInpaintingConditioning;

mod conditioning_console_debug;
pub use conditioning_console_debug::ConditioningConsoleDebug;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "conditioning")?;
    submodule.add_class::<FluxKontextInpaintingConditioning>()?;
    submodule.add_class::<ConditioningConsoleDebug>()?;
    Ok(submodule)
}

/// Conditioning node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
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
