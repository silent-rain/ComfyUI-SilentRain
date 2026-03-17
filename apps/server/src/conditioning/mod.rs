//! Conditioning

use pyo3::{
    Bound, PyResult, Python,
    types::{PyModule, PyModuleMethods},
};

use crate::core::node::NodeRegister;

mod flux_kontext_inpainting_conditioning;
pub use flux_kontext_inpainting_conditioning::FluxKontextInpaintingConditioning;

mod conditioning_console_debug;
pub use conditioning_console_debug::ConditioningConsoleDebug;

mod reference_latent_combo;
pub use reference_latent_combo::ReferenceLatentCombo;

mod flux2_reference_latent_combo;
pub use flux2_reference_latent_combo::Flux2ReferenceLatentCombo;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "conditioning")?;
    submodule.add_class::<FluxKontextInpaintingConditioning>()?;
    submodule.add_class::<ConditioningConsoleDebug>()?;
    submodule.add_class::<ReferenceLatentCombo>()?;
    submodule.add_class::<Flux2ReferenceLatentCombo>()?;
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
        NodeRegister(
            "ReferenceLatentCombo",
            py.get_type::<ReferenceLatentCombo>(),
            "Sr Reference Latent Combo",
        ),
        NodeRegister(
            "Flux2ReferenceLatentCombo",
            py.get_type::<Flux2ReferenceLatentCombo>(),
            "Sr Flux2 Reference Latent Combo",
        ),
    ];
    Ok(nodes)
}
