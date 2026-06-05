//! ComfyUI v3 Extension Crate
//!
//! Provides type-safe, Rust-native node development for ComfyUI v3.

use pyo3::prelude::*;

pub mod core;
pub mod error;
pub mod node;
pub mod prelude;
pub mod schema;
pub mod types;
pub mod utils;

use crate::schema::Output;

// --------------------------------------------------------------------------
// pyo3 module
// --------------------------------------------------------------------------

/// Python module initialisation.
///
/// When this crate is compiled as a `cdylib` and imported from Python, this
/// function registers all our Rust-backed types in the `comfyui_v3` namespace.
#[pymodule]
#[pyo3(name = "comfyui_v3")]
fn init_comfyui_v3(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize tracing
    let _ = tracing_subscriber::fmt()
        .with_ansi(true)
        .with_max_level(tracing::Level::DEBUG)
        .with_level(true)
        .with_file(true)
        .with_line_number(true)
        .with_target(false)
        .try_init();

    // Schema types (only pyclass-backed types can be registered)
    m.add_class::<Output>()?;

    Ok(())
}
