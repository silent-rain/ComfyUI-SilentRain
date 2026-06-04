//! Core infrastructure for `server_v3` nodes.
//!
//! Re-exports from `comfyui_v3` that every node implementation needs,
//! plus any crate-local base traits or helpers.

pub use comfyui_v3::node::{ComfyNode, ExtensionBuilder, PromptServer};
pub use comfyui_v3::schema::{
    NodeOutput, NodeSchema,
    hidden::Hidden,
    input::{BoolInput, ComboInput, FloatInput, ImageInput, IntInput, StringInput},
    output::Output,
};
