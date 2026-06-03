//! Schema definitions for ComfyUI v3 nodes.
//!
//! Provides type-safe builders for `io.Schema`, `io.NodeOutput`, and all
//! input / output descriptors.
pub mod hidden;
pub mod input;
pub mod node_schema;
pub mod output;
pub mod price_badge;

pub use hidden::Hidden;
pub use input::Input;
pub use node_schema::NodeSchema;
pub use output::{NodeOutput, Output, OutputType};
pub use price_badge::PriceBadge;
