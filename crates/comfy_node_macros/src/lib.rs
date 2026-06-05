//! Procedural macro crate for ComfyUI v3 node development.
use proc_macro::TokenStream;

mod comfy_node;

/// Attribute macro for ComfyUI v3 node structs.
///
/// This macro:
/// 1. Adds `#[pyclass]` attribute to the original struct
/// 2. Generates a `#[pymethods]` impl block with Python-exposed methods that
///    call into the `ComfyNode` trait methods
///
/// # Example
///
/// ```ignore
/// #[comfy_node]
/// pub struct MyNode;
///
/// impl ComfyNode for MyNode {
///     fn define_schema() -> Result<NodeSchema> { ... }
///     fn execute(py, args, kwargs) -> Result<NodeOutput> { ... }
/// }
/// ```
#[proc_macro_attribute]
pub fn comfy_node(args: TokenStream, input: TokenStream) -> TokenStream {
    comfy_node::comfy_node_impl(args, input)
}
