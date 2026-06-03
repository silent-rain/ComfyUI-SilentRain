use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

use crate::schema::{PriceBadge, hidden::Hidden, input::Input, output::Output};

// ---------------------------------------------------------------------------
// NodeSchema
// ---------------------------------------------------------------------------

/// Definition of V3 node properties.
///
/// Used on the **Rust side** to describe a node declaratively.  Internally
/// it holds only native Rust types so it is `Clone`, `Debug`, `Send` … and
/// can be manipulated without holding the GIL.
///
/// Once you are ready to register the node with the Python engine, call
/// [`NodeSchema::into_py_schema`] to produce the corresponding Python
/// `io.Schema` object.
#[pyclass(from_py_object)]
#[derive(Debug, Default, Clone)]
pub struct NodeSchema {
    /// ID of node - should be globally unique.
    /// If this is a custom node, add a prefix or postfix to avoid name clashes.
    pub node_id: String,

    /// Display name of node.
    pub display_name: Option<String>,

    /// The category of the node, as per the "Add Node" menu.
    pub category: String,

    /// Ordered list of visible node inputs.
    pub inputs: Vec<Input>,

    /// Ordered list of node outputs.
    pub outputs: Vec<Output>,

    /// Hidden inputs the engine injects automatically (e.g. PROMPT, UNIQUE_ID).
    pub hidden: Vec<Hidden>,

    /// Node description shown as a tooltip when hovering over the node.
    pub description: String,

    /// Alternative names for search. Useful for synonyms, abbreviations,
    /// or old names after renaming.
    pub search_aliases: Vec<String>,

    /// A flag indicating if this node implements the additional code necessary
    /// to deal with `OUTPUT_IS_LIST` nodes.
    ///
    /// All inputs of ``type`` will become ``list[type]``, regardless of how
    /// many items are passed in. This also affects `check_lazy_status`.
    ///
    /// See: <https://docs.comfy.org/custom-nodes/backend/lists\#list-processing\>
    pub is_input_list: bool,

    /// Flags this node as an output node, causing any inputs it requires
    /// to be executed. If a node is not connected to any output nodes,
    /// that node will not be executed.
    ///
    /// See: <https://docs.comfy.org/custom-nodes/backend/server_overview\#output-node\>
    pub is_output_node: bool,

    /// Flags a node as deprecated, indicating to users that they should
    /// find alternatives to this node.
    pub is_deprecated: bool,

    /// Flags a node as experimental, informing users that it may change
    /// or not work as expected.
    pub is_experimental: bool,

    /// Flags a node as dev-only, hiding it from search/menus unless
    /// dev mode is enabled.
    pub is_dev_only: bool,

    /// Flags a node as an API node.
    /// See: <https://docs.comfy.org/tutorials/api-nodes/overview\>
    pub is_api_node: bool,

    /// Optional client-evaluated pricing badge declaration for this node.
    pub price_badge: Option<PriceBadge>,

    /// Flags a node as not idempotent; when `true`, the node will run and
    /// not reuse the cached outputs when identical inputs are provided on a
    /// different node in the graph.
    pub not_idempotent: bool,

    /// Flags a node as expandable, allowing `NodeOutput` to include
    /// the 'expand' property.
    pub enable_expand: bool,

    /// When `true`, all inputs from the prompt will be passed to the node
    /// as kwargs, even if not defined in the schema.
    pub accept_all_inputs: bool,

    /// Optional category for the Essentials tab. Path-based like the
    /// `category` field (e.g. `"Basic"`, `"Image Tools/Editing"`).
    pub essentials_category: Option<String>,

    /// Flags this node as having intermediate output that should persist
    /// across page refreshes.
    ///
    /// Nodes with this flag behave like output nodes (their UI results are
    /// cached and resent to the frontend) but do NOT automatically get added
    /// to the execution list. They will only execute if they are on the
    /// dependency path of a real output node.
    pub has_intermediate_output: bool,
}

// ---------------------------------------------------------------------------
// NodeOutput — standardized output of a node execution
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Builder methods
// ---------------------------------------------------------------------------

impl NodeSchema {
    /// Create a new schema with the given node id.
    ///
    /// `category` defaults to `"sd"`.
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            category: "sd".to_string(),
            ..Default::default()
        }
    }

    pub fn with_display_name(mut self, display_name: impl Into<String>) -> Self {
        self.display_name = Some(display_name.into());
        self
    }

    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = category.into();
        self
    }

    /// Append an input to the schema (builder style).
    pub fn add_input(mut self, input: Input) -> Self {
        self.inputs.push(input);
        self
    }

    /// Append multiple inputs at once.
    pub fn with_inputs(mut self, inputs: impl IntoIterator<Item = Input>) -> Self {
        self.inputs = inputs.into_iter().collect();
        self
    }

    /// Append an output to the schema (builder style).
    pub fn add_output(mut self, output: Output) -> Self {
        self.outputs.push(output);
        self
    }

    /// Append multiple outputs at once.
    pub fn with_outputs(mut self, outputs: impl IntoIterator<Item = Output>) -> Self {
        self.outputs = outputs.into_iter().collect();
        self
    }

    /// Append a hidden field (builder style).
    pub fn add_hidden(mut self, hidden: Hidden) -> Self {
        self.hidden.push(hidden);
        self
    }

    /// Set all hidden fields at once.
    pub fn with_hidden(mut self, hidden: impl IntoIterator<Item = Hidden>) -> Self {
        self.hidden = hidden.into_iter().collect();
        self
    }

    /// Set the node description tooltip.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Append a search alias (builder style).
    pub fn add_search_alias(mut self, alias: impl Into<String>) -> Self {
        self.search_aliases.push(alias.into());
        self
    }

    /// Set all search aliases at once.
    pub fn with_search_aliases(mut self, aliases: impl IntoIterator<Item = String>) -> Self {
        self.search_aliases = aliases.into_iter().collect();
        self
    }

    /// Enable INPUT_IS_LIST behaviour.
    pub fn with_input_list(mut self, enabled: bool) -> Self {
        self.is_input_list = enabled;
        self
    }

    /// Mark node as an output node.
    pub fn with_output_node(mut self, enabled: bool) -> Self {
        self.is_output_node = enabled;
        self
    }

    /// Mark node as deprecated.
    pub fn with_deprecated(mut self, enabled: bool) -> Self {
        self.is_deprecated = enabled;
        self
    }

    /// Mark node as experimental.
    pub fn with_experimental(mut self, enabled: bool) -> Self {
        self.is_experimental = enabled;
        self
    }

    /// Mark node as dev-only.
    pub fn with_dev_only(mut self, enabled: bool) -> Self {
        self.is_dev_only = enabled;
        self
    }

    /// Mark node as an API node.
    pub fn with_api_node(mut self, enabled: bool) -> Self {
        self.is_api_node = enabled;
        self
    }

    /// Set the optional pricing badge.
    pub fn with_price_badge(mut self, badge: PriceBadge) -> Self {
        self.price_badge = Some(badge);
        self
    }

    /// Mark node as not idempotent.
    pub fn with_not_idempotent(mut self, enabled: bool) -> Self {
        self.not_idempotent = enabled;
        self
    }

    /// Enable expandable outputs.
    pub fn with_expand(mut self, enabled: bool) -> Self {
        self.enable_expand = enabled;
        self
    }

    /// Allow all prompt inputs to be forwarded as kwargs.
    pub fn with_accept_all_inputs(mut self, enabled: bool) -> Self {
        self.accept_all_inputs = enabled;
        self
    }

    /// Set the Essentials tab category.
    pub fn with_essentials_category(mut self, cat: impl Into<String>) -> Self {
        self.essentials_category = Some(cat.into());
        self
    }

    /// Mark node as having intermediate output.
    pub fn with_intermediate_output(mut self, enabled: bool) -> Self {
        self.has_intermediate_output = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// Python conversion
// ---------------------------------------------------------------------------

impl NodeSchema {
    /// Convert this pure-Rust `NodeSchema` into a real Python
    /// `comfy_api.latest.io.Schema` object.
    ///
    /// This acquires the GIL, imports `comfy_api.latest.io`, then
    /// dynamically invokes the Python constructors matching the v3 API.
    pub fn into_py_schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // import comfy_api.latest.io
        let io = py.import("comfy_api.latest")?.getattr("io")?;

        // --- build inputs list ---
        let inputs = PyList::empty(py);
        for inp in &self.inputs {
            inputs.append(inp.to_py_obj(py)?)?;
        }

        // --- build outputs list ---
        let outputs = PyList::empty(py);
        for out in &self.outputs {
            outputs.append(out.to_py_obj(py)?)?;
        }

        // --- build hidden list ---
        let hidden = PyList::empty(py);
        for h in &self.hidden {
            hidden.append(h.to_string())?;
        }

        // Assemble kwargs dict.
        // Fields with None / default values are omitted to keep kwargs lean,
        // because io.Schema constructor accepts missing optional args.
        let kwargs = PyDict::new(py);
        kwargs.set_item("node_id", &self.node_id)?;
        kwargs.set_item("display_name", &self.display_name)?;
        kwargs.set_item("category", &self.category)?;
        kwargs.set_item("inputs", inputs)?;
        kwargs.set_item("outputs", outputs)?;
        kwargs.set_item("hidden", hidden)?;

        if !self.description.is_empty() {
            kwargs.set_item("description", &self.description)?;
        }
        if !self.search_aliases.is_empty() {
            kwargs.set_item("search_aliases", &self.search_aliases)?;
        }
        if self.is_input_list {
            kwargs.set_item("is_input_list", true)?;
        }
        if self.is_output_node {
            kwargs.set_item("is_output_node", true)?;
        }
        if self.is_deprecated {
            kwargs.set_item("is_deprecated", true)?;
        }
        if self.is_experimental {
            kwargs.set_item("is_experimental", true)?;
        }
        if self.is_dev_only {
            kwargs.set_item("is_dev_only", true)?;
        }
        if self.is_api_node {
            kwargs.set_item("is_api_node", true)?;
        }
        if let Some(ref badge) = self.price_badge {
            let py_badge = badge.to_py_obj(py)?;
            kwargs.set_item("price_badge", py_badge)?;
        }
        if self.not_idempotent {
            kwargs.set_item("not_idempotent", true)?;
        }
        if self.enable_expand {
            kwargs.set_item("enable_expand", true)?;
        }
        if self.accept_all_inputs {
            kwargs.set_item("accept_all_inputs", true)?;
        }
        if let Some(ref cat) = self.essentials_category {
            kwargs.set_item("essentials_category", cat)?;
        }
        if self.has_intermediate_output {
            kwargs.set_item("has_intermediate_output", true)?;
        }

        io.getattr("Schema")?.call((), Some(&kwargs))
    }
}
