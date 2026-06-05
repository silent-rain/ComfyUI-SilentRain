use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::pythonize;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// InputSpec — shared metadata for all node inputs
// ---------------------------------------------------------------------------

/// Shared metadata for every ComfyUI node input.
///
/// Mirrors the common constructor parameters of Python's `io.Input` base
/// class: `id`, `display_name`, `optional`, `tooltip`, `lazy`, `raw_link`,
/// `advanced`.
///
/// # Example
/// ```no_run
/// let spec = InputSpec {
///     id: "image".to_string(),
///     display_name: Some("Source Image".to_string()),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InputSpec {
    /// Unique identifier for this input within the node.
    pub id: String,

    /// Display name shown on the node socket. Defaults to `id` when `None`.
    pub display_name: Option<String>,

    /// Whether the input is optional.
    pub optional: bool,

    /// Tooltip shown when hovering over the input socket.
    pub tooltip: Option<String>,

    /// Mark input as lazily evaluated.
    pub lazy: bool,

    /// When `true`, pass raw link information instead of parsed value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_link: Option<bool>,

    /// When `true`, input is hidden behind an "Advanced" toggle in the UI.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub advanced: Option<bool>,
}

// ---------------------------------------------------------------------------
// Macro — inject common builder methods into any input struct
// ---------------------------------------------------------------------------

/// Macro that injects the standard builder chain methods for `InputSpec`
/// fields into an input struct.
///
/// The struct must have a field named `spec` of type `InputSpec`.
macro_rules! impl_input_common {
    ($ty:ty) => {
        impl $ty {
            /// Set the display name.
            pub fn display_name(mut self, v: impl Into<String>) -> Self {
                self.spec.display_name = Some(v.into());
                self
            }

            /// Set whether the input is optional.
            pub fn optional(mut self, v: bool) -> Self {
                self.spec.optional = v;
                self
            }

            /// Set the tooltip.
            pub fn tooltip(mut self, v: impl Into<String>) -> Self {
                self.spec.tooltip = Some(v.into());
                self
            }

            /// Set whether the input is lazily evaluated.
            pub fn lazy(mut self, v: bool) -> Self {
                self.spec.lazy = v;
                self
            }

            /// Set the raw_link flag.
            pub fn raw_link(mut self, v: bool) -> Self {
                self.spec.raw_link = Some(v);
                self
            }

            /// Set the advanced flag.
            pub fn advanced(mut self, v: bool) -> Self {
                self.spec.advanced = Some(v);
                self
            }
        }
    };
}

// ---------------------------------------------------------------------------
// NumberDisplayMode
// ---------------------------------------------------------------------------

/// Display mode for numeric inputs (Int / Float).
///
/// Mirrors Python's `io.NumberDisplay`.
#[derive(Debug, Default, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub enum NumberDisplayMode {
    #[default]
    Number = 0,
    Slider = 1,
}

impl NumberDisplayMode {
    /// Return the corresponding Python `io.NumberDisplay` variant.
    pub fn to_py_variant<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let nd = io.getattr("NumberDisplay")?;
        match self {
            NumberDisplayMode::Number => nd.getattr("number"),
            NumberDisplayMode::Slider => nd.getattr("slider"),
        }
    }
}

// ---------------------------------------------------------------------------
// ControlAfterGenerate
// ---------------------------------------------------------------------------

/// Control behavior after generation for combo inputs.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub enum ControlAfterGenerate {
    Fixed,
    Increment,
    Decrement,
    Randomize,
}

impl ControlAfterGenerate {
    /// Return the corresponding Python `ControlAfterGenerate` variant.
    pub fn to_py_variant<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cag = io.getattr("ControlAfterGenerate")?;
        match self {
            ControlAfterGenerate::Fixed => cag.getattr("fixed"),
            ControlAfterGenerate::Increment => cag.getattr("increment"),
            ControlAfterGenerate::Decrement => cag.getattr("decrement"),
            ControlAfterGenerate::Randomize => cag.getattr("randomize"),
        }
    }
}

// ---------------------------------------------------------------------------
// UploadType
// ---------------------------------------------------------------------------

/// Upload type for combo inputs.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub enum UploadType {
    Image,
    Audio,
    Video,
    Model,
}

impl UploadType {
    /// Return the corresponding Python `UploadType` variant.
    pub fn to_py_variant<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let ut = io.getattr("UploadType")?;
        match self {
            UploadType::Image => ut.getattr("image"),
            UploadType::Audio => ut.getattr("audio"),
            UploadType::Video => ut.getattr("video"),
            UploadType::Model => ut.getattr("model"),
        }
    }
}

// ---------------------------------------------------------------------------
// FolderType
// ---------------------------------------------------------------------------

/// Folder type for combo inputs.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub enum FolderType {
    Input,
    Output,
    Temp,
}

impl FolderType {
    /// Return the corresponding Python `FolderType` variant.
    pub fn to_py_variant<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let ft = io.getattr("FolderType")?;
        match self {
            FolderType::Input => ft.getattr("input"),
            FolderType::Output => ft.getattr("output"),
            FolderType::Temp => ft.getattr("temp"),
        }
    }
}

// ---------------------------------------------------------------------------
// RemoteOptions
// ---------------------------------------------------------------------------

/// Remote source options for combo inputs.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RemoteOptions {
    pub route: String,
    pub refresh_button: bool,
    #[serde(default = "default_control_after_refresh")]
    pub control_after_refresh: String,
    pub timeout: Option<i64>,
    pub max_retries: Option<i64>,
    pub refresh: Option<i64>,
}

fn default_control_after_refresh() -> String {
    "first".to_string()
}

impl RemoteOptions {
    pub fn new(route: impl Into<String>, refresh_button: bool) -> Self {
        Self {
            route: route.into(),
            refresh_button,
            control_after_refresh: default_control_after_refresh(),
            timeout: None,
            max_retries: None,
            refresh: None,
        }
    }

    pub fn with_control_after_refresh(mut self, v: impl Into<String>) -> Self {
        self.control_after_refresh = v.into();
        self
    }
    pub fn with_timeout(mut self, v: i64) -> Self {
        self.timeout = Some(v);
        self
    }
    pub fn with_max_retries(mut self, v: i64) -> Self {
        self.max_retries = Some(v);
        self
    }
    pub fn with_refresh(mut self, v: i64) -> Self {
        self.refresh = Some(v);
        self
    }
}

// ---------------------------------------------------------------------------
// Input — unified enum for all input descriptors (storage only)
// ---------------------------------------------------------------------------

/// Every ComfyUI node input is represented by a variant of this enum.
///
/// All variants are pure Rust types, so a `Vec<Input>` (the list of
/// inputs on a `NodeSchema`) can be cloned without ever touching the GIL.
///
/// To convert to a Python `io.*.Input` object, pattern-match and call
/// `to_py_obj` on the inner struct.
#[derive(Debug, Clone)]
pub enum Input {
    // -- base types --
    Int(IntInput),
    Float(FloatInput),
    String(StringInput),
    Bool(BoolInput),
    // -- typed ComfyUI data-flow types --
    Combo(ComboInput),
    MultiCombo(MultiComboInput),
    Custom(CustomInput),
    Image(ImageInput),
    Model(TypedInput),
    Vae(TypedInput),
    Clip(TypedInput),
    Conditioning(TypedInput),
    Latent(TypedInput),
    Mask(TypedInput),
}

impl Input {
    /// Convert this input to a real Python `io.*.Input` object.
    ///
    /// Delegates to the per-struct `to_py_obj` implementation.
    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self {
            Input::Image(i) => i.to_py_obj(py),
            Input::Int(i) => i.to_py_obj(py),
            Input::Float(i) => i.to_py_obj(py),
            Input::String(i) => i.to_py_obj(py),
            Input::Combo(i) => i.to_py_obj(py),
            Input::Bool(i) => i.to_py_obj(py),
            Input::Model(i) => i.to_py_obj(py),
            Input::Vae(i) => i.to_py_obj(py),
            Input::Clip(i) => i.to_py_obj(py),
            Input::Conditioning(i) => i.to_py_obj(py),
            Input::Latent(i) => i.to_py_obj(py),
            Input::Mask(i) => i.to_py_obj(py),
            Input::MultiCombo(i) => i.to_py_obj(py),
            Input::Custom(i) => i.to_py_obj(py),
        }
    }
}

// ---------------------------------------------------------------------------
// Input structs (pure Rust, builder friendly)
// ---------------------------------------------------------------------------

/// Int input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct IntInput {
    #[serde(flatten)]
    pub spec: InputSpec,
    pub default: i64,
    pub min: Option<i64>,
    pub max: Option<i64>,
    pub step: Option<i64>,
    pub control_after_generate: bool,
    pub display_mode: NumberDisplayMode,
    pub socketless: bool,
    pub force_input: bool,
}

impl_input_common!(IntInput);

impl IntInput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub fn default_value(mut self, v: i64) -> Self {
        self.default = v;
        self
    }
    pub fn min(mut self, v: i64) -> Self {
        self.min = Some(v);
        self
    }
    pub fn max(mut self, v: i64) -> Self {
        self.max = Some(v);
        self
    }
    pub fn step(mut self, v: i64) -> Self {
        self.step = Some(v);
        self
    }
    pub fn control_after_generate(mut self, v: bool) -> Self {
        self.control_after_generate = v;
        self
    }
    pub fn display_mode(mut self, v: NumberDisplayMode) -> Self {
        self.display_mode = v;
        self
    }
    pub fn socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn force_input(mut self, v: bool) -> Self {
        self.force_input = v;
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("Int")?.getattr("Input")?;

        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        if self.display_mode != NumberDisplayMode::Number {
            let _ = kwargs.set_item("display_mode", self.display_mode.to_py_variant(py)?);
        }
        cls.call((), Some(&kwargs))
    }
}

/// Float input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FloatInput {
    #[serde(flatten)]
    pub spec: InputSpec,
    pub default: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub round: Option<f64>,
    pub display_mode: NumberDisplayMode,
    pub socketless: bool,
    pub force_input: bool,
}

impl_input_common!(FloatInput);

impl FloatInput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub fn default_value(mut self, v: f64) -> Self {
        self.default = v;
        self
    }
    pub fn min(mut self, v: f64) -> Self {
        self.min = Some(v);
        self
    }
    pub fn max(mut self, v: f64) -> Self {
        self.max = Some(v);
        self
    }
    pub fn step(mut self, v: f64) -> Self {
        self.step = Some(v);
        self
    }
    pub fn round(mut self, v: f64) -> Self {
        self.round = Some(v);
        self
    }
    pub fn display_mode(mut self, v: NumberDisplayMode) -> Self {
        self.display_mode = v;
        self
    }
    pub fn socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn force_input(mut self, v: bool) -> Self {
        self.force_input = v;
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("Float")?.getattr("Input")?;

        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        if self.display_mode != NumberDisplayMode::Number {
            let _ = kwargs.set_item("display_mode", self.display_mode.to_py_variant(py)?);
        }
        cls.call((), Some(&kwargs))
    }
}

/// String input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StringInput {
    #[serde(flatten)]
    pub spec: InputSpec,
    pub default: String,
    pub multiline: bool,
    pub placeholder: Option<String>,
    pub dynamic_prompts: bool,
    pub socketless: bool,
    pub force_input: bool,
}

impl_input_common!(StringInput);

impl StringInput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub fn default_value(mut self, v: impl Into<String>) -> Self {
        self.default = v.into();
        self
    }
    pub fn multiline(mut self, v: bool) -> Self {
        self.multiline = v;
        self
    }
    pub fn placeholder(mut self, v: impl Into<String>) -> Self {
        self.placeholder = Some(v.into());
        self
    }
    pub fn dynamic_prompts(mut self, v: bool) -> Self {
        self.dynamic_prompts = v;
        self
    }
    pub fn socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn force_input(mut self, v: bool) -> Self {
        self.force_input = v;
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("String")?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        cls.call((), Some(&kwargs))
    }
}

/// Boolean input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct BoolInput {
    #[serde(flatten)]
    pub spec: InputSpec,
    #[serde(default, rename = "default")]
    pub default: bool,
    pub label_on: Option<String>,
    pub label_off: Option<String>,
    pub socketless: bool,
    pub force_input: bool,
}

impl_input_common!(BoolInput);

impl BoolInput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub fn default_value(mut self, v: bool) -> Self {
        self.default = v;
        self
    }
    pub fn label_on(mut self, v: impl Into<String>) -> Self {
        self.label_on = Some(v.into());
        self
    }
    pub fn label_off(mut self, v: impl Into<String>) -> Self {
        self.label_off = Some(v.into());
        self
    }
    pub fn socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn force_input(mut self, v: bool) -> Self {
        self.force_input = v;
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("Boolean")?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        cls.call((), Some(&kwargs))
    }
}

/// Combo (dropdown) input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ComboInput {
    #[serde(flatten)]
    pub spec: InputSpec,
    pub options: Vec<String>,
    pub default: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub control_after_generate: Option<ControlAfterGenerate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upload: Option<UploadType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_folder: Option<FolderType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote: Option<RemoteOptions>,
    pub socketless: bool,
}

impl_input_common!(ComboInput);

impl ComboInput {
    pub fn new(
        id: impl Into<String>,
        options: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            options: options.into_iter().map(Into::into).collect(),
            ..Default::default()
        }
    }

    pub fn default_value(mut self, v: impl Into<String>) -> Self {
        self.default = Some(v.into());
        self
    }
    pub fn control_after_generate(mut self, v: ControlAfterGenerate) -> Self {
        self.control_after_generate = Some(v);
        self
    }
    pub fn upload(mut self, v: UploadType) -> Self {
        self.upload = Some(v);
        self
    }
    pub fn image_folder(mut self, v: FolderType) -> Self {
        self.image_folder = Some(v);
        self
    }
    pub fn remote(mut self, v: RemoteOptions) -> Self {
        self.remote = Some(v);
        self
    }
    pub fn socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("Combo")?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;

        // Convert ControlAfterGenerate enum to Python variant
        if let Some(cag) = &self.control_after_generate {
            let _ = kwargs.set_item("control_after_generate", cag.to_py_variant(py)?);
        }

        // Convert UploadType enum to Python variant
        if let Some(upload) = &self.upload {
            let _ = kwargs.set_item("upload", upload.to_py_variant(py)?);
        }

        // Convert FolderType enum to Python variant
        if let Some(folder) = &self.image_folder {
            let _ = kwargs.set_item("image_folder", folder.to_py_variant(py)?);
        }

        cls.call((), Some(&kwargs))
    }
}

/// Image input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ImageInput {
    #[serde(flatten)]
    pub spec: InputSpec,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_dict: Option<serde_json::Value>,
}

impl_input_common!(ImageInput);

impl ImageInput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    pub fn extra_dict(mut self, v: impl Into<serde_json::Value>) -> Self {
        self.extra_dict = Some(v.into());
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("Image")?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        cls.call((), Some(&kwargs))
    }
}

/// Generic typed input for flow types (MODEL, VAE, CLIP, CONDITIONING, LATENT, MASK).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct TypedInput {
    #[serde(flatten)]
    pub spec: InputSpec,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_dict: Option<serde_json::Value>,
    #[serde(skip)]
    _type_tag: &'static str,
}

impl_input_common!(TypedInput);

impl TypedInput {
    pub fn model(id: impl Into<String>) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            _type_tag: "MODEL",
            ..Default::default()
        }
    }
    pub fn vae(id: impl Into<String>) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            _type_tag: "VAE",
            ..Default::default()
        }
    }
    pub fn clip(id: impl Into<String>) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            _type_tag: "CLIP",
            ..Default::default()
        }
    }
    pub fn conditioning(id: impl Into<String>) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            _type_tag: "CONDITIONING",
            ..Default::default()
        }
    }
    pub fn latent(id: impl Into<String>) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            _type_tag: "LATENT",
            ..Default::default()
        }
    }
    pub fn mask(id: impl Into<String>) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            _type_tag: "MASK",
            ..Default::default()
        }
    }

    pub fn extra_dict(mut self, v: impl Into<serde_json::Value>) -> Self {
        self.extra_dict = Some(v.into());
        self
    }

    pub fn type_name(&self) -> &'static str {
        self._type_tag
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr(self._type_tag)?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        cls.call((), Some(&kwargs))
    }
}

/// Multi-select combo (dropdown) input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MultiComboInput {
    #[serde(flatten)]
    pub spec: InputSpec,
    pub options: Vec<String>,
    pub default: Vec<String>,
    pub placeholder: Option<String>,
    pub chip: bool,
    pub control_after_generate: bool,
    pub socketless: bool,
}

impl_input_common!(MultiComboInput);

impl MultiComboInput {
    pub fn new(
        id: impl Into<String>,
        options: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            spec: InputSpec {
                id: id.into(),
                ..Default::default()
            },
            options: options.into_iter().map(Into::into).collect(),
            ..Default::default()
        }
    }

    pub fn default_value(mut self, v: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.default = v.into_iter().map(Into::into).collect();
        self
    }
    pub fn placeholder(mut self, v: impl Into<String>) -> Self {
        self.placeholder = Some(v.into());
        self
    }
    pub fn chip(mut self, v: bool) -> Self {
        self.chip = v;
        self
    }
    pub fn control_after_generate(mut self, v: bool) -> Self {
        self.control_after_generate = v;
        self
    }
    pub fn socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("MultiCombo")?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        cls.call((), Some(&kwargs))
    }
}

/// Custom type input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CustomInput {
    pub id: String,
    pub io_type: String,
    #[serde(flatten)]
    pub spec: InputSpec,
    pub placeholder: Option<String>,
    pub socketless: bool,
}

impl_input_common!(CustomInput);

impl CustomInput {
    pub fn new(id: impl Into<String>, io_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            io_type: io_type.into(),
            spec: InputSpec::default(),
            ..Default::default()
        }
    }

    pub fn placeholder(mut self, v: impl Into<String>) -> Self {
        self.placeholder = Some(v.into());
        self
    }
    pub fn socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("Custom")?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        cls.call((&self.io_type, &self.id), Some(&kwargs))
    }
}

// -- typed ComfyUI data-flow types --------------------------------------------------

impl From<ImageInput> for Input {
    fn from(v: ImageInput) -> Self {
        Input::Image(v)
    }
}
impl From<IntInput> for Input {
    fn from(v: IntInput) -> Self {
        Input::Int(v)
    }
}
impl From<FloatInput> for Input {
    fn from(v: FloatInput) -> Self {
        Input::Float(v)
    }
}
impl From<StringInput> for Input {
    fn from(v: StringInput) -> Self {
        Input::String(v)
    }
}
impl From<ComboInput> for Input {
    fn from(v: ComboInput) -> Self {
        Input::Combo(v)
    }
}
impl From<BoolInput> for Input {
    fn from(v: BoolInput) -> Self {
        Input::Bool(v)
    }
}
impl From<TypedInput> for Input {
    fn from(v: TypedInput) -> Self {
        match v._type_tag {
            "MODEL" => Input::Model(v),
            "VAE" => Input::Vae(v),
            "CLIP" => Input::Clip(v),
            "CONDITIONING" => Input::Conditioning(v),
            "LATENT" => Input::Latent(v),
            "MASK" => Input::Mask(v),
            _ => panic!("unknown TypedInput type tag: {}", v._type_tag),
        }
    }
}
impl From<MultiComboInput> for Input {
    fn from(v: MultiComboInput) -> Self {
        Input::MultiCombo(v)
    }
}
impl From<CustomInput> for Input {
    fn from(v: CustomInput) -> Self {
        Input::Custom(v)
    }
}
