use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::pythonize;
use serde::{Deserialize, Serialize};

/*
## Input Types

### Basic Inputs

```python
# Integer input
io.Int.Input(
    id: str,
    display_name: str = None,
    optional: bool = False,
    tooltip: str = None,
    lazy: bool = None,
    default: int = None,
    min: int = None,
    max: int = None,
    step: int = None,
    control_after_generate: bool = None,
    display_mode: NumberDisplay = None,
    socketless: bool = None,
    force_input: bool = None
)

# Float input
io.Float.Input(
    id: str,
    display_name: str = None,
    optional: bool = False,
    tooltip: str = None,
    lazy: bool = None,
    default: float = None,
    min: float = None,
    max: float = None,
    step: float = None,
    round: float = None,
    display_mode: NumberDisplay = None,
    socketless: bool = None,
    force_input: bool = None
)

# String input
io.String.Input(
    id: str,
    display_name: str = None,
    optional: bool = False,
    tooltip: str = None,
    lazy: bool = None,
    multiline: bool = False,
    placeholder: str = None,
    default: str = None,
    dynamic_prompts: bool = None,
    socketless: bool = None,
    force_input: bool = None
)

# Boolean input
io.Boolean.Input(
    id: str,
    display_name: str = None,
    optional: bool = False,
    tooltip: str = None,
    lazy: bool = None,
    default: bool = None,
    label_on: str = None,
    label_off: str = None,
    socketless: bool = None,
    force_input: bool = None
)

# Combo (dropdown) input
io.Combo.Input(
    id: str,
    options: list[str] = None,
    display_name: str = None,
    optional: bool = False,
    tooltip: str = None,
    lazy: bool = None,
    default: str = None,
    control_after_generate: bool = None,
    upload: UploadType = None,
    image_folder: FolderType = None,
    remote: RemoteOptions = None,
    socketless: bool = None
)

# Multi-select combo
io.MultiCombo.Input(
    id: str,
    options: list[str],
    display_name: str = None,
    optional: bool = False,
    tooltip: str = None,
    lazy: bool = None,
    default: list[str] = None,
    placeholder: str = None,
    chip: bool = None,
    control_after_generate: bool = None,
    socketless: bool = None
)
# cusotm type
io.Custom(io_type="MY_TYPE").Input(
    id: str,
    display_name: str = None,
    optional: bool = False,
    tooltip: str = None,
    lazy: bool = None,
    placeholder: str = None,
)
```

### ComfyUI Types

```python
# Core types
io.Image.Input(id, ...)        # Type: torch.Tensor [B,H,W,C]
io.Mask.Input(id, ...)         # Type: torch.Tensor [H,W] or [B,H,W]
io.Latent.Input(id, ...)       # Type: dict with 'samples' tensor
io.Conditioning.Input(id, ...)  # Type: list[tuple[tensor, dict]]
io.Model.Input(id, ...)        # Type: ModelPatcher
io.Clip.Input(id, ...)         # Type: CLIP
io.Vae.Input(id, ...)          # Type: VAE
io.ControlNet.Input(id, ...)   # Type: ControlNet

# Sampling types
io.Sampler.Input(id, ...)      # Type: Sampler
io.Sigmas.Input(id, ...)       # Type: torch.Tensor
io.Noise.Input(id, ...)        # Type: torch.Tensor
io.Guider.Input(id, ...)       # Type: CFGGuider

# Additional types
io.ClipVision.Input(id, ...)         # Type: ClipVisionModel
io.ClipVisionOutput.Input(id, ...)   # Type: ClipVisionOutput
io.StyleModel.Input(id, ...)         # Type: StyleModel
io.Gligen.Input(id, ...)             # Type: ModelPatcher
io.UpscaleModel.Input(id, ...)       # Type: ImageModelDescriptor
io.Audio.Input(id, ...)              # Type: dict with 'waveform' and 'sample_rate'
io.Video.Input(id, ...)              # Type: VideoInput
io.Webcam.Input(id, ...)             # Type: str (filepath)
io.WanCameraEmbedding.Input(id, ...) # Type: torch.Tensor
io.LoraModel.Input(id, ...)          # Type: dict[str, Tensor]
io.Hooks.Input(id, ...)              # Type: HookGroup
io.HookKeyframes.Input(id, ...)      # Type: HookKeyframeGroup
io.SVG.Input(id, ...)                # Type: SVG (custom class)
io.Voxel.Input(id, ...)              # Type: Voxel data (custom class)
io.Mesh.Input(id, ...)               # Type: Mesh data (custom class)
```

### Advanced Inputs

```python
# Multi-type input (accepts multiple types)
io.MultiType.Input(
    id: str | InputV3,  # Can override from existing input
    types: list[type[ComfyType]],
    display_name: str = None,
    optional: bool = False,
    tooltip: str = None,
    lazy: bool = None
)

# Dynamic growing input
io.AutogrowDynamic.Input(
    id: str,
    template_input: InputV3,  # Template for each new input
    min: int = 1,             # Minimum inputs
    max: int = None           # Maximum inputs
)

# Custom type
@io.comfytype(io_type="MY_CUSTOM")
class MyCustom:
    Type = MyDataClass
    class Input(io.InputV3):
        ...
    class Output(io.OutputV3):
        ...
```

## Output Types

```python
# Basic output
io.Image.Output(
    id: str = None,
    display_name: str = None,
    tooltip: str = None,
    is_output_list: bool = False  # Output is list
)

# All ComfyUI types have corresponding outputs
io.Mask.Output(id, ...)
io.Latent.Output(id, ...)
io.Model.Output(id, ...)
io.Clip.Output(id, ...)
io.Vae.Output(id, ...)
io.Conditioning.Output(id, ...)
io.String.Output(id, ...)
io.Int.Output(id, ...)
io.Float.Output(id, ...)
io.Boolean.Output(id, ...)
# ... etc
```

## Hidden Inputs

```python
from comfy_api.latest import Hidden

# Available hidden inputs
Hidden.unique_id            # Node's unique ID
Hidden.prompt              # Complete prompt
Hidden.extra_pnginfo       # PNG metadata dict
Hidden.dynprompt           # Dynamic prompt object
Hidden.auth_token_comfy_org # ComfyOrg auth token
Hidden.api_key_comfy_org   # ComfyOrg API key

# Usage in schema
hidden=[
    Hidden.unique_id,
    Hidden.prompt
]

# Access in execute
unique_id = cls.hidden.unique_id
prompt = cls.hidden.prompt
```

*/

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
// NodeInput — unified enum for all input descriptors (storage only)
// ---------------------------------------------------------------------------

/// Every ComfyUI node input is represented by a variant of this enum.
///
/// All variants are pure Rust types, so a `Vec<NodeInput>` (the list of
/// inputs on a `NodeSchema`) can be cloned without ever touching the GIL.
///
/// To convert to a Python `io.*.Input` object, pattern-match and call
/// `to_py_obj` on the inner struct.
#[derive(Debug, Clone)]
pub enum Input {
    Int(IntInput),
    Float(FloatInput),
    String(StringInput),
    Bool(BoolInput),
    Combo(ComboInput),
    Image(ImageInput),
    // -- typed ComfyUI data-flow types --
    Model(TypedInput),
    Vae(TypedInput),
    Clip(TypedInput),
    Conditioning(TypedInput),
    Latent(TypedInput),
    Mask(TypedInput),
    MultiCombo(MultiComboInput),
    Custom(CustomInput),
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
    id: String,
    display_name: Option<String>,
    optional: bool,
    tooltip: Option<String>,
    lazy: bool,
    default: i64,
    min: Option<i64>,
    max: Option<i64>,
    step: Option<i64>,
    control_after_generate: bool,
    display_mode: NumberDisplayMode,
    socketless: bool,
    force_input: bool,
}

impl IntInput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    pub fn with_display_name(mut self, v: impl Into<String>) -> Self {
        self.display_name = Some(v.into());
        self
    }
    pub fn with_optional(mut self, v: bool) -> Self {
        self.optional = v;
        self
    }
    pub fn with_tooltip(mut self, v: impl Into<String>) -> Self {
        self.tooltip = Some(v.into());
        self
    }
    pub fn with_lazy(mut self, v: bool) -> Self {
        self.lazy = v;
        self
    }
    pub fn with_default(mut self, v: i64) -> Self {
        self.default = v;
        self
    }
    pub fn with_min(mut self, v: i64) -> Self {
        self.min = Some(v);
        self
    }
    pub fn with_max(mut self, v: i64) -> Self {
        self.max = Some(v);
        self
    }
    pub fn with_step(mut self, v: i64) -> Self {
        self.step = Some(v);
        self
    }
    pub fn with_control_after_generate(mut self, v: bool) -> Self {
        self.control_after_generate = v;
        self
    }
    pub fn with_display_mode(mut self, v: NumberDisplayMode) -> Self {
        self.display_mode = v;
        self
    }
    pub fn with_socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn with_force_input(mut self, v: bool) -> Self {
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
        cls.call((&self.id,), Some(&kwargs))
    }
}

/// Float input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct FloatInput {
    id: String,
    display_name: Option<String>,
    optional: bool,
    tooltip: Option<String>,
    lazy: bool,
    default: f64,
    min: Option<f64>,
    max: Option<f64>,
    step: Option<f64>,
    round: Option<f64>,
    display_mode: NumberDisplayMode,
    socketless: bool,
    force_input: bool,
}

impl FloatInput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    pub fn with_display_name(mut self, v: impl Into<String>) -> Self {
        self.display_name = Some(v.into());
        self
    }
    pub fn with_optional(mut self, v: bool) -> Self {
        self.optional = v;
        self
    }
    pub fn with_tooltip(mut self, v: impl Into<String>) -> Self {
        self.tooltip = Some(v.into());
        self
    }
    pub fn with_lazy(mut self, v: bool) -> Self {
        self.lazy = v;
        self
    }
    pub fn with_socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn with_force_input(mut self, v: bool) -> Self {
        self.force_input = v;
        self
    }
    pub fn with_default(mut self, v: f64) -> Self {
        self.default = v;
        self
    }
    pub fn with_min(mut self, v: f64) -> Self {
        self.min = Some(v);
        self
    }
    pub fn with_max(mut self, v: f64) -> Self {
        self.max = Some(v);
        self
    }
    pub fn with_step(mut self, v: f64) -> Self {
        self.step = Some(v);
        self
    }
    pub fn with_round(mut self, v: f64) -> Self {
        self.round = Some(v);
        self
    }
    pub fn with_display_mode(mut self, v: NumberDisplayMode) -> Self {
        self.display_mode = v;
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("Float")?.getattr("Input")?;

        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        if self.display_mode != NumberDisplayMode::Number {
            let _ = kwargs.set_item("display_mode", self.display_mode.to_py_variant(py)?);
        }
        cls.call((&self.id,), Some(&kwargs))
    }
}

/// String input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StringInput {
    id: String,
    display_name: Option<String>,
    optional: bool,
    tooltip: Option<String>,
    lazy: bool,
    default: String,
    multiline: bool,
    placeholder: Option<String>,
    dynamic_prompts: bool,
    socketless: bool,
    force_input: bool,
}

impl StringInput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    pub fn with_display_name(mut self, v: impl Into<String>) -> Self {
        self.display_name = Some(v.into());
        self
    }
    pub fn with_optional(mut self, v: bool) -> Self {
        self.optional = v;
        self
    }
    pub fn with_tooltip(mut self, v: impl Into<String>) -> Self {
        self.tooltip = Some(v.into());
        self
    }
    pub fn with_lazy(mut self, v: bool) -> Self {
        self.lazy = v;
        self
    }
    pub fn with_socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn with_force_input(mut self, v: bool) -> Self {
        self.force_input = v;
        self
    }
    pub fn with_default(mut self, v: impl Into<String>) -> Self {
        self.default = v.into();
        self
    }
    pub fn with_multiline(mut self, v: bool) -> Self {
        self.multiline = v;
        self
    }
    pub fn with_placeholder(mut self, v: impl Into<String>) -> Self {
        self.placeholder = Some(v.into());
        self
    }
    pub fn with_dynamic_prompts(mut self, v: bool) -> Self {
        self.dynamic_prompts = v;
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("String")?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        cls.call((&self.id,), Some(&kwargs))
    }
}

/// Boolean input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct BoolInput {
    id: String,
    display_name: Option<String>,
    optional: bool,
    tooltip: Option<String>,
    lazy: bool,
    default: bool,
    label_on: Option<String>,
    label_off: Option<String>,
    socketless: bool,
    force_input: bool,
}

impl BoolInput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    pub fn with_display_name(mut self, v: impl Into<String>) -> Self {
        self.display_name = Some(v.into());
        self
    }
    pub fn with_optional(mut self, v: bool) -> Self {
        self.optional = v;
        self
    }
    pub fn with_tooltip(mut self, v: impl Into<String>) -> Self {
        self.tooltip = Some(v.into());
        self
    }
    pub fn with_lazy(mut self, v: bool) -> Self {
        self.lazy = v;
        self
    }
    pub fn with_socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn with_force_input(mut self, v: bool) -> Self {
        self.force_input = v;
        self
    }
    pub fn with_default(mut self, v: bool) -> Self {
        self.default = v;
        self
    }
    pub fn with_label_on(mut self, v: impl Into<String>) -> Self {
        self.label_on = Some(v.into());
        self
    }
    pub fn with_label_off(mut self, v: impl Into<String>) -> Self {
        self.label_off = Some(v.into());
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("Bool")?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        cls.call((&self.id,), Some(&kwargs))
    }
}

/// Combo (dropdown) input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ComboInput {
    id: String,
    display_name: Option<String>,
    optional: bool,
    tooltip: Option<String>,
    lazy: bool,
    options: Vec<String>,
    default: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    control_after_generate: Option<ControlAfterGenerate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    upload: Option<UploadType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_folder: Option<FolderType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    remote: Option<RemoteOptions>,
    socketless: bool,
    force_input: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_link: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    advanced: Option<bool>,
}

impl ComboInput {
    pub fn new(
        id: impl Into<String>,
        options: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            id: id.into(),
            options: options.into_iter().map(Into::into).collect(),
            ..Default::default()
        }
    }

    pub fn with_display_name(mut self, v: impl Into<String>) -> Self {
        self.display_name = Some(v.into());
        self
    }
    pub fn with_optional(mut self, v: bool) -> Self {
        self.optional = v;
        self
    }
    pub fn with_tooltip(mut self, v: impl Into<String>) -> Self {
        self.tooltip = Some(v.into());
        self
    }
    pub fn with_lazy(mut self, v: bool) -> Self {
        self.lazy = v;
        self
    }
    pub fn with_socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn with_force_input(mut self, v: bool) -> Self {
        self.force_input = v;
        self
    }
    pub fn with_default(mut self, v: impl Into<String>) -> Self {
        self.default = Some(v.into());
        self
    }
    pub fn with_control_after_generate(mut self, v: ControlAfterGenerate) -> Self {
        self.control_after_generate = Some(v);
        self
    }
    pub fn with_upload(mut self, v: UploadType) -> Self {
        self.upload = Some(v);
        self
    }
    pub fn with_image_folder(mut self, v: FolderType) -> Self {
        self.image_folder = Some(v);
        self
    }
    pub fn with_remote(mut self, v: RemoteOptions) -> Self {
        self.remote = Some(v);
        self
    }
    pub fn with_raw_link(mut self, v: bool) -> Self {
        self.raw_link = Some(v);
        self
    }
    pub fn with_advanced(mut self, v: bool) -> Self {
        self.advanced = Some(v);
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

        cls.call((&self.id,), Some(&kwargs))
    }
}

/// Image input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct ImageInput {
    id: String,
    display_name: Option<String>,
    optional: bool,
    tooltip: Option<String>,
    lazy: bool,
    socketless: bool,
    force_input: bool,
}

impl ImageInput {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Default::default()
        }
    }

    pub fn with_display_name(mut self, v: impl Into<String>) -> Self {
        self.display_name = Some(v.into());
        self
    }
    pub fn with_optional(mut self, v: bool) -> Self {
        self.optional = v;
        self
    }
    pub fn with_tooltip(mut self, v: impl Into<String>) -> Self {
        self.tooltip = Some(v.into());
        self
    }
    pub fn with_lazy(mut self, v: bool) -> Self {
        self.lazy = v;
        self
    }
    pub fn with_socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn with_force_input(mut self, v: bool) -> Self {
        self.force_input = v;
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("Image")?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        cls.call((&self.id,), Some(&kwargs))
    }
}

/// Generic typed input for flow types (MODEL, VAE, CLIP, CONDITIONING, LATENT, MASK).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct TypedInput {
    id: String,
    display_name: Option<String>,
    optional: bool,
    tooltip: Option<String>,
    lazy: bool,
    socketless: bool,
    force_input: bool,
    #[serde(skip)]
    _type_tag: &'static str,
}

impl TypedInput {
    pub fn model(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            _type_tag: "MODEL",
            ..Default::default()
        }
    }
    pub fn vae(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            _type_tag: "VAE",
            ..Default::default()
        }
    }
    pub fn clip(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            _type_tag: "CLIP",
            ..Default::default()
        }
    }
    pub fn conditioning(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            _type_tag: "CONDITIONING",
            ..Default::default()
        }
    }
    pub fn latent(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            _type_tag: "LATENT",
            ..Default::default()
        }
    }
    pub fn mask(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            _type_tag: "MASK",
            ..Default::default()
        }
    }

    pub fn with_display_name(mut self, v: impl Into<String>) -> Self {
        self.display_name = Some(v.into());
        self
    }
    pub fn with_optional(mut self, v: bool) -> Self {
        self.optional = v;
        self
    }
    pub fn with_tooltip(mut self, v: impl Into<String>) -> Self {
        self.tooltip = Some(v.into());
        self
    }
    pub fn with_lazy(mut self, v: bool) -> Self {
        self.lazy = v;
        self
    }
    pub fn with_socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn with_force_input(mut self, v: bool) -> Self {
        self.force_input = v;
        self
    }

    pub fn type_name(&self) -> &'static str {
        self._type_tag
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr(self._type_tag)?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        cls.call((&self.id,), Some(&kwargs))
    }
}

/// Multi-select combo (dropdown) input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MultiComboInput {
    id: String,
    display_name: Option<String>,
    optional: bool,
    tooltip: Option<String>,
    lazy: bool,
    options: Vec<String>,
    default: Vec<String>,
    placeholder: Option<String>,
    chip: bool,
    control_after_generate: bool,
    socketless: bool,
}

impl MultiComboInput {
    pub fn new(
        id: impl Into<String>,
        options: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            id: id.into(),
            options: options.into_iter().map(Into::into).collect(),
            ..Default::default()
        }
    }

    pub fn with_display_name(mut self, v: impl Into<String>) -> Self {
        self.display_name = Some(v.into());
        self
    }
    pub fn with_optional(mut self, v: bool) -> Self {
        self.optional = v;
        self
    }
    pub fn with_tooltip(mut self, v: impl Into<String>) -> Self {
        self.tooltip = Some(v.into());
        self
    }
    pub fn with_lazy(mut self, v: bool) -> Self {
        self.lazy = v;
        self
    }
    pub fn with_socketless(mut self, v: bool) -> Self {
        self.socketless = v;
        self
    }
    pub fn with_default(mut self, v: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.default = v.into_iter().map(Into::into).collect();
        self
    }
    pub fn with_placeholder(mut self, v: impl Into<String>) -> Self {
        self.placeholder = Some(v.into());
        self
    }
    pub fn with_chip(mut self, v: bool) -> Self {
        self.chip = v;
        self
    }
    pub fn with_control_after_generate(mut self, v: bool) -> Self {
        self.control_after_generate = v;
        self
    }

    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("MultiCombo")?.getattr("Input")?;
        let kwargs = pythonize(py, &self)?.extract::<Bound<'py, PyDict>>()?;
        cls.call((&self.id,), Some(&kwargs))
    }
}

/// Custom type input.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CustomInput {
    id: String,
    io_type: String,
    display_name: Option<String>,
    optional: bool,
    tooltip: Option<String>,
    lazy: bool,
    placeholder: Option<String>,
    socketless: bool,
}

impl CustomInput {
    pub fn new(id: impl Into<String>, io_type: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            io_type: io_type.into(),
            ..Default::default()
        }
    }

    pub fn with_display_name(mut self, v: impl Into<String>) -> Self {
        self.display_name = Some(v.into());
        self
    }
    pub fn with_optional(mut self, v: bool) -> Self {
        self.optional = v;
        self
    }
    pub fn with_tooltip(mut self, v: impl Into<String>) -> Self {
        self.tooltip = Some(v.into());
        self
    }
    pub fn with_lazy(mut self, v: bool) -> Self {
        self.lazy = v;
        self
    }
    pub fn with_placeholder(mut self, v: impl Into<String>) -> Self {
        self.placeholder = Some(v.into());
        self
    }
    pub fn with_socketless(mut self, v: bool) -> Self {
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
