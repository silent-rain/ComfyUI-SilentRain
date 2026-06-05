//! output schema
use pyo3::{
    BoundObject,
    prelude::*,
    types::{PyDict, PyTuple},
};
use pythonize::pythonize;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// OutputSpec — shared metadata for all node outputs
// ---------------------------------------------------------------------------

/// Shared metadata for every ComfyUI node output.
///
/// Mirrors the common constructor parameters of Python's `io.Output` base
/// class: `id`, `display_name`, `tooltip`, `is_output_list`.
///
/// # Example
/// ```no_run
/// let spec = OutputSpec {
///     id: "result".to_string(),
///     display_name: Some("Processed Image".to_string()),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OutputSpec {
    /// Unique identifier for this output within the node.
    pub id: String,

    /// Display name shown on the node socket. Defaults to `id` when `None`.
    pub display_name: Option<String>,

    /// Tooltip shown when hovering over the output socket.
    pub tooltip: Option<String>,

    /// When `true`, the output is wrapped in a list regardless of how
    /// many items are actually produced.
    ///
    /// See Comfy docs on `OUTPUT_IS_LIST` for details.
    #[serde(rename = "is_output_list")]
    pub is_output_list: bool,
}

// ---------------------------------------------------------------------------
// OutputType — tag for the ComfyUI data-flow type
// ---------------------------------------------------------------------------

/// ComfyUI output type tag.
///
/// Most variants are zero-sized because all metadata lives in
/// [`OutputSpec`].  Only [`OutputType::Combo`] carries extra data
/// (`options`, matching Python's `Combo.Output`).
#[derive(Debug, Clone)]
pub enum OutputType {
    Image,
    Latent,
    Conditioning,
    Mask,
    Model,
    Vae,
    Clip,
    String,
    Int,
    Float,
    Boolean,
    /// Combo output with a fixed list of string options.
    Combo(Vec<String>),
    /// Custom / user-defined type.
    Custom(String),
}

impl OutputType {
    /// Return the `io_type` string used by ComfyUI (e.g. "IMAGE", "MODEL").
    pub fn io_type(&self) -> &str {
        match self {
            OutputType::Image => "IMAGE",
            OutputType::Latent => "LATENT",
            OutputType::Conditioning => "CONDITIONING",
            OutputType::Mask => "MASK",
            OutputType::Model => "MODEL",
            OutputType::Vae => "VAE",
            OutputType::Clip => "CLIP",
            OutputType::String => "STRING",
            OutputType::Int => "INT",
            OutputType::Float => "FLOAT",
            OutputType::Boolean => "BOOLEAN",
            OutputType::Combo(_) => "COMBO",
            OutputType::Custom(t) => t.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// NodeOutput — complete output descriptor (spec + type + builder)
// ---------------------------------------------------------------------------

/// Complete node output descriptor.
///
/// To create an output, pick a type-specific constructor and then chain
/// builder methods to set optional metadata.
///
/// # Example
/// ```no_run
/// use comfyui_v3::schema::output::NodeOutput;
///
/// let outputs = vec![
///     NodeOutput::image("imageout").with_display_name("Result"),
///     NodeOutput::combo("size", ["256", "512", "1024"]),
///     NodeOutput::string("stringout"),
///     NodeOutput::int("int").with_is_output_list(true),
/// ];
/// ```
#[pyclass(from_py_object)]
#[derive(Debug, Clone)]
pub struct Output {
    pub spec: OutputSpec,
    pub r#type: OutputType,
}

impl Output {
    // ----- type-specific constructors -----

    /// Image output (`io.Image.Output`).
    pub fn image(id: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Image,
        }
    }

    /// Model output (`io.Model.Output`).
    pub fn model(id: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Model,
        }
    }

    /// VAE output (`io.Vae.Output`).
    pub fn vae(id: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Vae,
        }
    }

    /// CLIP output (`io.Clip.Output`).
    pub fn clip(id: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Clip,
        }
    }

    /// Conditioning output (`io.Conditioning.Output`).
    pub fn conditioning(id: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Conditioning,
        }
    }

    /// Latent output (`io.Latent.Output`).
    pub fn latent(id: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Latent,
        }
    }

    /// Mask output (`io.Mask.Output`).
    pub fn mask(id: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Mask,
        }
    }

    /// String output (`io.String.Output`).
    pub fn string(id: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::String,
        }
    }

    /// Integer output (`io.Int.Output`).
    pub fn int(id: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Int,
        }
    }

    /// Float output (`io.Float.Output`).
    pub fn float(id: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Float,
        }
    }

    /// Boolean output (`io.Boolean.Output`).
    pub fn boolean(id: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Boolean,
        }
    }

    /// Combo (dropdown) output with a fixed set of options (`io.Combo.Output`).
    pub fn combo(
        id: impl Into<String>,
        options: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Combo(options.into_iter().map(Into::into).collect()),
        }
    }

    /// Custom / user-defined type output.
    pub fn custom(id: impl Into<String>, io_type: impl Into<String>) -> Self {
        Self {
            spec: OutputSpec {
                id: id.into(),
                ..Default::default()
            },
            r#type: OutputType::Custom(io_type.into()),
        }
    }

    // ----- builder chain -----

    /// Set display name.
    pub fn display_name(mut self, v: impl Into<String>) -> Self {
        self.spec.display_name = Some(v.into());
        self
    }

    /// Set tooltip.
    pub fn tooltip(mut self, v: impl Into<String>) -> Self {
        self.spec.tooltip = Some(v.into());
        self
    }

    /// Set `is_output_list` flag.
    pub fn is_output_list(mut self, v: bool) -> Self {
        self.spec.is_output_list = v;
        self
    }

    // ----- python conversion -----

    /// Convert this descriptor to a real Python `io.*.Output` object.
    ///
    /// Dynamically imports `comfy_api.latest.io`, locates the correct
    /// `Output` class, and calls it with kwargs generated by `pythonize`.
    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;

        // locate the Python class: io.Image.Output, io.Combo.Output, etc.
        let py_cls = match &self.r#type {
            OutputType::Image => io.getattr("Image")?.getattr("Output")?,
            OutputType::Latent => io.getattr("Latent")?.getattr("Output")?,
            OutputType::Conditioning => io.getattr("Conditioning")?.getattr("Output")?,
            OutputType::Mask => io.getattr("Mask")?.getattr("Output")?,
            OutputType::Model => io.getattr("Model")?.getattr("Output")?,
            OutputType::Vae => io.getattr("Vae")?.getattr("Output")?,
            OutputType::Clip => io.getattr("Clip")?.getattr("Output")?,
            OutputType::String => io.getattr("String")?.getattr("Output")?,
            OutputType::Int => io.getattr("Int")?.getattr("Output")?,
            OutputType::Float => io.getattr("Float")?.getattr("Output")?,
            OutputType::Boolean => io.getattr("Boolean")?.getattr("Output")?,
            OutputType::Combo(_) => io.getattr("Combo")?.getattr("Output")?,
            OutputType::Custom(io_type) => {
                // For custom types, use io.Custom(io_type).Output
                io.getattr("Custom")?.call1((io_type,))?.getattr("Output")?
            }
        };

        // Build kwargs dict from OutputSpec via pythonize.
        let kwargs = pythonize(py, &self.spec)?.extract::<Bound<'py, PyDict>>()?;

        // Handle type-specific extra fields (Combo needs "options").
        if let OutputType::Combo(opts) = &self.r#type {
            kwargs.set_item("options", opts)?;
        }

        py_cls.call((), Some(&kwargs))
    }
}

/// 标准化节点输出，对应 Python 的 `io.NodeOutput`。
///
/// 接受任意数量的位置参数，并支持可选的 `ui`、`expand`、`block_execution`
/// 关键字参数。
///
/// # Examples
/// ```no_run
/// use comfyui_v3::schema::NodeOutput;
///
/// let out = NodeOutput::new()
///     .add_arg(image)
///     .with_ui(json!({ "images": [...] }));
/// ```
#[derive(Debug, Default)]
pub struct NodeOutput {
    /// Positional arguments (actual node output data).
    pub args: Vec<Py<PyAny>>,

    /// Optional `ui` data (images, text, etc.) sent to the frontend.
    pub ui: Option<Py<PyAny>>,

    /// Optional dictionary for node expansion (used by `enable_expand`).
    pub expand: Option<Py<PyAny>>,

    /// If set, blocks execution with the given error message.
    pub block_execution: Option<String>,
}

impl NodeOutput {
    /// Create an empty `NodeOutput`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a positional argument (e.g. the processed image).
    pub fn add_arg(mut self, arg: Py<PyAny>) -> Self {
        self.args.push(arg);
        self
    }

    /// Add multiple positional arguments at once.
    pub fn add_args(mut self, args: impl IntoIterator<Item = Py<PyAny>>) -> Self {
        self.args.extend(args);
        self
    }

    pub fn add_arg_from_pyobject<T: Into<Py<PyAny>>>(mut self, arg: T) -> PyResult<Self> {
        let py_arg = arg.into();
        self.args.push(py_arg);
        Ok(self)
    }

    /// Add an argument from a value that can be converted into a Python object.
    ///
    /// This method accepts any type that implements `IntoPyObject`,
    /// such as `String`, `&str`, `i64`, `f64`, `bool`, etc.
    ///
    /// # Example
    /// ```no_run
    /// use comfyui_v3::schema::NodeOutput;
    ///
    /// let out = NodeOutput::new()
    ///     .add_arg_from(py, "hello".to_string())
    ///     .add_arg_from(py, 42_i64)
    ///     .add_arg_from(py, 3.14_f64);
    /// ```
    pub fn add_arg_from<'py, T>(mut self, py: Python<'py>, value: T) -> PyResult<Self>
    where
        T: IntoPyObject<'py>,
        <T as IntoPyObject<'py>>::Error: std::fmt::Debug,
        pyo3::PyErr: std::convert::From<<T as pyo3::IntoPyObject<'py>>::Error>,
    {
        let bound = value.into_pyobject(py)?;
        self.args.push(bound.into_any().unbind());
        Ok(self)
    }

    /// Add an argument from a serializable value, converting it to a Python object
    /// using `pythonize`.
    ///
    /// This is useful for complex types or when you want to pass JSON data.
    ///
    /// # Example
    /// ```no_run
    /// use comfyui_v3::schema::NodeOutput;
    /// use serde_json::json;
    ///
    /// let out = NodeOutput::new()
    ///     .add_arg_serializable(py, &json!({"key": "value"}))
    ///     .unwrap();
    /// ```
    pub fn add_arg_serializable<'py, T: Serialize>(
        mut self,
        py: Python<'py>,
        value: &T,
    ) -> PyResult<Self> {
        let py_obj = pythonize(py, value)?;
        self.args.push(py_obj.unbind());
        Ok(self)
    }

    /// Set the `ui` field with an arbitrary Python object.
    pub fn with_ui(mut self, ui: Py<PyAny>) -> Self {
        self.ui = Some(ui);
        self
    }

    /// Set the `expand` field (requires `enable_expand` on the schema).
    pub fn with_expand(mut self, expand: Py<PyAny>) -> Self {
        self.expand = Some(expand);
        self
    }

    /// Block execution with the given error message.
    pub fn with_block_execution(mut self, msg: impl Into<String>) -> Self {
        self.block_execution = Some(msg.into());
        self
    }

    /// Convert this `NodeOutput` into a real Python `io.NodeOutput` object.
    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Use the same approach as test_io_node_output which works
        let comfy_api = py.import("comfy_api.latest")?;
        let io = comfy_api.getattr("io")?;

        // Debug: print all attributes of io module
        // println!("io module dir: {:?}", io.dir()?);

        // Get NodeOutput class - same as test_io_node_output
        let cls = io.getattr("NodeOutput")?;

        // Debug: print cls info
        // println!("cls type: {}", cls.get_type().repr()?);
        // println!("cls repr: {}", cls.repr()?);
        // println!("cls is callable: {}", cls.is_callable());
        // println!("cls is None: {}", cls.is_none());

        // Debug: check if cls is actually callable (a class)
        if !cls.is_callable() {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "io.NodeOutput is not callable! type: {}, repr: {}",
                cls.get_type().repr()?,
                cls.repr()?
            )));
        }

        // Build args tuple from self.args
        let py_args = PyTuple::new(py, &self.args)?;

        // Build kwargs dict
        let kwargs = PyDict::new(py);
        if let Some(ref ui) = self.ui {
            kwargs.set_item("ui", ui)?;
        }
        if let Some(ref expand) = self.expand {
            kwargs.set_item("expand", expand)?;
        }
        if let Some(ref msg) = self.block_execution {
            kwargs.set_item("block_execution", msg)?;
        }

        // Call NodeOutput(*args, **kwargs) - same as test_io_node_output
        if kwargs.is_empty() {
            cls.call1(py_args)
        } else {
            cls.call(py_args, Some(&kwargs))
        }
    }
}
