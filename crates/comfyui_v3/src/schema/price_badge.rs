//! Extra v3 schema types: PriceBadge, Hidden, etc.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PriceBadgeDepends
// ---------------------------------------------------------------------------

/// Dependencies for evaluating a PriceBadge expression.
///
/// Mirrors Python `comfy_api.latest.io.PriceBadgeDepends`.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PriceBadgeDepends {
    /// List of input IDs whose values are required by the expression.
    pub inputs: Vec<String>,
    /// List of external resource IDs whose values are required by the expression.
    pub resources: Vec<String>,
    /// List of cached value keys required by the expression.
    pub cached: Vec<String>,
}

impl PriceBadgeDepends {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_inputs(mut self, inputs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.inputs = inputs.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_resources(
        mut self,
        resources: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.resources = resources.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_cached(mut self, cached: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.cached = cached.into_iter().map(Into::into).collect();
        self
    }

    /// Validate depends_on configuration.
    pub fn validate(&self) -> PyResult<()> {
        Ok(())
    }

    /// Convert to Python dict matching `PriceBadgeDepends.as_dict()`.
    pub fn to_py_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("inputs", &self.inputs)?;
        dict.set_item("resources", &self.resources)?;
        dict.set_item("cached", &self.cached)?;
        Ok(dict)
    }
}

// ---------------------------------------------------------------------------
// PriceBadge
// ---------------------------------------------------------------------------

/// Optional client-evaluated pricing badge declaration for a node.
///
/// Mirrors Python `comfy_api.latest.io.PriceBadge`.
#[pyclass(from_py_object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceBadge {
    /// JSONata expression used to evaluate the price dynamically.
    pub expr: String,
    /// Dependencies required for evaluating the expression.
    pub depends_on: PriceBadgeDepends,
    /// Expression engine used to evaluate the badge. Only "jsonata" is supported.
    pub engine: String,
}

impl Default for PriceBadge {
    fn default() -> Self {
        Self {
            expr: String::new(),
            depends_on: PriceBadgeDepends::default(),
            engine: "jsonata".to_string(),
        }
    }
}

impl PriceBadge {
    /// Create a new PriceBadge with the given expression.
    pub fn new(expr: impl Into<String>) -> Self {
        Self {
            expr: expr.into(),
            ..Default::default()
        }
    }

    pub fn with_depends_on(mut self, depends_on: PriceBadgeDepends) -> Self {
        self.depends_on = depends_on;
        self
    }

    pub fn with_engine(mut self, engine: impl Into<String>) -> Self {
        self.engine = engine.into();
        self
    }

    /// Validate the PriceBadge configuration.
    ///
    /// Corresponds to Python `PriceBadge.validate()`.
    pub fn validate(&self) -> PyResult<()> {
        if self.engine != "jsonata" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported PriceBadge.engine '{}'. Only 'jsonata' is supported.",
                self.engine
            )));
        }
        if self.expr.trim().is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "PriceBadge.expr must be a non-empty string.",
            ));
        }
        self.depends_on.validate()?;
        Ok(())
    }

    /// Convert to a Python `io.PriceBadge` object.
    ///
    /// Corresponds to Python `PriceBadge.as_dict()`.
    /// The `schema_inputs` parameter is kept for API parity but is not used
    /// in the current implementation (matching the Python fallback behavior).
    pub fn to_py_obj<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let io = py.import("comfy_api.latest")?.getattr("io")?;
        let cls = io.getattr("PriceBadge")?;

        let kwargs = PyDict::new(py);
        kwargs.set_item("expr", &self.expr)?;
        kwargs.set_item("depends_on", self.depends_on.to_py_dict(py)?)?;
        kwargs.set_item("engine", &self.engine)?;

        cls.call((), Some(&kwargs))
    }
}
