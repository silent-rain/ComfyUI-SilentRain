//! Flux Kontext Inpainting Conditioning
//!
//! 引用: https://github.com/ZenAI-Vietnam/ComfyUI-Kontext-Inpainting
//!

use candle_core::pickle::Object;
use log::{error, info};
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyList, PyType},
    Bound, Py, PyAny, PyErr, PyObject, PyResult, Python,
};
use serde::{Deserialize, Serialize};

use crate::{
    core::category::CATEGORY_CONDITIONING,
    error::Error,
    wrapper::comfyui::{
        types::{NODE_BOOLEAN, NODE_CONDITIONING, NODE_IMAGE, NODE_LATENT, NODE_MASK, NODE_VAE},
        PromptServer,
    },
};

/// Flux Kontext Inpainting Conditioning
#[pyclass(subclass)]
pub struct FluxKontextInpaintingConditioning {}

impl PromptServer for FluxKontextInpaintingConditioning {}

#[pymethods]
impl FluxKontextInpaintingConditioning {
    #[new]
    fn new() -> Self {
        Self {}
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str) {
        (NODE_CONDITIONING, NODE_LATENT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("conditioning", "latent")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_CONDITIONING;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Kontext Inpainting"
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);

                required.set_item(
                    "conditioning",
                    (NODE_CONDITIONING, {
                        let conditioning = PyDict::new(py);
                        conditioning
                    }),
                )?;
                required.set_item(
                    "vae",
                    (NODE_VAE, {
                        let vae = PyDict::new(py);
                        vae
                    }),
                )?;
                required.set_item(
                    "pixels",
                    (NODE_IMAGE, {
                        let pixels = PyDict::new(py);
                        pixels
                    }),
                )?;
                required.set_item(
                    "mask",
                    (NODE_MASK, {
                        let mask = PyDict::new(py);
                        mask
                    }),
                )?;
                required.set_item(
                    "noise_mask",
                    (NODE_BOOLEAN, {
                        let noise_mask = PyDict::new(py);
                        noise_mask.set_item("default", true)?;
                        noise_mask.set_item(
                            "tooltip",
                            "Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model."
                            )?;
                        noise_mask
                    }),
                )?;

                required
            })?;
            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        conditioning: Bound<'py, PyAny>,
        vae: Bound<'py, PyAny>,
        pixels: Bound<'py, PyAny>,
        mask: Bound<'py, PyAny>,
        noise_mask: bool,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        info!("kontextInpaint: {:#?}", conditioning);
        let results = self.encode(py, conditioning, vae, pixels, mask, noise_mask);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("kontextInpaint error, {e}");
                if let Err(e) = self.send_error(py, "kontextInpaint".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl FluxKontextInpaintingConditioning {
    /// Encode the conditioning and pixels into a latent vector
    fn encode<'py>(
        &self,
        py: Python<'py>,
        conditioning: Bound<'py, PyAny>,
        vae: Bound<'py, PyAny>,
        pixels: Bound<'py, PyAny>,
        mask: Bound<'py, PyAny>,
        noise_mask: bool,
    ) -> Result<(Bound<'py, PyAny>, Bound<'py, PyAny>), Error> {
        let x = PyList::empty(py).into_any();
        let y = PyList::empty(py).into_any();
        Ok((x, y))
    }
}
