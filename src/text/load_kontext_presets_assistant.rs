//! Kontext Presets 预设助手
//!
//!
//! 部分提示词由 AI 辅助生成, 未经过完全测试

use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyErr, PyResult, Python,
};
use serde::{Deserialize, Serialize};

use crate::{
    core::category::CATEGORY_TEXT,
    error::Error,
    wrapper::comfyui::{types::NODE_STRING, PromptServer},
};

const KONTEXT_PRESETS_DATA: &str = include_str!("../../resources/kontext_presets_assistant.json");

/// 预设内容
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Preset {
    name: String,
    brief: String,
    brief_zh: String,
}

/// Kontext 预设助手
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PresetAssistant {
    prefix: String,
    prefix_zh: String,
    presets: Vec<Preset>,
    suffix: String,
    suffix_zh: String,
}

/// Kontext Presets 预设助手
#[pyclass(subclass)]
pub struct LoadKontextPresetsAssistant {}

impl PromptServer for LoadKontextPresetsAssistant {}

#[pymethods]
impl LoadKontextPresetsAssistant {
    #[new]
    fn new() -> Self {
        Self {}
    }

    #[classattr]
    #[pyo3(name = "EXPERIMENTAL")]
    fn experimental() -> bool {
        true
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str) {
        (NODE_STRING, NODE_STRING)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("Prompt", "Prompt_Zh")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_TEXT;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Kontext Presets Assistant.
        Pre set prompt words extracted through the official Kontext interface.
        It also includes some AI generated prompt words that have not been fully validated."
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

                let preset_assistant: PresetAssistant =
                    serde_json::from_str(KONTEXT_PRESETS_DATA).unwrap();
                let preset_options = preset_assistant
                    .presets
                    .iter()
                    .map(|preset| preset.name.clone())
                    .collect::<Vec<_>>();

                required.set_item(
                    "preset",
                    (preset_options, {
                        let preset = PyDict::new(py);
                        preset.set_item("default", "")?;
                        preset.set_item("tooltip", "Kontext Presets preset prompt list.")?;
                        preset
                    }),
                )?;

                required
            })?;
            Ok(dict.into())
        })
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(&mut self, py: Python<'py>, preset: &str) -> PyResult<(String, String)> {
        let results = self.get_preset(preset);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("LoadKontextPresetsAssistant error, {e}");
                if let Err(e) =
                    self.send_error(py, "LoadKontextPresetsAssistant".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LoadKontextPresetsAssistant {
    /// 获取预设
    fn get_preset(&self, preset_name: &str) -> Result<(String, String), Error> {
        let preset_assistant: PresetAssistant = serde_json::from_str(KONTEXT_PRESETS_DATA)?;
        for preset in preset_assistant.presets {
            if preset.name == preset_name {
                let assistant = preset_assistant.prefix
                    + "\n"
                    + &preset.brief
                    + "\n"
                    + &preset_assistant.suffix;
                let assistant_zh = preset_assistant.prefix_zh
                    + "\n"
                    + &preset.brief_zh
                    + "\n"
                    + &preset_assistant.suffix_zh;

                return Ok((assistant, assistant_zh));
            }
        }

        Ok(("".to_string(), "".to_string()))
    }
}
