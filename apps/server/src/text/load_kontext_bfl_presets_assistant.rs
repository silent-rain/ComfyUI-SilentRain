//! Kontext Presets 预设助手 - BFL 官方版
//!
//! 通过官方的API接口返回的信息提取的提示词。
//! 官方: https://playground.bfl.ai/image/edit
//! 官方提示词指南: https://docs.bfl.ai/guides/prompting_guide_kontext_i2i
//!

use log::error;
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use serde::{Deserialize, Serialize};

use crate::{
    core::category::CATEGORY_TEXT,
    error::Error,
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_INT, NODE_STRING},
    },
};

const KONTEXT_PRESETS_DATA: &str =
    include_str!("../../resources/kontext_bfl_presets_assistant.json");

/// 预设内容
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Preset {
    name: String,
    description: String,
    brief: String,
    fallback_prompts: Vec<String>,
}

/// Kontext 预设助手
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PresetAssistant {
    prefix: String,
    presets: Vec<Preset>,
    suffix: String,
}

/// Kontext Presets 预设助手 - BFL 官方版
#[pyclass(subclass)]
pub struct LoadKontextBflPresetsAssistant {}

impl PromptServer for LoadKontextBflPresetsAssistant {}

#[pymethods]
impl LoadKontextBflPresetsAssistant {
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
    fn return_types() -> (&'static str, &'static str, &'static str) {
        (NODE_STRING, NODE_STRING, NODE_STRING)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str, &'static str) {
        ("prompt", "description", "fallback_prompts")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool, bool) {
        (false, false, true)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_TEXT;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Kontext Presets Assistant.
        Pre set prompt words extracted through the official Kontext interface."
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
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

                // {{NUM_PROMPTS}}
                required.set_item(
                    "num_prompts",
                    (NODE_INT, {
                        let pattern = PyDict::new(py);
                        pattern.set_item("default", 1)?;
                        pattern.set_item("tooltip", "Number of prompts.")?;
                        pattern
                    }),
                )?;

                // {{SUBJECT}}
                required.set_item(
                    "subject",
                    (NODE_STRING, {
                        let subject = PyDict::new(py);
                        subject.set_item("default", "")?;
                        subject.set_item(
                            "tooltip",
                            "Subject of the prompt. Only Zoom/Remove anything",
                        )?;
                        subject
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
        preset: &str,
        num_prompts: usize,
        subject: &str,
    ) -> PyResult<(String, String, Vec<String>)> {
        let results = self.get_preset(preset, num_prompts, subject);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("LoadKontextBflPresetsAssistant error, {e}");
                if let Err(e) = self.send_error(
                    py,
                    "LoadKontextBflPresetsAssistant".to_string(),
                    e.to_string(),
                ) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LoadKontextBflPresetsAssistant {
    /// 获取预设
    fn get_preset(
        &self,
        preset_name: &str,
        num_prompts: usize,
        subject: &str,
    ) -> Result<(String, String, Vec<String>), Error> {
        let preset_assistant: PresetAssistant = serde_json::from_str(KONTEXT_PRESETS_DATA)?;
        for preset in preset_assistant.presets {
            if preset.name != preset_name {
                continue;
            }

            let prompt =
                preset_assistant.prefix + "\n" + &preset.brief + "\n" + &preset_assistant.suffix;

            let prompt = prompt
                .replace("{{NUM_PROMPTS}}", &num_prompts.to_string())
                .replace("{{SUBJECT}}", subject);

            let description = preset.description;
            let fallback_prompts = preset.fallback_prompts;

            return Ok((prompt, description, fallback_prompts));
        }

        Ok(("".to_string(), "".to_string(), Vec::new()))
    }
}
