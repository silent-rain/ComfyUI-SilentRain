//! Kontext Presets 常用预设

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

const KONTEXT_PRESETS_DATA: &str = include_str!("../../resources/kontext_presets.json");

/// 预设内容
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Preset {
    name: String,
    brief: String,
    brief_zh: String,
}

/// Kontext 综合预设
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PresetComprehensive {
    /// 样式转换
    style_conversions: Vec<Preset>,
    /// 保持特征
    preservations: Vec<Preset>,
    /// 其他
    other: Vec<Preset>,
}

/// Kontext Presets 常用预设
#[pyclass(subclass)]
pub struct LoadKontextPresets {}

impl PromptServer for LoadKontextPresets {}

#[pymethods]
impl LoadKontextPresets {
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
        "Kontext Presets Assistant."
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

                let presets: PresetComprehensive =
                    serde_json::from_str(KONTEXT_PRESETS_DATA).unwrap();
                let mut style_conversions = presets
                    .style_conversions
                    .iter()
                    .map(|preset| preset.name.clone())
                    .collect::<Vec<_>>();
                style_conversions.insert(0, "none".to_string());

                let mut preservations = presets
                    .preservations
                    .iter()
                    .map(|preset| preset.name.clone())
                    .collect::<Vec<_>>();
                preservations.insert(0, "none".to_string());

                let mut other = presets
                    .other
                    .iter()
                    .map(|preset| preset.name.clone())
                    .collect::<Vec<_>>();
                other.insert(0, "none".to_string());

                required.set_item(
                    "style_conversion",
                    (style_conversions, {
                        let style_conversions = PyDict::new(py);
                        style_conversions.set_item("default", "none")?;
                        style_conversions
                            .set_item("tooltip", "List of prompt words for style conversion.")?;
                        style_conversions
                    }),
                )?;
                required.set_item(
                    "preservation",
                    (preservations, {
                        let preservation = PyDict::new(py);
                        preservation.set_item("default", "none")?;
                        preservation.set_item(
                            "tooltip",
                            "List of prompt words that retain their original features.",
                        )?;
                        preservation
                    }),
                )?;
                required.set_item(
                    "other",
                    (other, {
                        let other = PyDict::new(py);
                        other.set_item("default", "none")?;
                        other.set_item(
                            "tooltip",
                            "List of prompt words related to specific functions.",
                        )?;
                        other
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
        style_conversion: &str,
        preservation: &str,
        other: &str,
    ) -> PyResult<(String, String)> {
        let results = self.get_preset(style_conversion, preservation, other);

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

impl LoadKontextPresets {
    /// 获取预设
    fn get_preset(
        &self,
        style_conversion: &str,
        preservation: &str,
        other: &str,
    ) -> Result<(String, String), Error> {
        let presets: PresetComprehensive = serde_json::from_str(KONTEXT_PRESETS_DATA)?;

        let mut prompts = Vec::new();
        let mut prompt_zhs = Vec::new();

        if style_conversion != "none" {
            let preset = presets
                .style_conversions
                .iter()
                .find(|preset| preset.name == style_conversion);
            if let Some(preset) = preset {
                prompts.push(preset.brief.clone());
                prompt_zhs.push(preset.brief_zh.clone());
            }
        }

        if preservation != "none" {
            let preset = presets
                .preservations
                .iter()
                .find(|preset| preset.name == preservation);
            if let Some(preset) = preset {
                prompts.push(preset.brief.clone());
                prompt_zhs.push(preset.brief_zh.clone());
            }
        }

        if other != "none" {
            let preset = presets.other.iter().find(|preset| preset.name == other);
            if let Some(preset) = preset {
                prompts.push(preset.brief.clone());
                prompt_zhs.push(preset.brief_zh.clone());
            }
        }

        // 提示词尾部以 "." or "," 结尾
        let prompts = prompts
            .into_iter()
            .map(|prompt| {
                if prompt.ends_with('.') || prompt.ends_with(',') {
                    prompt
                } else {
                    format!("{prompt}.")
                }
            })
            .collect::<Vec<_>>();
        let prompt_zhs = prompt_zhs
            .into_iter()
            .map(|prompt| {
                if prompt.ends_with('.') || prompt.ends_with(',') {
                    prompt
                } else {
                    format!("{prompt}.")
                }
            })
            .collect::<Vec<_>>();

        Ok((prompts.join("\n\n"), prompt_zhs.join("\n\n")))
    }
}
