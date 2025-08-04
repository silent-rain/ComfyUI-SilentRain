//! Wan Presets 预设
//!
//! 提示词由 AI 辅助生成, 未经过完全测试

use std::collections::HashMap;

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

const PRESETS_DATA: &str = include_str!("../../resources/wan_presets.json");

/// 预设内容
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Preset {
    name: String,
    description: String,
    brief: String,
    prompts: Vec<String>,
}

/// 预设类别
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PresetCategory {
    identifier: String,
    name: String,
    presets: Vec<Preset>,
}

/// Wan Presets 预设
#[pyclass(subclass)]
pub struct LoadWanPresets {}

impl PromptServer for LoadWanPresets {}

#[pymethods]
impl LoadWanPresets {
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
    fn return_types() -> (&'static str, &'static str, &'static str) {
        (NODE_STRING, NODE_STRING, NODE_STRING)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str, &'static str) {
        ("prompt", "description", "prompts")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() -> (&'static str, &'static str, &'static str) {
        (
            "Preset prompt words",
            "Preset Description: When multiple presets are selected, a set of preset prompt words will be displayed here",
            "Preset content, when multiple presets are selected, the preset content collection is displayed here",
        )
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
        "Wan Presets."
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

                let preset_categorys: Vec<PresetCategory> =
                    serde_json::from_str(PRESETS_DATA).unwrap();

                for preset_category in preset_categorys {
                    let identifier = preset_category.identifier.clone();
                    let name = preset_category.name.clone();
                    let preset_options = preset_category
                        .presets
                        .iter()
                        .map(|preset| preset.name.clone())
                        .collect::<Vec<_>>();

                    required.set_item(
                        identifier,
                        (preset_options, {
                            let attribute = PyDict::new(py);
                            attribute.set_item("default", "none")?;
                            attribute.set_item("tooltip", name)?;
                            attribute
                        }),
                    )?;
                }

                required
            })?;
            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        camera_movement: &str,
        advanced_camera_movement: &str,
        shot_type: &str,
        camera_distance: &str,
        camera_level: &str,
        action: &str,
        composition: &str,
        emotion: &str,
        mood: &str,
        tone: &str,
        time_light: &str,
        light_class: &str,
        light_type: &str,
        weather: &str,
        background: &str,
        wedding: &str,
        wedding_photography: &str,
        painting_style: &str,
    ) -> PyResult<(String, String, Vec<String>)> {
        let preset_dict = HashMap::from([
            ("camera_movement", camera_movement),
            ("advanced_camera_movement", advanced_camera_movement),
            ("shot_type", shot_type),
            ("camera_distance", camera_distance),
            ("camera_level", camera_level),
            ("action", action),
            ("composition", composition),
            ("emotion", emotion),
            ("mood", mood),
            ("tone", tone),
            ("time_light", time_light),
            ("light_class", light_class),
            ("light_type", light_type),
            ("weather", weather),
            ("background", background),
            ("wedding", wedding),
            ("wedding_photography", wedding_photography),
            ("painting_style", painting_style),
        ]);
        let results = self.get_preset(preset_dict);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("LoadWanPresets error, {e}");
                if let Err(e) = self.send_error(py, "LoadWanPresets".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LoadWanPresets {
    /// 获取预设
    fn get_preset(
        &self,
        preset_dict: HashMap<&str, &str>,
    ) -> Result<(String, String, Vec<String>), Error> {
        let mut results = Vec::new();
        let preset_categorys: Vec<PresetCategory> = serde_json::from_str(PRESETS_DATA)?;
        for (identifier, preset_name) in preset_dict {
            if preset_name == "none" {
                continue;
            }
            for preset_category in &preset_categorys {
                if identifier != preset_category.identifier {
                    continue;
                }

                for preset in preset_category.presets.clone() {
                    if preset.name != preset_name {
                        continue;
                    }
                    results.push(preset);
                }
            }
        }

        if results.len() == 1 {
            let preset = results[0].clone();
            return Ok((preset.brief.clone(), preset.brief, preset.prompts));
        }

        let briefs = results
            .iter()
            .map(|preset| preset.brief.clone())
            .collect::<Vec<String>>();

        let descriptions = results
            .iter()
            .map(|preset| preset.description.clone())
            .collect::<Vec<String>>();

        let prompts = results
            .iter()
            .map(|preset| preset.prompts.join(";"))
            .collect::<Vec<String>>();

        Ok((briefs.join(","), descriptions.join("\n"), prompts))
    }
}
