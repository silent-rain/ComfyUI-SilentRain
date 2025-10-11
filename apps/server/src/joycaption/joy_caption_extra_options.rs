//! JoyCaption Extra Options
//!
//! rust 语言的实现
//! 引用: https://github.com/judian17/ComfyUI-joycaption-beta-one-GGUF

use std::collections::HashMap;

use log::error;
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};

use crate::{
    core::category::CATEGORY_JOY_CAPTION,
    error::Error,
    wrapper::comfyui::{
        PromptServer,
        types::{NODE_BOOLEAN, NODE_STRING},
    },
};

const INPUT_NAMES: [&str; 27] = [
    "refer_character_name",
    "exclude_people_info",
    "include_lighting",
    "include_camera_angle",
    "include_watermark_info",
    "include_JPEG_artifacts",
    "include_exif",
    "exclude_sexual",
    "exclude_image_resolution",
    "include_aesthetic_quality",
    "include_composition_style",
    "exclude_text",
    "specify_depth_field",
    "specify_lighting_sources",
    "do_not_use_ambiguous_language",
    "include_nsfw_rating",
    "only_describe_most_important_elements",
    "do_not_include_artist_name_or_title",
    "identify_image_orientation",
    "use_vulgar_slang_and_profanity",
    "do_not_use_polite_euphemisms",
    "include_character_age",
    "include_camera_shot_type",
    "exclude_mood_feeling",
    "include_camera_vantage_height",
    "mention_watermark_explicitly",
    "avoid_meta_descriptive_phrases",
    // "character_name",
];

/// 额外选项
fn build_extra_options() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        (
            "refer_character_name",
            "If there is a person/character in the image you must refer to them as {name}.",
        ),
        (
            "exclude_people_info",
            "Do NOT include information about people/characters that cannot be changed (like ethnicity, gender, etc), but do still include changeable attributes (like hair style).",
        ),
        ("include_lighting", "Include information about lighting."),
        (
            "include_camera_angle",
            "Include information about camera angle.",
        ),
        (
            "include_watermark_info",
            "Include information about whether there is a watermark or not.",
        ),
        (
            "include_JPEG_artifacts",
            "Include information about whether there are JPEG artifacts or not.",
        ),
        (
            "include_exif",
            "If it is a photo you MUST include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
        ),
        (
            "exclude_sexual",
            "Do NOT include anything sexual; keep it PG.",
        ),
        (
            "exclude_image_resolution",
            "Do NOT mention the image's resolution.",
        ),
        (
            "include_aesthetic_quality",
            "You MUST include information about the subjective aesthetic quality of the image from low to very high.",
        ),
        (
            "include_composition_style",
            "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
        ),
        (
            "exclude_text",
            "Do NOT mention any text that is in the image.",
        ),
        (
            "specify_depth_field",
            "Specify the depth of field and whether the background is in focus or blurred.",
        ),
        (
            "specify_lighting_sources",
            "If applicable, mention the likely use of artificial or natural lighting sources.",
        ),
        (
            "do_not_use_ambiguous_language",
            "Do NOT use any ambiguous language.",
        ),
        (
            "include_nsfw_rating",
            "Include whether the image is sfw, suggestive, or nsfw.",
        ),
        (
            "only_describe_most_important_elements",
            "ONLY describe the most important elements of the image.",
        ),
        (
            "do_not_include_artist_name_or_title",
            "If it is a work of art, do not include the artist's name or the title of the work.",
        ),
        (
            "identify_image_orientation",
            "Identify the image orientation (portrait, landscape, or square) and aspect ratio if obvious.",
        ),
        (
            "use_vulgar_slang_and_profanity",
            r#"Use vulgar slang and profanity, such as (but not limited to) "fucking," "slut," "cock," etc."#,
        ),
        (
            "do_not_use_polite_euphemisms",
            "Do NOT use polite euphemisms—lean into blunt, casual phrasing.",
        ),
        (
            "include_character_age",
            "Include information about the ages of any people/characters when applicable.",
        ),
        (
            "include_camera_shot_type",
            "Mention whether the image depicts an extreme close-up, close-up, medium close-up, medium shot, cowboy shot, medium wide shot, wide shot, or extreme wide shot.",
        ),
        (
            "exclude_mood_feeling",
            "Do not mention the mood/feeling/etc of the image.",
        ),
        (
            "include_camera_vantage_height",
            "Explicitly specify the vantage height (eye-level, low-angle worm’s-eye, bird’s-eye, drone, rooftop, etc.).",
        ),
        (
            "mention_watermark_explicitly",
            "If there is a watermark, you must mention it.",
        ),
        (
            "avoid_meta_descriptive_phrases",
            r#"Your response will be used by a text-to-image model, so avoid useless meta phrases like “This image shows…”, "You are looking at..."), etc."#,
        ),
    ])
}

/// JoyCaption Extra Options
#[pyclass(subclass)]
pub struct JoyCaptionExtraOptions {}

impl PromptServer for JoyCaptionExtraOptions {}

#[pymethods]
impl JoyCaptionExtraOptions {
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
    fn return_types() -> (&'static str,) {
        (NODE_STRING,)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("extra_options",)
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() {}

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_JOY_CAPTION;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "JoyCaption Extra Options"
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

                for name in INPUT_NAMES {
                    required.set_item(
                        name,
                        (NODE_BOOLEAN, {
                            let attribute = PyDict::new(py);
                            attribute.set_item("default", false)?;
                            attribute
                        }),
                    )?;
                }

                required.set_item(
                    "character_name",
                    (NODE_STRING, {
                        let character_name = PyDict::new(py);
                        character_name.set_item("placeholder", "e.g., 'Skywalker'")?;
                        character_name.set_item("tooltip", "If there is a person/character in the image you must refer to them as {{character_name}}.")?;
                        character_name
                    }),
                )?;

                required
            })?;
            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute", signature = (character_name, **kwargs))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        character_name: &str,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<(Vec<String>,)> {
        let mut preset_dict = HashMap::new();
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.into_iter() {
                let key: String = key.extract()?;
                let value: bool = value.extract()?;
                preset_dict.insert(key, value);
            }
        }

        let results = self.get_extra_options(character_name, preset_dict);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("JoyCaptionExtraOptions error, {e}");
                if let Err(e) =
                    self.send_error(py, "JoyCaptionExtraOptions".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl JoyCaptionExtraOptions {
    /// 获取额外选项
    fn get_extra_options(
        &self,
        character_name: &str,
        preset_dict: HashMap<String, bool>,
    ) -> Result<(Vec<String>,), Error> {
        let mut extra_options = Vec::new();
        let extra_option_dict: HashMap<String, &'static str> = build_extra_options()
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        for (name, value) in preset_dict {
            if !value {
                continue;
            }
            if let Some(extra_option) = extra_option_dict.get(&name) {
                if character_name.is_empty() {
                    extra_options.push(extra_option.to_string().replace("{name}", "{NAME}"));
                } else {
                    extra_options.push(extra_option.to_string().replace("{name}", character_name));
                }
            }
        }

        Ok((extra_options,))
    }
}
