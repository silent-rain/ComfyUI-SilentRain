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
            "如果图像中有人物/角色，你必须将其称为{姓名}。",
        ),
        (
            "exclude_people_info",
            "请勿包含关于无法改变的人物/角色信息（如种族、性别等），但仍需包含可改变的属性（如发型）。",
        ),
        ("include_lighting", "包含关于光照的信息。"),
        ("include_camera_angle", "包含关于拍摄角度的信息。"),
        ("include_watermark_info", "包含关于是否存在水印的信息。"),
        (
            "include_JPEG_artifacts",
            "包含关于是否存在JPEG压缩痕迹的信息。",
        ),
        (
            "include_exif",
            "如果是照片，你必须包含可能使用的相机型号以及光圈、快门速度、ISO等细节信息。",
        ),
        ("exclude_sexual", "请勿包含任何性相关内容；保持内容健康。"),
        ("exclude_image_resolution", "请勿提及图像的分辨率。"),
        (
            "include_aesthetic_quality",
            "你必须包含对图像主观美学质量的评价，从低到极高。",
        ),
        (
            "include_composition_style",
            "包含关于图像构图风格的信息，如引导线、三分法或对称构图。",
        ),
        ("exclude_text", "请勿提及图像中的任何文字。"),
        (
            "specify_depth_field",
            "明确景深，并说明背景是清晰还是模糊。",
        ),
        (
            "specify_lighting_sources",
            "如果适用，请提及可能使用的人造或自然光源。",
        ),
        (
            "do_not_use_ambiguous_language",
            "请勿使用任何模糊不清的语言。",
        ),
        (
            "include_nsfw_rating",
            "包含该图像是安全的、具有暗示性的还是不适合工作场合的评级。",
        ),
        (
            "only_describe_most_important_elements",
            "只描述图像中最重要的元素。",
        ),
        (
            "do_not_include_artist_name_or_title",
            "如果是艺术作品，请勿包含艺术家姓名或作品标题。",
        ),
        (
            "identify_image_orientation",
            "说明图像的朝向（人像、风景或方形）以及明显可辨的宽高比。",
        ),
        (
            "use_vulgar_slang_and_profanity",
            r#"使用粗俗的俚语和脏话，例如（但不限于）“他妈的”、“婊子”、“鸡巴”等。"#,
        ),
        (
            "do_not_use_polite_euphemisms",
            "请勿使用礼貌的委婉说法——应采用直率、随意的措辞。",
        ),
        (
            "include_character_age",
            "在适用时，包含任何人物/角色的年龄信息。",
        ),
        (
            "include_camera_shot_type",
            "说明图像呈现的是特写、近景、中近景、中景、牛仔镜头、中远景、远景还是大远景。",
        ),
        ("exclude_mood_feeling", "请勿提及图像的情绪/感觉等。"),
        (
            "include_camera_vantage_height",
            "明确指定视点高度（眼平、低角度仰视、鸟瞰、无人机视角、屋顶视角等）。",
        ),
        ("mention_watermark_explicitly", "如果有水印，你必须提及它。"),
        (
            "avoid_meta_descriptive_phrases",
            r#"你的回答将用于文生图模型，因此请避免使用无用的元描述短语，如“这张图片展示…”、“你正在看…”等。"#,
        ),
    ])
}

/// JoyCaption Extra Options
#[pyclass(subclass)]
pub struct JoyCaptionExtraOptionsZh {}

impl PromptServer for JoyCaptionExtraOptionsZh {}

#[pymethods]
impl JoyCaptionExtraOptionsZh {
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

impl JoyCaptionExtraOptionsZh {
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
