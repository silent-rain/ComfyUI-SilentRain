//! JoyCaption Ollama Prompter
//!
//! rust 语言的实现
//! 引用: https://github.com/judian17/ComfyUI-JoyCaption-beta-one-hf-llava-Prompt_node

use std::{collections::HashMap, vec};

use log::error;
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyErr, PyResult, Python,
};

use crate::{
    core::category::CATEGORY_JOY_CAPTION,
    error::Error,
    wrapper::comfyui::{types::NODE_STRING, PromptServer},
};

/// caption length prompts
///
/// - Very short: 25
/// - Short: 50
/// - Medium Length: 100
/// - Long: 150
/// - Very Long: 200
fn caption_length_map() -> HashMap<&'static str, &'static str> {
    HashMap::from([
        ("very short", "25"),
        ("short", "50"),
        ("medium-length", "100"),
        ("long", "150"),
        ("very long", "200"),
    ])
}

/// Constants for caption generation, copied from the original JoyCaption GGUF node
pub fn caption_type_map() -> HashMap<&'static str, Vec<&'static str>> {
    HashMap::from([
    (
    "Descriptive", vec![
            "Write a detailed description for this image.",
            "Write a detailed description for this image in {word_count} words or less.",
            "Write a {length} detailed description for this image.",
        ],
    ),
    (
    "Descriptive (Casual)", vec![
            "Write a descriptive caption for this image in a casual tone.",
            "Write a descriptive caption for this image in a casual tone within {word_count} words.",
            "Write a {length} descriptive caption for this image in a casual tone.",
        ],
    ),
    ("Straightforward", vec![
            "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
            "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
            "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        ],
    ),
    ("Stable Diffusion Prompt", vec![
            "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
            "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
            "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        ]
    ),
    (
    "MidJourney", vec![
            "Write a MidJourney prompt for this image.",
            "Write a MidJourney prompt for this image within {word_count} words.",
            "Write a {length} MidJourney prompt for this image.",
        ],
    ),
    (
        "Danbooru tag list", vec![
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
            "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
        ],
    ),
    (
        "e621 tag list", vec![
            "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
            "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
            "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        ],
    ),
    (
        "Rule34 tag list", vec![
            "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
            "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
            "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
        ],
    ),
    (
        "Booru-like tag list", vec![
            "Write a list of Booru-like tags for this image.",
            "Write a list of Booru-like tags for this image within {word_count} words.",
            "Write a {length} list of Booru-like tags for this image.",
        ],
    ),
    (
    "Art Critic", vec![
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
            "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
        ],
    ),
    (
    "Product Listing", vec![
            "Write a caption for this image as though it were a product listing.",
            "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
            "Write a {length} caption for this image as though it were a product listing.",
        ],
    ),
    (
        "Social Media Post", vec![
            "Write a caption for this image as if it were being used for a social media post.",
            "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
            "Write a {length} caption for this image as if it were being used for a social media post.",
        ],
    ),
    ])
}

pub fn caption_length_choices() -> Vec<String> {
    let mut choices: Vec<String> = [
        "any",
        "very short",
        "short",
        "medium-length",
        "long",
        "very long",
    ]
    .iter()
    .map(|v| v.to_string())
    .collect();

    let numbers: Vec<String> = (20..=261).step_by(10).map(|i| i.to_string()).collect();

    choices.extend(numbers);

    choices
}

/// Select system prompt based on caption type
fn system_prompt(caption_type: &str) -> String {
    let system_prompts;
    let caption_type = caption_type.to_lowercase();
    if caption_type.contains("tag list") {
        if caption_type.contains("danbooru") {
            system_prompts = vec![
                "You are a Danbooru tag generator. Generate ONLY comma-separated tags in lowercase with underscores.",
                "Follow this exact order: artist:, copyright:, character:, meta:, then general tags.",
                "Include precise counts (1girl, 2boys), specific details about appearance, clothing, accessories, pose, expression, actions, and background.",
                "Use EXACT Danbooru syntax. NO explanatory text or natural language.",
            ];
        } else if caption_type.contains("e621") {
            system_prompts = vec![
                "You are an e621 tag generator. Generate ONLY comma-separated tags in alphabetical order.",
                "Follow this exact order: artist:, copyright:, character:, species:, meta:, lore:, then general tags.",
                "Be extremely precise with tag formatting. NO explanatory text.",
            ];
        } else if caption_type.contains("rule34") {
            system_prompts =  vec![
                "You are a Rule34 tag generator. Generate ONLY comma-separated tags in alphabetical order.",
                "Follow this exact order: artist:, copyright:, character:, meta:, then general tags.",
                "Be extremely precise and use proper tag syntax. NO explanatory text.",
            ];
        } else {
            system_prompts = vec![
                "You are a booru tag generator. Generate ONLY comma-separated descriptive tags.",
                "Focus on visual elements, character traits, clothing, pose, setting, and actions.",
                "Use consistent formatting with underscores for multi-word tags. NO explanatory text.",
            ];
        }
    } else if caption_type == "stable diffusion prompt" {
        system_prompts = vec![
                "You are a Stable Diffusion prompt engineer. Create prompts that work well with Stable Diffusion.",
                "Focus on visual details, artistic style, camera angles, lighting, and composition.",
                "Use common SD syntax and keywords. Separate key elements with commas.",
                "Keep strictly within the specified length limit.",
            ];
    } else if caption_type == "midjourney" {
        system_prompts = vec![
            "You are a MidJourney prompt expert. Create prompts optimized for MidJourney.",
            "Use MidJourney's specific syntax and parameter style.",
            "Include artistic style, camera view, lighting, and composition.",
            "Keep strictly within the specified length limit.",
        ];
    } else if caption_type == "straightforward" {
        system_prompts =vec![
                "You are a precise image descriptor. Focus on concrete, observable details.",
                "Begin with main subject and medium. Describe pivotal elements using confident language.",
                "Focus on color, shape, texture, and spatial relationships.",
                "Omit speculation and mood. Quote any text exactly. Note technical details like watermarks.",
                "Keep strictly within word limits. Never use phrases like 'This image shows...'",
            ];
    } else {
        system_prompts = vec![
            "You are an adaptive image description assistant.",
            "Adjust your style to match the requested caption type exactly.",
            "Strictly adhere to specified word limits and formatting requirements.",
            "Be precise, clear, and follow the given style guidelines exactly.",
        ];
    }

    system_prompts.join("\n")
}

/// Select user prompt based on caption type
pub fn user_prompt(
    caption_type: &str,
    caption_length: &str,
    extra_options: Option<Vec<String>>,
) -> String {
    let caption_type_map = caption_type_map();
    let prompt_templates = if let Some(prompt_templates) = caption_type_map.get(caption_type) {
        prompt_templates
    } else {
        let default_template_key = caption_type_map
            .keys()
            .next()
            .map_or("Descriptive (Casual)", |v| v);
        caption_type_map.get(default_template_key).unwrap()
    };

    let mut map_idx;
    if caption_length == "any" {
        map_idx = 0;
    } else if caption_length.parse::<i32>().is_ok() {
        map_idx = 1;
    } else {
        map_idx = 2;
    };

    if map_idx >= prompt_templates.len() as i32 {
        map_idx = 0;
    }

    let mut prompt = prompt_templates[map_idx as usize].to_string();
    if let Some(extra_options) = extra_options {
        prompt += &(" ".to_string() + " " + &extra_options.join(" "));
    }

    prompt = prompt
        .replace("{length}", caption_length)
        .replace("{word_count}", caption_length);

    prompt
}

/// JoyCaption Ollama Prompter
#[pyclass(subclass)]
pub struct JoyCaptionOllamaPrompter {}

impl PromptServer for JoyCaptionOllamaPrompter {}

#[pymethods]
impl JoyCaptionOllamaPrompter {
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
        ("system_prompt", "user_prompt")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() {}

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_JOY_CAPTION;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "JoyCaption is a set of text descriptions for Olama templates that can generate images using AI generated titles."
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

                let caption_type_map = caption_type_map();
                let caption_type_keys = caption_type_map.keys().cloned().collect::<Vec<_>>();

                let caption_lengths = caption_length_choices();

                required.set_item(
                    "caption_type",
                    (caption_type_keys, {
                        let caption_type = PyDict::new(py);
                        caption_type.set_item("default", "Descriptive (Casual)")?;
                        caption_type
                    }),
                )?;

                required.set_item(
                    "caption_length",
                    (caption_lengths, {
                        let caption_length = PyDict::new(py);
                        caption_length.set_item("default", "medium-length")?;
                        caption_length
                    }),
                )?;

                required
            })?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);
                optional.set_item(
                    "extra_options",
                    (NODE_STRING, {
                        let extra_options = PyDict::new(py);
                        extra_options.set_item("forceInput", true)?;
                        extra_options
                    }),
                )?;

                optional
            })?;
            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        caption_type: &str,
        caption_length: &str,
        extra_options: Option<Vec<String>>,
    ) -> PyResult<(String, String)> {
        let results = self.generate_prompts(caption_type, caption_length, extra_options);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("JoyCaptionOllamaPrompter error, {e}");
                if let Err(e) =
                    self.send_error(py, "JoyCaptionOllamaPrompter".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl JoyCaptionOllamaPrompter {
    /// Generate the user and system prompts
    fn generate_prompts(
        &self,
        caption_type: &str,
        caption_length: &str,
        extra_options: Option<Vec<String>>,
    ) -> Result<(String, String), Error> {
        let caption_length_map = caption_length_map();

        // system prompt
        let mut system_prompt = system_prompt(caption_type).to_string();
        // Add length enforcement to system prompt
        if caption_length.parse::<i32>().is_ok() {
            system_prompt +=
                format!("\nIMPORTANT: Your response MUST NOT exceed {caption_length} words.")
                    .as_str();
        } else if let Some(length) = caption_length_map.get(caption_length) {
            system_prompt +=
                format!("\nIMPORTANT: Keep your response approximately {length} words.").as_str();
        }

        // user prompt
        let user_prompt = user_prompt(caption_type, caption_length, extra_options);

        Ok((system_prompt, user_prompt))
    }
}
