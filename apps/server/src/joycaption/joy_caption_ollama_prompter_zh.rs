//! JoyCaption Ollama Prompter
//!
//! rust 语言的实现
//! 引用: https://github.com/judian17/ComfyUI-JoyCaption-beta-one-hf-llava-Prompt_node

use std::{collections::HashMap, vec};

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
    wrapper::comfyui::{PromptServer, types::NODE_STRING},
};

/// caption length prompts
///
/// - Very short: 25
/// - Short: 50
/// - Medium Length: 100
/// - Long: 150
/// - Very Long: 200
pub fn caption_length_map() -> HashMap<&'static str, &'static str> {
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
            "描述性",
            vec![
                "为这张图片写一段详细的描述。",
                "为这张图片写一段详细的描述，字数在{word_count}字以内。",
                "为这张图片写一段{length}的详细描述。",
            ],
        ),
        (
            "描述性 (非正式)",
            vec![
                "以非正式的口吻为这张图片写一段描述性文字说明。",
                "以非正式的口吻为这张图片写一段描述性文字说明，字数在{word_count}字以内。",
                "以非正式的口吻为这张图片写一段{length}的描述性文字说明。",
            ],
        ),
        (
            "直截了当",
            vec![
                "为这张图片写一段直截了当的文字说明。开头说明主要拍摄对象和媒介。用自信、确定的语言提及关键元素——人物、物体、场景。关注颜色、形状、纹理、空间关系等具体细节。展示元素的互动方式。省略情绪和推测性措辞。如果存在文字，请准确引用。注意任何水印、签名或压缩痕迹。切勿提及不存在的内容、分辨率或不可观察的细节。变化句式结构，保持描述简洁，不要以“这张图片是…”或类似短语开头。",
                "在{word_count}字以内，为这张图片写一段直截了当的文字说明。开头说明主要拍摄对象和媒介。用自信、确定的语言提及关键元素——人物、物体、场景。关注颜色、形状、纹理、空间关系等具体细节。展示元素的互动方式。省略情绪和推测性措辞。如果存在文字，请准确引用。注意任何水印、签名或压缩痕迹。切勿提及不存在的内容、分辨率或不可观察的细节。变化句式结构，保持描述简洁，不要以“这张图片是…”或类似短语开头。",
                "为这张图片写一段{length}的、直截了当的文字说明。开头说明主要拍摄对象和媒介。用自信、确定的语言提及关键元素——人物、物体、场景。关注颜色、形状、纹理、空间关系等具体细节。展示元素的互动方式。省略情绪和推测性措辞。如果存在文字，请准确引用。注意任何水印、签名或压缩痕迹。切勿提及不存在的内容、分辨率或不可观察的细节。变化句式结构，保持描述简洁，不要以“这张图片是…”或类似短语开头。",
            ],
        ),
        (
            "Stable Diffusion 提示词",
            vec![
                "生成一个与真实的Stable Diffusion提示词无异的Stable Diffusion提示词。",
                "生成一个与真实的Stable Diffusion提示词无异的Stable Diffusion提示词，字数在{word_count}字以内。",
                "生成一个{length}的、与真实的Stable Diffusion提示词无异的Stable Diffusion提示词。",
            ],
        ),
        (
            "MidJourney 提示词",
            vec![
                "为这张图片写一个MidJourney提示词。",
                "为这张图片写一个MidJourney提示词，字数在{word_count}字以内。",
                "为这张图片写一个{length}的MidJourney提示词。",
            ],
        ),
        (
            "Danbooru 标签列表",
            vec![
                "仅生成逗号分隔的Danbooru标签 (小写_下划线)。严格顺序：`artist:`, `copyright:`, `character:`, `meta:`，然后是常规标签。包含数量(1girl)、外貌、服装、配饰、姿势、表情、动作、背景。使用精确的Danbooru语法。不添加额外文本。",
                "仅生成逗号分隔的Danbooru标签 (小写_下划线)。严格顺序：`artist:`, `copyright:`, `character:`, `meta:`，然后是常规标签。包含数量(1girl)、外貌、服装、配饰、姿势、表情、动作、背景。使用精确的Danbooru语法。不添加额外文本。字数在{word_count}字以内。",
                "仅生成逗号分隔的Danbooru标签 (小写_下划线)。严格顺序：`artist:`, `copyright:`, `character:`, `meta:`，然后是常规标签。包含数量(1girl)、外貌、服装、配饰、姿势、表情、动作、背景。使用精确的Danbooru语法。不添加额外文本。长度为{length}。",
            ],
        ),
        (
            "e621 标签列表",
            vec![
                "为这张图片按字母顺序写一个逗号分隔的e621标签列表。如果有的话，以带‘artist:’、‘copyright:’、‘character:’、‘species:’、‘meta:’、‘lore:’前缀的艺术家、版权、角色、物种、元数据和设定标签开头。然后是所有常规标签。",
                "为这张图片按字母顺序写一个逗号分隔的e621标签列表。如果有的话，以带‘artist:’、‘copyright:’、‘character:’、‘species:’、‘meta:’、‘lore:’前缀的艺术家、版权、角色、物种、元数据和设定标签开头。然后是所有常规标签。字数控制在{word_count}字以内。",
                "为这张图片按字母顺序写一个{length}的、逗号分隔的e621标签列表。如果有的话，以带‘artist:’、‘copyright:’、‘character:’、‘species:’、‘meta:’、‘lore:’前缀的艺术家、版权、角色、物种、元数据和设定标签开头。然后是所有常规标签。",
            ],
        ),
        (
            "Rule34 标签列表",
            vec![
                "为这张图片按字母顺序写一个逗号分隔的rule34标签列表。如果有的话，以带‘artist:’、‘copyright:’、‘character:’、‘meta:’前缀的艺术家、版权、角色、元数据标签开头。然后是所有常规标签。",
                "为这张图片按字母顺序写一个逗号分隔的rule34标签列表。如果有的话，以带‘artist:’、‘copyright:’、‘character:’、‘meta:’前缀的艺术家、版权、角色、元数据标签开头。然后是所有常规标签。字数控制在{word_count}字以内。",
                "为这张图片按字母顺序写一个{length}的、逗号分隔的rule34标签列表。如果有的话，以带‘artist:’、‘copyright:’、‘character:’、‘meta:’前缀的艺术家、版权、角色、元数据标签开头。然后是所有常规标签。",
            ],
        ),
        (
            "Booru式 标签列表",
            vec![
                "为这张图片写一个Booru式标签列表。",
                "在{word_count}字以内，为这张图片写一个Booru式标签列表。",
                "为这张图片写一个{length}的Booru式标签列表。",
            ],
        ),
        (
            "艺术评论",
            vec![
                "像艺术评论家一样分析这张图片，包括其构图、风格、象征意义、色彩运用、光线、可能所属的艺术运动等信息。",
                "像艺术评论家一样分析这张图片，包括其构图、风格、象征意义、色彩运用、光线、可能所属的艺术运动等信息。字数控制在{word_count}字以内。",
                "像艺术评论家一样分析这张图片，包括其构图、风格、象征意义、色彩运用、光线、可能所属的艺术运动等信息。保持{length}的篇幅。",
            ],
        ),
        (
            "产品列表",
            vec![
                "为这张图片写一段文字说明，仿照产品列表的描述方式。",
                "为这张图片写一段文字说明，仿照产品列表的描述方式，字数在{word_count}字以内。",
                "为这张图片写一段{length}的文字说明，仿照产品列表的描述方式。",
            ],
        ),
        (
            "社交媒体帖子",
            vec![
                "为这张图片写一段文字说明，仿佛它将用于社交媒体帖子。",
                "为这张图片写一段文字说明，仿佛它将用于社交媒体帖子，将说明文字限制在{word_count}字以内。",
                "为这张图片写一段{length}的文字说明，仿佛它将用于社交媒体帖子。",
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
pub fn system_prompt(caption_type: &str) -> String {
    let system_prompts;
    let caption_type = caption_type.to_lowercase();
    if caption_type.contains("tag list") {
        if caption_type.contains("danbooru") {
            system_prompts = vec![
                "你是一个Danbooru标签生成器。只生成用下划线连接的小写、逗号分隔的标签。",
                "遵循此精确顺序：artist:, copyright:, character:, meta:，然后是常规标签。",
                "包含精确的数量（1girl, 2boys）、外貌、服装、配饰、姿势、表情、动作和背景的具体细节。",
                "使用精确的Danbooru语法。不要解释性文字或自然语言描述。",
            ];
        } else if caption_type.contains("e621") {
            system_prompts = vec![
                "你是一个e621标签生成器。只生成按字母顺序排列的、逗号分隔的标签。",
                "遵循此精确顺序：artist:, copyright:, character:, species:, meta:, lore:，然后是常规标签。",
                "标签格式要极其精确。不要解释性文字。",
            ];
        } else if caption_type.contains("rule34") {
            system_prompts = vec![
                "你是一个Rule34标签生成器。只生成按字母顺序排列的、逗号分隔的标签。",
                "遵循此精确顺序：artist:, copyright:, character:, meta:，然后是常规标签。",
                "要极其精确并使用正确的标签语法。不要解释性文字。",
            ];
        } else {
            system_prompts = vec![
                "你是一个booru式标签生成器。只生成逗号分隔的描述性标签。",
                "关注视觉元素、角色特征、服装、姿势、场景和动作。",
                "使用一致的格式，多词标签用下划线连接。不要解释性文字。",
            ];
        }
    } else if caption_type == "stable diffusion prompt" {
        system_prompts = vec![
            "你是一位Stable Diffusion提示词工程师。创建适用于Stable Diffusion的提示词。",
            "关注视觉细节、艺术风格、拍摄角度、光线和构图。",
            "使用常见的SD语法和关键词。用逗号分隔关键元素。",
            "严格遵守指定的长度限制。",
        ];
    } else if caption_type == "midjourney" {
        system_prompts = vec![
            "你是一位MidJourney提示词专家。创建为MidJourney优化的提示词。",
            "使用MidJourney特定的语法和参数风格。",
            "包含艺术风格、摄像机视角、光线和构图。",
            "严格遵守指定的长度限制。",
        ];
    } else if caption_type == "straightforward" {
        system_prompts = vec![
            "你是一个精确的图像描述者。专注于具体的、可观察的细节。",
            "以主要拍摄对象和媒介开头。使用自信的语言描述关键元素。",
            "专注于颜色、形状、纹理和空间关系。",
            "省略推测和情绪。如有文字，请准确引用。注意水印等技术细节。",
            "严格遵守字数限制。切勿使用“这张图片展示了...”之类的短语。",
        ];
    } else {
        system_prompts = vec![
            "你是一个自适应的图像描述助手。",
            "调整你的风格以精确匹配请求的文字说明类型。",
            "严格遵守指定的字数限制和格式要求。",
            "做到精确、清晰，并完全遵循给定的风格指南。",
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
pub struct JoyCaptionOllamaPrompterZh {}

impl PromptServer for JoyCaptionOllamaPrompterZh {}

#[pymethods]
impl JoyCaptionOllamaPrompterZh {
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
        Python::attach(|py| {
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

impl JoyCaptionOllamaPrompterZh {
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
