//! Wan2.2 官方提示词选择器
//!
//! 通义万相AI生视频—使用指南zh: https://alidocs.dingtalk.com/i/nodes/jb9Y4gmKWrx9eo4dCql9LlbYJGXn6lpz
//! 通义万相AI生视频—使用指南en:https://alidocs.dingtalk.com/i/nodes/EpGBa2Lm8aZxe5myC99MelA2WgN7R35y

use std::collections::HashMap;

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
    wrapper::comfyui::{PromptServer, types::NODE_STRING},
};

/// 预设内容
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Preset {
    name: &'static str,
    brief: &'static str,
    brief_zh: &'static str,
    prompts: Vec<&'static str>,
    prompt_zhs: Vec<&'static str>,
}

/// Wan 2.2 提示词
fn build_prompt() -> Vec<Preset> {
    vec![
        // 影视级美学控制(Aesthetic Control)
        Preset {
            name: "light_source",
            brief: "Light Source",
            brief_zh: "光源类型",
            prompts: vec![
                "none",
                "Sunny Lighting",
                "Artificial Lighting",
                "Moonlighting",
                "Practical Lighting",
                "Firelighting",
                "Fluorescent Lighting",
                "Overcast Lighting",
                "Mixed Lighting",
                "Sunny Lighting",
            ],
            prompt_zhs: vec![
                "none",
                "日光",
                "人工光",
                "月光",
                "实用光",
                "火光",
                "荧光",
                "阴天光",
                "混合光",
                "晴天光",
            ],
        },
        Preset {
            name: "lighting_type",
            brief: "Lighting Type",
            brief_zh: "光线类型",
            prompts: vec![
                "none",
                "Soft Lighting",
                "Hard Lighting",
                "Top Lighting",
                "Side Lighting",
                "Medium Lens",
                "Underlighting",
                "Edge Lighting",
                "Silhouette Lighting",
                "Low Contrast Lighting",
                "High Contrast Lighting",
            ],
            prompt_zhs: vec![
                "none",
                "柔光",
                "硬光",
                "顶光",
                "侧光",
                "背光",
                "底光",
                "边缘光",
                "剪影",
                "低对比度",
                "高对比度",
            ],
        },
        Preset {
            name: "time_of_day",
            brief: "Time of Day",
            brief_zh: "时间段",
            prompts: vec![
                "none",
                "Sunrise Time",
                "Night Time",
                "Dusk Time",
                "Sunset Time",
                "Dawn Time",
                "Sunrise Time",
            ],
            prompt_zhs: vec!["none", "白天", "夜晚", "黄昏", "日落", "黎明", "日出"],
        },
        Preset {
            name: "shot_size",
            brief: "Shot Size",
            brief_zh: "景别",
            prompts: vec![
                "none",
                "Extreme Close-up Shot",
                "Close-up Shot",
                "Medium Shot",
                "Medium Close-up Shot",
                "Medium Wide Shot",
                "Wide Shot",
                "Wide-angle Lens",
            ],
            prompt_zhs: vec![
                "none",
                "特写",
                "近景",
                "中景",
                "中近景",
                "中全景",
                "全景",
                "广角",
            ],
        },
        Preset {
            name: "composition",
            brief: "Composition",
            brief_zh: "构图",
            prompts: vec![
                "none",
                "Center Composition",
                "Balanced Composition",
                "Left/right Weighted Composition",
                "Symmetrical Composition",
                "Short-side Composition",
            ],
            prompt_zhs: vec![
                "none",
                "中心构图",
                "平衡构图",
                "右/左侧重构图",
                "对称构图",
                "短边构图",
            ],
        },
        // 镜头(Lens)
        Preset {
            name: "focal_length",
            brief: "Focal Length",
            brief_zh: "镜头焦段",
            prompts: vec![
                "none",
                "Medium Lens",
                "Wide Lens",
                "Long-focus Lens",
                "Telephoto Lens",
                "Fisheye Lens",
            ],
            prompt_zhs: vec!["none", "中焦距", "广角", "长焦", "望远", "超广角-鱼眼"],
        },
        Preset {
            name: "camera_angle",
            brief: "Camera Angle",
            brief_zh: "机位角度",
            prompts: vec![
                "none",
                "Over-the-shoulder Shot",
                "High Angle Shot",
                "Low Angle Shot",
                "Dutch Angle Shot",
                "Aerial Shot",
                "Hgh Angle Shot",
            ],
            prompt_zhs: vec![
                "none",
                "过肩角度",
                "高角度",
                "低角度",
                "倾斜角度",
                "航拍",
                "俯视角度",
            ],
        },
        Preset {
            name: "lens_type",
            brief: "Lens Type",
            brief_zh: "镜头类型",
            prompts: vec![
                "none",
                "Clean Single Shot",
                "Two Shot",
                "Three Shot",
                "Group Shot",
                "Establishing Shot",
            ],
            prompt_zhs: vec![
                "none",
                "干净的单人镜头",
                "双人镜头",
                "三人镜头",
                "群像镜头",
                "定场镜头",
            ],
        },
        // 色调(Color Tone)
        Preset {
            name: "color_tone",
            brief: "Color Tone",
            brief_zh: "色调",
            prompts: vec![
                "none",
                "Warm Colors",
                "Cool Colors",
                "Saturated Colors",
                "Desaturated Colors",
            ],
            prompt_zhs: vec!["none", "暖色调", "冷色调", "高饱和度", "低饱和度"],
        },
        // 动态控制(Dynamic Control)
        Preset {
            name: "motion",
            brief: "Motion",
            brief_zh: "运动",
            prompts: vec![
                "none",
                "Street Dance",
                "Running",
                "Skateboarding",
                "Soccer",
                "Tennis",
                "Table Tennis",
                "Snowboarder",
                "Basketball",
                "Rugby Field",
                "Bowl Dance",
                "Aerial Cartwheel",
            ],
            prompt_zhs: vec![
                "none",
                "街舞",
                "跑步",
                "滑滑板",
                "踢足球",
                "网球",
                "乒乓球",
                "滑雪",
                "篮球",
                "橄榄球",
                "顶碗舞",
                "侧手翻",
            ],
        },
        Preset {
            name: "character_emotion",
            brief: "Character Emotion",
            brief_zh: "人物情绪",
            prompts: vec!["none", "Angrily", "Fear", "Happy", "Sadly", "Surprised"],
            prompt_zhs: vec!["none", "愤怒", "恐惧", "高兴", "悲伤", "惊讶"],
        },
        Preset {
            name: "basic_camera_movement",
            brief: "Basic Camera Movement",
            brief_zh: "基础运镜",
            prompts: vec![
                "none",
                "Camera Pushes In For A Close-up",
                "Camera Pulls Back",
                "Camera Pans To The Right",
                "Camera Moves To The Left",
                "Camera Tilts Up",
            ],
            prompt_zhs: vec![
                "none",
                "镜头推进",
                "镜头拉远",
                "镜头向右移动",
                "镜头向左移动",
                "镜头上摇",
            ],
        },
        Preset {
            name: "advanced_camera_movement",
            brief: "Advanced Camera Movement",
            brief_zh: "高级运镜",
            prompts: vec![
                "none",
                "Handheld Camera",
                "Compound Move",
                "Tracking Shot",
                "Arc Shot",
            ],
            prompt_zhs: vec!["none", "手持镜头", "复合运镜", "跟随镜头", "环绕运镜"],
        },
        // 风格化表现(Stylization)
        Preset {
            name: "visual_style",
            brief: "Visual Style",
            brief_zh: "视觉风格",
            prompts: vec![
                "none",
                "Felt Style",
                "3D Cartoon Style",
                "Pixel Art Style",
                "Puppet Animation",
                "3D Game Scene",
                "Claymation Style",
                "2D Anime Style",
                "Watercolor Painting",
                "Black And White Animation Segment",
                "Oil Painting Style",
            ],
            prompt_zhs: vec![
                "none",
                "毛毡风格",
                "3D卡通",
                "像素风格",
                "木偶动画",
                "3D游戏",
                "黏土风格",
                "二次元",
                "水彩画",
                "黑白动画",
                "油画风格",
            ],
        },
        Preset {
            name: "visual_effect",
            brief: "Visual Effects",
            brief_zh: "特效镜头",
            prompts: vec!["none", "Tilt-shift Photography", "Time-lapse"],
            prompt_zhs: vec!["none", "移轴摄影", "延时拍摄"],
        },
        Preset {
            // AI 扩展的内容
            name: "background",
            brief: "Background",
            brief_zh: "背景风格",
            prompts: vec![
                "none",
                "Simple Studio Background",
                "Natural Background",
                "Abstract Background",
                "Natural Background",
                "Drama Background",
                "Stylish Background",
                "Realistic Background",
                "Abstract Background",
                "Light Backgroung",
                "Stylish Background",
            ],
            prompt_zhs: vec![
                "none",
                "简单工作室背景",
                "自然背景",
                "抽象背景",
                "自然背景",
                "戏剧性背景",
                "时髦背景",
                "写实背景",
                "抽象背景",
                "亮背景",
                "时髦背景",
            ],
        },
        Preset {
            // AI 扩展的内容
            name: "location",
            brief: "Location",
            brief_zh: "场景",
            prompts: vec![
                "none",
                "Street",
                "Home Interior",
                "Indoor",
                "Outdoor",
                "Park",
                "Urban",
                "Urban Night",
            ],
            prompt_zhs: vec![
                "none",
                "街边",
                "家中",
                "室内",
                "室外",
                "公园",
                "城市",
                "城市夜景",
            ],
        },
    ]
}

/// Wan2.2 官方提示词选择器
#[pyclass(subclass)]
pub struct Wan22OfficialPromptSelector {}

impl PromptServer for Wan22OfficialPromptSelector {}

#[pymethods]
impl Wan22OfficialPromptSelector {
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
        ("prompt", "prompt_zh")
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
    const CATEGORY: &'static str = CATEGORY_TEXT;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Wan2.2 official preset prompt word selector."
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

                let presets = build_prompt();

                for preset in presets {
                    let name = preset.name;
                    let brief = preset.brief;
                    let brief_zh = preset.brief_zh;
                    let preset_options = preset.prompt_zhs;

                    let mut tips = format!("{}({})", brief_zh, brief);
                    // AI 扩展的提示词
                    let ai_ext = ["background", "location"];
                    if ai_ext.contains(&name) {
                        tips.push_str(", The unofficial prompt words for this category were expanded and filled in by AI, but have not been fully verified.");
                    }

                    required.set_item(
                        name,
                        (preset_options, {
                            let attribute = PyDict::new(py);
                            attribute.set_item("default", "none")?;
                            attribute.set_item("tooltip", tips)?;
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
    #[pyo3(name = "execute", signature = (**kwargs))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<(String, String)> {
        let mut preset_dict = HashMap::new();
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs.into_iter() {
                let key_str: String = key.extract()?;
                preset_dict.insert(key_str, value.to_string());
            }
        }

        let results = self.get_preset(preset_dict);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("Wan22OfficialPromptSelector error, {e}");
                if let Err(e) =
                    self.send_error(py, "Wan22OfficialPromptSelector".to_string(), e.to_string())
                {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl Wan22OfficialPromptSelector {
    /// 获取预设
    fn get_preset(&self, preset_dict: HashMap<String, String>) -> Result<(String, String), Error> {
        let mut prompts = Vec::new();
        let mut prompt_zhs = Vec::new();
        let presets = build_prompt();
        for (name, name_prompt_zh) in preset_dict {
            if name_prompt_zh == "none" {
                continue;
            }
            for preset in &presets {
                if name != preset.name {
                    continue;
                }

                for (i, prompt_zh) in preset.prompt_zhs.clone().into_iter().enumerate() {
                    if prompt_zh != name_prompt_zh {
                        continue;
                    }
                    prompts.push(preset.prompts[i]);
                    prompt_zhs.push(prompt_zh);
                }
            }
        }

        Ok((prompts.join(","), prompt_zhs.join(",")))
    }
}
