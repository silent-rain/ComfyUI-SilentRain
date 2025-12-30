//! 纯预设尺寸节点
//! 根据选择的宽高比和分辨率级别输出对应的宽度和高度
//!
//! 预置尺寸包括：
//! - 1:1 (1024x1024, 2048x2048, 4096x4096)
//! - 2:3 (848x1264, 1696x2528, 3392x5056)
//! - 3:2 (1264x848, 2528x1696, 5056x3392)
//! - 3:4 (896x1200, 1792x2400, 3584x4800)
//! - 4:3 (1200x896, 2400x1792, 4800x3584)
//! - 4:5 (928x1152, 1856x2304, 3712x4608)
//! - 5:4 (1152x928, 2304x1856, 4608x3712)
//! - 9:16 (768x1376, 1536x2752, 3072x5504)
//! - 16:9 (1376x768, 2752x1536, 5504x3072)
//! - 21:9 (1584x672, 3168x1344, 6336x2688)

use pyo3::{
    prelude::*,
    types::{PyDict, PyType},
};
use std::collections::HashMap;
use strum_macros::{Display, EnumString};

use crate::{
    core::category::CATEGORY_IMAGE,
    wrapper::comfyui::types::{NODE_INT, NODE_STRING},
};

/// 宽高比枚举
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum AspectRatio {
    /// 1:1 正方形
    #[strum(to_string = "1:1")]
    Ratio1_1,

    /// 2:3 竖版
    #[strum(to_string = "2:3")]
    Ratio2_3,

    /// 3:2 横版
    #[strum(to_string = "3:2")]
    Ratio3_2,

    /// 3:4 竖版
    #[strum(to_string = "3:4")]
    Ratio3_4,

    /// 4:3 横版
    #[strum(to_string = "4:3")]
    Ratio4_3,

    /// 4:5 竖版
    #[strum(to_string = "4:5")]
    Ratio4_5,

    /// 5:4 横版
    #[strum(to_string = "5:4")]
    Ratio5_4,

    /// 9:16 竖版手机
    #[strum(to_string = "9:16")]
    Ratio9_16,

    /// 16:9 横版宽屏
    #[strum(to_string = "16:9")]
    Ratio16_9,

    /// 21:9 超宽屏
    #[strum(to_string = "21:9")]
    Ratio21_9,
}

/// 分辨率级别枚举
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum ResolutionLevel {
    /// 1K 分辨率
    #[strum(to_string = "1K")]
    K1,

    /// 2K 分辨率
    #[strum(to_string = "2K")]
    K2,

    /// 4K 分辨率
    #[strum(to_string = "4K")]
    K4,
}

/// 纯预设尺寸节点
#[pyclass(subclass)]
pub struct ImagePresetResolution;

#[pymethods]
impl ImagePresetResolution {
    #[new]
    fn new() -> Self {
        Self
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str, &'static str, &'static str) {
        (NODE_INT, NODE_INT, NODE_STRING, NODE_STRING)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str, &'static str, &'static str) {
        ("width", "height", "aspect_ratio", "preset_name")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool, bool, bool) {
        (false, false, false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_IMAGE;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Output preset resolution dimensions based on selected aspect ratio and resolution level. Includes 1K, 2K, and 4K resolutions for common aspect ratios."
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

                // 宽高比选择
                required.set_item(
                    "aspect_ratio",
                    (
                        vec![
                            AspectRatio::Ratio1_1.to_string(),
                            AspectRatio::Ratio2_3.to_string(),
                            AspectRatio::Ratio3_2.to_string(),
                            AspectRatio::Ratio3_4.to_string(),
                            AspectRatio::Ratio4_3.to_string(),
                            AspectRatio::Ratio4_5.to_string(),
                            AspectRatio::Ratio5_4.to_string(),
                            AspectRatio::Ratio9_16.to_string(),
                            AspectRatio::Ratio16_9.to_string(),
                            AspectRatio::Ratio21_9.to_string(),
                        ],
                        {
                            let mut options = HashMap::new();
                            options.insert("default", AspectRatio::Ratio1_1.to_string());
                            options
                        },
                    ),
                )?;

                // 分辨率级别选择
                required.set_item(
                    "resolution_level",
                    (
                        vec![
                            ResolutionLevel::K1.to_string(),
                            ResolutionLevel::K2.to_string(),
                            ResolutionLevel::K4.to_string(),
                        ],
                        {
                            let mut options = HashMap::new();
                            options.insert("default", ResolutionLevel::K1.to_string());
                            options
                        },
                    ),
                )?;

                required
            })?;

            Ok(dict.into())
        })
    }

    fn execute(
        &self,
        aspect_ratio: String,
        resolution_level: String,
    ) -> PyResult<(usize, usize, String, String)> {
        // 解析宽高比和分辨率级别
        let aspect_ratio = aspect_ratio
            .parse::<AspectRatio>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let resolution_level = resolution_level
            .parse::<ResolutionLevel>()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // 获取目标尺寸
        let (width, height, preset_name) = self.get_target_size(aspect_ratio, resolution_level)?;

        Ok((width, height, aspect_ratio.to_string(), preset_name))
    }
}

impl ImagePresetResolution {
    /// 获取目标尺寸
    fn get_target_size(
        &self,
        aspect_ratio: AspectRatio,
        resolution_level: ResolutionLevel,
    ) -> Result<(usize, usize, String), PyErr> {
        // 根据宽高比和分辨率级别计算尺寸
        let (width, height) = match aspect_ratio {
            AspectRatio::Ratio1_1 => match resolution_level {
                ResolutionLevel::K1 => (1024, 1024),
                ResolutionLevel::K2 => (2048, 2048),
                ResolutionLevel::K4 => (4096, 4096),
            },
            AspectRatio::Ratio2_3 => match resolution_level {
                ResolutionLevel::K1 => (848, 1264),
                ResolutionLevel::K2 => (1696, 2528),
                ResolutionLevel::K4 => (3392, 5056),
            },
            AspectRatio::Ratio3_2 => match resolution_level {
                ResolutionLevel::K1 => (1264, 848),
                ResolutionLevel::K2 => (2528, 1696),
                ResolutionLevel::K4 => (5056, 3392),
            },
            AspectRatio::Ratio3_4 => match resolution_level {
                ResolutionLevel::K1 => (896, 1200),
                ResolutionLevel::K2 => (1792, 2400),
                ResolutionLevel::K4 => (3584, 4800),
            },
            AspectRatio::Ratio4_3 => match resolution_level {
                ResolutionLevel::K1 => (1200, 896),
                ResolutionLevel::K2 => (2400, 1792),
                ResolutionLevel::K4 => (4800, 3584),
            },
            AspectRatio::Ratio4_5 => match resolution_level {
                ResolutionLevel::K1 => (928, 1152),
                ResolutionLevel::K2 => (1856, 2304),
                ResolutionLevel::K4 => (3712, 4608),
            },
            AspectRatio::Ratio5_4 => match resolution_level {
                ResolutionLevel::K1 => (1152, 928),
                ResolutionLevel::K2 => (2304, 1856),
                ResolutionLevel::K4 => (4608, 3712),
            },
            AspectRatio::Ratio9_16 => match resolution_level {
                ResolutionLevel::K1 => (768, 1376),
                ResolutionLevel::K2 => (1536, 2752),
                ResolutionLevel::K4 => (3072, 5504),
            },
            AspectRatio::Ratio16_9 => match resolution_level {
                ResolutionLevel::K1 => (1376, 768),
                ResolutionLevel::K2 => (2752, 1536),
                ResolutionLevel::K4 => (5504, 3072),
            },
            AspectRatio::Ratio21_9 => match resolution_level {
                ResolutionLevel::K1 => (1584, 672),
                ResolutionLevel::K2 => (3168, 1344),
                ResolutionLevel::K4 => (6336, 2688),
            },
        };

        // 确保尺寸是8的倍数（符合深度学习模型要求）
        let width = (width / 8) * 8;
        let height = (height / 8) * 8;

        let name = format!("{} - {}x{}", aspect_ratio, width, height);

        Ok((width, height, name))
    }
}
