//! 自定义尺寸节点 - 基于数学公式的动态计算
//! 支持自定义宽高比和分辨率缩放，避免硬编码
//!
//! 核心设计原则：
//! - 基准边长：1K = 1024
//! - 分辨率缩放因子：1-8（边长倍数，如2表示2048，4表示4096）
//! - 先计算理论尺寸，然后四舍五入到整数，再向上取整到16的倍数
//! - 支持自定义宽高比（范围：1-21）
//!
//! 输入参数：
//! - aspect_width: 宽度比（整数，1-21）
//! - aspect_height: 高度比（整数，1-21）
//! - resolution_scale: 分辨率缩放因子（边长倍数，整数，1-8）

use pyo3::{
    prelude::*,
    types::{PyDict, PyType},
};
use std::collections::HashMap;

use crate::{core::category::CATEGORY_IMAGE, wrapper::comfyui::types::NODE_INT};

/// 基准边长（1K）
const BASE_SIZE: f64 = 1024.0;

/// 自定义尺寸节点
#[pyclass(subclass)]
pub struct ImageCustomResolution {}

#[pymethods]
impl ImageCustomResolution {
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
        (NODE_INT, NODE_INT)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("width", "height")
    }

    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (false, false)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_IMAGE;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Custom resolution calculator based on aspect ratio and scale. Computes dimensions dynamically with 16-pixel alignment."
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

                // 宽高分子输入
                required.set_item(
                    "aspect_width",
                    (NODE_INT, {
                        let mut options = HashMap::new();
                        options.insert("default", 1);
                        options.insert("min", 1);
                        options.insert("max", 21);
                        options.insert("step", 1);
                        options
                    }),
                )?;

                // 宽高分母输入
                required.set_item(
                    "aspect_height",
                    (NODE_INT, {
                        let mut options = HashMap::new();
                        options.insert("default", 1);
                        options.insert("min", 1);
                        options.insert("max", 21);
                        options.insert("step", 1);
                        options
                    }),
                )?;

                // 分辨率缩放因子输入
                required.set_item(
                    "resolution_scale",
                    (NODE_INT, {
                        let mut options = HashMap::new();
                        options.insert("default", 1);
                        options.insert("min", 1);
                        options.insert("max", 8);
                        options.insert("step", 1);
                        options
                    }),
                )?;

                required
            })?;

            Ok(dict.into())
        })
    }

    fn execute(
        &self,
        aspect_width: usize,
        aspect_height: usize,
        resolution_scale: usize,
    ) -> PyResult<(usize, usize)> {
        // 验证宽高比输入
        if aspect_width == 0 || aspect_height == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Aspect ratio values must be greater than 0",
            ));
        }
        if aspect_width > 21 || aspect_height > 21 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Aspect ratio values must not exceed 21",
            ));
        }

        // 验证分辨率缩放因子
        if resolution_scale == 0 || resolution_scale > 8 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Resolution scale must be between 1 and 8",
            ));
        }

        // 动态计算尺寸
        let (width, height) =
            self.calculate_dimensions(aspect_width, aspect_height, resolution_scale)?;

        Ok((width, height))
    }
}

impl ImageCustomResolution {
    /// 基于宽高比和分辨率缩放因子动态计算尺寸
    fn calculate_dimensions(
        &self,
        aspect_width: usize,
        aspect_height: usize,
        resolution_scale: usize,
    ) -> Result<(usize, usize), PyErr> {
        // 计算宽高比
        let ratio = aspect_width as f64 / aspect_height as f64;

        // 先计算 1K 的基准尺寸（1024×1024 对应的尺寸）
        // 然后按 resolution_scale 缩放
        //
        // 计算公式：
        // - 设基准边长为 base = 1024
        // - 对于非正方形，需要找到 (width, height) 使得：
        //   - width / height = ratio
        //   - width × height ≈ base × base
        // - 解得：width = base × sqrt(ratio), height = base / sqrt(ratio)
        // - 最后按 scale 缩放：final_width = width × scale, final_height = height × scale

        let base_size = BASE_SIZE;
        let sqrt_ratio = ratio.sqrt();

        let width = base_size * sqrt_ratio;
        let height = base_size / sqrt_ratio;

        // 先四舍五入到整数，再分别向上取整到16的倍数（符合原始预设值的对齐规则）
        let width = ((width.round() / 16.0).ceil() * 16.0) as usize;
        let height = ((height.round() / 16.0).ceil() * 16.0) as usize;

        // 按 resolution_scale 缩放
        let width = width * resolution_scale;
        let height = height * resolution_scale;

        // 确保尺寸不为0
        let width = width.max(64); // 最小64像素
        let height = height.max(64);

        Ok((width, height))
    }

    /// 计算实际像素数（用于调试）
    #[allow(dead_code)]
    fn actual_pixels(&self, width: usize, height: usize) -> usize {
        width * height
    }

    /// 计算与基准像素数的偏差百分比
    #[allow(dead_code)]
    fn deviation_percentage(&self, width: usize, height: usize, resolution_scale: usize) -> f64 {
        let actual = self.actual_pixels(width, height) as f64;
        let target = BASE_SIZE * BASE_SIZE * resolution_scale as f64;
        ((actual - target) / target) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1k_square() {
        let node = ImageCustomResolution {};

        let (width, height) = node.calculate_dimensions(1, 1, 1).unwrap();
        assert_eq!(width, 1024);
        assert_eq!(height, 1024);
    }

    #[test]
    fn test_1k_portrait_2_3() {
        let node = ImageCustomResolution {};

        let (width, height) = node.calculate_dimensions(2, 3, 1).unwrap();
        // 动态计算结果
        assert_eq!(width, 848);
        assert_eq!(height, 1264);
    }

    #[test]
    fn test_2k_square() {
        let node = ImageCustomResolution::new();

        let (width, height) = node.calculate_dimensions(1, 1, 2).unwrap();
        assert_eq!(width, 2048);
        assert_eq!(height, 2048);
    }

    #[test]
    fn test_4k_square() {
        let node = ImageCustomResolution::new();

        let (width, height) = node.calculate_dimensions(1, 1, 4).unwrap();
        assert_eq!(width, 4096);
        assert_eq!(height, 4096);
    }

    #[test]
    fn test_pixel_consistency() {
        let node = ImageCustomResolution::new();

        // 常用宽高比测试
        let test_ratios = [
            (1, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (4, 5),
            (5, 4),
            (9, 16),
            (16, 9),
            (21, 9),
        ];

        for (aspect_w, aspect_h) in test_ratios {
            let (width, height) = node.calculate_dimensions(aspect_w, aspect_h, 1).unwrap();
            let pixels = node.actual_pixels(width, height);
            let deviation = node.deviation_percentage(width, height, 1);

            // 所有宽高比的像素数应在基准值的±5%范围内
            assert!(
                deviation.abs() < 5.0,
                "Aspect ratio {}:{} has deviation {}% ({}×{} = {} pixels)",
                aspect_w,
                aspect_h,
                deviation,
                width,
                height,
                pixels
            );
        }
    }

    #[test]
    fn test_alignment_to_8() {
        let node = ImageCustomResolution::new();

        let test_ratios = [
            (1, 1),
            (2, 3),
            (3, 2),
            (3, 4),
            (4, 3),
            (4, 5),
            (5, 4),
            (9, 16),
            (16, 9),
            (21, 9),
        ];

        for (aspect_w, aspect_h) in test_ratios {
            let (width, height) = node.calculate_dimensions(aspect_w, aspect_h, 1).unwrap();

            // 确保尺寸是8的倍数
            assert_eq!(width % 8, 0, "Width {} is not aligned to 8", width);
            assert_eq!(height % 8, 0, "Height {} is not aligned to 8", height);
        }
    }

    #[test]
    fn test_various_scales() {
        let node = ImageCustomResolution::new();

        // 测试不同的缩放因子
        let scales = [1, 2, 4, 8];
        for scale in scales {
            let (width, height) = node.calculate_dimensions(1, 1, scale).unwrap();
            let expected = 1024 * scale;
            assert_eq!(
                width, expected,
                "Scale {}: width should be {}",
                scale, expected
            );
            assert_eq!(
                height, expected,
                "Scale {}: height should be {}",
                scale, expected
            );
        }
    }
}
