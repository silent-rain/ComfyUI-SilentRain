//! 节点分类

use std::fmt;

use crate::constant::CATEGORY_PREFIX;

/// ComfyUI 节点分类枚举。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    /// 文本
    Text,
    /// 逻辑
    Logic,
    /// 实用工具
    Utils,
    /// 列表
    List,
    /// 图片
    Image,
    /// 遮罩
    Mask,
    /// 条件
    Conditioning,
    /// 模型
    Model,
    /// JoyCaption
    JoyCaption,
    /// llama.cpp
    LlamaCpp,
    /// Math
    Math,
    /// Experimental
    Experimental,
    /// 示例
    Example,
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Category::Text => f.write_str(&format!("{CATEGORY_PREFIX}/Text")),
            Category::Logic => f.write_str(&format!("{CATEGORY_PREFIX}/Logic")),
            Category::Utils => f.write_str(&format!("{CATEGORY_PREFIX}/Utils")),
            Category::List => f.write_str(&format!("{CATEGORY_PREFIX}/List")),
            Category::Image => f.write_str(&format!("{CATEGORY_PREFIX}/Image")),
            Category::Mask => f.write_str(&format!("{CATEGORY_PREFIX}/Mask")),
            Category::Conditioning => f.write_str(&format!("{CATEGORY_PREFIX}/conditioning")),
            Category::Model => f.write_str(&format!("{CATEGORY_PREFIX}/Model")),
            Category::JoyCaption => f.write_str(&format!("{CATEGORY_PREFIX}/JoyCaption")),
            Category::LlamaCpp => f.write_str(&format!("{CATEGORY_PREFIX}/LlamaCpp")),
            Category::Math => f.write_str(&format!("{CATEGORY_PREFIX}/Math")),
            Category::Experimental => f.write_str(&format!("{CATEGORY_PREFIX}/Experimental")),
            Category::Example => f.write_str(&format!("{CATEGORY_PREFIX}/Example")),
        }
    }
}

impl From<Category> for String {
    fn from(value: Category) -> Self {
        value.to_string()
    }
}
