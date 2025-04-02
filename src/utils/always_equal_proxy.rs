//! 任意参数类型代理

use std::ops::Deref;
use std::{cmp::PartialEq, fmt};

use pyo3::{pyclass, pymethods};
use serde::Serialize;

/// 任意参数类型代理
#[pyclass]
#[pyo3(name = "AlwaysEqualProxy")]
#[derive(Debug, Serialize)]
pub struct AlwaysEqualProxy(String);

#[pymethods]
impl AlwaysEqualProxy {
    #[new]
    pub fn new(s: String) -> Self {
        Self(s)
    }
}

// 添加Display实现以便友好输出
impl fmt::Display for AlwaysEqualProxy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// 实现Deref以便像字符串一样使用
impl Deref for AlwaysEqualProxy {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// 为所有类型T实现特殊比较逻辑
impl<T> PartialEq<T> for AlwaysEqualProxy {
    fn eq(&self, _: &T) -> bool {
        true
    }
}

// 便捷构造函数
impl From<&str> for AlwaysEqualProxy {
    fn from(s: &str) -> Self {
        AlwaysEqualProxy(s.to_string())
    }
}

/// 任意类型
#[macro_export]
macro_rules! any_type {
    () => {
        AlwaysEqualProxy("*".to_string())
    };
    ($s:expr) => {
        AlwaysEqualProxy($s.to_string())
    };
}

/// 任意类型
// pub static ANY_TYPE: LazyLock<AlwaysEqualProxy> = LazyLock::new(|| AlwaysEqualProxy::from("*"));
// pub static ANY_TYPE: AlwaysEqualProxy = AlwaysEqualProxy::new("*".to_string());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_always_equal_proxy() -> anyhow::Result<()> {
        let any_type = AlwaysEqualProxy::from("*");

        // 验证比较行为
        assert!(any_type == "string"); // 与字符串比较
        assert!(any_type == 42); // 与整数比较
        assert!(any_type == 3.1); // 与浮点数比较
        assert!(!(any_type != "anything")); // 否定比较始终为false

        println!("output: {}", any_type);
        Ok(())
    }
}
