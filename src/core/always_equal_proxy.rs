//! 任意类型代理
//!
//! 将以下代码用rust实现
//! ```python
//! class AlwaysEqualProxy(str):
//!         def __eq__(self, _):
//!             return True
//!         def __ne__(self, _):
//!             return False
//! ```
//!
//! 注意: 在python端, 使用json.dumps()会报错

use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
    ops::Deref,
};

use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyString},
    Bound, Py, PyAny, PyResult, Python,
};
use serde::Serialize;

/// 任意参数类型代理
///
/// 注意: 在python端, 使用json.dumps()会报错
#[pyclass(name = "AlwaysEqualProxy")]
#[derive(Debug, Clone)]
pub struct AlwaysEqualProxy(String);

#[pymethods]
impl AlwaysEqualProxy {
    #[new]
    pub fn new(s: String) -> Self {
        Self(s)
    }

    fn __eq__(&self, _other: &Bound<'_, PyAny>) -> bool {
        true
    }

    fn __ne__(&self, _other: &Bound<'_, PyAny>) -> bool {
        false
    }

    fn __str__(&self) -> String {
        self.0.clone()
    }

    fn __repr__(&self) -> String {
        format!("'{}'", self.0)
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.0.hash(&mut hasher);
        hasher.finish()
    }

    #[getter]
    fn __dict__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("value", &self.0)?;
        Ok(dict.into())
    }

    #[getter]
    fn __json__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("value", &self.0)?;
        Ok(dict.into())
    }

    // 代理所有方法到内部 PyAny
    fn __getattr__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        PyString::new(py, &self.0).getattr(name)
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

// 确保在序列化时被视为字符串
impl Serialize for AlwaysEqualProxy {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

// 便捷构造函数
impl From<&str> for AlwaysEqualProxy {
    fn from(s: &str) -> Self {
        AlwaysEqualProxy(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_always_equal_proxy() -> anyhow::Result<()> {
        let any_type = AlwaysEqualProxy::from("*");

        // 验证比较行为
        assert!(any_type == "string"); // 与字符串比较
        assert!(any_type == 42); // 与整数比较
        assert!(any_type == 3.1); // 与浮点数比较
        assert!(!(any_type != "anything")); // 否定比较始终为false

        println!("output: {any_type}");
        Ok(())
    }
}
