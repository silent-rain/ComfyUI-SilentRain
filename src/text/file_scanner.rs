//! 文件扫描

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use chardet::{charset2encoding, detect};
use encoding::label::encoding_from_whatwg_label;
use encoding::DecoderTrap;
use log::error;
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyModule, PyType},
    Bound, Py, PyErr, PyResult, PyTypeInfo, Python,
};
use walkdir::WalkDir;

use crate::error::Error;

#[pyclass(name = "FileScanner", subclass)] // 允许子类化
pub struct FileScanner {}

#[pymethods]
impl FileScanner {
    #[new]
    fn new() -> Self {
        Self {}
    }

    /// 使用 #[classmethod] 实现 INPUT_TYPES
    /// ```py
    /// def INPUT_TYPES(cls):
    ///      return {
    ///          "required": {
    ///             "directory": ("STRING", {
    ///                "default": "./input",
    ///                "tooltip": "扫描目录路径"
    ///             }),
    ///            "recursive": (["enable", "disable"], {"default": "disable"}),
    ///            "encoding": (["utf-8", "gbk", "big5", "latin1"], {
    ///               "default": "utf-8",
    ///               "tooltip": "文件编码格式"
    ///            })
    ///           }
    ///       }
    /// ```
    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);
                required.set_item(
                    "directory",
                    ("STRING", {
                        let directory = PyDict::new(py);
                        directory.set_item("default", "./input")?;
                        directory.set_item("tooltip", "扫描目录路径")?;
                        directory
                    }),
                )?;
                required.set_item(
                    "recursive",
                    (vec!["enable", "disable"], {
                        let recursive = PyDict::new(py);
                        recursive.set_item("default", "disable")?;
                        recursive
                    }),
                )?;
                required.set_item(
                    "encoding",
                    (vec!["utf-8", "gbk", "auto"], {
                        let mut encoding = HashMap::new();
                        encoding.insert("default", "auto");
                        encoding.insert("tooltip", "文件编码格式");
                        encoding
                    }),
                )?;
                required
            })?;
            Ok(dict.into())
        })
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str) {
        ("LIST", "LIST")
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("paths", "contents")
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = "SilentRain/text";

    #[pyo3(name = "execute")]
    fn execute(
        &self,
        py: Python,
        directory: String,
        recursive: String,
        encoding: String,
    ) -> (Vec<String>, Vec<String>) {
        let results = self.scan_files(&directory, &recursive, &encoding);

        match results {
            Ok(v) => v,
            Err(e) => {
                println!("scan files failed, {e}");
                if let Err(e) = self.send_error(py, "FILE_READ_ERROR".to_string(), e.to_string()) {
                    println!("send error failed, {e}");
                    return (vec![], vec![]);
                };
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string());
                (vec![], vec![])
            }
        }
    }

    fn send_error(&self, py: Python, error_type: String, message: String) -> PyResult<()> {
        // 初始化时获取 PromptServer 实例
        let server = PyModule::import(py, "server")?
            .getattr("PromptServer")?
            .getattr("instance")?;

        // 构建错误数据字典
        let error_data = PyDict::new(py);
        error_data.set_item("type", &error_type)?;
        error_data.set_item("message", message)?;
        error_data.set_item("node", self.get_class_name(py)?)?;

        // 调用 Python 端方法
        server
            .getattr("send_sync")?
            .call1((error_type, error_data))?;

        Ok(())
    }

    fn get_class_name(&self, py: Python) -> PyResult<String> {
        // 需要 Clone
        // let py_self = self.as_ref().into_pyobject(py)?;
        // py_self.getattr("__name__")?.extract()

        FileScanner::type_object(py)
            .getattr("__name__")?
            .extract::<String>()
    }
}

// 实现 AsRef 的核心逻辑
impl AsRef<FileScanner> for FileScanner {
    fn as_ref(&self) -> &FileScanner {
        self
    }
}

impl FileScanner {
    /// 主扫描方法
    fn scan_files(
        &self,
        directory: &str,
        recursive: &str,
        encoding: &str,
    ) -> Result<(Vec<String>, Vec<String>), Error> {
        let path = Path::new(directory);
        if !path.is_dir() {
            return Err(Error::InvalidDirectory(directory.to_string()));
        }

        // 获取文件路径列表
        let file_paths = if recursive.to_lowercase() == "enable" {
            self.recursive_scan(path)?
        } else {
            self.single_scan(path)?
        };

        // 读取文件内容
        let mut contents = Vec::with_capacity(file_paths.len());
        for fp in &file_paths {
            let content = if encoding.to_ascii_lowercase() == "auto" {
                self.read_file_with_auto_encoding(&fp)?
            } else {
                self.read_file_with_encoding(&fp, encoding)?
            };
            contents.push(content);
        }

        Ok((
            file_paths
                .iter()
                .map(|v| v.to_string_lossy().to_string())
                .collect::<Vec<String>>(),
            contents,
        ))
    }

    /// 递归扫描实现
    fn recursive_scan(&self, dir: &Path) -> Result<Vec<PathBuf>, Error> {
        let walker = WalkDir::new(dir)
            .max_depth(100) // 防止无限递归
            .into_iter();

        let mut files = Vec::new();
        for entry in walker.filter_map(|e| e.ok()) {
            if entry.file_type().is_file() && self.is_txt_file(entry.path()) {
                files.push(entry.into_path());
            }
        }
        Ok(files)
    }

    /// 单层扫描实现
    fn single_scan(&self, dir: &Path) -> Result<Vec<PathBuf>, Error> {
        let mut files = Vec::new();
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && self.is_txt_file(&path) {
                files.push(path);
            }
        }
        Ok(files)
    }

    /// 文本文件检测
    fn is_txt_file(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|s| s.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("txt"))
            .unwrap_or(false)
    }

    /// 带编码转换的文件读取
    fn read_file_with_encoding(&self, path: &Path, encoding: &str) -> Result<String, Error> {
        let bytes = fs::read(path)?;

        // 优先尝试 UTF-8 解码
        if let Ok(s) = std::str::from_utf8(&bytes) {
            return Ok(s.to_string());
        }

        // 检测常见编码（GBK/ISO-8859-1）
        if let Some(coder) = encoding_from_whatwg_label(charset2encoding(&encoding.to_string())) {
            let utf8reader = coder.decode(&bytes, DecoderTrap::Ignore).map_err(|e| {
                error!("decode error, {e}");
                Error::DecodeError(e.to_string())
            })?;

            return Ok(utf8reader);
        }

        error!("file auto decode failed");
        Err(Error::DecodeError("file decode failed".to_string()))
    }

    /// 读取文件内容， 自动匹配文件编码
    fn read_file_with_auto_encoding(&self, path: &Path) -> Result<String, Error> {
        let bytes = fs::read(path)?;

        // 优先尝试 UTF-8 解码
        if let Ok(s) = std::str::from_utf8(&bytes) {
            return Ok(s.to_string());
        }

        // detect charset of the file
        let result = detect(&bytes);

        // 检测常见编码（GBK/ISO-8859-1）
        if let Some(coder) = encoding_from_whatwg_label(charset2encoding(&result.0)) {
            let utf8reader = coder.decode(&bytes, DecoderTrap::Ignore).map_err(|e| {
                error!("decode error, {e}");
                Error::DecodeError(e.to_string())
            })?;

            return Ok(utf8reader);
        }

        error!("file auto decode failed");
        Err(Error::DecodeError("file auto decode failed".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_file_with_auto_encoding() -> anyhow::Result<()> {
        let file_scanner = FileScanner::new();
        let content = file_scanner.read_file_with_auto_encoding(Path::new("README.md"))?;
        println!("content: {content}");
        Ok(())
    }
}
