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
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyErr, PyResult, Python,
};
use walkdir::WalkDir;

use crate::{
    core::{
        category::CATEGORY_UTILS,
        types::{NODE_BOOLEAN, NODE_STRING},
        PromptServer,
    },
    error::Error,
};

/// 文件扫描器
#[pyclass(subclass)] // 允许子类化
pub struct FileScanner {
    file_extension: String,
}

impl PromptServer for FileScanner {}

#[pymethods]
impl FileScanner {
    #[new]
    fn new() -> Self {
        // 初始化全局日志
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_level(true)
            .with_file(true)
            .with_line_number(true)
            .try_init();
        Self {
            file_extension: "".to_string(),
        }
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
    ///            "recursive": ("BOOLEAN", {"default":,"label_on": "enabled", "label_off": "disabled"}),
    ///            "encoding": (["utf-8", "gbk", "big5", "latin1"], {
    ///               "default": "utf-8",
    ///               "tooltip": "文件编码格式"
    ///            })
    ///            "file_extension": ("STRING", {
    ///               "default": "txt",
    ///               "tooltip": "过滤文件扩展名"
    ///            }),
    ///          }
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
                    (NODE_STRING, {
                        let directory = PyDict::new(py);
                        directory.set_item("default", "./input")?;
                        directory.set_item("tooltip", "Scan directory path")?;
                        directory
                    }),
                )?;

                required.set_item(
                    "recursive",
                    (NODE_BOOLEAN, {
                        let recursive = PyDict::new(py);
                        recursive.set_item("default", false)?;
                        recursive.set_item("label_on", "enabled")?;
                        recursive.set_item("label_off", "disabled")?;
                        recursive
                    }),
                )?;
                required.set_item(
                    "encoding",
                    (vec!["utf-8", "gbk", "auto"], {
                        let mut encoding = HashMap::new();
                        encoding.insert("default", "auto");
                        encoding.insert("tooltip", "File encoding format");
                        encoding
                    }),
                )?;
                required.set_item(
                    "file_extension",
                    (NODE_STRING, {
                        let directory = PyDict::new(py);
                        directory.set_item("default", "txt")?;
                        directory.set_item("tooltip", "Filter file extensions")?;
                        directory
                    }),
                )?;
                required
            })?;
            Ok(dict.into())
        })
    }

    // 输入列表, 可选
    // #[classattr]
    // #[pyo3(name = "INPUT_IS_LIST")]
    // fn input_is_list() -> bool {
    //     false
    // }

    // 输出节点, 可选
    // #[classattr]
    // #[pyo3(name = "OUTPUT_NODE")]
    // fn output_node() -> bool {
    //     false
    // }

    // 节点描述, 可选
    // #[classattr]
    // #[pyo3(name = "DESCRIPTION")]
    // fn description() -> &'static str {
    //     ""
    // }

    // 过时标记, 可选
    // #[classattr]
    // #[pyo3(name = "DEPRECATED")]
    // fn deprecated() -> bool {
    //     false
    // }

    // 实验性的, 可选
    // #[classattr]
    // #[pyo3(name = "EXPERIMENTAL")]
    // fn experimental() -> bool {
    //     false
    // }

    // 返回参数类型
    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str) {
        (NODE_STRING, NODE_STRING)
    }

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str) {
        ("paths", "contents")
    }

    // 输出列表, 可选
    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool) {
        (true, true)
    }

    // 调研方法函数名称
    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    // 节点分类
    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_UTILS;

    #[pyo3(name = "execute")]
    fn execute(
        &mut self,
        py: Python,
        directory: String,
        file_extension: String,
        recursive: bool,
        encoding: String,
    ) -> PyResult<(Vec<String>, Vec<String>)> {
        self.file_extension = file_extension;

        let results = self.scan_files(&directory, recursive, &encoding);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("scan files failed, {e}");
                if let Err(e) = self.send_error(py, "SCAN_FILES_ERROR".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        e.to_string(),
                    ));
                };
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    e.to_string(),
                ))
            }
        }
    }
}

impl FileScanner {
    /// 主扫描方法
    fn scan_files(
        &self,
        directory: &str,
        recursive: bool,
        encoding: &str,
    ) -> Result<(Vec<String>, Vec<String>), Error> {
        let path = Path::new(directory);
        if !path.is_dir() {
            return Err(Error::InvalidDirectory(directory.to_string()));
        }

        // 获取文件路径列表
        let file_paths = if recursive {
            self.recursive_scan(path)?
        } else {
            self.single_scan(path)?
        };

        // 读取文件内容
        let mut contents = Vec::with_capacity(file_paths.len());
        for fp in &file_paths {
            let content = if encoding.eq_ignore_ascii_case("auto") {
                self.read_file_with_auto_encoding(fp)?
            } else {
                self.read_file_with_encoding(fp, encoding)?
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
            .map(|ext| ext.eq_ignore_ascii_case(&self.file_extension))
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
                Error::Decode(e.to_string())
            })?;

            return Ok(utf8reader);
        }

        error!("file auto decode failed");
        Err(Error::Decode("file decode failed".to_string()))
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
                Error::Decode(e.to_string())
            })?;

            return Ok(utf8reader);
        }

        error!("file auto decode failed");
        Err(Error::Decode("file auto decode failed".to_string()))
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
