//! 保存文本

use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
};

use log::error;
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyErr, PyResult, Python,
};
use strum_macros::{Display, EnumString};

use crate::{
    core::category::CATEGORY_TEXT,
    error::Error,
    wrapper::comfyui::{types::NODE_STRING, PromptServer},
};

/// 保存文本的模式
///
/// "overwrite", "append", "append_new_line"
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum SaveTextMode {
    /// 覆盖
    #[strum(to_string = "Overwrite")]
    Overwrite,
    /// 追加
    #[strum(to_string = "Append")]
    Append,
    /// 追加新行
    #[strum(to_string = "Append New Line")]
    AppendNewLine,
}

/// 保存文本
#[pyclass(subclass)]
pub struct SaveText {}

impl PromptServer for SaveText {}

#[pymethods]
impl SaveText {
    #[new]
    fn new() -> Self {
        // 初始化全局日志
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_level(true)
            .with_file(true)
            .with_line_number(true)
            .try_init();
        Self {}
    }

    // 输入列表, 可选
    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    // 返回参数类型
    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_STRING,)
    }

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("captions",)
    }

    // 返回参数提示
    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() -> (&'static str,) {
        ("captions",)
    }

    // 输出列表, 可选
    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool,) {
        (false,)
    }

    // 节点分类
    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_TEXT;

    // 节点描述, 可选
    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Save Text Node."
    }

    // 调研方法函数名称
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
                required.set_item(
                    "captions",
                    (NODE_STRING, {
                        let captions = PyDict::new(py);
                        captions.set_item("default", "")?;
                        captions.set_item("tooltip", "Saved Text Content")?;
                        captions
                    }),
                )?;
                required.set_item(
                    "filepath",
                    (NODE_STRING, {
                        let filepath = PyDict::new(py);
                        filepath.set_item("default", "output_folder")?;
                        filepath.set_item("tooltip", "Save file path")?;
                        filepath
                    }),
                )?;
                required.set_item(
                    "filename",
                    (NODE_STRING, {
                        let filename = PyDict::new(py);
                        filename.set_item("default", "file")?;
                        filename.set_item("tooltip", "Save file name")?;
                        filename
                    }),
                )?;
                required.set_item(
                    "filename_prefix",
                    (NODE_STRING, {
                        let filename_prefix = PyDict::new(py);
                        filename_prefix.set_item("default", "")?;
                        filename_prefix.set_item("tooltip", "The prefix of the saved file name")?;
                        filename_prefix
                    }),
                )?;
                required.set_item(
                    "filename_suffix",
                    (NODE_STRING, {
                        let filename_prefix = PyDict::new(py);
                        filename_prefix.set_item("default", "")?;
                        filename_prefix.set_item("tooltip", "The suffix of the saved file name")?;
                        filename_prefix
                    }),
                )?;
                required.set_item(
                    "file_extension",
                    (NODE_STRING, {
                        let file_extension = PyDict::new(py);
                        file_extension.set_item("default", ".txt")?;
                        file_extension.set_item("tooltip", "Filter file extensions")?;
                        file_extension
                    }),
                )?;

                required.set_item(
                    "mode",
                    (
                        vec![
                            SaveTextMode::Overwrite.to_string(),
                            SaveTextMode::Append.to_string(),
                            SaveTextMode::AppendNewLine.to_string(),
                        ],
                        {
                            let mode = PyDict::new(py);
                            mode.set_item("default", SaveTextMode::Overwrite.to_string())?;
                            mode.set_item("tooltip", "Saved Content Mode")?;
                            mode
                        },
                    ),
                )?;

                required
            })?;

            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute")]
    fn execute(
        &mut self,
        py: Python,
        captions: String,
        filepath: String,
        filename: String,
        filename_prefix: String,
        filename_suffix: String,
        file_extension: String,
        mode: String,
    ) -> PyResult<(String,)> {
        let results = self.save_text(
            captions,
            filepath,
            filename,
            filename_prefix,
            filename_suffix,
            file_extension,
            mode,
        );

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("SaveText error, {e}");
                if let Err(e) = self.send_error(py, "SaveText".to_string(), e.to_string()) {
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

impl SaveText {
    /// 保存文本
    #[allow(clippy::too_many_arguments)]
    fn save_text(
        &self,
        captions: String,
        filepath: String,
        filename: String,
        filename_prefix: String,
        filename_suffix: String,
        file_extension: String,
        mode: String,
    ) -> Result<(String,), Error> {
        // 判断文件路径是否存在
        if !Path::new(&filepath).exists() {
            return Err(Error::FilePathNotExist(format!(
                "{}: File path does not exist",
                filepath
            )));
        }

        let file_path =
            format!("{filepath}/{filename_prefix}{filename}{filename_suffix}{file_extension}");

        // 解析枚举并执行模式匹配
        let mode = mode
            .parse::<SaveTextMode>()
            .map_err(|e| Error::ParseEnumString(e.to_string()))?;

        // 保存文本
        match mode {
            SaveTextMode::Overwrite => {
                let mut file = File::create(file_path)?;
                file.write_all(captions.as_bytes())?;
            }
            SaveTextMode::Append => {
                let mut file = OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(file_path)?;
                file.write_all(captions.as_bytes())?;
            }
            SaveTextMode::AppendNewLine => {
                let mut file = OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(file_path)?;
                file.write_all(b"\n")?;
                file.write_all(captions.as_bytes())?;
            }
        }

        Ok((captions,))
    }
}
