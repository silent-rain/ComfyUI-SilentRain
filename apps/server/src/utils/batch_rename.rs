//! 批量重命名
//!
//! 文件夹/文件/混合重命名

use std::{
    collections::HashMap,
    fs::{self, DirEntry},
    path::{Path, PathBuf},
};

use log::{error, info, warn};
use pyo3::{
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyErr, PyResult, Python,
};
use strum_macros::{Display, EnumString};

use crate::{
    core::category::CATEGORY_UTILS,
    error::Error,
    wrapper::comfyui::{
        types::{NODE_INT, NODE_INT_MAX, NODE_STRING},
        PromptServer,
    },
};

/// 重命名的模式
///
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum RenameMode {
    /// 仅文件夹
    #[strum(to_string = "OnlyFolder")]
    OnlyFolder,
    /// 仅文件
    #[strum(to_string = "OnlyFile")]
    OnlyFile,
    /// 文件夹和文件
    #[strum(to_string = "FolderAndFile")]
    FolderAndFile,
}

/// 批量重命名
#[pyclass(subclass)]
pub struct BatchRename {
    name: String,
    current_index: usize,
    padding_width: usize,
    select_every_nth: usize,
}

impl PromptServer for BatchRename {}

#[pymethods]
impl BatchRename {
    #[new]
    fn new() -> Self {
        Self {
            name: "".to_string(),
            current_index: 0,
            padding_width: 0,
            select_every_nth: 1,
        }
    }

    // 输入列表, 可选
    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }
    // 输出节点, 可选
    #[classattr]
    #[pyo3(name = "OUTPUT_NODE")]
    fn output_node() -> bool {
        true
    }

    // 返回参数类型
    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() {}

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() {}

    // 返回参数提示
    #[classattr]
    #[pyo3(name = "OUTPUT_TOOLTIPS")]
    fn output_tooltips() {}

    // 输出列表, 可选
    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() {}

    // 节点分类
    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_UTILS;

    // 节点描述, 可选
    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Specify the path and rename the files or folders within it."
    }

    // 调研方法函数名称
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
                required.set_item(
                    "folder",
                    (NODE_STRING, {
                        let folder = PyDict::new(py);
                        folder.set_item("default", "")?;
                        folder
                            .set_item("tooltip", "Rename the contents of the specified folder")?;
                        folder
                    }),
                )?;
                required.set_item(
                    "name",
                    (NODE_STRING, {
                        let name = PyDict::new(py);
                        name.set_item("default", "new_name_%d")?;
                        name.set_item("tooltip", "New name, use '%d' for index replacement")?;
                        name
                    }),
                )?;

                required.set_item(
                    "mode",
                    (
                        vec![
                            RenameMode::OnlyFolder.to_string(),
                            RenameMode::OnlyFile.to_string(),
                            RenameMode::FolderAndFile.to_string(),
                        ],
                        {
                            let mode = PyDict::new(py);
                            mode.set_item("default", RenameMode::OnlyFolder.to_string())?;
                            mode.set_item("tooltip", "rename mode")?;
                            mode
                        },
                    ),
                )?;

                required
            })?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);
                optional.set_item(
                    "start_index",
                    (NODE_INT, {
                        let start_index = PyDict::new(py);
                        start_index.set_item("default", 0)?;
                        start_index.set_item("tooltip", "Starting index number")?;
                        start_index
                    }),
                )?;
                optional.set_item(
                    "select_every_nth",
                    (NODE_INT, {
                        let select_every_nth = PyDict::new(py);
                        select_every_nth.set_item("default", 1)?;
                        select_every_nth.set_item("step", 1)?;
                        select_every_nth.set_item("min", 1)?;
                        select_every_nth.set_item("max", NODE_INT_MAX)?;
                        select_every_nth.set_item("tooltip", "Select every n times")?;
                        select_every_nth
                    }),
                )?;
                optional.set_item(
                    "padding_width",
                    (NODE_INT, {
                        let padding_width = PyDict::new(py);
                        padding_width.set_item("default", 5)?;
                        padding_width.set_item("step", 1)?;
                        padding_width.set_item("min", 0)?;
                        padding_width.set_item("tooltip", "Maintain consistent file name length for easy sorting, dynamic width zero padding, for example: dynamic frame_00002.jpg")?;
                        padding_width
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
        folder: &str,
        name: &str,
        mode: &str,
        start_index: usize,
        select_every_nth: usize,
        padding_width: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.name = name.to_string();
        self.current_index = start_index;
        self.padding_width = padding_width;
        self.select_every_nth = select_every_nth;

        let results = self.batch_rename(folder, mode);

        match results {
            Ok(_v) => self.node_result(py),
            Err(e) => {
                error!("BatchRename error, {e}");
                if let Err(e) = self.send_error(py, "BatchRename".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl BatchRename {
    /// 组合为前端需要的数据结构
    fn node_result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("result", HashMap::<String, String>::new())?;
        Ok(dict)
    }
}

impl BatchRename {
    /// 批量替换名称
    pub fn batch_rename(&mut self, folder: &str, mode: &str) -> Result<(), Error> {
        let folder = Path::new(&folder);
        // 判断文件路径是否存在
        if !folder.exists() {
            return Err(Error::InvalidDirectory(format!(
                "{}: folder does not exist",
                folder.to_string_lossy()
            )));
        }

        let entries: Vec<DirEntry> = fs::read_dir(folder)?
            .filter_map(|entry| entry.ok())
            .collect();

        let mut dirs: Vec<PathBuf> = entries
            .iter()
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();
        dirs.sort();

        let mut files: Vec<PathBuf> = entries
            .iter()
            .map(|dir_entry| dir_entry.path())
            .filter(|path| path.is_file())
            .collect();
        files.sort();

        // 解析枚举并执行模式匹配
        let mode = mode
            .parse::<RenameMode>()
            .map_err(|e| Error::ParseEnumString(e.to_string()))?;

        match mode {
            RenameMode::OnlyFolder => {
                self.file_rename(dirs)?;
            }
            RenameMode::OnlyFile => {
                self.file_rename(files)?;
            }
            RenameMode::FolderAndFile => {
                self.file_rename(dirs)?;
                self.file_rename(files)?;
            }
        }

        Ok(())
    }

    /// 文件/文件夹进行重命名
    fn file_rename(&mut self, files: Vec<PathBuf>) -> Result<(), Error> {
        if files.is_empty() {
            return Ok(());
        }

        let total = files.len();
        let mut used_names = Vec::new(); // 记录已使用名称

        for path in &files {
            // 如果当前文件存在规划的路径中则跳过
            if used_names.contains(path) {
                continue;
            }

            let new_path = loop {
                // 获取新的文件路径
                let new_path = self.get_new_path(path, self.current_index, total);

                // 递增到下一个序号
                self.current_index += self.select_every_nth;

                if new_path == *path {
                    break new_path;
                }

                // 检查是否冲突
                if !new_path.exists() {
                    break new_path;
                }

                used_names.push(new_path.clone());

                warn!("skip {:?}", new_path.to_string_lossy().to_string());
            };

            // 执行重命名
            fs::rename(path, &new_path)?;
            // used_indices.push(self.current_index);

            // 递增到下一个序号
            // self.current_index += self.select_every_nth;

            info!(
                "rename {:?} {:?}",
                path.to_string_lossy().to_string(),
                new_path.to_string_lossy().to_string(),
            );
        }

        Ok(())
    }

    /// 保持文件名长度一致，便于排序,动态宽度零填充
    fn format_index(&self, current_index: usize, total: usize) -> String {
        // 计算需要的位数
        let padding_width =
            std::cmp::max(self.padding_width, (total as f64).log10().ceil() as usize);

        // format!("{:0width$}", current_index, width = padding_width)
        format!("{current_index:0padding_width$}")
    }

    /// 获取新的文件名称
    fn get_new_path(&self, path: &Path, current_index: usize, file_total: usize) -> PathBuf {
        // 格式化索引为N位数字，前面补零
        let formatted_index = self.format_index(current_index, file_total);

        let new_name = if path.is_file() {
            // 获取文件扩展名
            let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

            // 重命名
            format!(
                "{}.{}",
                self.name.replace("%d", &formatted_index),
                extension
            )
        } else {
            // 重命名
            self.name.replace("%d", &formatted_index).to_string()
        };

        path.with_file_name(&new_name)
    }
}
