//! 从文件夹加载图像列表

use std::{
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, RgbaImage};
use log::error;
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
    Bound, Py, PyAny, PyErr, PyResult, Python,
};
use walkdir::WalkDir;

use crate::{
    core::{category::CATEGORY_IMAGE, utils::image_to_tensor},
    error::Error,
    wrapper::{
        comfyui::{
            types::{NODE_BOOLEAN, NODE_IMAGE, NODE_INT, NODE_MASK, NODE_STRING},
            PromptServer,
        },
        torch::tensor::TensorWrapper,
    },
};

/// 从文件夹加载图像列表
#[pyclass(subclass)]
pub struct LoadImagesFromFolder {
    device: Device,
    file_extensions: Vec<String>,
}

impl PromptServer for LoadImagesFromFolder {}

#[pymethods]
impl LoadImagesFromFolder {
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
            device: Device::Cpu,
            file_extensions: ["png", "jpg", "jpeg", "bmp", "webp"]
                .iter()
                .map(|v| v.to_string())
                .collect(),
        }
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
    fn return_types() -> (
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    ) {
        (NODE_IMAGE, NODE_MASK, NODE_STRING, NODE_STRING, NODE_INT)
    }

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (
        &'static str,
        &'static str,
        &'static str,
        &'static str,
        &'static str,
    ) {
        ("images", "masks", "filenames", "filepaths", "count")
    }

    // 输出列表, 可选
    #[classattr]
    #[pyo3(name = "OUTPUT_IS_LIST")]
    fn output_is_list() -> (bool, bool, bool, bool, bool) {
        (true, true, true, true, false)
    }

    // 节点分类
    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_IMAGE;

    // 节点描述, 可选
    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Load images from folder."
    }

    // 调研方法函数名称
    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    /// 使用 #[classmethod] 实现 INPUT_TYPES
    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);
                required.set_item(
                    "folder",
                    (NODE_STRING, {
                        let folder = PyDict::new(py);
                        folder.set_item("default", "./input")?;
                        folder.set_item("tooltip", "Scan folder path")?;
                        folder
                    }),
                )?;
                required
            })?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);
                optional.set_item(
                    "include_subfolders",
                    (NODE_BOOLEAN, {
                        let include_subfolders = PyDict::new(py);
                        include_subfolders.set_item("default", false)?;
                        include_subfolders.set_item("label_on", "enabled")?;
                        include_subfolders.set_item("label_off", "disabled")?;
                        include_subfolders
                    }),
                )?;
                optional.set_item(
                    "start_index",
                    (NODE_INT, {
                        let start_index = PyDict::new(py);
                        start_index.set_item("default", 0)?;
                        start_index.set_item("min", 0)?;
                        start_index.set_item("step", 0)?;
                        start_index
                    }),
                )?;
                optional.set_item(
                    "limit",
                    (NODE_INT, {
                        let limit = PyDict::new(py);
                        limit.set_item("default", -1)?;
                        limit.set_item("min", -1)?;
                        limit.set_item("step", 1)?;
                        limit
                    }),
                )?;

                optional
            })?;
            Ok(dict.into())
        })
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        folder: String,
        include_subfolders: bool,
        start_index: usize,
        limit: i32,
    ) -> PyResult<(
        Vec<Bound<'py, PyAny>>,
        Vec<Bound<'py, PyAny>>,
        Vec<String>,
        Vec<String>,
        usize,
    )> {
        let results = self.scan_files(py, &folder, include_subfolders, start_index, limit);

        match results {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("LoadImagesFromFolder error, {e}");
                if let Err(e) =
                    self.send_error(py, "LoadImagesFromFolder".to_string(), e.to_string())
                {
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

impl LoadImagesFromFolder {
    /// 主扫描方法
    #[allow(clippy::type_complexity)]
    fn scan_files<'py>(
        &self,
        py: Python<'py>,
        folder: &str,
        include_subfolders: bool,
        start_index: usize,
        limit: i32,
    ) -> Result<
        (
            Vec<Bound<'py, PyAny>>,
            Vec<Bound<'py, PyAny>>,
            Vec<String>,
            Vec<String>,
            usize,
        ),
        Error,
    > {
        let path = Path::new(folder);
        if !path.is_dir() {
            return Err(Error::InvalidDirectory(folder.to_string()));
        }

        // 获取文件路径列表
        let (file_paths, file_names) =
            self.get_file_paths(folder, include_subfolders, start_index, limit)?;

        // 读取图片
        let images = self.get_images(&file_paths)?;

        let image_tensors = self.get_image_tensors(py, &images)?;

        let image_masks = self.get_image_masks(py, &images)?;

        Ok((
            image_tensors,
            image_masks,
            file_paths
                .iter()
                .map(|v| v.to_string_lossy().to_string())
                .collect::<Vec<String>>(),
            file_names,
            file_paths.len(),
        ))
    }

    /// 获取文件路径列表
    fn get_file_paths(
        &self,
        folder: &str,
        include_subfolders: bool,
        start_index: usize,
        limit: i32,
    ) -> Result<(Vec<PathBuf>, Vec<String>), Error> {
        let path = Path::new(folder);
        if !path.is_dir() {
            return Err(Error::InvalidDirectory(folder.to_string()));
        }

        // 获取文件路径列表
        let mut file_paths: Vec<PathBuf> = if include_subfolders {
            self.recursive_scan(path)?
        } else {
            self.single_scan(path)?
        };

        // 截取指定范围的路径
        let end_index = if limit <= 0 {
            file_paths.len()
        } else {
            limit as usize
        }
        .min(file_paths.len());
        file_paths = file_paths[start_index..end_index].to_vec();

        // 获取文件名列表（不带路径）
        let file_names: Vec<String> = file_paths
            .iter()
            .filter_map(|path| path.file_name()) // 获取文件名部分
            .map(|name| name.to_string_lossy().to_string()) // 转换为String
            .collect();

        Ok((file_paths, file_names))
    }

    /// 递归扫描实现
    fn recursive_scan(&self, dir: &Path) -> Result<Vec<PathBuf>, Error> {
        let walker = WalkDir::new(dir)
            .max_depth(1000) // 防止无限递归
            .into_iter();

        let mut files = Vec::new();
        for entry in walker.filter_map(|e| e.ok()) {
            if entry.file_type().is_file() && self.is_img_file(entry.path()) {
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
            if path.is_file() && self.is_img_file(&path) {
                files.push(path);
            }
        }
        Ok(files)
    }

    /// 图像文件检测
    fn is_img_file(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|s| s.to_str())
            .map(|ext| {
                self.file_extensions
                    .iter()
                    .any(|e| ext.eq_ignore_ascii_case(e))
            })
            .unwrap_or(false)
    }

    /// 读取图片
    fn get_images(&self, paths: &[PathBuf]) -> Result<Vec<DynamicImage>, Error> {
        let images: Vec<DynamicImage> = paths
            .iter()
            .map(image::open)
            .collect::<Result<Vec<DynamicImage>, _>>()?;

        Ok(images)
    }

    /// 获取图片张量
    fn get_image_tensors<'py>(
        &self,
        py: Python<'py>,
        paths: &[DynamicImage],
    ) -> Result<Vec<Bound<'py, PyAny>>, Error> {
        let images: Vec<Bound<'_, PyAny>> = paths
            .iter()
            .map(|img| {
                // HWC
                let tensor = image_to_tensor(img, &self.device)?;

                // HWC -> NHWC
                let tensor = tensor.unsqueeze(0)?;

                // Tensor -> TensorWrapper
                let tensor_wrapper: TensorWrapper<f32> = tensor.into();

                tensor_wrapper.to_py_tensor(py).map_err(Error::PyErr)
            })
            .collect::<Result<Vec<Bound<'_, PyAny>>, _>>()?;

        Ok(images)
    }

    /// 获取图片mask
    fn get_image_masks<'py>(
        &self,
        py: Python<'py>,
        images: &[DynamicImage],
    ) -> Result<Vec<Bound<'py, PyAny>>, Error> {
        let masks: Vec<Bound<'_, PyAny>> = images
            .iter()
            .map(|img| {
                // 转换为RGBA格式获取alpha通道
                let rgba = img.to_rgba8();
                let (target_width, target_height) = rgba.dimensions();

                // 提取alpha通道并转换为mask
                let mask = if let Some(mask_data) = self.rgba_to_alpha_mask(&rgba) {
                    // 转换为tensor [H, W]
                    let tensor = Tensor::from_vec(
                        mask_data,
                        (target_height as usize, target_width as usize),
                        &self.device,
                    )?
                    .to_dtype(DType::F32)?;
                    // [H, W] -> [1, H, W]
                    tensor.unsqueeze(0)?
                } else {
                    // 如果没有alpha通道，创建全零mask
                    // [1, H, W]
                    Tensor::zeros(
                        (1, target_height as usize, target_width as usize),
                        DType::F32,
                        &self.device,
                    )?
                };

                // 反转mask (1 - alpha)
                let mask = (1.0 - mask)?;

                // 转换为PyTorch tensor并返回
                let tensor_wrapper: TensorWrapper<f32> = mask.into();
                tensor_wrapper.to_py_tensor(py).map_err(Error::PyErr)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(masks)
    }

    /// 从RGBA图像中提取alpha通道作为灰度图像
    fn rgba_to_alpha_mask(&self, rgba: &RgbaImage) -> Option<Vec<f32>> {
        if rgba.pixels().all(|p| p.0[3] == 255) {
            // 所有像素都是不透明的，没有alpha通道
            None
        } else {
            // 提取alpha通道
            let mask_data: Vec<f32> = rgba.pixels().map(|p| p[3] as f32 / 255.0).collect();

            Some(mask_data)
        }
    }
}
