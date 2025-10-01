//! 文件夹路径

use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use lazy_static::lazy_static;
use log::{error, warn};

use crate::{
    core::utils::directory::{filter_files_extensions, recursive_search},
    error::Error,
    wrapper::comfy::file_list_cache::{CacheEntry, FileListCache},
};

// 支持的模型文件扩展名
lazy_static! {
    static ref SUPPORTED_PT_EXTENSIONS: HashSet<&'static str> = {
        let mut set = HashSet::new();
        set.insert(".ckpt");
        set.insert(".pt");
        set.insert(".pt2");
        set.insert(".bin");
        set.insert(".pth");
        set.insert(".safetensors");
        set.insert(".pkl");
        set.insert(".sft");
        set
    };
}

/// 文件夹路径配置结构体
#[allow(clippy::type_complexity)]
#[derive(Debug)]
pub struct FolderPaths {
    /// 基础路径
    base_path: PathBuf,
    /// 模型路径
    model_path: PathBuf,
    /// folders, extensions
    /// 文件夹名称和路径映射
    folder_names_and_paths: BTreeMap<&'static str, (Vec<PathBuf>, HashSet<&'static str>)>,
    /// 输出目录
    output_directory: PathBuf,
    /// 临时目录
    temp_directory: PathBuf,
    /// 输入目录
    input_directory: PathBuf,
    /// 用户目录
    user_directory: PathBuf,
}

impl Default for FolderPaths {
    /// 创建一个默认的 FolderPaths 实例
    fn default() -> Self {
        let base_path = std::env::current_dir().expect("Failed to get current directory");
        let models_dir = base_path.join("models");
        let folder_names_and_paths = Self::init_folder_names_and_paths(&base_path, &models_dir);

        Self {
            base_path: base_path.clone(),
            model_path: models_dir,
            folder_names_and_paths,
            output_directory: base_path.join("output"),
            temp_directory: base_path.join("temp"),
            input_directory: base_path.join("input"),
            user_directory: base_path.join("user"),
        }
    }
}

impl FolderPaths {
    /// 创建新的FolderPaths实例
    pub fn from_base_directory(base_directory: &str) -> Self {
        let base_path = PathBuf::from(base_directory);

        let models_dir = base_path.join(&base_path);
        let folder_names_and_paths = Self::init_folder_names_and_paths(&base_path, &models_dir);

        Self {
            base_path: base_path.clone(),
            model_path: models_dir,
            folder_names_and_paths,
            output_directory: base_path.join("output"),
            temp_directory: base_path.join("temp"),
            input_directory: base_path.join("input"),
            user_directory: base_path.join("user"),
        }
    }

    fn init_folder_names_and_paths(
        base_path: &Path,
        models_dir: &Path,
    ) -> BTreeMap<&'static str, (Vec<PathBuf>, HashSet<&'static str>)> {
        let mut folder_names_and_paths = BTreeMap::new();

        // 添加各种模型路径配置
        folder_names_and_paths.insert(
            "checkpoints",
            (
                vec![models_dir.join("checkpoints")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "configs",
            (vec![models_dir.join("configs")], {
                let mut set = HashSet::new();
                set.insert(".yaml");
                set
            }),
        );

        // 添加其他路径配置...
        folder_names_and_paths.insert(
            "loras",
            (
                vec![models_dir.join("loras")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "vae",
            (
                vec![models_dir.join("vae")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "text_encoders",
            (
                vec![models_dir.join("text_encoders"), models_dir.join("clip")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "diffusion_models",
            (
                vec![models_dir.join("unet"), models_dir.join("diffusion_models")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "clip_vision",
            (
                vec![models_dir.join("clip_vision")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "style_models",
            (
                vec![models_dir.join("style_models")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "embeddings",
            (
                vec![models_dir.join("embeddings")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "diffusers",
            (vec![models_dir.join("diffusers")], {
                let mut set = HashSet::new();
                set.insert("folder");
                set
            }),
        );

        folder_names_and_paths.insert(
            "vae_approx",
            (
                vec![models_dir.join("vae_approx")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "controlnet",
            (
                vec![
                    models_dir.join("controlnet"),
                    models_dir.join("t2i_adapter"),
                ],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "gligen",
            (
                vec![models_dir.join("gligen")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "upscale_models",
            (
                vec![models_dir.join("upscale_models")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "custom_nodes",
            (vec![base_path.join("custom_nodes")], HashSet::new()),
        );

        folder_names_and_paths.insert(
            "hypernetworks",
            (
                vec![models_dir.join("hypernetworks")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "photomaker",
            (
                vec![models_dir.join("photomaker")],
                SUPPORTED_PT_EXTENSIONS.clone(),
            ),
        );

        folder_names_and_paths.insert(
            "classifiers",
            (vec![models_dir.join("classifiers")], {
                let mut set = HashSet::new();
                set.insert("");
                set
            }),
        );

        folder_names_and_paths.insert(
            "LLM",
            (vec![models_dir.join("LLM")], {
                let mut set = HashSet::new();
                set.insert("");
                set
            }),
        );

        folder_names_and_paths
    }

    /// 获取基础路径
    pub fn base_path(&self) -> PathBuf {
        self.base_path.clone()
    }

    pub fn model_path(&self) -> PathBuf {
        self.model_path.clone()
    }

    /// 获取文件夹路径映射
    pub fn folder_names_and_paths(
        &self,
    ) -> &BTreeMap<&'static str, (Vec<PathBuf>, HashSet<&'static str>)> {
        &self.folder_names_and_paths
    }

    /// 获取输出目录
    pub fn output_directory(&self) -> PathBuf {
        self.output_directory.clone()
    }

    /// 获取临时目录
    pub fn temp_directory(&self) -> PathBuf {
        self.temp_directory.clone()
    }

    /// 获取输入目录
    pub fn input_directory(&self) -> PathBuf {
        self.input_directory.clone()
    }

    /// 获取用户目录
    pub fn user_directory(&self) -> PathBuf {
        self.user_directory.clone()
    }
}

impl FolderPaths {
    /// 旧文件夹名称映射
    pub fn map_legacy(folder_name: &str) -> &str {
        match folder_name {
            "unet" => "diffusion_models",
            "clip" => "text_encoders",
            _ => folder_name,
        }
    }

    /// 将相对路径转换为绝对路径
    pub fn to_absolute_path(&self, relative_path: &Path) -> PathBuf {
        self.base_path.join(relative_path)
    }

    /// 将绝对路径转换为相对路径
    pub fn to_relative_path(&self, absolute_path: &Path) -> Option<PathBuf> {
        absolute_path
            .strip_prefix(&self.base_path)
            .ok()
            .map(|p| p.to_path_buf())
    }

    /// 检查路径是否为绝对路径
    pub fn is_absolute_path(&self, path: &Path) -> bool {
        path.is_absolute()
    }
}

impl FolderPaths {
    /// 添加模型文件夹路径
    ///
    /// # 参数
    /// * `folder_name` - 文件夹名称
    /// * `full_folder_path` - 完整的文件夹路径
    /// * `is_default` - 是否设为默认路径（放在列表首位）
    pub fn add_model_folder_path(
        &mut self,
        folder_name: &'static str,
        full_folder_path: PathBuf,
        is_default: bool,
    ) {
        let folder_name = Self::map_legacy(folder_name);

        if let Some((paths, _exts)) = self.folder_names_and_paths.get_mut(folder_name) {
            // 检查路径是否已存在
            if paths.contains(&full_folder_path) {
                if is_default && paths[0] != full_folder_path {
                    // 如果路径不是列表中的第一个，则移动到开头
                    let index = paths.iter().position(|p| p == &full_folder_path).unwrap();
                    paths.remove(index);
                    paths.insert(0, full_folder_path);
                }
            } else {
                // 添加新路径
                if is_default {
                    paths.insert(0, full_folder_path);
                } else {
                    paths.push(full_folder_path);
                }
            }
        } else {
            // 创建新的文件夹配置
            let mut extensions = HashSet::new();
            // 默认使用支持的模型文件扩展名
            if folder_name != "configs"
                && folder_name != "diffusers"
                && folder_name != "classifiers"
                && folder_name != "custom_nodes"
            {
                extensions = SUPPORTED_PT_EXTENSIONS.clone();
            }
            self.folder_names_and_paths
                .insert(folder_name, (vec![full_folder_path], extensions));
        }
    }

    /// 获取完整文件路径
    pub fn get_full_path(
        &self,
        folder_name: &str,
        filename: &str,
    ) -> Result<Option<PathBuf>, Error> {
        let folder_name = Self::map_legacy(folder_name);

        // 获取基础路径列表
        let (file_paths, _) = self
            .folder_names_and_paths()
            .get(folder_name)
            .ok_or_else(|| Error::InvalidDirectory(format!("folder {folder_name} not found")))?;

        // 规范化文件名路径
        let normalized_filename = Path::new("/")
            .join(filename)
            .strip_prefix("/")
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|_| PathBuf::from(filename));

        // 在基础路径中查找文件
        for file_path in file_paths {
            let full_path = self.base_path.join(file_path).join(&normalized_filename);

            // 检查文件是否存在
            if let Ok(metadata) = fs::symlink_metadata(&full_path) {
                if metadata.is_file() {
                    return Ok(Some(full_path));
                } else if metadata.file_type().is_symlink() {
                    // 检查符号链接是否有效
                    if fs::metadata(&full_path).is_err() {
                        warn!(
                            "WARNING path {} exists but doesn't link anywhere, skipping.",
                            full_path.display()
                        );
                    }
                }
            }
        }

        Ok(None)
    }

    /// 获取文件名列表
    pub fn get_filename_list(&self, folder_name: &str) -> Vec<String> {
        let folder_name = Self::map_legacy(folder_name);

        if let Some(entry) = self.cached_filename_list(folder_name) {
            return entry.files.clone();
        }

        // 更新缓存
        let entry = self.get_filename_list_(folder_name);
        // 处理可能的错误，但不影响返回结果
        if let Err(e) = FileListCache.set(folder_name.to_string(), entry.clone()) {
            error!("Failed to update file list cache: {}", e);
        }

        entry.files
    }

    /// 从缓存中获取文件列表
    fn cached_filename_list(&self, folder_name: &str) -> Option<CacheEntry> {
        let folder_name = Self::map_legacy(folder_name);

        // 处理 Result<Option<CacheEntry>, Error>
        let entry = match FileListCache.get(folder_name) {
            Ok(Some(entry)) => entry,
            Ok(None) | Err(_) => return None,
        };

        // 检查目录修改时间是否变化
        if !entry.is_valid() {
            return None;
        }

        // 检查是否有新增目录
        if let Some((dir_paths, _)) = self.folder_names_and_paths.get(folder_name) {
            for dir_path in dir_paths {
                // 判断是否为目录
                if !dir_path.is_dir() {
                    continue;
                }
                // 检查目录是否发生变化
                if !entry
                    .dir_mtimes
                    .contains_key(&dir_path.to_string_lossy().to_string())
                {
                    return None;
                }
            }
        }

        Some(entry.clone())
    }

    /// 获取文件名列表
    fn get_filename_list_(&self, folder_name: &str) -> CacheEntry {
        let folder_name = Self::map_legacy(folder_name);
        let mut output_list = HashSet::new();
        let mut dir_mtimes = BTreeMap::new();

        if let Some((dir_paths, extensions)) = self.folder_names_and_paths.get(folder_name) {
            for dir_path in dir_paths {
                let (files, dirs) =
                    recursive_search(dir_path.to_string_lossy().as_ref(), &[".git"]);
                dir_mtimes.extend(dirs);

                let extensions_vec: Vec<String> = extensions
                    .iter()
                    .map(|s| s.to_string())
                    .filter(|ext| !ext.is_empty())
                    .collect();
                let filtered = filter_files_extensions(&files, &extensions_vec);
                output_list.extend(filtered);
            }
        }

        let mut sorted_list: Vec<String> = output_list.into_iter().collect();
        sorted_list.sort_unstable();

        CacheEntry {
            files: sorted_list,
            dir_mtimes,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_folder_paths_initialization() {
        let folder_paths = FolderPaths::default();
        assert!(folder_paths.base_path().exists());
        assert!(folder_paths
            .folder_names_and_paths()
            .contains_key("checkpoints"));
    }
}
