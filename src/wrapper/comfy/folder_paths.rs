//! 文件夹路径

use lazy_static::lazy_static;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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
    base_path: PathBuf,
    folder_names_and_paths: HashMap<&'static str, (Vec<PathBuf>, HashSet<&'static str>)>,
    output_directory: PathBuf,
    temp_directory: PathBuf,
    input_directory: PathBuf,
    user_directory: PathBuf,
    filename_list_cache: HashMap<String, (Vec<String>, HashMap<String, f64>, f64)>,
}

impl FolderPaths {
    /// 创建新的FolderPaths实例
    pub fn new(base_directory: Option<&str>) -> Self {
        let base_path = match base_directory {
            Some(dir) => PathBuf::from(dir),
            None => std::env::current_dir().expect("Failed to get current directory"),
        };

        let models_dir = base_path.join("models");

        let mut folder_names_and_paths = HashMap::new();

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

        Self {
            base_path: base_path.clone(),
            folder_names_and_paths,
            output_directory: base_path.join("output"),
            temp_directory: base_path.join("temp"),
            input_directory: base_path.join("input"),
            user_directory: base_path.join("user"),
            filename_list_cache: HashMap::new(),
        }
    }

    /// 获取基础路径
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// 获取文件夹路径映射
    pub fn folder_names_and_paths(
        &self,
    ) -> &HashMap<&'static str, (Vec<PathBuf>, HashSet<&'static str>)> {
        &self.folder_names_and_paths
    }

    /// 获取输出目录
    pub fn output_directory(&self) -> &Path {
        &self.output_directory
    }

    /// 获取临时目录
    pub fn temp_directory(&self) -> &Path {
        &self.temp_directory
    }

    /// 获取输入目录
    pub fn input_directory(&self) -> &Path {
        &self.input_directory
    }

    /// 获取用户目录
    pub fn user_directory(&self) -> &Path {
        &self.user_directory
    }

    /// 更新文件名缓存
    pub fn update_filename_cache(&mut self, key: String, filenames: Vec<String>) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let mut mtimes = HashMap::new();
        for filename in &filenames {
            // 这里应该添加获取文件修改时间的逻辑
            mtimes.insert(filename.clone(), timestamp);
        }

        self.filename_list_cache
            .insert(key, (filenames, mtimes, timestamp));
    }

    /// 获取文件名缓存
    pub fn get_filename_cache(
        &self,
        key: &str,
    ) -> Option<&(Vec<String>, HashMap<String, f64>, f64)> {
        self.filename_list_cache.get(key)
    }

    /// 检查缓存是否有效
    pub fn is_cache_valid(&self, key: &str, max_age_seconds: f64) -> bool {
        match self.filename_list_cache.get(key) {
            Some((_, _, timestamp)) => {
                let current_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
                current_time - timestamp <= max_age_seconds
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_folder_paths_initialization() {
        let folder_paths = FolderPaths::new(None);
        assert!(folder_paths.base_path().exists());
        assert!(folder_paths
            .folder_names_and_paths()
            .contains_key("checkpoints"));
    }
}
