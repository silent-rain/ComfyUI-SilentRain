//! 目录操作

use std::{collections::HashMap, path::Path, time::UNIX_EPOCH};

use walkdir::{DirEntry, WalkDir};

use crate::error::Error;

/// 检查目录是否存在
pub fn is_directory(path: &str) -> bool {
    Path::new(path).is_dir()
}

/// 递归搜索目录
pub fn recursive_search(
    directory: &str,
    excluded_dir_names: &[&str],
) -> (Vec<String>, HashMap<String, f64>) {
    let mut files = Vec::new();
    let mut dirs = HashMap::new();

    if !Path::new(directory).is_dir() {
        return (files, dirs);
    }

    let walker = WalkDir::new(directory)
        .into_iter()
        .filter_entry(|e| !is_excluded_dir(e, excluded_dir_names));

    for entry in walker.filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            if let Some(path) = entry.path().to_str() {
                files.push(path.to_string());
            }
        } else if entry.file_type().is_dir() {
            if let Some(path) = entry.path().to_str() {
                if let Ok(mtime) = get_mtime(path) {
                    dirs.insert(path.to_string(), mtime);
                }
            }
        }
    }

    (files, dirs)
}

/// 检查是否为排除目录
pub fn is_excluded_dir(entry: &DirEntry, excluded_names: &[&str]) -> bool {
    if !entry.file_type().is_dir() {
        return false;
    }

    entry
        .file_name()
        .to_str()
        .map(|name| excluded_names.contains(&name))
        .unwrap_or(false)
}

/// 过滤文件扩展名
pub fn filter_files_extensions(files: &[String], extensions: &[String]) -> Vec<String> {
    if extensions.is_empty() {
        return files.to_vec();
    }

    files
        .iter()
        .filter(|file| {
            Path::new(file)
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| extensions.iter().any(|e| e.eq_ignore_ascii_case(ext)))
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}

/// 获取目录修改时间
pub fn get_mtime(path: &str) -> Result<f64, Error> {
    let metadata = std::fs::metadata(path)?;
    let mtime = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)?
        .as_secs_f64();
    Ok(mtime)
}
