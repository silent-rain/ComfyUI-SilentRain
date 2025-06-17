//! 文件列表缓存
//!

use std::{
    collections::HashMap,
    sync::{Mutex, MutexGuard, OnceLock},
};

use log::error;

use crate::{core::utils::directory::get_mtime, error::Error};

// 全局文件列表缓存实例
static FILE_LIST_CACHE: OnceLock<Mutex<FileListCache>> = OnceLock::new();

// 缓存项结构
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// 文件列表
    pub files: Vec<String>,
    /// 目录修改时间
    pub dir_mtimes: HashMap<String, f64>,
    /// 时间戳
    pub timestamp: f64,
}

impl CacheEntry {
    /// 检查缓存是否有效
    ///
    /// 检查文件本身的时间是否与缓存中的时间一致
    pub fn is_valid(&self) -> bool {
        // 检查目录修改时间是否变化
        for (folder_name, cached_time) in self.dir_mtimes.clone() {
            let current_mtime = match get_mtime(&folder_name) {
                Ok(v) => v,
                Err(e) => {
                    error!("error: {}", e);
                    return false;
                }
            };
            if (current_mtime as f64 - cached_time).abs() > f64::EPSILON {
                return false;
            }
        }
        true
    }
}

// 文件列表缓存
#[derive(Default)]
pub struct FileListCache {
    filename_list_cache: HashMap<String, CacheEntry>,
}

impl FileListCache {
    pub fn new() -> Result<MutexGuard<'static, FileListCache>, Error> {
        let cache = FILE_LIST_CACHE.get_or_init(|| Mutex::new(FileListCache::default()));
        let obj = cache.lock().map_err(|e| Error::LockError(e.to_string()))?;
        Ok(obj)
    }

    /// 更新文件名缓存
    pub fn set(&mut self, key: String, entry: CacheEntry) {
        self.filename_list_cache.insert(key, entry);
    }

    /// 获取文件名缓存
    pub fn get(&self, key: &str) -> Option<&CacheEntry> {
        self.filename_list_cache.get(key)
    }

    /// 获取文件名缓存的克隆
    pub fn get_cloned(&self, key: &str) -> Option<CacheEntry> {
        self.filename_list_cache.get(key).cloned()
    }

    /// 检查缓存是否有效
    ///
    /// 检查文件本身的时间是否与缓存中的时间一致
    pub fn is_valid(&self, key: &str) -> bool {
        match self.get(key) {
            Some(cache_entry) => cache_entry.is_valid(),
            None => false,
        }
    }
}
