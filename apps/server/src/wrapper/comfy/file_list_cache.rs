//! 文件列表缓存
//!

use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock, RwLock},
};

use log::error;

use crate::{core::utils::directory::get_mtime, error::Error};

// 全局文件列表缓存实例
static FILE_LIST_CACHE: OnceLock<Arc<RwLock<BTreeMap<String, CacheEntry>>>> = OnceLock::new();

// 缓存项结构
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// 文件列表
    pub files: Vec<String>,
    /// 目录修改时间
    pub dir_mtimes: BTreeMap<String, f64>,
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
                    error!("error: {e}");
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
pub struct FileListCache;

impl FileListCache {
    /// 更新文件名缓存
    pub fn set(&mut self, key: String, entry: CacheEntry) -> Result<(), Error> {
        let cache = FILE_LIST_CACHE.get_or_init(|| Arc::new(RwLock::new(BTreeMap::new())));
        let mut cache_guard = cache.write().map_err(|e| Error::LockError(e.to_string()))?;
        cache_guard.insert(key, entry);

        Ok(())
    }

    /// 获取文件名缓存
    pub fn get(&self, key: &str) -> Result<Option<CacheEntry>, Error> {
        let cache = FILE_LIST_CACHE.get_or_init(|| Arc::new(RwLock::new(BTreeMap::new())));
        let cache_guard = cache.read().map_err(|e| Error::LockError(e.to_string()))?;
        Ok(cache_guard.get(key).cloned())
    }

    /// 检查缓存是否有效
    ///
    /// 检查文件本身的时间是否与缓存中的时间一致
    pub fn is_valid(&self, key: &str) -> bool {
        match self.get(key) {
            Ok(Some(cache_entry)) => cache_entry.is_valid(),
            Ok(None) | Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_file_list_cache() -> anyhow::Result<()> {
        let entry = FileListCache.get("key")?;
        println!("entry: {:?}", entry);
        Ok(())
    }
}
