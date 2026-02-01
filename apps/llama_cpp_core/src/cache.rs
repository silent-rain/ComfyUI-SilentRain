//! 模型缓存管理器
use std::{
    any::Any,
    collections::HashMap,
    fmt,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, OnceLock, RwLock},
};

use tracing::error;

use crate::error::Error;

static GLOBAL_CACHE: OnceLock<Arc<CacheManager>> = OnceLock::new();

/// 缓存条目
#[derive(Clone)]
pub struct CacheEntry<T: Any + Send + Sync> {
    data: T,
    params_hash: u64,
}

/// Debug 展示
impl<T: Any + Send + Sync> fmt::Debug for CacheEntry<T> {
    /// 格式化显示缓存条目的参数哈希值
    ///
    /// # 参数
    /// * `f` - 格式化写入器
    ///
    /// # 返回
    /// 格式化操作结果
    ///
    /// # 示例
    /// ```
    /// let entry = CacheEntry { params_hash: 123 };
    /// println!("{}", entry);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CacheEntry params_hash: {}", self.params_hash)
    }
}

/// 全局单例的模型缓存
pub struct CacheManager {
    pool: RwLock<HashMap<String, CacheEntry<Arc<dyn Any + Send + Sync>>>>,
}

impl Default for CacheManager {
    fn default() -> Self {
        Self {
            pool: RwLock::new(HashMap::new()),
        }
    }
}

impl fmt::Debug for CacheManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pool = self.pool.read().map_err(|_| fmt::Error)?;
        f.debug_struct("CacheManager")
            .field("pool_size", &pool.len())
            .finish()
    }
}

impl CacheManager {
    // 单例模式，全局唯一的模型缓存
    pub fn global() -> &'static Arc<CacheManager> {
        GLOBAL_CACHE.get_or_init(|| Arc::new(CacheManager::default()))
    }

    /// 添加或更新模型（仅当 hash 值变化时更新）
    pub fn insert<T>(
        &self,
        key: &str,
        params: &[T],
        data: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), Error>
    where
        T: Hash,
    {
        let params_hash = Self::compute_hash(params);
        let mut pool = self.pool.write().map_err(|e| {
            error!("get cache write lock error: {e:?}");
            Error::LockError(e.to_string())
        })?;

        if let Some(cache) = pool.get(key)
            && cache.params_hash == params_hash
        {
            // hash 值未变化，不更新
            return Ok(());
        }
        pool.insert(key.to_string(), CacheEntry { params_hash, data });

        Ok(())
    }

    /// 强制更新模型（无论 hash 值是否变化）
    pub fn force_update<T>(
        &self,
        key: &str,
        params: &[T],
        data: Arc<dyn Any + Send + Sync>,
    ) -> Result<(), Error>
    where
        T: Hash,
    {
        let params_hash = Self::compute_hash(params);
        let mut pool = self.pool.write().map_err(|e| {
            error!("get cache write lock error: {e:?}");
            Error::LockError(e.to_string())
        })?;
        pool.insert(key.to_string(), CacheEntry { params_hash, data });

        Ok(())
    }

    /// 获取或添加或更新模型（通过闭包初始化）并转换为具体类型
    pub fn get_or_insert<F, T, P>(
        &self,
        key: &str,
        params: &[P],
        handler: F,
    ) -> Result<CacheEntry<Arc<T>>, Error>
    where
        F: FnOnce() -> Result<Arc<T>, Error>,
        P: Hash,
        T: Send + Sync + 'static,
    {
        let params_hash = Self::compute_hash(params);

        // 先检查缓存（读锁）
        {
            let pool = self.pool.read().map_err(|e| {
                error!("get cache read lock error: {:?}", e);
                Error::LockError(e.to_string())
            })?;
            if let Some(cache) = pool.get(key)
                && cache.params_hash == params_hash
            {
                // hash 值未变化，尝试转换为具体类型并返回
                drop(pool); // 释放读锁
                return self.get::<T>(key);
            }
        }

        // hash 值不存在或不一致，初始化新模型并更新缓存（写锁）
        let data = handler().map_err(|e| {
            error!("get cache data error: {:?}", e);
            e
        })?;
        let cache = CacheEntry {
            params_hash,
            data: data.clone() as Arc<dyn Any + Send + Sync>,
        };
        let mut pool = self.pool.write().map_err(|e| {
            error!("get cache write lock error: {e:?}");
            Error::LockError(e.to_string())
        })?;
        pool.insert(key.to_string(), cache.clone());

        // 返回正确类型的 CacheEntry
        Ok(CacheEntry { params_hash, data })
    }

    /// 获取模型并转换为具体类型
    pub fn get<T: Send + Sync + 'static>(&self, key: &str) -> Result<CacheEntry<Arc<T>>, Error> {
        let pool = self.pool.read().map_err(|e| {
            error!("get cache read lock error: {e:?}");
            Error::LockError(e.to_string())
        })?;
        let cache = pool
            .get(key)
            .and_then(|entry| {
                // 从 CacheEntry 中提取 data (Arc<dyn Any + Send + Sync>) 并下转为 Arc<T>
                Arc::downcast::<T>(entry.data.clone())
                    .ok()
                    .map(|typed_data| CacheEntry {
                        data: typed_data,
                        params_hash: entry.params_hash,
                    })
            })
            .ok_or_else(|| {
                Error::CacheNotInitialized(format!("Cache for key '{}' is not initialized", key))
            })?;

        Ok(cache)
    }

    /// 计算参数的 hash
    fn compute_hash<T: Hash>(params: &[T]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for param in params {
            param.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// 删除模型
    pub fn remove(
        &self,
        key: &str,
    ) -> Result<Option<CacheEntry<Arc<dyn Any + Send + Sync>>>, Error> {
        let mut pool = self.pool.write().map_err(|e| {
            error!("get cache write lock error: {e:?}");
            Error::LockError(e.to_string())
        })?;
        Ok(pool.remove(key))
    }

    /// 获取数据并转换为具体类型
    pub fn get_data<T: Send + Sync + 'static>(&self, key: &str) -> Result<Option<Arc<T>>, Error> {
        let pool = self.pool.read().map_err(|e| {
            error!("get cache read lock error: {e:?}");
            Error::LockError(e.to_string())
        })?;
        let data = pool
            .get(key)
            .and_then(|arc_any| Arc::downcast::<T>(arc_any.data.clone()).ok());

        Ok(data)
    }

    /// 获取模型 hash 值
    pub fn get_params_hash(&self, key: &str) -> Option<u64> {
        let pool = self
            .pool
            .read()
            .map_err(|e| {
                error!("get cache read lock error: {e:?}");
                Error::LockError(e.to_string())
            })
            .ok()?;
        pool.get(key).map(|v| v.params_hash)
    }

    /// 获取所有的缓存 key
    pub fn get_keys(&self) -> Result<Vec<String>, Error> {
        let pool = self.pool.read().map_err(|e| {
            error!("get cache read lock error: {e:?}");
            Error::LockError(e.to_string())
        })?;
        Ok(pool.keys().cloned().collect())
    }

    /// 清空所有模型
    pub fn clear(&self) -> Result<(), Error> {
        self.pool
            .write()
            .map_err(|e| {
                error!("get cache write lock error: {e:?}");
                Error::LockError(e.to_string())
            })?
            .clear();

        Ok(())
    }

    /// 获取缓存大小
    pub fn len(&self) -> Result<usize, Error> {
        let pool = self.pool.read().map_err(|e| {
            error!("get cache read lock error: {e:?}");
            Error::LockError(e.to_string())
        })?;
        Ok(pool.len())
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> Result<bool, Error> {
        let pool = self.pool.read().map_err(|e| {
            error!("get cache read lock error: {e:?}");
            Error::LockError(e.to_string())
        })?;

        Ok(pool.is_empty())
    }

    /// 将 `Arc<dyn Any + Send + Sync>` 转换为 `Arc<T>`，如果类型匹配则返回 `Some(Arc<T>)`，否则返回 `None`
    pub fn cast_arc<T: Send + Sync + 'static>(
        arc_any: Arc<dyn Any + Send + Sync>,
    ) -> Option<Arc<T>> {
        Arc::downcast::<T>(arc_any).ok()
    }
}

/// 便捷函数：获取全局缓存
pub fn global_cache() -> &'static Arc<CacheManager> {
    CacheManager::global()
}

#[cfg(test)]
mod tests {
    use super::*;

    // 定义一个模型类型
    #[derive(Debug, Clone)]
    struct MyModel {
        value: i32,
    }

    #[test]
    fn test_simple() -> anyhow::Result<()> {
        // 测试插入
        {
            let cache = CacheManager::global();

            cache.insert("model1", &[1, 2, 3], Arc::new(MyModel { value: 42 }))?;
            assert!(cache.get::<MyModel>("model1").is_ok());
        }

        // 测试获取
        {
            let cache = CacheManager::global();

            let model = cache.get::<MyModel>("model1").unwrap();
            assert_eq!(model.data.value, 42);
        }

        Ok(())
    }

    #[test]
    fn test_hash_based_update() -> anyhow::Result<()> {
        let cache = CacheManager::global();

        // 测试插入
        {
            cache.insert("model1", &[1, 2, 3], Arc::new(MyModel { value: 42 }))?;
            assert!(cache.get::<MyModel>("model1").is_ok());
        }

        // 测试 hash 值未变化时不更新
        {
            cache.insert("model1", &[1, 2, 3], Arc::new(MyModel { value: 100 }))?;
            let model = cache.get::<MyModel>("model1").unwrap();
            assert_eq!(model.data.value, 42); // 仍然是旧值，因为 hash 未变化
        }

        // 测试 hash 值变化时更新
        {
            cache.insert("model1", &[1, 2, 4], Arc::new(MyModel { value: 100 }))?;
            let model = cache.get::<MyModel>("model1").unwrap();
            assert_eq!(model.data.value, 100); // 新值，因为 hash 变化了
        }

        Ok(())
    }

    #[test]
    fn test_get_or_insert() -> anyhow::Result<()> {
        let cache = CacheManager::global();

        let params = &[1, 2, 4];

        // 第一次调用，初始化模型
        {
            let model = cache
                .get_or_insert("model1", params, || Ok(Arc::new(MyModel { value: 42 })))
                .unwrap();
            assert_eq!(model.data.value, 42);
        }

        // 第二次调用，hash 一致，返回现有模型
        {
            let model = cache
                .get_or_insert("model1", params, || {
                    println!("This should not be printed!");
                    Ok(Arc::new(MyModel { value: 80 }))
                })
                .unwrap();
            assert_eq!(model.data.value, 42);
        }

        // 第三次调用，hash 不一致，更新模型
        {
            let model = cache
                .get_or_insert("model1", params, || Ok(Arc::new(MyModel { value: 100 })))
                .unwrap();
            assert_eq!(model.data.value, 100);
        }

        Ok(())
    }
}
