use std::{
    any::Any,
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, OnceLock, RwLock},
};

use crate::error::Error;

static GLOBAL_CACHE: OnceLock<RwLock<ModelManager>> = OnceLock::new();

/// 全局单例的模型缓存
#[derive(Debug, Default)]
pub struct ModelManager {
    pool: HashMap<String, Arc<dyn Any + Send + Sync>>,
    hashes: HashMap<String, String>, // 存储每个 key 对应的 hash 值
}

impl ModelManager {
    // 单例模式，全局唯一的模型缓存
    pub fn global() -> &'static RwLock<ModelManager> {
        GLOBAL_CACHE.get_or_init(|| RwLock::new(ModelManager::default()))
    }

    /// 添加或更新模型（仅当 hash 值变化时更新）
    pub fn insert(&mut self, key: &str, model: Arc<dyn Any + Send + Sync>, model_hash: &str) {
        if let Some(current_hash) = self.hashes.get(key)
            && current_hash == model_hash
        {
            return; // hash 值未变化，不更新
        }
        self.pool.insert(key.to_string(), model);
        self.hashes.insert(key.to_string(), model_hash.to_string());
    }

    /// 强制更新模型（无论 hash 值是否变化）
    pub fn force_update(&mut self, key: &str, model: Arc<dyn Any + Send + Sync>, model_hash: &str) {
        self.pool.insert(key.to_string(), model);
        self.hashes.insert(key.to_string(), model_hash.to_string());
    }

    /// 获取或添加或更新模型（通过闭包初始化）并转换为具体类型
    pub fn get_or_insert<F, T: Send + Sync + 'static>(
        &mut self,
        key: &str,
        model_hash: &str,
        handler: F,
    ) -> Result<Arc<T>, Error>
    where
        F: FnOnce() -> Result<Arc<T>, Error>,
    {
        // 检查当前 hash 是否存在
        if let Some(current_hash) = self.hashes.get(key)
            && current_hash == model_hash
        {
            // hash 值未变化，直接返回现有模型
            let model = self.get::<T>(key).ok_or_else(|| {
                Error::ModelNotInitialized(format!("Model for key '{}' is not initialized", key))
            })?;
            return Ok(model);
        }

        // hash 值不存在或不一致，初始化新模型并更新缓存
        let model = handler()?;
        self.pool.insert(
            key.to_string(),
            Arc::clone(&model) as Arc<dyn Any + Send + Sync>,
        );
        self.hashes.insert(key.to_string(), model_hash.to_string());
        Ok(model)
    }

    /// 删除模型
    pub fn remove(&mut self, key: &str) -> Option<Arc<dyn Any + Send + Sync>> {
        self.hashes.remove(key);
        self.pool.remove(key)
    }

    /// 获取模型并转换为具体类型
    pub fn get<T: Send + Sync + 'static>(&self, key: &str) -> Option<Arc<T>> {
        self.pool.get(key).and_then(|arc_any| {
            let arc_any = Arc::clone(arc_any);
            Arc::downcast::<T>(arc_any).ok()
        })
    }

    /// 计算 Vec<T> 的哈希键
    pub fn cal_hash<T: Hash>(&self, params: &[T]) -> String {
        let mut hasher = DefaultHasher::new();
        for param in params {
            param.hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }

    /// 获取模型 hash 值
    pub fn get_hash(&self, key: &str) -> Option<String> {
        self.hashes.get(key).cloned()
    }

    /// 清空所有模型
    pub fn clear(&mut self) {
        self.pool.clear();
        self.hashes.clear();
    }

    /// 将 `Arc<dyn Any + Send + Sync>` 转换为 `Arc<T>`，如果类型匹配则返回 `Some(Arc<T>)`，否则返回 `None`
    pub fn cast_arc<T: Send + Sync + 'static>(
        arc_any: Arc<dyn Any + Send + Sync>,
    ) -> Option<Arc<T>> {
        Arc::downcast::<T>(arc_any).ok()
    }
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
            let cache = ModelManager::global();
            let mut cache = cache.write().unwrap();

            let model_hash = cache.cal_hash(&[1, 2, 3]);

            cache.insert("model1", Arc::new(MyModel { value: 42 }), &model_hash);
            assert!(cache.get::<MyModel>("model1").is_some());
            assert_eq!(cache.get_hash("model1"), Some(model_hash));
        }

        // 测试获取
        {
            let cache = ModelManager::global().read().unwrap();

            let model = cache.get::<MyModel>("model1").unwrap();
            assert_eq!(model.value, 42);
        }

        Ok(())
    }

    #[test]
    fn test_hash_based_update() -> anyhow::Result<()> {
        let cache = ModelManager::global();

        // 测试插入
        {
            let mut cache = cache.write().unwrap();

            let model_hash = cache.cal_hash(&[1, 2, 3]);

            cache.insert("model1", Arc::new(MyModel { value: 42 }), &model_hash);
            assert!(cache.get::<MyModel>("model1").is_some());
            assert_eq!(cache.get_hash("model1"), Some(model_hash));
        }

        // 测试 hash 值未变化时不更新
        {
            let mut cache = cache.write().unwrap();

            let model_hash = cache.cal_hash(&[1, 2, 3]);

            cache.insert("model1", Arc::new(MyModel { value: 100 }), &model_hash);
            let model = cache.get::<MyModel>("model1").unwrap();
            assert_eq!(model.value, 42); // 仍然是旧值，因为 hash 未变化
        }

        // 测试 hash 值变化时更新
        {
            let mut cache = cache.write().unwrap();

            let model_hash = cache.cal_hash(&[1, 2, 4]);

            cache.insert("model1", Arc::new(MyModel { value: 100 }), &model_hash);
            let model = cache.get::<MyModel>("model1").unwrap();
            assert_eq!(model.value, 100); // 新值，因为 hash 变化了
        }

        Ok(())
    }

    #[test]
    fn test_get_or_insert() -> anyhow::Result<()> {
        let cache = ModelManager::global();

        // 第一次调用，初始化模型
        {
            let mut cache = cache.write().unwrap();
            let model = cache
                .get_or_insert("model1", "hash123", || Ok(Arc::new(MyModel { value: 42 })))
                .unwrap();
            assert_eq!(model.value, 42);
        }

        // 第二次调用，hash 一致，返回现有模型
        {
            let mut cache = cache.write().unwrap();
            let model = cache
                .get_or_insert("model1", "hash123", || {
                    println!("This should not be printed!");
                    Ok(Arc::new(MyModel { value: 80 }))
                })
                .unwrap();
            assert_eq!(model.value, 42);
        }

        // 第三次调用，hash 不一致，更新模型
        {
            let mut cache = cache.write().unwrap();
            let model = cache
                .get_or_insert("model1", "hash456", || Ok(Arc::new(MyModel { value: 100 })))
                .unwrap();
            assert_eq!(model.value, 100);
        }

        Ok(())
    }
}
