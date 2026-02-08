//! 模型

use std::{
    collections::hash_map::DefaultHasher,
    ffi::CString,
    hash::{Hash, Hasher},
    pin::pin,
    sync::Arc,
};

use llama_cpp_2::{
    LlamaBackendDevice, list_llama_ggml_backend_devices,
    llama_backend::LlamaBackend,
    model::{
        LlamaModel,
        params::{LlamaModelParams, LlamaSplitMode},
    },
    mtmd::{MtmdContext, MtmdContextParams, mtmd_default_marker},
};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::{CacheManager, global_cache};
use crate::{cache::CacheType, error::Error};

/// 生成模型路径的唯一哈希键
fn hash_model_key(path: &str) -> String {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    format!("model:{:016x}", hasher.finish())
}

/// 生成 mmproj 路径的唯一哈希键
fn hash_mmproj_key(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    format!("mmproj:{:016x}", hasher.finish())
}

/// 模型参数配置
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the model file (e.g., "ggml-model.bin")
    #[serde(default)]
    pub model_path: String,

    /// Path to the multimodal projection file (e.g., "mmproj-model.bin")
    /// Required for models with multimodal capabilities (e.g., vision or audio).
    #[serde(default)]
    pub mmproj_path: String,

    /// Index of the main GPU to use.
    /// Relevant for multi-GPU systems.
    #[serde(default)]
    pub main_gpu: i32,
    /// 设备索引列表
    /// This option overrides `main-gpu` and enables multi-GPU.
    /// Set devices to use by index, separated by commas (e.g. --devices 0,1,2). Overrides main-gpu and enables multi-GPU.
    #[serde(default)]
    pub devices: Vec<usize>,
    /// Number of GPU layers to offload.
    /// 0 = CPU only mode, >0 = GPU mode with specified layers
    #[serde(default)]
    pub n_gpu_layers: u32,

    /// Keep MoE layers on CPU
    #[serde(default)]
    pub cmoe: bool,

    /// Number of threads to use during generation.
    /// Set to a specific value to limit CPU usage.
    #[serde(default)]
    pub n_threads: i32,

    /// Media marker. If not provided, the default marker will be used.
    #[serde(default)]
    pub media_marker: Option<String>,

    /// 是否打印详细信息
    #[serde(default)]
    pub verbose: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            mmproj_path: String::new(),
            main_gpu: 0,
            n_gpu_layers: 0, // 0 = CPU only mode
            cmoe: false,
            devices: Vec::new(),
            media_marker: Some("<__media__>".to_string()), // 默认媒体标记
            n_threads: 4,
            verbose: false,
        }
    }
}

impl ModelConfig {
    pub fn with_model_path(mut self, path: impl Into<String>) -> Self {
        self.model_path = path.into();
        self
    }

    pub fn with_cmoe(mut self, cmoe: bool) -> Self {
        self.cmoe = cmoe;
        self
    }

    pub fn with_main_gpu(mut self, gpu: i32) -> Self {
        self.main_gpu = gpu;
        self
    }

    /// 设置 GPU 层数（0 = CPU 模式）
    pub fn with_gpu_layers(mut self, layers: u32) -> Self {
        self.n_gpu_layers = layers;
        self
    }

    /// 检查是否使用 GPU
    /// n_gpu_layers > 0 表示启用 GPU
    pub fn use_gpu(&self) -> bool {
        self.n_gpu_layers > 0
    }

    pub fn with_media_marker(mut self, media_marker: impl Into<String>) -> Self {
        self.media_marker = Some(media_marker.into());
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    ///  vec to str, e.g. "0,1,2"
    pub fn devices_str(&self) -> String {
        self.devices
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<String>>()
            .join(",")
    }
}

/// 模型
#[derive(Debug, Clone)]
pub struct Model {
    config: ModelConfig,
    cache: Arc<CacheManager>,
    cache_model_key: String,
    cache_mmproj_key: String,
}

impl Model {
    pub fn new(model_path: impl Into<String>, mmproj_path: impl Into<Option<String>>) -> Self {
        let cache = global_cache().clone();
        let model_path = model_path.into();
        let mmproj_path = mmproj_path.into().unwrap_or_default();

        // 基于路径生成唯一缓存键，确保不同模型不会互相覆盖
        let cache_model_key = hash_model_key(&model_path);
        let cache_mmproj_key = hash_mmproj_key(&mmproj_path);

        Self {
            config: ModelConfig {
                model_path: model_path.clone(),
                mmproj_path: mmproj_path.clone(),
                ..ModelConfig::default()
            },
            cache,
            cache_model_key,
            cache_mmproj_key,
        }
    }

    pub fn from_config(config: ModelConfig) -> Self {
        let cache = global_cache().clone();

        // 基于路径生成唯一缓存键
        let cache_model_key = hash_model_key(&config.model_path);
        let cache_mmproj_key = hash_mmproj_key(&config.mmproj_path);

        Self {
            config: config.clone(),
            cache,
            cache_model_key,
            cache_mmproj_key,
        }
    }

    pub fn with_cache_key(
        mut self,
        cache_model_key: impl Into<String>,
        cache_mmproj_key: Option<impl Into<String>>,
    ) -> Self {
        self.cache_model_key = cache_model_key.into();
        if let Some(cache_mmproj_key) = cache_mmproj_key {
            self.cache_mmproj_key = cache_mmproj_key.into();
        }

        self
    }
}

impl Model {
    /// 获取可用设备列表
    pub fn devices(&self) -> Vec<LlamaBackendDevice> {
        let devices = list_llama_ggml_backend_devices();

        if self.config.verbose {
            for (i, dev) in devices.iter().enumerate() {
                println!("Device {i:>2}: {}", dev.name);
                println!("           Description: {}", dev.description);
                println!("           Device Type: {:?}", dev.device_type);
                println!("           Backend: {}", dev.backend);
                println!(
                    "           Memory total: {:?} MiB",
                    dev.memory_total / 1024 / 1024
                );
                println!(
                    "           Memory free:  {:?} MiB",
                    dev.memory_free / 1024 / 1024
                );
            }
        }

        devices
    }

    /// 加载模型
    pub fn load_llama_model(&self, backend: &LlamaBackend) -> Result<LlamaModel, Error> {
        // 加载新模型
        info!("Loading model: {:?}", self.config.model_path);

        let mut model_params = LlamaModelParams::default();

        if !self.config.devices.is_empty() {
            // 多 GPU 模式
            model_params = model_params
                .with_devices(&self.config.devices)
                .map_err(|e| {
                    error!("Failed to set devices: {:?}", e);
                    e
                })?;
        } else if self.config.n_gpu_layers > 0 {
            // 单 GPU 模式
            model_params = model_params.with_main_gpu(self.config.main_gpu);
            // Enable single GPU mode
            model_params = model_params.with_split_mode(LlamaSplitMode::None);
        }

        if self.config.n_gpu_layers > 0 {
            model_params = model_params.with_n_gpu_layers(self.config.n_gpu_layers);
            // 设置 use_mlock
            // model_params = model_params.with_use_mlock(self.config.use_mlock);
        }

        let mut model_params = pin!(model_params);

        // for (k, v) in &key_value_overrides {
        //     let k = CString::new(k.as_bytes())?;
        //     model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        // }

        if self.config.cmoe {
            model_params.as_mut().add_cpu_moe_override();
        }

        // Load model
        info!("Loading model with params: {:?}", model_params);
        let model = LlamaModel::load_from_file(backend, &self.config.model_path, &model_params)
            .map_err(|e| {
                error!("Failed to load model: {:?}", e);
                e
            })?;

        Ok(model)
    }

    /// 加载模型（带缓存）
    pub fn load_cache_llama_model(&self, backend: &LlamaBackend) -> Result<Arc<LlamaModel>, Error> {
        if !true {
            self.cache.remove(&self.cache_model_key)?;
            return Ok(Arc::new(self.load_llama_model(backend)?));
        }

        // 尝试从缓存获取
        if let Some(entry) = self.cache.get_data::<LlamaModel>(&self.cache_model_key)? {
            info!("Model cache hit: {:?}", self.config.model_path);
            return Ok(entry);
        }

        // 加载新模型
        let model = Arc::new(self.load_llama_model(backend)?);

        // 缓存参数
        let params = vec![
            self.config.model_path.clone(),
            self.config.main_gpu.to_string(),
            self.config.n_gpu_layers.to_string(),
            self.config.cmoe.to_string(),
            self.config
                .devices
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<String>>()
                .join(","),
        ];

        // 缓存模型
        self.cache.insert_or_update(
            &self.cache_model_key,
            CacheType::Model,
            &params,
            model.clone() as Arc<dyn std::any::Any + Send + Sync>,
        )?;

        Ok(model)
    }

    /// Load MTMD context
    pub fn load_mtmd_mtmd_context(&self, model: Arc<LlamaModel>) -> Result<MtmdContext, Error> {
        let use_gpu = self.config.n_gpu_layers > 0;

        // Create media marker CString
        let media_marker = CString::new(
            self.config
                .media_marker
                .as_ref()
                .unwrap_or(&mtmd_default_marker().to_string())
                .clone(),
        )
        .map_err(|e| Error::InvalidInput {
            field: "media_marker".into(),
            message: e.to_string(),
        })?;

        // 确保 n_threads 有合理的值（mmproj 编码需要足够线程）
        let n_threads = if self.config.n_threads > 0 {
            self.config.n_threads
        } else {
            std::thread::available_parallelism()
                .map(|p| p.get() as i32)
                .unwrap_or(4)
        };

        let mtmd_params = MtmdContextParams {
            use_gpu,
            print_timings: self.config.verbose,
            n_threads,
            media_marker,
        };
        let mtmd_context =
            MtmdContext::init_from_file(&self.config.mmproj_path, &model, &mtmd_params)?;

        Ok(mtmd_context)
    }

    /// Load cache MTMD context
    pub fn load_cache_mtmd_context(
        &self,
        model: Arc<LlamaModel>,
    ) -> Result<Arc<MtmdContext>, Error> {
        if !true {
            self.cache.remove(&self.cache_mmproj_key)?;
            let ctx = self.load_mtmd_mtmd_context(model)?;
            return Ok(ctx.into());
        }

        // 尝试从缓存获取
        if let Some(entry) = self
            .cache
            .get_data::<SendableMtmdContext>(&self.cache_mmproj_key)?
        {
            info!("MtmdContext cache hit: {:?}", self.config.mmproj_path);
            return Ok(entry.0.clone());
        }

        // 加载新模型
        let ctx = self.load_mtmd_mtmd_context(model)?;
        let sendable_ctx = Arc::new(SendableMtmdContext(ctx.into()));

        // 缓存参数
        let params = vec![
            self.config.mmproj_path.clone(),
            (self.config.n_gpu_layers == 0).to_string(),
            self.config.media_marker.clone().unwrap_or_default(),
            self.config.n_threads.to_string(),
            self.config.verbose.to_string(),
        ];

        // 缓存模型
        self.cache.insert_or_update(
            &self.cache_mmproj_key,
            CacheType::Model,
            &params,
            sendable_ctx.clone() as Arc<dyn std::any::Any + Send + Sync>,
        )?;

        Ok(sendable_ctx.0.clone())
    }
}

/// 包装 MtmdContext 使其支持 Send
///
/// 注意：MtmdContext 内部包含 NonNull 指针，但 llama.cpp 的 mtmd_context
/// 实际上可以安全地在线程间移动（只要不在多个线程同时访问）
pub struct SendableMtmdContext(Arc<MtmdContext>);

// 不安全地实现 Send trait
// 安全前提：MtmdContext 只能在单线程中使用，但可以在不同线程间传递
// 只要确保不会同时从多个线程访问即可
unsafe impl Send for SendableMtmdContext {}
unsafe impl Sync for SendableMtmdContext {}
/// 智能指针解引用
impl std::ops::Deref for SendableMtmdContext {
    type Target = Arc<MtmdContext>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
