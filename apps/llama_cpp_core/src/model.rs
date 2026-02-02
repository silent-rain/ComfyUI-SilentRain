//! 模型

use std::{pin::pin, sync::Arc};

use llama_cpp_2::{
    LlamaBackendDevice, list_llama_ggml_backend_devices,
    llama_backend::LlamaBackend,
    model::{
        LlamaModel,
        params::{LlamaModelParams, LlamaSplitMode},
    },
};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::error::Error;
use crate::{CacheManager, global_cache};

/// 模型参数配置
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Path to the model file (e.g., "ggml-model.bin")
    #[serde(default)]
    pub model_path: String,

    /// Disable offloading layers to the gpu
    #[serde(default)]
    pub disable_gpu: bool,
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
    /// Higher values offload more work to the GPU.
    #[serde(default)]
    pub n_gpu_layers: u32,

    /// Keep MoE layers on CPU
    #[serde(default)]
    pub cmoe: bool,

    /// Force system to keep model in RAM (use mlock)
    #[serde(default)]
    pub use_mlock: bool,

    /// 是否缓存模型
    #[serde(default)]
    pub cache_model: bool,
    /// 是否打印详细信息
    #[serde(default)]
    pub verbose: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            disable_gpu: true,
            main_gpu: 0,
            n_gpu_layers: 0,
            cmoe: false,
            use_mlock: false,
            devices: Vec::new(),
            cache_model: false,
            verbose: false,
        }
    }
}

/// 模型
#[derive(Debug, Clone)]
pub struct Model {
    config: ModelConfig,
    cache: Arc<CacheManager>,
    cache_key: String,
}

impl Model {
    pub fn new(model_path: impl Into<String>) -> Self {
        let cache = global_cache().clone();
        let model_path = model_path.into();
        Self {
            config: ModelConfig {
                model_path: model_path.clone(),
                ..ModelConfig::default()
            },
            cache,
            cache_key: format!("model:{model_path}"),
        }
    }

    pub fn from_config(config: ModelConfig) -> Self {
        let cache = global_cache().clone();
        Self {
            config: config.clone(),
            cache,
            cache_key: format!("model:{}", config.model_path),
        }
    }

    pub fn with_cache_key(mut self, cache_key: impl Into<String>) -> Self {
        self.cache_key = cache_key.into();
        self
    }

    pub fn with_model_path(mut self, path: impl Into<String>) -> Self {
        self.config.model_path = path.into();
        self
    }

    pub fn with_disable_gpu(mut self, disable_gpu: bool) -> Self {
        self.config.disable_gpu = disable_gpu;
        self
    }

    pub fn with_cmoe(mut self, cmoe: bool) -> Self {
        self.config.cmoe = cmoe;
        self
    }

    pub fn with_main_gpu(mut self, gpu: i32) -> Self {
        self.config.main_gpu = gpu;
        self
    }

    pub fn with_n_gpu_layers(mut self, layers: u32) -> Self {
        self.config.n_gpu_layers = layers;
        self
    }

    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.config.use_mlock = use_mlock;
        self
    }

    pub fn with_cache_model(mut self, cache_model: bool) -> Self {
        self.config.cache_model = cache_model;
        self
    }

    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
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
    pub fn load_model(&self, backend: &LlamaBackend) -> Result<LlamaModel, Error> {
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
        } else if !self.config.disable_gpu {
            // 单 GPU 模式
            model_params = model_params.with_main_gpu(self.config.main_gpu);
            // Enable single GPU mode
            model_params = model_params.with_split_mode(LlamaSplitMode::None);
        }

        if !self.config.disable_gpu {
            model_params = model_params.with_n_gpu_layers(self.config.n_gpu_layers);
        }

        // 设置 use_mlock
        model_params = model_params.with_use_mlock(self.config.use_mlock);

        let mut model_params = pin!(model_params);

        // for (k, v) in &key_value_overrides {
        //     let k = CString::new(k.as_bytes())?;
        //     model_params.as_mut().append_kv_override(k.as_c_str(), *v);
        // }

        if !self.config.disable_gpu && self.config.cmoe {
            model_params.as_mut().add_cpu_moe_override();
        }

        // Load model
        let model = LlamaModel::load_from_file(backend, &self.config.model_path, &model_params)
            .map_err(|e| {
                error!("Failed to load model: {:?}", e);
                e
            })?;

        Ok(model)
    }

    /// 加载模型（带缓存）
    pub fn load_cache_model(&self, backend: &LlamaBackend) -> Result<Arc<LlamaModel>, Error> {
        if !self.config.cache_model {
            return Ok(Arc::new(self.load_model(backend)?));
        }

        // 尝试从缓存获取
        if let Some(entry) = self.cache.get_data::<LlamaModel>(&self.cache_key)? {
            info!("Model cache hit: {:?}", self.config.model_path);
            return Ok(entry);
        }

        // 加载新模型
        let model = Arc::new(self.load_model(backend)?);

        let params = vec![
            self.config.model_path.clone(),
            self.config.main_gpu.to_string(),
            self.config.n_gpu_layers.to_string(),
        ];

        // 缓存模型
        self.cache.insert(
            &self.cache_key,
            &params,
            model.clone() as Arc<dyn std::any::Any + Send + Sync>,
        )?;

        Ok(model)
    }
}
