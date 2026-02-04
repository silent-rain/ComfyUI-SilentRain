//! llama.cpp Vision v2
//!
//! v2 版本改进：
//! - 支持从 LlamaCppModelv2 接收模型缓存 key
//! - 自动处理图像批处理
//! - 支持进度条显示
//! - 使用 llama_cpp_core 的 Pipeline 进行推理

use log::{error, info};
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use std::{path::PathBuf, sync::Arc};

use llama_cpp_core::{
    CacheManager, GenerateRequest, Pipeline, sampler::SamplerConfig, types::MediaData,
};

use crate::{
    core::{category::CATEGORY_LLAMA_CPP, utils::image::tensor_to_raw_data},
    error::Error,
    wrapper::{
        comfy::folder_paths::FolderPaths,
        comfyui::{
            PromptServer,
            types::{NODE_BOOLEAN, NODE_IMAGE, NODE_INT, NODE_INT_MAX, NODE_SEED_MAX, NODE_STRING},
        },
    },
};

/// LlamaCpp Vision v2
///
/// ### 版本差异
/// - **v1**: 独立加载模型，单图处理
/// - **v2**:
///   - 复用 LlamaCppModelv2 加载的模型
///   - 支持图像批处理和进度条
///   - 使用 llama_cpp_core 的高性能 Pipeline
///   - 支持 keep_context 保持视觉对话上下文
#[pyclass(subclass)]
pub struct LlamaCppVisionv2 {
    cache: Arc<CacheManager>,
    pipeline: Arc<Pipeline>,
}

impl PromptServer for LlamaCppVisionv2 {}

#[pymethods]
impl LlamaCppVisionv2 {
    #[new]
    fn new() -> PyResult<Self> {
        let cache = CacheManager::global();
        let pipeline = Pipeline::with_cache(cache.clone()).map_err(|e| {
            PyErr::new::<PyRuntimeError, _>(format!("Failed to create pipeline: {}", e))
        })?;

        Ok(Self {
            cache,
            pipeline: Arc::new(pipeline),
        })
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str, &'static str, &'static str) {
        (NODE_IMAGE, NODE_STRING, NODE_STRING) // (images, captions, context_key)
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str, &'static str, &'static str) {
        ("images", "captions", "context_key")
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LLAMA_CPP;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "llama.cpp vision v2 - Image understanding with model reuse"
    }

    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classattr]
    #[pyo3(name = "EXPERIMENTAL")]
    fn experimental() -> bool {
        true
    }

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);

                // 图像输入
                required.set_item(
                    "images",
                    (NODE_IMAGE, {
                        let params = PyDict::new(py);
                        params.set_item("forceInput", true)?;
                        params.set_item("tooltip", "Input images [batch, H, W, C]")?;
                        params
                    }),
                )?;

                // 模型句柄
                required.set_item(
                    "model_handle",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("tooltip", "Model cache key from LlamaCppModelv2")?;
                        params
                    }),
                )?;

                // MMPROJ 路径
                let mmproj_list = Self::get_mmproj_model_list();
                required.set_item(
                    "mmproj_path",
                    (mmproj_list, {
                        let params = PyDict::new(py);
                        params.set_item("tooltip", "MMPROJ model path for vision")?;
                        params
                    }),
                )?;

                required.set_item(
                    "system_prompt",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", "Describe the image in detail.")?;
                        params.set_item("multiline", true)?;
                        params.set_item("tooltip", "System prompt for image description")?;
                        params
                    }),
                )?;

                required.set_item(
                    "user_prompt",
                    (NODE_STRING, {
                        let params = PyDict::new(py);
                        params.set_item("default", "What do you see in this image?")?;
                        params.set_item("multiline", true)?;
                        params.set_item("tooltip", "User prompt about the image")?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_ctx",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 4096)?;
                        params.set_item("min", 256)?;
                        params.set_item("max", NODE_INT_MAX)?;
                        params.set_item("step", 1)?;
                        params.set_item("tooltip", "Context window size")?;
                        params
                    }),
                )?;

                required.set_item(
                    "n_predict",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", 512)?;
                        params.set_item("min", -1)?;
                        params.set_item("max", NODE_INT_MAX)?;
                        params.set_item("step", 1)?;
                        params.set_item("tooltip", "Max tokens to predict")?;
                        params
                    }),
                )?;

                required.set_item(
                    "seed",
                    (NODE_INT, {
                        let params = PyDict::new(py);
                        params.set_item("default", -1)?;
                        params.set_item("min", -1)?;
                        params.set_item("max", NODE_SEED_MAX)?;
                        params.set_item("step", 1)?;
                        params.set_item("tooltip", "Random seed (-1 = random)")?;
                        params
                    }),
                )?;

                required.set_item(
                    "keep_context",
                    (NODE_BOOLEAN, {
                        let params = PyDict::new(py);
                        params.set_item("default", false)?;
                        params.set_item("tooltip", "Keep multi-image context")?;
                        params
                    }),
                )?;

                required
            })?;

            Ok(dict.into())
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute", signature = (images, model_handle, mmproj_path, system_prompt, user_prompt, n_ctx, n_predict, seed, keep_context))]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        images: Bound<'py, PyAny>,
        model_handle: String,
        mmproj_path: String,
        system_prompt: String,
        user_prompt: String,
        n_ctx: u32,
        n_predict: i32,
        seed: i32,
        keep_context: bool,
    ) -> PyResult<(Bound<'py, PyAny>, Vec<String>, String)> {
        let result = self.generate(
            py,
            images,
            model_handle,
            mmproj_path,
            system_prompt,
            user_prompt,
            n_ctx,
            n_predict,
            seed,
            keep_context,
        );

        match result {
            Ok((images_out, captions, context_key)) => Ok((images_out, captions, context_key)),
            Err(e) => {
                error!("LlamaCppVisionv2 error: {e}");
                if let Err(e) = self.send_error(py, "LlamaCppVisionv2".to_string(), e.to_string()) {
                    error!("send error failed: {e}");
                }
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LlamaCppVisionv2 {
    /// 生成图像描述
    #[allow(clippy::too_many_arguments)]
    fn generate<'py>(
        &self,
        py: Python<'py>,
        images: Bound<'py, PyAny>,
        model_handle: String,
        mmproj_path: String,
        system_prompt: String,
        user_prompt: String,
        n_ctx: u32,
        n_predict: i32,
        seed: i32,
        keep_context: bool,
    ) -> Result<(Bound<'py, PyAny>, Vec<String>, String), Error> {
        // 解析图像数据
        let image_raw_datas = tensor_to_raw_data(&images)?;
        let batch_size = image_raw_datas.len();

        info!("Processing {} images with vision v2", batch_size);

        // 解析模型句柄
        let (model_path, model_config) = self.resolve_model_handle(&model_handle)?;

        // 解析 mmproj 路径
        let base_models_dir = FolderPaths::default().model_path();
        let mmproj_full_path = base_models_dir.join("LLM").join(&mmproj_path);

        if !mmproj_full_path.exists() {
            return Err(Error::FileNotFound(mmproj_path));
        }

        // 创建进度条
        let pbar = py
            .import("comfy")?
            .getattr("utils")?
            .call_method1("ProgressBar", (batch_size,))?;

        // 构建采样配置
        let sampler_config = SamplerConfig::default().seed(seed);

        let mut captions = Vec::with_capacity(batch_size);

        // 处理每张图像
        for (idx, image_data) in image_raw_datas.into_iter().enumerate() {
            info!("Processing image {}/{}", idx + 1, batch_size);

            // 构建视觉推理请求
            let request = GenerateRequest::text(&user_prompt)
                .with_system(&system_prompt)
                .with_media(MediaData::new_image(image_data))
                .with_keep_context(keep_context);

            // 执行推理
            let response = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current()
                    .block_on(async { self.pipeline.generate(&request).await })
            })
            .map_err(|e| Error::InvalidInput(format!("Generation failed: {}", e)))?;

            captions.push(response.text);
            pbar.call_method1("update", (1,))?;
        }

        // 生成上下文 key（用于多图对话）
        let context_key = if keep_context && !captions.is_empty() {
            format!(
                "vision_ctx:{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            )
        } else {
            String::new()
        };

        info!(
            "Vision inference complete, generated {} captions",
            captions.len()
        );

        Ok((images, captions, context_key))
    }

    /// 解析模型句柄
    fn resolve_model_handle(
        &self,
        handle: &str,
    ) -> Result<(PathBuf, llama_cpp_core::model::ModelConfig), Error> {
        if let Some((model_path, config)) = self
            .cache
            .get_data::<(String, llama_cpp_core::model::ModelConfig)>(handle)?
        {
            let base_models_dir = FolderPaths::default().model_path();
            let full_path = base_models_dir.join("LLM").join(&model_path);
            return Ok((full_path, *config));
        }

        let base_models_dir = FolderPaths::default().model_path();
        let full_path = base_models_dir.join("LLM").join(handle);

        if full_path.exists() {
            Ok((full_path, llama_cpp_core::model::ModelConfig::default()))
        } else {
            Err(Error::FileNotFound(handle.to_string()))
        }
    }

    /// 获取 mmproj 模型列表
    fn get_mmproj_model_list() -> Vec<String> {
        let llm_list = FolderPaths::default().get_filename_list("LLM");
        let text_encoders_list = FolderPaths::default().get_filename_list("text_encoders");

        [llm_list, text_encoders_list]
            .concat()
            .into_iter()
            .filter(|v| v.ends_with(".gguf"))
            .filter(|v| v.contains("mmproj"))
            .collect()
    }
}
