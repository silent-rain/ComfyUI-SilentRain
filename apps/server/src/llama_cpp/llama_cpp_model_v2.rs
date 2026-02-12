//! llama.cpp Model Loader v2

use log::error;
use pyo3::{
    Bound, Py, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use pythonize::pythonize;
use serde::{Deserialize, Serialize};

use crate::{
    core::{
        category::CATEGORY_LLAMA_CPP,
        node_base::{InputSpec, InputType},
    },
    error::Error,
    wrapper::{
        comfy::folder_paths::FolderPaths,
        comfyui::{PromptServer, types::NODE_LLAMA_CPP_MODEL_V2},
    },
};

/// LlamaCpp模型参数结构体
#[derive(Debug, Default, Clone, Deserialize, Serialize)]
pub struct LlamaCppModelParams {
    pub model_path: String,
    pub mmproj_path: String,
    pub cache_model: bool,
    pub cache_model_key: String,
}

/// LlamaCpp Model Loader v2
#[pyclass(subclass)]
pub struct LlamaCppModelv2 {}

impl PromptServer for LlamaCppModelv2 {}

#[pymethods]
impl LlamaCppModelv2 {
    #[new]
    fn new() -> Self {
        Self {}
    }

    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() -> (&'static str,) {
        (NODE_LLAMA_CPP_MODEL_V2,) // 返回模型缓存 key
    }

    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() -> (&'static str,) {
        ("model",)
    }

    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_LLAMA_CPP;

    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "llama.cpp model loader v2"
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
        let model_list = Self::get_model_list();

        InputSpec::new()
            // 模型路径
            .with_required(
                "model_path",
                InputType::list(model_list.clone()).tooltip("Path to the GGUF model file"),
            )
            .with_required(
                "mmproj_path",
                InputType::list(model_list).tooltip("Path to the GGUF model file"),
            )
            .with_required(
                "cache_model",
                InputType::bool()
                    .default(false)
                    .tooltip("Cache model in memory for reuse"),
            )
            .build()
    }

    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        model_path: String,
        mmproj_path: String,
        cache_model: bool,
    ) -> PyResult<(Bound<'py, PyDict>,)> {
        let result = self.load_model(py, model_path, mmproj_path, cache_model);

        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                error!("LlamaCppModelv2 error: {e}");
                if let Err(e) = self.send_error(py, "LlamaCppModelv2".to_string(), e.to_string()) {
                    error!("send error failed: {e}");
                }
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl LlamaCppModelv2 {
    /// 加载模型并返回缓存 key
    fn load_model<'py>(
        &self,
        py: Python<'py>,
        model_path: String,
        mmproj_path: String,
        cache_model: bool,
    ) -> Result<(Bound<'py, PyDict>,), Error> {
        // 解析完整路径
        let base_models_dir = FolderPaths::default().model_path();
        let model_path = base_models_dir.join("LLM").join(&model_path);
        let mmproj_path = base_models_dir.join("LLM").join(&mmproj_path);

        if !model_path.exists() {
            return Err(Error::InvalidPath(format!(
                "Model not found: {}",
                model_path.display()
            )));
        }

        let cache_model_key = format!(
            "comfyui:model:{}",
            model_path.file_name().unwrap_or_default().to_string_lossy()
        );

        let llama_cpp_model_params = LlamaCppModelParams {
            model_path: model_path.to_string_lossy().to_string(),
            mmproj_path: mmproj_path.to_string_lossy().to_string(),
            cache_model,
            cache_model_key,
        };
        let py_params = pythonize(py, &llama_cpp_model_params)?
            .extract::<Bound<'py, PyDict>>()
            .map_err(|e| {
                error!("pythonize error, {e}");
                Error::Pyo3CastError(e.to_string())
            })?;

        Ok((py_params,))
    }

    /// 获取模型列表
    fn get_model_list() -> Vec<String> {
        let llm_list = FolderPaths::default().get_filename_list("LLM");
        let text_encoders_list = FolderPaths::default().get_filename_list("text_encoders");

        let mut models: Vec<String> = [llm_list, text_encoders_list]
            .concat()
            .into_iter()
            .filter(|v| v.ends_with(".gguf"))
            .collect();

        models.insert(0, "None".to_string());
        models
    }
}
