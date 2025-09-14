//! 静态资源文件
#![allow(unused)]
use std::{io::Write, path::PathBuf};

use rust_embed::RustEmbed;
use tempfile::{Builder, NamedTempFile};
use tracing::info;

use crate::error::Error;

/// Resources 静态资源
#[derive(Debug, Default, RustEmbed)]
#[folder = "./resources/"]
#[prefix = "/"]
pub struct Resources;

impl Resources {
    pub fn to_bytes(path: &str) -> Option<Vec<u8>> {
        let asset = Self::get(&path)?;
        Some(asset.data.to_vec())
    }

    /// 获取文件类型
    pub fn mimetype(path: &str) -> Option<String> {
        let asset = Self::get(&path)?;
        Some(asset.metadata.mimetype().to_string())
    }

    /// 获取临时文件路径
    pub fn temp_gguf_path(path: &str) -> Result<(PathBuf, NamedTempFile), Error> {
        // 创建临时文件
        let mut temp_file = Builder::new()
            .prefix("tmp")
            .suffix(&".gguf")
            .rand_bytes(12)
            .tempfile()?;

        temp_file.write_all(
            &Self::get(&path)
                .ok_or_else(|| Error::FileNotFound(format!("resource file not found: {}", path)))?
                .data,
        )?;
        let temp_path = temp_file.path().to_path_buf();
        temp_file.flush()?;
        info!("create temp file: {}", temp_path.display());

        if PathBuf::from(temp_path.clone()).exists() {
            println!("model file exists");
        } else {
            println!("model file not exists");
        }

        Ok((temp_path, temp_file))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// LD_LIBRARY_PATH=~/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib \
    /// cargo test --package comfyui_silentrain --lib -- asset::tests::test_simple --exact --show-output
    #[test]
    // #[ignore]
    fn test_simple() -> anyhow::Result<()> {
        let (model_path, _temp_file) = Resources::temp_gguf_path("/llm/llama_cpp_empty.gguf")
            .expect("Failed to get llama.cpp model path");

        let model_path = model_path.to_string_lossy().to_string();
        println!("{model_path:?}");

        if PathBuf::from(model_path).exists() {
            println!("model file exists");
        } else {
            println!("model file not exists");
        }
        Ok(())
    }
}
