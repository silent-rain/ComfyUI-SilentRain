//! 保存图像与文本

use std::{collections::HashMap, fs::File, io::BufWriter, path::Path};

use candle_core::Device;
use encoding::{DecoderTrap, EncoderTrap, Encoding, all::ISO_8859_1};
use image::{DynamicImage, ImageFormat};
use log::error;
use png::{Compression, Encoder};
use pyo3::{
    Bound, Py, PyAny, PyErr, PyResult, Python,
    exceptions::PyRuntimeError,
    pyclass, pymethods,
    types::{PyAnyMethods, PyDict, PyType},
};
use serde_json::Value;

use crate::{
    core::{category::CATEGORY_IMAGE, utils::image::tensor_to_images2},
    error::Error,
    text::{SaveText, SaveTextMode},
    wrapper::{
        comfyui::{
            PromptServer,
            node_input::InputPrompt,
            types::{NODE_BOOLEAN, NODE_IMAGE, NODE_STRING},
        },
        torch::tensor::TensorWrapper,
    },
};

/// 保存图像与文本
#[pyclass(subclass)]
pub struct SaveImageText {
    device: Device,
    image_extension: String,
}

impl PromptServer for SaveImageText {}

#[pymethods]
impl SaveImageText {
    #[new]
    fn new() -> Self {
        Self {
            device: Device::Cpu,
            image_extension: "png".to_string(),
        }
    }

    // 输入列表, 可选
    #[classattr]
    #[pyo3(name = "INPUT_IS_LIST")]
    fn input_is_list() -> bool {
        false
    }

    // 输出节点, 可选
    #[classattr]
    #[pyo3(name = "OUTPUT_NODE")]
    fn output_node() -> bool {
        true
    }

    // 返回参数类型
    #[classattr]
    #[pyo3(name = "RETURN_TYPES")]
    fn return_types() {}

    // 返回参数名称
    #[classattr]
    #[pyo3(name = "RETURN_NAMES")]
    fn return_names() {}

    // 输出列表, 可选
    // #[classattr]
    // #[pyo3(name = "OUTPUT_IS_LIST")]
    // fn output_is_list() {}

    // 节点分类
    #[classattr]
    #[pyo3(name = "CATEGORY")]
    const CATEGORY: &'static str = CATEGORY_IMAGE;

    // 节点描述, 可选
    #[classattr]
    #[pyo3(name = "DESCRIPTION")]
    fn description() -> &'static str {
        "Batch Save Images."
    }

    // 调研方法函数名称
    #[classattr]
    #[pyo3(name = "FUNCTION")]
    const FUNCTION: &'static str = "execute";

    #[classmethod]
    #[pyo3(name = "INPUT_TYPES")]
    fn input_types(_cls: &Bound<'_, PyType>) -> PyResult<Py<PyDict>> {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            dict.set_item("required", {
                let required = PyDict::new(py);
                required.set_item("image", (NODE_IMAGE, { PyDict::new(py) }))?;
                required.set_item(
                    "captions",
                    (NODE_STRING, {
                        let captions = PyDict::new(py);
                        captions.set_item("default", "")?;
                        captions.set_item("tooltip", "Saved Text Content")?;
                        captions
                    }),
                )?;
                required.set_item(
                    "filename",
                    (NODE_STRING, {
                        let filename = PyDict::new(py);
                        filename.set_item("default", "")?;
                        filename.set_item("tooltip", "Save file name")?;
                        filename
                    }),
                )?;
                required.set_item(
                    "filepath",
                    (NODE_STRING, {
                        let filepath = PyDict::new(py);
                        filepath.set_item("default", "output")?;
                        // filepath.set_item("placeholder", "output folder")?;
                        filepath.set_item("tooltip", "Save file path")?;
                        filepath
                    }),
                )?;

                required.set_item(
                    "text_extension",
                    (NODE_STRING, {
                        let text_extension = PyDict::new(py);
                        text_extension.set_item("default", ".txt")?;
                        text_extension.set_item("tooltip", "Save file extension")?;
                        text_extension
                    }),
                )?;

                required.set_item(
                    "text_mode",
                    (
                        vec![
                            SaveTextMode::Overwrite.to_string(),
                            SaveTextMode::Append.to_string(),
                            SaveTextMode::AppendNewLine.to_string(),
                        ],
                        {
                            let text_mode = PyDict::new(py);
                            text_mode.set_item("default", SaveTextMode::Overwrite.to_string())?;
                            text_mode.set_item("tooltip", "Saved Content Mode")?;
                            text_mode
                        },
                    ),
                )?;

                required
            })?;

            dict.set_item("optional", {
                let optional = PyDict::new(py);
                optional.set_item(
                    "filename_prefix",
                    (NODE_STRING, {
                        let filename_prefix = PyDict::new(py);
                        filename_prefix.set_item("default", "")?;
                        filename_prefix.set_item("tooltip", "The prefix of the saved file name")?;
                        filename_prefix
                    }),
                )?;
                optional.set_item(
                    "filename_suffix",
                    (NODE_STRING, {
                        let filename_prefix = PyDict::new(py);
                        filename_prefix.set_item("default", "")?;
                        filename_prefix.set_item("tooltip", "The suffix of the saved file name")?;
                        filename_prefix
                    }),
                )?;

                optional.set_item(
                    "save_workflow",
                    (NODE_BOOLEAN, {
                        let save_workflow = PyDict::new(py);
                        save_workflow.set_item("default", false)?;
                        save_workflow.set_item("label_on", "enabled")?;
                        save_workflow.set_item("label_off", "disabled")?;
                        save_workflow
                            .set_item("tooltip", "Save workflow information in the image.")?;
                        save_workflow
                    }),
                )?;

                optional
            })?;

            dict.set_item("hidden", {
                let hidden = PyDict::new(py);
                hidden.set_item("prompt", "PROMPT")?;
                hidden.set_item("extra_pnginfo", "EXTRA_PNGINFO")?;

                hidden
            })?;
            Ok(dict.into())
        })
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "execute")]
    fn execute<'py>(
        &mut self,
        py: Python<'py>,
        image: Bound<'py, PyAny>,
        captions: &str,
        filename: &str,
        filepath: &str,
        text_extension: &str,
        text_mode: &str,
        filename_prefix: &str,
        filename_suffix: &str,
        save_workflow: bool,
        prompt: Bound<'py, PyDict>,
        extra_pnginfo: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let results = self.save_image_and_text(
            &image,
            captions,
            filename,
            filepath,
            text_extension,
            text_mode,
            filename_prefix,
            filename_suffix,
            save_workflow,
            prompt,
            extra_pnginfo,
        );

        match results {
            Ok(_v) => self.node_result(py),
            Err(e) => {
                error!("SaveImageText error, {e}");
                if let Err(e) = self.send_error(py, "SaveImageText".to_string(), e.to_string()) {
                    error!("send error failed, {e}");
                    return Err(PyErr::new::<PyRuntimeError, _>(e.to_string()));
                };
                Err(PyErr::new::<PyRuntimeError, _>(e.to_string()))
            }
        }
    }
}

impl SaveImageText {
    /// 组合为前端需要的数据结构
    fn node_result<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("result", HashMap::<String, String>::new())?;
        Ok(dict)
    }
}

impl SaveImageText {
    /// 保存图像
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn save_image_and_text<'py>(
        &self,
        image: &Bound<'py, PyAny>,
        captions: &str,
        filename: &str,
        filepath: &str,
        text_extension: &str,
        text_mode: &str,
        filename_prefix: &str,
        filename_suffix: &str,
        save_workflow: bool,
        prompt: Bound<'py, PyDict>,
        extra_pnginfo: Bound<'py, PyDict>,
    ) -> Result<(), Error> {
        // 判断文件路径是否存在
        if !Path::new(&filepath).exists() {
            return Err(Error::InvalidDirectory(format!(
                "{filepath}: File path does not exist"
            )));
        }

        // 保存图像路径
        let image_extension = self.image_extension.clone();
        let image_path =
            format!("{filepath}/{filename_prefix}{filename}{filename_suffix}.{image_extension}");

        // 转换张量为图像
        let image = TensorWrapper::<f32>::new(image, &self.device)?.into_tensor();
        let images = tensor_to_images2(&image)?;
        if images.is_empty() {
            return Err(Error::ListEmpty);
        }
        let image = &images[0];

        if !save_workflow {
            // 保存图片
            if let Err(e) = image.save_with_format(&image_path, ImageFormat::Png) {
                error!("save image failed, {e}");
                return Ok(());
            }
        } else {
            // 获取文本框列表
            let text_chunck = Self::get_text_chunck(prompt, extra_pnginfo)?;

            // 保存图片
            if let Err(e) = Self::save_image_with_metadata(image, &image_path, &text_chunck) {
                error!("save image failed, {e}");
            }
        }

        // 保存文本
        if let Err(e) = SaveText::save_text(
            captions,
            filepath,
            filename,
            filename_prefix,
            filename_suffix,
            text_extension,
            text_mode,
        ) {
            error!("save text failed, {e}");
        }

        Ok(())
    }

    /// 保存图片的同时嵌入工作流
    ///
    /// 保存为 png 图片
    fn save_image_with_metadata<Q>(
        image: &DynamicImage,
        path: &Q,
        metadata: &[TextChunk],
    ) -> Result<(), Error>
    where
        Q: AsRef<Path>,
    {
        let (width, height) = (image.width(), image.height());
        let color_type = match image {
            DynamicImage::ImageLuma8(_) => png::ColorType::Grayscale,
            DynamicImage::ImageLumaA8(_) => png::ColorType::GrayscaleAlpha,
            DynamicImage::ImageRgb8(_) => png::ColorType::Rgb,
            DynamicImage::ImageRgba8(_) => png::ColorType::Rgba,
            _ => png::ColorType::Rgb, // 其他类型转换为RGB
        };

        let bit_depth = png::BitDepth::Eight;
        let data = image.clone().into_bytes();

        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        let mut encoder = Encoder::new(writer, width, height);
        encoder.set_color(color_type);
        encoder.set_depth(bit_depth);
        encoder.set_compression(Compression::default());

        // 添加元数据
        for chunk in metadata {
            encoder.add_text_chunk(chunk.keyword.clone(), chunk.text.clone())?;
        }

        let mut writer = encoder.write_header()?;
        writer.write_image_data(&data)?;

        Ok(())
    }

    /// 获取文本框列表
    fn get_text_chunck<'py>(
        prompt: Bound<'py, PyDict>,
        extra_pnginfo: Bound<'py, PyDict>,
    ) -> Result<Vec<TextChunk>, Error> {
        let mut results = Vec::new();

        let prompt_str = InputPrompt::new(prompt)?.to_string();
        results.push(TextChunk {
            keyword: "prompt".to_string(),
            text: Self::convert_to_iso_8859_1(&prompt_str)?,
        });

        for (key, value) in extra_pnginfo.into_iter() {
            let keyword: String = key.extract()?;
            let text: Value = pythonize::depythonize(&value)?;
            results.push(TextChunk {
                keyword,
                text: Self::convert_to_iso_8859_1(&text.to_string())?,
            });
        }

        Ok(results)
    }

    /// 将文本转换为 ISO-8859-1 编码的字节向量
    ///
    /// # 参数
    /// - `text`: 要转换的文本（UTF-8 格式）
    ///
    /// # 返回
    /// - 成功: ISO-8859-1 编码的字节向量
    /// - 失败: 错误信息
    pub fn convert_to_iso_8859_1(text: &str) -> Result<String, Error> {
        // 使用 ISO-8859-1 编码器
        let encoder = ISO_8859_1;

        // 编码文本
        let bytes = encoder
            .encode(text, EncoderTrap::Replace)
            .map_err(|e| Error::Encode(format!("Encoding to ISO-8859-1 failed: {e}")))?;

        encoder
            .decode(&bytes, DecoderTrap::Ignore)
            .map_err(|e| Error::Decode(format!("Decoding from ISO-8859-1 failed: {e}")))
    }
}

/// 文本块
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub keyword: String,
    pub text: String,
}
