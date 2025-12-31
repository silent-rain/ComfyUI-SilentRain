//! Workflow Image Export - 将工作流导出为带元数据的图片
//!
//! 复现自: https://github.com/pythongosssss/ComfyUI-Custom-Scripts/blob/main/web/js/workflowImage.js
//!
//! 支持的导出格式：
//! - PNG: 使用 tEXt 块嵌入 workflow JSON 数据
//! - SVG: 使用 canvas2svg 库，在 <desc> 标签中嵌入 workflow

use base64::{Engine as _, engine::general_purpose::STANDARD};
use comfy_app::ComfyApp;
use js_sys::{Array, Float64Array, Function, JSON, Object, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue, prelude::*};

/// Workflow 图片导出器基础类
pub struct WorkflowImage {
    app: ComfyApp,
    state: Option<CanvasState>,
}

/// 画布状态
#[derive(Clone)]
struct CanvasState {
    scale: f64,
    offset_x: f64,
    offset_y: f64,
    width: u32,
    height: u32,
}

impl WorkflowImage {
    /// 创建新的 WorkflowImage 实例
    pub fn new(app: ComfyApp) -> Self {
        Self { app, state: None }
    }

    /// 获取 canvas 对象
    fn get_canvas(&self) -> Result<Object, JsValue> {
        Reflect::get(&self.app.app(), &"canvas".into())?.dyn_into::<Object>()
    }

    /// 获取 graph 对象
    fn get_graph(&self) -> Result<Object, JsValue> {
        self.app.graph()?.dyn_into::<Object>()
    }

    /// 计算图中所有节点的边界
    ///
    /// 返回: [min_x, min_y, max_x, max_y]
    pub fn get_bounds(&self) -> Result<[f64; 4], JsValue> {
        let graph = self.get_graph()?;
        let nodes = Reflect::get(&graph, &"_nodes".into())?.dyn_into::<Array>()?;

        let mut bounds: [f64; 4] = [99999.0, 99999.0, -99999.0, -99999.0];

        for i in 0..nodes.length() {
            let node_val = nodes.get(i);
            let node_obj = node_val.dyn_into::<Object>()?;

            // 获取节点位置
            let pos_val = Reflect::get(&node_obj, &"pos".into())?;

            let (x, y) = if let Ok(arr) = pos_val.clone().dyn_into::<Array>() {
                (
                    arr.get(0).as_f64().unwrap_or(0.0),
                    arr.get(1).as_f64().unwrap_or(0.0),
                )
            } else if let Ok(float_arr) = pos_val.clone().dyn_into::<Float64Array>() {
                if float_arr.length() >= 2 {
                    (float_arr.get_index(0), float_arr.get_index(1))
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };

            // 获取节点尺寸
            let get_bounding =
                Reflect::get(&node_obj, &"getBounding".into())?.dyn_into::<Function>()?;
            let node_bounds_val = get_bounding.call0(&node_obj)?;

            let (width, height) = if let Ok(arr) = node_bounds_val.clone().dyn_into::<Array>() {
                (
                    arr.get(2).as_f64().unwrap_or(0.0),
                    arr.get(3).as_f64().unwrap_or(0.0),
                )
            } else if let Ok(float_arr) = node_bounds_val.clone().dyn_into::<Float64Array>() {
                if float_arr.length() >= 4 {
                    (float_arr.get_index(2), float_arr.get_index(3))
                } else {
                    (0.0, 0.0)
                }
            } else {
                (0.0, 0.0)
            };

            let r = x + width;
            let b = y + height;

            if x < bounds[0] {
                bounds[0] = x;
            }
            if y < bounds[1] {
                bounds[1] = y;
            }
            if r > bounds[2] {
                bounds[2] = r;
            }
            if b > bounds[3] {
                bounds[3] = b;
            }
        }

        // 添加边距
        bounds[0] -= 100.0;
        bounds[1] -= 100.0;
        bounds[2] += 100.0;
        bounds[3] += 100.0;

        Ok(bounds)
    }

    /// 保存画布状态
    pub fn save_state(&mut self) -> Result<(), JsValue> {
        let canvas = self.get_canvas()?;
        let ds = Reflect::get(&canvas, &"ds".into())?;
        let scale = Reflect::get(&ds, &"scale".into())?.as_f64().unwrap_or(1.0);

        let offset = Reflect::get(&ds, &"offset".into())?.dyn_into::<Array>()?;
        let offset_x = offset.get(0).as_f64().unwrap_or(0.0);
        let offset_y = offset.get(1).as_f64().unwrap_or(0.0);

        let canvas_val = Reflect::get(&canvas, &"canvas".into())?;
        let canvas_el = canvas_val.dyn_into::<web_sys::HtmlCanvasElement>()?;
        let width = canvas_el.width();
        let height = canvas_el.height();

        self.state = Some(CanvasState {
            scale,
            offset_x,
            offset_y,
            width,
            height,
        });

        Ok(())
    }

    /// 恢复画布状态
    pub fn restore_state(&self) -> Result<(), JsValue> {
        let state = self
            .state
            .as_ref()
            .ok_or(JsValue::from_str("No saved state"))?;

        let canvas = self.get_canvas()?;
        let ds = Reflect::get(&canvas, &"ds".into())?;

        Reflect::set(&ds, &"scale".into(), &JsValue::from_f64(state.scale))?;

        let offset = Array::new();
        offset.push(&JsValue::from_f64(state.offset_x));
        offset.push(&JsValue::from_f64(state.offset_y));
        Reflect::set(&ds, &"offset".into(), &offset)?;

        let canvas_val = Reflect::get(&canvas, &"canvas".into())?;
        let canvas_el = canvas_val.dyn_into::<web_sys::HtmlCanvasElement>()?;
        canvas_el.set_width(state.width);
        canvas_el.set_height(state.height);

        Ok(())
    }

    /// 更新视图以适应边界
    pub fn update_view(&self, bounds: [f64; 4]) -> Result<(), JsValue> {
        let canvas = self.get_canvas()?;
        let canvas_el_val = Reflect::get(&canvas, &"canvas".into())?;

        let canvas_el: web_sys::HtmlCanvasElement = canvas_el_val
            .clone()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .map_err(|_| JsValue::from_str("Could not find canvas element"))?;

        let ds = Reflect::get(&canvas, &"ds".into())?;

        let device_pixel_ratio = web_sys::window()
            .ok_or(JsValue::from_str("No window"))?
            .device_pixel_ratio() as f64;

        let content_width = bounds[2] - bounds[0];
        let content_height = bounds[3] - bounds[1];

        let scale = device_pixel_ratio;
        let width = content_width as u32;
        let height = content_height as u32;

        canvas_el.set_width(width);
        canvas_el.set_height(height);

        Reflect::set(&ds, &"scale".into(), &JsValue::from_f64(scale))?;

        let offset = Array::new();
        offset.push(&JsValue::from_f64(-bounds[0]));
        offset.push(&JsValue::from_f64(-bounds[1]));
        Reflect::set(&ds, &"offset".into(), &offset)?;

        Ok(())
    }

    /// 绘制画布
    pub fn draw(&self, clear_canvas: bool, draw_foreground: bool) -> Result<(), JsValue> {
        let canvas = self.get_canvas()?;

        if clear_canvas {
            let canvas_el_val = Reflect::get(&canvas, &"canvas".into())?;
            let canvas_el = canvas_el_val.dyn_into::<web_sys::HtmlCanvasElement>()?;
            let context = canvas_el
                .get_context("2d")?
                .ok_or(JsValue::from_str("Failed to get canvas context"))?;
            let ctx = web_sys::CanvasRenderingContext2d::from(JsValue::from(context));
            ctx.clear_rect(
                0.0,
                0.0,
                canvas_el.width() as f64,
                canvas_el.height() as f64,
            );
        }

        let draw = Reflect::get(&canvas, &"draw".into())?.dyn_into::<Function>()?;
        draw.call2(
            &canvas,
            &JsValue::from_bool(clear_canvas),
            &JsValue::from_bool(draw_foreground),
        )?;

        Ok(())
    }

    /// 获取工作流 JSON 数据
    pub fn get_workflow_json(&self) -> Result<String, JsValue> {
        let graph = self.get_graph()?;

        let workflow_json = if let Ok(serialize) =
            Reflect::get(&graph, &"serialize".into()).and_then(|f| f.dyn_into::<Function>())
        {
            let workflow_data = serialize.call0(&graph)?;
            let json_str = JSON::stringify(&workflow_data)?;
            if json_str.is_undefined() || json_str.is_null() {
                return Err(JsValue::from_str("Failed to stringify workflow"));
            }
            json_str
                .as_string()
                .ok_or(JsValue::from_str("Failed to convert to string"))?
        } else {
            let json_str = JSON::stringify(&graph)?;
            if json_str.is_undefined() || json_str.is_null() {
                return Err(JsValue::from_str("Failed to stringify graph"));
            }
            json_str
                .as_string()
                .ok_or(JsValue::from_str("Failed to convert to string"))?
        };

        Ok(workflow_json)
    }

    /// 下载文件
    pub fn download_blob(&self, blob: &web_sys::Blob, filename: &str) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or(JsValue::from_str("No window"))?;
        let document = window.document().ok_or(JsValue::from_str("No document"))?;

        let url = web_sys::Url::create_object_url_with_blob(blob)?;

        let a = document
            .create_element("a")?
            .dyn_into::<web_sys::HtmlAnchorElement>()?;

        a.set_href(&url);
        a.set_download(filename);

        let body = document.body().ok_or(JsValue::from_str("No body"))?;
        body.append_child(&a)?;
        a.click();

        let _ = body.remove_child(&a);

        // 延迟释放 URL
        let _ = Closure::once_into_js(move || {
            let _ = web_sys::Url::revoke_object_url(&url);
        });

        Ok(())
    }
}

/// PNG 工作流图片导出器
pub struct PngWorkflowImage {
    workflow_image: WorkflowImage,
}

impl PngWorkflowImage {
    pub fn new(app: ComfyApp) -> Self {
        Self {
            workflow_image: WorkflowImage::new(app),
        }
    }

    /// 导出工作流为 PNG
    pub fn export(&mut self, include_workflow: bool) -> Result<(), JsValue> {
        self.workflow_image.save_state()?;

        // 克隆保存的状态，用于后续恢复
        let saved_state = self.workflow_image.state.clone();

        let bounds = self.workflow_image.get_bounds()?;
        self.workflow_image.update_view(bounds)?;
        self.workflow_image.draw(true, true)?;

        let window = web_sys::window().ok_or(JsValue::from_str("No window"))?;
        let canvas = self.workflow_image.get_canvas()?;

        let filename = "workflow.png";

        if include_workflow {
            let workflow = self.workflow_image.get_workflow_json()?;

            let callback = Closure::once_into_js(move || {
                let canvas_val = Reflect::get(&canvas, &"canvas".into()).unwrap();
                let canvas_el = canvas_val.dyn_into::<web_sys::HtmlCanvasElement>().unwrap();

                let data_url = canvas_el.to_data_url_with_type("image/png").unwrap();
                let blob = Self::embed_workflow_to_png(&data_url, &workflow).unwrap();

                let wf_image = WorkflowImage {
                    app: ComfyApp::new().unwrap(),
                    state: saved_state,
                };

                wf_image.download_blob(&blob, filename).unwrap();

                // 恢复画布状态
                let _ = wf_image.restore_state();
                let _ = wf_image.draw(true, true);
            });

            window.request_animation_frame(callback.as_ref().unchecked_ref())?;
        } else {
            let callback = Closure::once_into_js(move || {
                let canvas_val = Reflect::get(&canvas, &"canvas".into()).unwrap();
                let canvas_el = canvas_val.dyn_into::<web_sys::HtmlCanvasElement>().unwrap();

                let data_url = canvas_el.to_data_url_with_type("image/png").unwrap();
                let blob = Self::data_url_to_blob_internal(&data_url).unwrap();

                let wf_image = WorkflowImage {
                    app: ComfyApp::new().unwrap(),
                    state: saved_state,
                };

                wf_image.download_blob(&blob, filename).unwrap();

                // 恢复画布状态
                let _ = wf_image.restore_state();
                let _ = wf_image.draw(true, true);
            });

            window.request_animation_frame(callback.as_ref().unchecked_ref())?;
        }
        Ok(())
    }

    /// 静态方法：将 data URL 转换为 Blob
    fn data_url_to_blob_internal(data_url: &str) -> Result<web_sys::Blob, JsValue> {
        let parts: Array = Array::new();
        let base64_data = data_url
            .strip_prefix("data:image/png;base64,")
            .ok_or(JsValue::from_str("Invalid data URL format"))?;

        let bytes = STANDARD
            .decode(base64_data)
            .map_err(|e| JsValue::from_str(&format!("Failed to decode base64: {}", e)))?;

        let uint8_array = Uint8Array::from(&bytes[..]);
        parts.push(&uint8_array);

        let blob_props = web_sys::BlobPropertyBag::new();
        blob_props.set_type("image/png");

        web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &blob_props)
            .map_err(|_| JsValue::from_str("Failed to create blob"))
    }

    /// 在 PNG 中嵌入工作流数据（使用 tEXt 块）
    fn embed_workflow_to_png(data_url: &str, workflow: &str) -> Result<web_sys::Blob, JsValue> {
        let base64_data = data_url
            .strip_prefix("data:image/png;base64,")
            .ok_or(JsValue::from_str("Invalid data URL format"))?;

        let png_bytes = STANDARD
            .decode(base64_data)
            .map_err(|e| JsValue::from_str(&format!("Failed to decode base64: {}", e)))?;

        let workflow_bytes = workflow.as_bytes();
        let mut new_png = Vec::new();

        // PNG 签名
        new_png.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

        // 找到 IEND 块位置
        let mut pos = 8;
        while pos < png_bytes.len().saturating_sub(8) {
            let chunk_len = u32::from_be_bytes([
                png_bytes[pos],
                png_bytes[pos + 1],
                png_bytes[pos + 2],
                png_bytes[pos + 3],
            ]) as usize;
            let chunk_type = &png_bytes[pos + 4..pos + 8];
            pos += 8 + chunk_len + 4;
            if chunk_type == b"IEND" {
                break;
            }
        }

        // 在 IEND 之前插入 tEXt 块
        let header_end = pos - 12;
        new_png.extend_from_slice(&png_bytes[8..header_end]);

        // 创建 tEXt 块
        let keyword = b"workflow";
        let mut chunk_data = Vec::new();
        chunk_data.extend_from_slice(keyword);
        chunk_data.push(0);
        chunk_data.extend_from_slice(workflow_bytes);

        let chunk_len = chunk_data.len() as u32;
        new_png.extend_from_slice(&chunk_len.to_be_bytes());
        new_png.extend_from_slice(b"tEXt");
        new_png.extend_from_slice(&chunk_data);

        // 计算 CRC
        let mut crc_data: Vec<u8> = Vec::new();
        crc_data.extend_from_slice(b"tEXt");
        crc_data.extend_from_slice(&chunk_data);
        let crc = crc32fast::hash(&crc_data);
        new_png.extend_from_slice(&crc.to_be_bytes());

        // 写入 IEND 块
        new_png.extend_from_slice(&png_bytes[header_end..]);

        // 创建 Blob
        let parts: Array = Array::new();
        let uint8_array = Uint8Array::from(&new_png[..]);
        parts.push(&uint8_array);

        let blob_props = web_sys::BlobPropertyBag::new();
        blob_props.set_type("image/png");

        web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &blob_props)
            .map_err(|_| JsValue::from_str("Failed to create blob"))
    }
}

/// SVG 工作流图片导出器
pub struct SvgWorkflowImage {
    workflow_image: WorkflowImage,
}

impl SvgWorkflowImage {
    pub fn new(app: ComfyApp) -> Self {
        Self {
            workflow_image: WorkflowImage::new(app),
        }
    }

    /// 导出工作流为 SVG
    pub fn export(&mut self, include_workflow: bool) -> Result<(), JsValue> {
        // 注意: SVG 导出需要 canvas2svg 库的支持
        // 这里简化实现，仅作为示例

        self.workflow_image.save_state()?;

        // 克隆保存的状态，用于后续恢复
        let saved_state = self.workflow_image.state.clone();

        let bounds = self.workflow_image.get_bounds()?;
        self.workflow_image.update_view(bounds)?;
        self.workflow_image.draw(true, true)?;

        let window = web_sys::window().ok_or(JsValue::from_str("No window"))?;
        let canvas = self.workflow_image.get_canvas()?;
        let canvas_val = Reflect::get(&canvas, &"canvas".into())?;
        let canvas_el = canvas_val.dyn_into::<web_sys::HtmlCanvasElement>()?;

        // 将 canvas 转换为 SVG (简化版本)
        let width = canvas_el.width();
        let height = canvas_el.height();
        let data_url = canvas_el.to_data_url_with_type("image/png")?;

        // 构建 SVG (嵌入 PNG 图片)
        let svg_content = if include_workflow {
            let workflow = self.workflow_image.get_workflow_json()?;
            let escaped_workflow = Self::escape_xml(&workflow);
            format!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">
  <image width="{}" height="{}" href="{}"/>
  <desc>{}</desc>
</svg>"#,
                width, height, width, height, data_url, escaped_workflow
            )
        } else {
            format!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}">
  <image width="{}" height="{}" href="{}"/>
</svg>"#,
                width, height, width, height, data_url
            )
        };

        let filename = "workflow.svg";

        let callback = Closure::once_into_js(move || {
            // 创建 Blob
            let parts: Array = Array::new();
            parts.push(&JsValue::from_str(&svg_content));

            let blob_props = web_sys::BlobPropertyBag::new();
            blob_props.set_type("image/svg+xml");

            let blob =
                web_sys::Blob::new_with_str_sequence_and_options(&parts, &blob_props).unwrap();

            let wf_image = WorkflowImage {
                app: ComfyApp::new().unwrap(),
                state: saved_state,
            };

            wf_image.download_blob(&blob, filename).unwrap();

            // 恢复画布状态
            let _ = wf_image.restore_state();
            let _ = wf_image.draw(true, true);
        });

        window.request_animation_frame(callback.as_ref().unchecked_ref())?;

        Ok(())
    }

    /// 转义 XML 特殊字符
    fn escape_xml(s: &str) -> String {
        s.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&apos;")
    }
}

/// 注册工作流图片导出功能菜单
pub fn register_workflow_image_export() -> Result<Array, JsValue> {
    let menu_items = Array::new();

    // 添加分隔符
    menu_items.push(&JsValue::NULL);

    // Workflow Image 主菜单
    let workflow_submenu = Array::new();

    // PNG 导出
    let png_submenu = Array::new();

    let png_with_callback = Closure::wrap(Box::new(move || {
        let app_obj = ComfyApp::new().unwrap();
        let mut exporter = PngWorkflowImage::new(app_obj);
        if let Err(e) = exporter.export(true) {
            web_sys::console::error_1(&JsValue::from_str(&format!(
                "Failed to export PNG: {:?}",
                e
            )));
        }
    }) as Box<dyn FnMut()>);
    let png_with_item = create_menu_item(
        "png (with embedded workflow)",
        png_with_callback.as_ref().unchecked_ref(),
    )?;
    png_submenu.push(&png_with_item);
    png_with_callback.forget();

    let png_without_callback = Closure::wrap(Box::new(move || {
        let app_obj = ComfyApp::new().unwrap();
        let mut exporter = PngWorkflowImage::new(app_obj);
        if let Err(e) = exporter.export(false) {
            web_sys::console::error_1(&JsValue::from_str(&format!(
                "Failed to export PNG: {:?}",
                e
            )));
        }
    }) as Box<dyn FnMut()>);
    let png_without_item = create_menu_item(
        "png (no embedded workflow)",
        png_without_callback.as_ref().unchecked_ref(),
    )?;
    png_submenu.push(&png_without_item);
    png_without_callback.forget();

    let png_menu = create_menu_with_submenu("png", &png_submenu)?;
    workflow_submenu.push(&png_menu);

    // SVG 导出
    let svg_submenu = Array::new();

    let svg_with_callback = Closure::wrap(Box::new(move || {
        let app_obj = ComfyApp::new().unwrap();
        let mut exporter = SvgWorkflowImage::new(app_obj);
        if let Err(e) = exporter.export(true) {
            web_sys::console::error_1(&JsValue::from_str(&format!(
                "Failed to export SVG: {:?}",
                e
            )));
        }
    }) as Box<dyn FnMut()>);
    let svg_with_item = create_menu_item(
        "svg (with embedded workflow)",
        svg_with_callback.as_ref().unchecked_ref(),
    )?;
    svg_submenu.push(&svg_with_item);
    svg_with_callback.forget();

    let svg_without_callback = Closure::wrap(Box::new(move || {
        let app_obj = ComfyApp::new().unwrap();
        let mut exporter = SvgWorkflowImage::new(app_obj);
        if let Err(e) = exporter.export(false) {
            web_sys::console::error_1(&JsValue::from_str(&format!(
                "Failed to export SVG: {:?}",
                e
            )));
        }
    }) as Box<dyn FnMut()>);
    let svg_without_item = create_menu_item(
        "svg (no embedded workflow)",
        svg_without_callback.as_ref().unchecked_ref(),
    )?;
    svg_submenu.push(&svg_without_item);
    svg_without_callback.forget();

    let svg_menu = create_menu_with_submenu("svg", &svg_submenu)?;
    workflow_submenu.push(&svg_menu);

    let workflow_menu = create_menu_with_submenu("Workflow Image2", &workflow_submenu)?;
    menu_items.push(&workflow_menu);

    Ok(menu_items)
}

/// 创建带子菜单的菜单项
fn create_menu_with_submenu(content: &str, submenu: &Array) -> Result<Object, JsValue> {
    let menu_item = Object::new();
    Reflect::set(&menu_item, &"content".into(), &JsValue::from_str(content))?;
    let submenu_obj = Object::new();
    Reflect::set(&submenu_obj, &"options".into(), submenu)?;
    Reflect::set(&menu_item, &"submenu".into(), &submenu_obj)?;
    Ok(menu_item)
}

/// 创建菜单项
fn create_menu_item(content: &str, callback: &Function) -> Result<Object, JsValue> {
    let menu_item = Object::new();
    Reflect::set(&menu_item, &"content".into(), &JsValue::from_str(content))?;
    Reflect::set(&menu_item, &"callback".into(), callback)?;
    Ok(menu_item)
}
