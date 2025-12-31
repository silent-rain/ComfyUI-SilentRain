//! Workflow Image Export - 将工作流导出为带元数据的PNG图片
//!
//! 复现自: https://github.com/BobRandomNumber/ComfyUI-QoL-Pack/blob/main/web/js/QoL_WorkflowImage.js

use base64::{Engine as _, engine::general_purpose::STANDARD};
use comfy_app::ComfyApp;
use js_sys::{Array, Float64Array, Function, JSON, Object, Promise, Reflect};
use wasm_bindgen::{JsCast, JsValue, prelude::*};
use wasm_bindgen_futures::JsFuture;

/// 异步睡眠函数 - 在指定时间后唤醒
async fn _async_sleep(timeout: i32) -> Result<(), JsValue> {
    // 创建一个 Promise，在指定时间后 resolve
    let promise = Promise::new(&mut |resolve, _reject| {
        let _ = web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                &resolve, // 定时器结束后触发 resolve
                timeout,
            );
    });

    // 等待 Promise 完成
    JsFuture::from(promise).await?;
    Ok(())
}

/// Workflow图片导出器
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
    /// 创建新的WorkflowImage实例
    pub fn new(app: ComfyApp) -> Self {
        Self { app, state: None }
    }

    /// 获取canvas对象
    fn get_canvas(&self) -> Result<Object, JsValue> {
        Reflect::get(&self.app.app(), &"canvas".into())?.dyn_into::<Object>()
    }

    /// 获取graph对象
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

            // pos 可能是 Array 或 Float64Array
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

        // 获取原始 canvas 尺寸
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

        // 恢复原始 canvas 尺寸
        let canvas_val = Reflect::get(&canvas, &"canvas".into())?;
        let canvas_el = canvas_val.dyn_into::<web_sys::HtmlCanvasElement>()?;

        canvas_el.set_width(state.width);
        canvas_el.set_height(state.height);

        Ok(())
    }

    /// 更新视图以适应边界
    pub fn update_view(&self, bounds: [f64; 4]) -> Result<(), JsValue> {
        let canvas = self.get_canvas()?;

        // 尝试获取真实的 canvas 元素
        let canvas_el_val = Reflect::get(&canvas, &"canvas".into())?;

        let canvas_el: web_sys::HtmlCanvasElement = if let Ok(el) = canvas_el_val
            .clone()
            .dyn_into::<web_sys::HtmlCanvasElement>(
        ) {
            el
        } else {
            return Err(JsValue::from_str("Could not find canvas element"));
        };

        let ds = Reflect::get(&canvas, &"ds".into())?;

        let device_pixel_ratio = web_sys::window()
            .ok_or(JsValue::from_str("No window"))?
            .device_pixel_ratio() as f64;

        // 计算内容尺寸
        let content_width = bounds[2] - bounds[0];
        let content_height = bounds[3] - bounds[1];

        // 设置合适的缩放比例
        // 使用设备像素比确保在高清屏幕上图像清晰
        let scale = device_pixel_ratio;

        // 设置新的 canvas 尺寸，使用内容尺寸
        let width = content_width as u32;
        let height = content_height as u32;

        canvas_el.set_width(width);
        canvas_el.set_height(height);

        // 设置 scale
        Reflect::set(&ds, &"scale".into(), &JsValue::from_f64(scale))?;

        // 设置 offset（移动到左上角）
        let offset = Array::new();
        offset.push(&JsValue::from_f64(-bounds[0]));
        offset.push(&JsValue::from_f64(-bounds[1]));
        Reflect::set(&ds, &"offset".into(), &offset)?;

        Ok(())
    }

    /// 绘制画布
    pub fn draw(&self, clear_canvas: bool, draw_foreground: bool) -> Result<(), JsValue> {
        let canvas = self.get_canvas()?;

        // 清除画布（如果需要）
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

        // 调用原始 draw 方法
        let draw = Reflect::get(&canvas, &"draw".into())?.dyn_into::<Function>()?;
        draw.call2(
            &canvas,
            &JsValue::from_bool(clear_canvas),
            &JsValue::from_bool(draw_foreground),
        )?;

        Ok(())
    }
}

/// PNG图片导出器（带workflow元数据）
pub struct PngWorkflowImage {
    workflow_image: WorkflowImage,
}

impl PngWorkflowImage {
    /// 创建新的PngWorkflowImage实例
    pub fn new(app: ComfyApp) -> Self {
        Self {
            workflow_image: WorkflowImage::new(app),
        }
    }

    /// 导出工作流为PNG
    pub fn export(&mut self, include_workflow: bool) -> Result<(), JsValue> {
        self.workflow_image.save_state()?;

        let bounds = self.workflow_image.get_bounds()?;
        self.workflow_image.update_view(bounds)?;

        // 先不清除画布，只绘制前景，保留之前的渲染内容
        self.workflow_image.draw(false, true)?;

        let window = web_sys::window().ok_or(JsValue::from_str("No window"))?;
        let canvas = self.workflow_image.get_canvas()?;

        // 下载图片
        let filename = "workflow.png";
        if include_workflow {
            let workflow = self.workflow_json(self.workflow_image.app.clone())?;

            let timeout_callback = Closure::once_into_js(move || {
                let data_url = Self::image_embed_workflow(canvas, &workflow).unwrap();
                Self::download_canvas(filename, &data_url).unwrap();
            });

            // 使用 setTimeout 延迟执行
            window.set_timeout_with_callback_and_timeout_and_arguments_0(
                timeout_callback.as_ref().unchecked_ref(),
                1000,
            )?;
        } else {
            let canvas_val = Reflect::get(&canvas, &"canvas".into())?;
            let canvas_el = canvas_val.dyn_into::<web_sys::HtmlCanvasElement>()?;
            let data_url = canvas_el.to_data_url_with_type("image/png")?;

            // 使用 requestAnimationFrame 等待渲染完成
            let callback = Closure::once_into_js(move || {
                Self::download_canvas(filename, &data_url).unwrap();
            });

            // 注册 requestAnimationFrame 回调
            window.request_animation_frame(callback.as_ref().unchecked_ref())?;
        }

        // 恢复画布状态
        self.workflow_image.restore_state()?;
        self.workflow_image.draw(true, true)?;

        Ok(())
    }

    /// 获取工作流Json数据
    fn workflow_json(&self, app: ComfyApp) -> Result<String, JsValue> {
        // 获取工作流数据 - 通过 app 获取 graph 的序列化数据
        let graph = app.graph()?.dyn_into::<Object>()?;

        // 尝试获取 serialize 方法
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
            // 如果没有 serialize 方法，尝试直接序列化
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

    /// 图像嵌入工作流数据
    fn image_embed_workflow(canvas: Object, workflow: &str) -> Result<String, JsValue> {
        // let canvas = self.workflow_image.get_canvas()?;
        let canvas_val = Reflect::get(&canvas, &"canvas".into())?;
        let canvas_el = canvas_val.dyn_into::<web_sys::HtmlCanvasElement>()?;

        // 获取 data URL
        let data_url = canvas_el.to_data_url_with_type("image/png")?;

        // 解析 base64 数据
        let base64_data = data_url
            .strip_prefix("data:image/png;base64,")
            .ok_or(JsValue::from_str("Invalid data URL format"))?;

        // 解码 base64
        let png_bytes = STANDARD
            .decode(base64_data)
            .map_err(|e| JsValue::from_str(&format!("Failed to decode base64: {}", e)))?;

        // 创建包含工作流数据的 PNG (使用 tEXt 块)
        let workflow_bytes = workflow.as_bytes();
        let mut new_png = Vec::new();

        // PNG 签名
        new_png.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

        // 复制 IHDR 块 (前 8 字节是签名 + 4 字节长度 + 4 字节类型 + 13 字节数据 + 4 字节 CRC)
        // 找到第一个 IEND 块
        let mut pos = 8;
        while pos < png_bytes.len().saturating_sub(8) {
            let chunk_len = u32::from_be_bytes([
                png_bytes[pos],
                png_bytes[pos + 1],
                png_bytes[pos + 2],
                png_bytes[pos + 3],
            ]) as usize;
            let chunk_type = &png_bytes[pos + 4..pos + 8];
            pos += 8 + chunk_len + 4; // length + type + data + crc
            if chunk_type == b"IEND" {
                break;
            }
        }

        // 在 IEND 之前插入 tEXt 块
        let header_end = pos - 12; // 减去 IEND 的长度(4) + 类型(4) + CRC(4)
        new_png.extend_from_slice(&png_bytes[8..header_end]);

        // 创建 tEXt 块: keyword="workflow", text=workflow_json
        let keyword = b"workflow";
        let mut chunk_data = Vec::new();
        chunk_data.extend_from_slice(keyword);
        chunk_data.push(0); // null 分隔符
        chunk_data.extend_from_slice(workflow_bytes);

        // 计算块长度
        let chunk_len = chunk_data.len() as u32;

        // 写入块长度
        new_png.extend_from_slice(&chunk_len.to_be_bytes());
        // 写入块类型
        new_png.extend_from_slice(b"tEXt");
        // 写入块数据
        new_png.extend_from_slice(&chunk_data);
        // 计算 CRC (类型 + 数据)
        let mut crc_data: Vec<u8> = Vec::new();
        crc_data.extend_from_slice(b"tEXt");
        crc_data.extend_from_slice(&chunk_data);
        let crc = crc32fast::hash(&crc_data);
        new_png.extend_from_slice(&crc.to_be_bytes());

        // 写入 IEND 块
        new_png.extend_from_slice(&png_bytes[header_end..]);

        // 重新编码为 base64
        let new_base64 = STANDARD.encode(&new_png);
        let new_data_url = format!("data:image/png;base64,{}", new_base64);
        Ok(new_data_url)
    }

    /// 下载canvas内容
    fn download_canvas(filename: &str, data_url: &str) -> Result<(), JsValue> {
        // 创建下载链接
        let window = web_sys::window().ok_or(JsValue::from_str("No window"))?;
        let document = window.document().ok_or(JsValue::from_str("No document"))?;

        let a = document
            .create_element("a")?
            .dyn_into::<web_sys::HtmlAnchorElement>()?;

        a.set_href(data_url);
        a.set_download(filename);

        let body = document.body().ok_or(JsValue::from_str("No body"))?;
        body.append_child(&a)?;
        a.click();

        // 清理
        let _ = body.remove_child(&a);

        Ok(())
    }
}

/// 注册工作流图片导出功能注册菜单
pub fn register_workflow_image_export() -> Result<JsValue, JsValue> {
    // 添加"Save Workflow as PNG"菜单项
    let save_item = Object::new();
    Reflect::set(
        &save_item,
        &"content".into(),
        &JsValue::from_str("Save Workflow as PNG"),
    )?;

    let callback = Closure::wrap(Box::new(move || {
        let app_obj = ComfyApp::new().unwrap();
        let mut exporter = PngWorkflowImage::new(app_obj);

        if let Err(e) = exporter.export(true) {
            web_sys::console::error_1(&JsValue::from_str(&format!(
                "Failed to export workflow: {:?}",
                e
            )));
        }
    }) as Box<dyn FnMut()>);

    Reflect::set(&save_item, &"callback".into(), &callback.as_ref().clone())?;
    callback.forget();

    Ok(save_item.into())
}
