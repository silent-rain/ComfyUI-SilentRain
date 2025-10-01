//! StringDynList

use comfy_app::{Node, WidgetValue};
use wasm_bindgen::JsValue;
use web_sys::console;

pub struct StringDynList {}

impl StringDynList {
    pub fn add_widget_callback(node: &Node) -> Result<(), JsValue> {
        let node_c = node.clone();
        // string_num 添加回调函数
        node.add_widget_callback(0, move |value, _canvas, _node, _pos, _pointer_event| {
            console::log_1(&format!("🚀 小部件值变化: {:#?}", value).into());

            if let WidgetValue::Int(value) = value {
                console::log_1(&format!("🚀 String Dyn List widget value: {:#?}", value).into());

                // 添加组件
                let widgets = node_c.widgets()?;
                let string_widget_len = widgets.length() as usize - 2;
                console::log_1(
                    &format!("🚀string widgets length: {:#?}", string_widget_len).into(),
                );

                // string_*, index: 2
                // 添加组件
                if value > string_widget_len as i64 {
                    for i in string_widget_len..(value as usize) {
                        let mut widget = node_c.get_widget(2)?;
                        widget.name = format!("string_{}", i + 1);
                        widget.label = format!("string_{}", i + 1);
                        widget.value = Some(WidgetValue::String("".to_string()));
                        widgets.push(&widget.to_js()?);
                    }
                    node_c.set_widgets(&widgets)?;
                } else if value < string_widget_len as i64 && string_widget_len > 2 {
                    // 删除组件
                    for _i in value..(string_widget_len as i64) {
                        widgets.pop();
                    }
                    node_c.set_widgets(&widgets)?;
                }
            }

            Ok(JsValue::undefined())
        })?;

        let string_num = node.get_widget_by_name("string_num")?;

        console::log_1(&format!("🚀 string_num: {:#?}", string_num).into());

        Ok(())
    }
}
