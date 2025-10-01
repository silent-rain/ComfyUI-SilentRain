//! StringDynList

use comfy_app::{Node, NodeType, WidgetValue};
use js_sys::Reflect;
use serde_json::json;
use wasm_bindgen::JsValue;
use web_sys::console;

pub struct StringDynList {}

impl StringDynList {
    pub fn _init_widget(node: &Node) -> Result<(), JsValue> {
        let widget = node.get_widget(0)?;

        // 添加组件
        let widgets = node.widgets()?;
        let string_widget_len = widgets.length() as usize - 2;
        console::log_1(&format!("🚀string widgets length: {:#?}", string_widget_len).into());

        if let Some(WidgetValue::Int(value)) = widget.value {
            // string_*, index: 2
            // 添加组件
            if value > string_widget_len as i64 {
                // for i in string_widget_len..(value as usize) {
                //     let mut widget = node.get_widget(2)?;
                //     widget.name = format!("string_{}", i + 1);
                //     widget.label = format!("string_{}", i + 1);
                //     widget.value = Some(WidgetValue::String("".to_string()));

                //     widgets.push(&widget.to_js()?);
                // }
                // node.set_widgets(&widgets)?;

                // 添加组件2
                // let widget = node.get_widget(2)?;
                for i in string_widget_len..(value as usize) {
                    let options = json!({
                        // "serialize":false,
                        "y": 8,
                    });
                    let options = serde_wasm_bindgen::to_value(&options)?;
                    node.add_widget(
                        // &widget.r#type.clone(),
                        "STRING",
                        &format!("string_{}", i + 1),
                        JsValue::from_str(""),
                        None,
                        options,
                    )?;
                }
            }
        }

        let widgets = node.widgets()?;
        console::log_1(&format!("🚀string widgets: {:#?}", widgets).into());

        Ok(())
    }

    pub fn on_node_created(node_type: &NodeType) -> Result<(), JsValue> {
        node_type.prototype()?.on_node_created(|node| {
            let widgets = node.widgets()?;
            console::log_1(&format!("🚀string widgets on_node_created: {:#?}", widgets).into());

            {
                let widgets = node.widgets()?;
                let widget = widgets.get(0);
                let v = Reflect::get(&widget, &"value".into())?;
                console::log_1(&format!("🚀string widget 0 on_node_created: {:#?}", v).into());
            }

            // string_num 添加回调函数
            Self::add_widget_callback(&node)?;

            Ok(())
        })?;

        Ok(())
    }

    fn add_widget_callback(node: &Node) -> Result<(), JsValue> {
        let node_c = node.clone();

        // string_num 添加回调函数
        node.add_widget_callback(0, move |value, _canvas, _node, _pos, _pointer_event| {
            if let WidgetValue::Int(value) = value {
                // 添加组件
                let widgets = node_c.widgets()?;
                let string_widget_len = widgets.length() as usize - 2;
                console::log_1(
                    &format!("🚀string widgets length: {:#?}", string_widget_len).into(),
                );

                // string_*, index: 2
                // 添加组件
                if value > string_widget_len as i64 {
                    // for i in string_widget_len..(value as usize) {
                    //     let mut widget = node_c.get_widget(2)?;
                    //     widget.name = format!("string_{}", i + 1);
                    //     widget.label = format!("string_{}", i + 1);
                    //     widget.value = Some(WidgetValue::String("".to_string()));
                    //     widgets.push(&widget.to_js()?);
                    // }
                    // node_c.set_widgets(&widgets)?;

                    let widget = node_c.get_widget(2)?;
                    for i in string_widget_len..(value as usize) {
                        let options = json!({
                            "serialize":false,
                        });
                        let options = serde_wasm_bindgen::to_value(&options)?;
                        node_c.add_widget(
                            &widget.r#type.clone(),
                            &format!("string_{}", i + 1),
                            JsValue::from_str(""),
                            None,
                            options,
                        )?;
                    }
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

        Ok(())
    }
}
