//! StringDynList

use comfy_app::{Node, NodeType, WidgetValue};
use serde_json::json;
use wasm_bindgen::JsValue;
use web_sys::console;

pub struct StringDynList {}

impl StringDynList {
    pub fn init_widget(node: &Node) -> Result<(), JsValue> {
        // string_num widget
        let widget = node.get_widget(0)?;
        let cur_value = match widget.value {
            Some(WidgetValue::Int(v)) => v,
            _ => 0,
        };

        // 删除组件
        {
            let widgets = node.widgets()?;
            let widgets_len = widgets.length() as i64;

            // 2: 其他的组件
            for _i in (cur_value + 2)..widgets_len {
                widgets.pop();
            }
            node.set_widgets(&widgets)?;
        }

        let size = node.size()?;
        let compute_size = node.compute_size()?;
        node.set_size(size[0], compute_size[1])?;

        Ok(())
    }

    pub fn on_node_created(node_type: &NodeType) -> Result<(), JsValue> {
        node_type.prototype()?.on_node_created(|node| {
            // 默认隐藏

            // string_num 添加回调函数
            Self::add_widget_callback(&node)?;

            Ok(())
        })?;

        Ok(())
    }

    // 这是一个动态添加组件的方案，但是无法后端联动，UI层面异常
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
                    // {
                    //     // 添加组件, 无法后端联动, 输入框无法输入值
                    //     for i in string_widget_len..(value as usize) {
                    //         let mut widget = node_c.get_widget(2)?;
                    //         widget.name = format!("string_{}", i + 1);
                    //         widget.label = format!("string_{}", i + 1);
                    //         widget.value = Some(WidgetValue::String("".to_string()));
                    //         widget.computed_height = Some(0.0);
                    //         widget.r#type = "string".to_string(); // 非指定的参数，widget 会被忽略

                    //         let widget_js: Object = widget.to_js()?.into();
                    //         // Reflect::set(&widget_js, &"node".into(), &node_c.get_inner())?;

                    //         widgets.push(&widget_js);
                    //     }
                    //     node_c.set_widgets(&widgets)?;

                    //     let size = node_c.size()?;
                    //     let compute_size = node_c.compute_size()?;
                    //     node_c.set_size(size[0], compute_size[1])?;
                    //     console::log_1(
                    //         &format!("🚀 Node size updated: {:#?} {:#?}", size, compute_size)
                    //             .into(),
                    //     );
                    // }

                    // {
                    //     // 添加组件, 无法后端联动， UI层面异常
                    //     let widgets = node_c.widgets()?;
                    //     for i in string_widget_len..(value as usize) {
                    //         let widget = widgets.get(2).clone();
                    //         Reflect::set(
                    //             &widget,
                    //             &"name".into(),
                    //             &format!("string_{}", i + 1).into(),
                    //         )?;
                    //         Reflect::set(
                    //             &widget,
                    //             &"label".into(),
                    //             &format!("string_{}", i + 1).into(),
                    //         )?;
                    //         Reflect::set(&widget, &"value".into(), &"".into())?;

                    //         widgets.push(&widget);
                    //     }
                    //     node_c.set_widgets(&widgets)?;
                    // }

                    {
                        // 添加组件, 无法后端联动
                        // 但是如果存在过原始组件的情况下（即有卡槽的情况下），输入框可以输入值
                        let widget = node_c.get_widget(2)?;
                        for i in string_widget_len..(value as usize) {
                            let options = json!({
                                "serialize":false,
                            });
                            let options = serde_wasm_bindgen::to_value(&options)?;
                            node_c.add_widget(
                                &widget.r#type.clone(),
                                // "number", // STRING/text/number
                                &format!("string_{}", i + 1),
                                JsValue::from_str(""),
                                None,
                                options,
                            )?;
                        }
                    }
                } else if value < string_widget_len as i64 && string_widget_len > 2 {
                    // 删除组件
                    {
                        for _i in value..(string_widget_len as i64) {
                            widgets.pop();
                        }
                        node_c.set_widgets(&widgets)?;
                    }

                    // 重置窗口大小
                    {
                        // let size = node_c.size()?;
                        // let compute_size = node_c.compute_size()?;
                        // node_c.set_size(size[0], compute_size[1])?;

                        node_c.reset_size()?;
                    }
                }

                // 打印组件
                // let widgets = node_c.widgets()?;
                // console::log_1(&format!("🚀string widgets on_node_created: {:#?}", widgets).into());
            }

            Ok(JsValue::undefined())
        })?;

        Ok(())
    }

    // 动态隐藏的方案
    fn _add_widget_callback2(node: &Node) -> Result<(), JsValue> {
        let node_c = node.clone();

        // string_num 添加回调函数
        node.add_widget_callback(0, move |value, _canvas, _node, _pos, _pointer_event| {
            if let WidgetValue::Int(value) = value {
                let widgets = node_c.widgets()?;
                let string_widget_len = widgets.length() as usize - 2;
                console::log_1(
                    &format!("🚀string widgets length: {:#?}", string_widget_len).into(),
                );

                // string_*, index: 2
                // 显示组件
                if value > string_widget_len as i64 {
                    {
                        let mut hidden_height = 0.0;
                        for i in string_widget_len..(value as usize) {
                            let mut widget = node_c.get_widget(i)?;

                            // 移除 _ 前缀
                            widget.r#type = widget.r#type.replace("_", "").clone(); // 非指定的参数，widget 会被忽略

                            let widget_js = widget.to_js()?;
                            widgets.set(i as u32, widget_js);

                            hidden_height += widget.computed_height.unwrap_or(0.0);
                        }
                        node_c.set_widgets(&widgets)?;

                        let size = node_c.size()?;
                        let compute_size = node_c.compute_size()?;
                        // node_c.set_size(size[0], compute_size[1])?;
                        console::log_1(
                            &format!("🚀 Node size updated: {:#?} {:#?}", size, compute_size)
                                .into(),
                        );
                        console::log_1(&format!("🚀 Hidden height: {:#?}", hidden_height).into());
                    }
                } else if value < string_widget_len as i64 && string_widget_len > 2 {
                    // 隐藏组件
                    {
                        let mut hidden_height = 0.0;
                        for i in value..(string_widget_len as i64) {
                            let mut widget = node_c.get_widget(i as usize)?;

                            // 添加 _ 前缀
                            widget.r#type = format!("_{}", widget.r#type); // 非指定的参数，widget 会被忽略

                            let widget_js = widget.to_js()?;
                            widgets.set(i as u32, widget_js);

                            hidden_height += widget.computed_height.unwrap_or(0.0);
                        }
                        node_c.set_widgets(&widgets)?;

                        let size = node_c.size()?;
                        let compute_size = node_c.compute_size()?;
                        console::log_1(
                            &format!("🚀 Node size updated: {:#?} {:#?}", size, compute_size)
                                .into(),
                        );
                        console::log_1(&format!("🚀 Hidden height: {:#?}", hidden_height).into());
                    }

                    // 重置窗口大小
                    {
                        // let size = node_c.size()?;
                        // let compute_size = node_c.compute_size()?;
                        // node_c.set_size(size[0], compute_size[1])?;

                        node_c.reset_size()?;
                    }
                }

                // 打印组件
                let widgets = node_c.widgets()?;
                console::log_1(&format!("🚀string widgets on_node_created: {:#?}", widgets).into());
            }

            Ok(JsValue::undefined())
        })?;

        Ok(())
    }
}
