//! AnyDynList

use comfy_app::{ConnectionType, NodeType};
use serde_json::json;
use wasm_bindgen::JsValue;
use web_sys::console;

pub struct AnyDynList {}

impl AnyDynList {
    pub fn on_connections_change(node_type: &NodeType) -> Result<(), JsValue> {
        node_type.prototype()?.on_connections_change(
            move |mut this, r#type, index, connected, link_info| {
                if link_info.is_none() {
                    return Ok(());
                }

                if r#type == ConnectionType::Input {
                    // connect input

                    if connected {
                        let inputs = this.get_inputs()?;
                        let inputs_length = inputs.len();

                        let has_empty_slot = inputs
                            .into_iter()
                            .filter(|v| v.name.starts_with("any_"))
                            .any(|v| v.link.is_none());

                        if inputs_length >= 2 && !has_empty_slot {
                            // 添加空卡槽
                            let extra_info = json!({
                                // "widget": {"name":format!("any_{}", inputs_length )},
                                // "shape": 7,
                            }); // TODO 此处参数未生效，需要确认
                            let extra_info = serde_wasm_bindgen::to_value(&extra_info)?;
                            let first_any_input = this.get_input(1)?;
                            this.add_input(
                                &format!("any_{}", inputs_length + 1),
                                &first_any_input.r#type,
                                extra_info,
                            )
                            .map_err(|e| {
                                console::log_1(&format!("🔗 添加空卡槽失败: {:#?}", e).into());
                                e
                            })?;
                        }
                    }

                    if !connected {
                        // 移除当前空卡槽
                        {
                            let inputs = this.get_inputs()?;
                            let inputs_length = inputs.len();

                            if inputs_length > 2 {
                                this.remove_input(index)?;
                            }
                        }

                        // 重命名"any_"卡槽
                        {
                            let inputs = this.get_inputs()?;
                            let mut input_count = 0;
                            for (i, input) in inputs.iter().enumerate() {
                                let mut input = input.clone();

                                if !input.name.starts_with("any_") {
                                    continue;
                                }
                                input_count += 1;

                                input.name = format!("any_{}", input_count);
                                input.label = Some(format!("any_{}", input_count));
                                this.set_input(i, input.clone())?;
                            }
                        }
                    }
                } else {
                    // connect output
                }

                Ok(())
            },
        )?;

        Ok(())
    }
}
