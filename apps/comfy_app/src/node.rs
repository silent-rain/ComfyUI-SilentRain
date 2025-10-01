//! Node

use js_sys::{Array, Function, Object, Reflect};
use wasm_bindgen::{
    JsCast, JsValue,
    prelude::{Closure, wasm_bindgen},
};

use crate::{Input, Output, Widget, widget::WidgetValue};

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct Node {
    inner: Object,
}

#[wasm_bindgen]
impl Node {
    #[wasm_bindgen(constructor)]
    pub fn new(inner: Object) -> Self {
        Self { inner }
    }

    /// 获取inner对象
    pub fn get_inner(&self) -> Object {
        self.inner.clone()
    }
}
/*
{
"title":"Sr String Dyn List",
"graph":null,
"id":-1,
"type":"",
"inputs":[
    {"name":"string_num","localized_name":"string_num","label":"string_num","type":"INT","widget":{"name":"string_num"},"boundingRect":{"0":0,"1":0,"2":0,"3":0},"link":null},
    {"name":"delimiter","localized_name":"delimiter","label":"delimiter","type":"STRING","widget":{"name":"delimiter"},"boundingRect":{"0":0,"1":0,"2":0,"3":0},"link":null},
    {"name":"string_1","localized_name":"string_1","label":"string_1","type":"STRING","shape":7,"widget":{"name":"string_1"},"boundingRect":{"0":0,"1":0,"2":0,"3":0},"link":null},
    {"name":"string_2","localized_name":"string_2","label":"string_2","type":"STRING","shape":7,"widget":{"name":"string_2"},"boundingRect":{"0":0,"1":0,"2":0,"3":0},"link":null}
],
"outputs":[
    {"name":"prompt_strings","localized_name":"prompt_strings","label":"prompt_strings","type":"STRING","shape":6,"boundingRect":{"0":0,"1":0,"2":0,"3":0},"links":null},
    {"name":"strings","localized_name":"strings","label":"strings","type":"STRING","boundingRect":{"0":0,"1":0,"2":0,"3":0},"links":null},
    {"name":"total","localized_name":"total","label":"total","type":"INT","boundingRect":{"0":0,"1":0,"2":0,"3":0},"links":null}
],
"properties":{},
"properties_info":[],
"flags":{},
"widgets":[
    {"name":"string_num","options":{"min":2,"max":20,"step":10,"step2":1,"precision":0},"label":"string_num","type":"number","y":0},
    {"name":"delimiter","options":{},"label":"delimiter","type":"text","y":0},
    {"name":"string_1","options":{},"label":"string_1","type":"text","y":0},
    {"name":"string_2","options":{},"label":"string_2","type":"text","y":0}
],
"order":0,
"mode":0,
"serialize_widgets":true,"strokeStyles":{},"badges":[null],
"title_buttons":[],
"badgePosition":"top-right",
"_posSize":{"0":10,"1":10,"2":270,"3":170},
"_pos":{"0":10,"1":10},
"_size":{"0":270,"1":170}
}
*/

#[wasm_bindgen]
impl Node {
    /// title
    #[wasm_bindgen(getter)]
    pub fn title(&self) -> Result<String, JsValue> {
        Reflect::get(&self.inner, &"title".into()).and_then(|v| v.as_string().ok_or(JsValue::NULL))
    }

    /// 设置title
    #[wasm_bindgen(setter)]
    pub fn set_title(&self, title: &str) -> Result<bool, JsValue> {
        Reflect::set(&self.inner, &"title".into(), &JsValue::from_str(title))
    }

    /// 获取输入槽位列表
    #[wasm_bindgen(getter)]
    pub fn inputs(&self) -> Result<Array, JsValue> {
        Reflect::get(&self.inner, &"inputs".into())?.dyn_into::<Array>()
    }

    /// 设置输入槽位列表
    #[wasm_bindgen(setter)]
    pub fn set_inputs(&self, inputs: &Array) -> Result<bool, JsValue> {
        Reflect::set(&self.inner, &"inputs".into(), inputs)
    }

    /// 获取输出槽位列表
    #[wasm_bindgen(getter)]
    pub fn outputs(&self) -> Result<Array, JsValue> {
        Reflect::get(&self.inner, &"outputs".into())?.dyn_into::<Array>()
    }

    /// 设置输出槽位列表
    #[wasm_bindgen(setter)]
    pub fn set_outputs(&self, outputs: &Array) -> Result<bool, JsValue> {
        Reflect::set(&self.inner, &"outputs".into(), outputs)
    }

    /// 获取小部件列表
    #[wasm_bindgen(getter)]
    pub fn widgets(&self) -> Result<Array, JsValue> {
        Reflect::get(&self.inner, &"widgets".into())?.dyn_into::<Array>()
    }

    /// 设置小部件列表
    #[wasm_bindgen(setter)]
    pub fn set_widgets(&self, widgets: &Array) -> Result<bool, JsValue> {
        Reflect::set(&self.inner, &"widgets".into(), widgets)
    }

    /// 添加输入槽位
    ///
    /// Args:
    /// * name: 输入槽位名称
    /// * r#type: 输入槽位类型
    /// * extra_info: 额外信息
    pub fn add_input(&self, name: &str, r#type: &str, extra_info: JsValue) -> Result<(), JsValue> {
        let add_input_fn = Reflect::get(&self.inner, &"addInput".into())?.dyn_into::<Function>()?;

        add_input_fn.call3(
            &self.inner,
            &JsValue::from_str(name),
            &JsValue::from_str(r#type),
            &extra_info,
        )?;

        Ok(())
    }

    /// 移除输入槽位
    pub fn remove_input(&self, index: usize) -> Result<(), JsValue> {
        let remove_input_fn =
            Reflect::get(&self.inner, &"removeInput".into())?.dyn_into::<Function>()?;

        remove_input_fn.call1(&self.inner, &JsValue::from_f64(index as f64))?;

        Ok(())
    }
}

/// 便捷接口
impl Node {
    /// 获取输入槽位列表
    pub fn get_inputs(&self) -> Result<Vec<Input>, JsValue> {
        let js_value = Reflect::get(&self.inner, &"inputs".into())?;
        let results: Vec<Input> = serde_wasm_bindgen::from_value(js_value)?;
        Ok(results)
    }

    /// 获取输入槽位
    pub fn get_input(&self, index: usize) -> Result<Input, JsValue> {
        let inputs = self.inputs()?;
        let input = inputs.get(index as u32);
        Input::from_js(input)
    }

    /// 获取输出槽位列表
    pub fn get_outputs(&self) -> Result<Vec<Output>, JsValue> {
        let js_value = Reflect::get(&self.inner, &"outputs".into())?;
        let results: Vec<Output> = serde_wasm_bindgen::from_value(js_value)?;
        Ok(results)
    }

    /// 获取输出槽位
    pub fn get_output(&self, index: usize) -> Result<Output, JsValue> {
        let outputs = self.outputs()?;
        let output = outputs.get(index as u32);
        Output::from_js(output)
    }

    /// 获取小部件列表
    pub fn get_widgets(&self) -> Result<Vec<Widget>, JsValue> {
        let js_value = Reflect::get(&self.inner, &"widgets".into())?;
        let results: Vec<Widget> = serde_wasm_bindgen::from_value(js_value)?;
        Ok(results)
    }

    /// 获取小部件
    pub fn get_widget(&self, index: usize) -> Result<Widget, JsValue> {
        let widgets = self.widgets()?;
        let widget = widgets.get(index as u32);
        Widget::from_js(widget)
    }

    /// 设置输入槽位
    pub fn set_input(&mut self, index: usize, input: Input) -> Result<(), JsValue> {
        let inputs = self.inputs()?;
        inputs.set(index as u32, serde_wasm_bindgen::to_value(&input)?);
        self.set_inputs(&inputs)?;
        Ok(())
    }

    /// 设置输出槽位
    pub fn set_output(&mut self, index: usize, output: Output) -> Result<(), JsValue> {
        let outputs = self.outputs()?;
        outputs.set(index as u32, serde_wasm_bindgen::to_value(&output)?);
        self.set_outputs(&outputs)?;
        Ok(())
    }

    /// 设置小部件
    pub fn set_widget(&mut self, index: usize, widget: Widget) -> Result<(), JsValue> {
        let widgets = self.widgets()?;
        widgets.set(index as u32, serde_wasm_bindgen::to_value(&widget)?);
        self.set_widgets(&widgets)?;
        Ok(())
    }

    // export const getWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);
    pub fn get_widget_by_name(&self, name: &str) -> Result<Widget, JsValue> {
        let widgets = self.get_widgets()?;
        let widget = widgets
            .into_iter()
            .find(|w| w.name == name)
            .ok_or(JsValue::from_str("widget not found"))?;

        Ok(widget)
    }

    /// 为指定索引的小部件添加回调函数
    ///
    /// node.widgets[0].callback = (value, canvas, node, pos, e) => {}
    pub fn add_widget_callback<F>(&self, index: usize, handler: F) -> Result<(), JsValue>
    where
        F: Fn(WidgetValue, Object, Node, Object, Object) -> Result<JsValue, JsValue> + 'static,
    {
        let widgets = self.widgets()?;
        let widget = widgets.get(index as u32);

        let handler = Closure::wrap(Box::new(
            move |value: JsValue,
                  canvas: Object,
                  node: Object,
                  pos: Object,
                  pointer_event: Object| {
                let value = WidgetValue::from_js(value)?;
                let node = Node::new(node);

                handler(value, canvas, node, pos, pointer_event)
            },
        )
            as Box<dyn Fn(JsValue, Object, Object, Object, Object) -> Result<JsValue, JsValue>>);

        Reflect::set(&widget, &"callback".into(), &handler.as_ref().clone())?;

        // 保持闭包生命周期
        handler.forget();

        widgets.set(index as u32, widget);
        self.set_widgets(&widgets)?;
        Ok(())
    }
}

impl Node {
    /// 添加小部件
    ///
    /// addWidget(type, name2, value, callback, options2)
    pub fn add_widget(
        &self,
        r#type: &str,
        name: &str,
        value: JsValue,
        callback: Option<&Function>,
        options: JsValue,
    ) -> Result<(), JsValue> {
        let add_widget_fn =
            Reflect::get(&self.inner, &"addWidget".into())?.dyn_into::<Function>()?;

        add_widget_fn.call5(
            &self.inner,
            &JsValue::from_str(r#type),
            &JsValue::from_str(name),
            &value,
            &callback.cloned().unwrap_or(JsValue::NULL.into()).into(),
            &options,
        )?;

        Ok(())
    }
}

impl From<Node> for Object {
    fn from(node: Node) -> Self {
        node.get_inner()
    }
}
