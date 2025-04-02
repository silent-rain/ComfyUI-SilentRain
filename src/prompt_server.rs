//! Prompt Server

use pyo3::{
    types::{PyAnyMethods, PyDict, PyModule},
    PyResult, PyTypeInfo, Python,
};

/// comfyui PromptServer wrapper
pub trait PromptServer: PyTypeInfo {
    /// 发送日志信息到ComfyUI
    ///
    /// 当前案例, 当节点执行出现异常时通知前端
    fn send_error(&self, py: Python, error_type: String, message: String) -> PyResult<()> {
        // 初始化时获取 PromptServer 实例
        let server = PyModule::import(py, "server")?
            .getattr("PromptServer")?
            .getattr("instance")?;

        // 构建错误数据字典
        let error_data = PyDict::new(py);
        error_data.set_item("type", &error_type)?;
        error_data.set_item("node", self.get_class_name(py)?)?;
        error_data.set_item("message", message)?;

        // 调用 Python 端方法
        server
            .getattr("send_sync")?
            .call1(("silentrain", error_data))?;

        Ok(())
    }

    /// Class 名称
    fn get_class_name(&self, py: Python) -> PyResult<String> {
        // 需要 Clone
        // let py_self = self.as_ref().into_pyobject(py)?;
        // py_self.getattr("__name__")?.extract()

        Self::type_object(py)
            .getattr("__name__")?
            .extract::<String>()
    }
}
