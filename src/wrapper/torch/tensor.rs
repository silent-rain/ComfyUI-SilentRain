//! Convert to Python object wrapper
//! 依赖:
//! - python: torch

use std::marker::PhantomData;

use candle_core::{DType, Device, Shape, Tensor, WithDType};
use numpy::{
    ndarray::{Dim, IxDynImpl},
    Element, PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArray, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::PyRuntimeError, types::PyAnyMethods, Bound, IntoPyObject, PyAny, PyErr, PyResult,
    Python,
};

use crate::error::Error;

pub struct TensorWrapper<T>
where
    T: Element + WithDType,
{
    tensor: Tensor,
    _marker: PhantomData<T>,
}

impl<T> TensorWrapper<T>
where
    T: Element + WithDType,
{
    pub fn new<'py>(py_any: &Bound<'py, PyAny>, device: &Device) -> PyResult<Self> {
        let tensor = Self::torch_to_candle(py_any, device)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

        Ok(Self {
            tensor,
            _marker: PhantomData,
        })
    }

    /// The dimension size for this tensor on each axis.
    pub fn dims(&self) -> &[usize] {
        self.tensor.dims()
    }

    pub fn from_tensor(tensor: Tensor) -> Self {
        Self {
            tensor,
            _marker: PhantomData,
        }
    }

    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }
}

impl<T> TensorWrapper<T>
where
    T: Element + WithDType,
{
    /// 转换为PyArray
    /// 使用更高效的numpy接口直接获取数据
    pub fn extract_ndarray<'py>(
        py_any: &Bound<'py, PyAny>,
    ) -> PyResult<PyReadonlyArray<'py, T, Dim<IxDynImpl>>> {
        // 用 .numpy() 得到 numpy array
        let numpy_any = py_any.call_method0("numpy")?;
        let array_view = numpy_any
            .downcast::<PyArrayDyn<T>>()? // 直接获取numpy数组
            .readonly(); // 获取只读视图

        Ok(array_view)
    }

    /// 从 Python torch.Tensor 转为 Rust candle_core::Tensor
    ///
    /// Tensor::from_vec
    fn torch_to_candle<'py>(
        torch_tensor: &Bound<'py, PyAny>,
        device: &Device,
    ) -> Result<Tensor, Error> {
        // 1. 获取 numpy 数组
        let np = torch_tensor.call_method0("numpy")?;

        // 2. 使用 downcast 而不是 extract
        let arr = np
            .downcast::<PyArrayDyn<T>>()
            .map_err(|e| Error::PyDowncastError(e.to_string()))?;

        // 3. 获取形状
        let shape = arr.shape().to_vec();

        // 4. 获取数据切片
        let data = arr.to_vec()?;

        // 5. 创建 tensor
        let tensor = Tensor::from_vec(data, shape, device)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

        // 6. 返回 Tensor
        Ok(tensor)
    }

    /// 从 Python torch.Tensor 转为 Rust candle_core::Tensor
    ///
    /// Tensor::from_slice
    fn _torch_to_candle2<'py>(
        torch_tensor: &Bound<'py, PyAny>,
        device: &Device,
    ) -> Result<Tensor, Error> {
        // 1. 获取 numpy 数组
        let np = torch_tensor.call_method0("numpy")?;

        // 2. 使用 downcast 而不是 extract
        let arr = np
            .downcast::<PyArrayDyn<T>>()
            .map_err(|e| Error::PyDowncastError(e.to_string()))?;

        // 3. 获取形状
        let shape = arr.shape().to_vec();

        // 4. 获取数据切片
        let slice = unsafe { arr.as_slice()? };

        // 5. 创建 tensor
        let tensor = Tensor::from_slice(slice, Shape::from(shape), device)?;

        // 6. 返回 Tensor
        Ok(tensor)
    }

    /// 从 Python torch.Tensor 转为 Rust candle_core::Tensor
    ///
    /// Tensor::from_raw_buffer
    fn _torch_to_candle3<'py>(
        torch_tensor: &Bound<'py, PyAny>,
        device: &Device,
    ) -> Result<Tensor, Error> {
        // 1. 获取 numpy 数组
        let np = torch_tensor.call_method0("numpy")?;

        // 2. 使用 downcast 而不是 extract
        let arr = np
            .downcast::<PyArrayDyn<T>>()
            .map_err(|e| Error::PyDowncastError(e.to_string()))?;

        // 3. 获取形状
        let shape = arr.shape().to_vec();

        // 4. 获取数据切片
        // let slice = unsafe { arr.as_slice()? };

        // 5. 创建 tensor
        // let tensor = Tensor::from_slice(slice, Shape::from(shape), device)?;

        let slice: &[u8] = unsafe {
            let slice_t = arr.as_slice()?; // 获取 &[T]
            std::slice::from_raw_parts(
                slice_t.as_ptr() as *const u8,  // 转换为 u8 指针
                std::mem::size_of_val(slice_t), // 计算字节长度
            )
        };
        let tensor = Tensor::from_raw_buffer(slice, DType::F32, &shape, device)?;

        // 6. 返回 Tensor
        Ok(tensor)
    }
}

impl<T> TensorWrapper<T>
where
    T: Element + WithDType,
{
    /// 转换为python对象
    ///
    /// 将数组转换为 python 的 tensor
    /// ```python,ignore
    /// import torch
    /// tensor = torch.tensor(data)
    /// ```
    pub fn to_py_tensor<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let data = self.into_pyobject(py)?;

        let torch = py.import("torch")?;
        torch.getattr("tensor")?.call1((data,))
    }
}

impl<T> From<Tensor> for TensorWrapper<T>
where
    T: Element + WithDType,
{
    fn from(value: Tensor) -> Self {
        TensorWrapper::from_tensor(value)
    }
}

impl<T> TensorWrapper<T>
where
    T: Element + WithDType,
{
    /// 将 tensor 转换为 vec4
    ///
    /// 假设Tensor的形状是[batch, channels, height, width]
    pub fn to_vec4(&self) -> Result<Vec<Vec<Vec<Vec<f32>>>>, candle_core::Error> {
        let shape = self.tensor.dims4()?;

        let data = self.tensor.flatten_all()?.to_vec1::<f32>()?;

        let mut result = Vec::with_capacity(shape.0);
        let elements_per_batch = shape.1 * shape.2 * shape.3;

        for batch in 0..shape.0 {
            let mut channels = Vec::with_capacity(shape.1);
            for channel in 0..shape.1 {
                let mut rows = Vec::with_capacity(shape.2);
                for row in 0..shape.2 {
                    let start =
                        batch * elements_per_batch + channel * shape.2 * shape.3 + row * shape.3;
                    let end = start + shape.3;
                    rows.push(data[start..end].to_vec());
                }
                channels.push(rows);
            }
            result.push(channels);
        }

        Ok(result)
    }
}

impl<'py, T> IntoPyObject<'py> for TensorWrapper<T>
where
    T: Element + WithDType,
{
    type Target = PyArrayDyn<T>; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = PyErr; // the conversion error type, has to be convertable to `PyErr`

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let tensor = self.into_tensor();
        let shape = tensor.dims();

        // 直接访问底层数据指针
        let data = tensor
            .flatten_all()
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?
            .to_vec1::<_>()
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

        // 创建数组并重新排列维度
        let array = PyArray::from_iter(py, data)
            .reshape(shape)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use pyo3::types::PyList;

    use super::*;

    #[test]
    #[ignore]
    fn test_new() -> anyhow::Result<()> {
        Python::with_gil(|py| {
            let binding = PyList::empty(py);
            let py_any = binding.as_any();
            let tensor_wrapper = TensorWrapper::<f32>::new(py_any, &Device::Cpu).unwrap();
            let tensor = tensor_wrapper.into_tensor();
            println!("dims: {:?}", tensor.dims());
        });

        Ok(())
    }

    #[test]
    #[ignore]
    fn test_extract_ndarray() -> anyhow::Result<()> {
        Python::with_gil(|py| {
            let binding = PyList::empty(py);
            let py_any = binding.as_any();
            let ndarray = TensorWrapper::<f32>::extract_ndarray(py_any).unwrap();
            println!("ndarray: {:?}", ndarray.to_vec());
        });

        Ok(())
    }
}
