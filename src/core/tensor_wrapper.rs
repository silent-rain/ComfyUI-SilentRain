//! Convert to Python object wrapper

use candle_core::{Device, Tensor, WithDType};
use numpy::{
    ndarray::{Dim, IxDynImpl},
    Element, PyArray, PyArrayDyn, PyArrayMethods, PyReadonlyArray, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::PyRuntimeError, types::PyAnyMethods, Bound, IntoPyObject, PyAny, PyErr, PyResult,
    Python,
};

pub struct TensorWrapper {
    tensor: Tensor,
}

impl TensorWrapper {
    pub fn new<'py, T: Element + WithDType>(
        py_any: &Bound<'py, PyAny>,
        device: &Device,
    ) -> PyResult<Self> {
        let ndarray = Self::extract_ndarray::<T>(py_any)?;
        let shape = ndarray.shape();
        let tensor = Tensor::from_vec(ndarray.to_vec()?, shape, device)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { tensor })
    }

    /// The dimension size for this tensor on each axis.
    pub fn dims(&self) -> &[usize] {
        self.tensor.dims()
    }

    pub fn from_tensor(tensor: Tensor) -> Self {
        Self { tensor }
    }

    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }
}

impl TensorWrapper {
    /// 转换为PyArray
    /// 使用更高效的numpy接口直接获取数据
    pub fn extract_ndarray<'py, T: Element>(
        py_any: &Bound<'py, PyAny>,
    ) -> PyResult<PyReadonlyArray<'py, T, Dim<IxDynImpl>>> {
        let numpy_any = py_any.call_method0("numpy")?;
        let array_view = numpy_any
            .downcast::<PyArrayDyn<T>>()? // 直接获取numpy数组
            .readonly(); // 获取只读视图

        Ok(array_view)
    }
}

impl TensorWrapper {
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

impl From<Tensor> for TensorWrapper {
    fn from(value: Tensor) -> Self {
        TensorWrapper::from_tensor(value)
    }
}

impl TensorWrapper {
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

impl<'py> IntoPyObject<'py> for TensorWrapper {
    // type Target = PyArray<f32, numpy::ndarray::Dim<numpy::ndarray::IxDynImpl>>; // the Python type
    type Target = PyArrayDyn<f32>; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = PyErr; // the conversion error type, has to be convertable to `PyErr`

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let tensor = self.into_tensor();
        let shape = tensor.dims();

        // 直接访问底层数据指针
        let data = tensor
            .flatten_all()
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?
            .to_vec1::<f32>()
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

    // #[test]
    // fn test_tensor() -> anyhow::Result<()> {
    //     let data: [u32; 3] = [1u32, 2, 3];
    //     let tensor = TensorWrapper::new(&data, &Device::Cpu)?;
    //     println!("tensor: {:?}", tensor.to_vec1::<u32>()?);

    //     let nested_data: [[u32; 3]; 3] = [[1u32, 2, 3], [4, 5, 6], [7, 8, 9]];
    //     let nested_tensor = TensorWrapper::new(&nested_data, &Device::Cpu)?;
    //     println!("nested_tensor: {:?}", nested_tensor.to_vec2::<u32>()?);

    //     Ok(())
    // }

    #[test]
    fn test_new() -> anyhow::Result<()> {
        Python::with_gil(|py| {
            let binding = PyList::empty(py);
            let py_any = binding.as_any();
            let tensor_wrapper = TensorWrapper::new::<f32>(py_any, &Device::Cpu).unwrap();
            let tensor = tensor_wrapper.into_tensor();
            println!("dims: {:?}", tensor.dims());
        });

        Ok(())
    }

    #[test]
    fn test_extract_ndarray() -> anyhow::Result<()> {
        Python::with_gil(|py| {
            let binding = PyList::empty(py);
            let py_any = binding.as_any();
            let ndarray = TensorWrapper::extract_ndarray::<f32>(py_any).unwrap();
            println!("ndarray: {:?}", ndarray.to_vec());
        });

        Ok(())
    }
}
