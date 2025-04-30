//! Convert to Python object wrapper

use std::ops::Deref;

use candle_core::{Device, NdArray};
use pyo3::{exceptions::PyTypeError, Bound, IntoPyObject, PyAny, PyErr, PyObject, Python};

use crate::error::Error;

// impl IntoPy<PyObject> for Tensor {
//     fn into_py(self, py: Python<'_>) -> PyObject {
//         let shape = self.dims().iter().map(|x| *x as usize).collect::<Vec<_>>();
//         let data = self.flatten_all().unwrap().to_vec1::<f32>().unwrap();

//         // 转换为numpy数组
//         PyArrayDyn::from_vec(py, shape, data).into_py(py)
//     }
// }

pub struct Tensor(candle_core::Tensor);

impl Tensor {
    #[allow(clippy::new_ret_no_self)]
    pub fn new<A>(array: A, device: &Device) -> Result<candle_core::Tensor, Error>
    where
        A: NdArray,
    {
        Ok(candle_core::Tensor::new(array, device)?)
    }
}

impl Deref for Tensor {
    type Target = candle_core::Tensor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'py> IntoPyObject<'py> for Tensor {
    type Target = PyAny; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = PyErr; // the conversion error type, has to be convertable to `PyErr`
                        // Err(PyErr::new::<PyTypeError, _>("Error message"));

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        todo!()
    }
}
