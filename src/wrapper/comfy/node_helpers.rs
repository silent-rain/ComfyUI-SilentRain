//! node_helpers

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use pyo3::{
    types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods},
    Bound, Python,
};

use crate::{error::Error, wrapper::torch::tensor::TensorWrapper};

/// 条件
#[derive(Debug, Clone)]
pub struct Conditioning(pub Tensor, pub HashMap<String, Tensor>);

impl Conditioning {
    /// 条件形状
    pub fn shape(&self) -> ConditioningShape {
        let shape = self.0.dims().to_vec();
        let mut shape_dict = HashMap::new();
        for (k, v) in &self.1 {
            shape_dict.insert(k.to_string(), v.dims().to_vec());
        }
        ConditioningShape(shape, shape_dict)
    }
}

/// 条件形状
#[derive(Debug, Clone)]
pub struct ConditioningShape(pub Vec<usize>, pub HashMap<String, Vec<usize>>);

/// Set values in conditionings
pub fn conditioning_set_values(
    conditionings: Vec<Conditioning>,
    values: HashMap<String, Tensor>,
    append: bool,
) -> Result<Vec<Conditioning>, Error> {
    let mut c = Vec::new();
    for conditioning in conditionings {
        let mut n = conditioning.clone();
        for (k, val) in &values {
            let mut t_val = val.clone();
            if append {
                if let Some(old_val) = conditioning.1.get(k) {
                    t_val = old_val.add(val)?;
                }
            }
            n.1.insert(k.to_string(), t_val);
        }
        c.push(n);
    }
    Ok(c)
}

/// Convert Python's conditioning to Rust type
pub fn conditionings_py2rs<'py>(
    conditioning: Bound<'py, PyList>,
    device: &Device,
) -> Result<Vec<Conditioning>, Error> {
    let mut conditionings = Vec::with_capacity(conditioning.len());
    for conditioning_item in conditioning.iter() {
        let py_tensor = conditioning_item.get_item(0)?;
        let tensor = TensorWrapper::<f32>::new(&py_tensor, device)?.into_tensor();

        let dict_any = conditioning_item.get_item(1)?;
        let dict_py = dict_any
            .downcast::<PyDict>()
            .map_err(|e| Error::PyDowncastError(e.to_string()))?;

        // py dict to rust map
        let mut dict_rs = HashMap::new();
        for (k, v) in dict_py {
            let k_rs = k.to_string();
            let v_rs = TensorWrapper::<f32>::new(&v, device)?.into_tensor();
            dict_rs.insert(k_rs, v_rs);
        }

        conditionings.push(Conditioning(tensor, dict_rs));
    }

    Ok(conditionings)
}

/// Convert Rust's conditioning to Python type
pub fn conditionings_rs2py<'py>(
    py: Python<'py>,
    conditioning: Vec<Conditioning>,
) -> Result<Bound<'py, PyList>, Error> {
    let mut list = Vec::new();
    for Conditioning(tensor, conditioning_dict) in conditioning.iter() {
        let tensor_py = TensorWrapper::<f32>::from_tensor(tensor.clone()).to_py_tensor(py)?;
        let dict_py = PyDict::new(py);
        for (k, v) in conditioning_dict {
            let tensor_py = TensorWrapper::<f32>::from_tensor(v.clone()).to_py_tensor(py)?;
            dict_py.set_item(k, tensor_py)?;
        }

        let elements = vec![tensor_py, dict_py.into_any()];
        let ele_list = PyList::new(py, elements)?;

        list.push(ele_list);
    }

    let results = PyList::new(py, list)?;
    Ok(results)
}

/// Convert Rust's conditioning shape to Python type
pub fn conditionings_shape_rs2py<'py>(
    py: Python<'py>,
    conditionings: Vec<Conditioning>,
) -> Result<Bound<'py, PyList>, Error> {
    let mut list = Vec::new();
    for conditioning in conditionings.iter() {
        let ConditioningShape(shape, shape_dict) = conditioning.shape();
        let shape_py = PyList::new(py, shape)?;

        let dict_py = PyDict::new(py);
        for (k, v) in shape_dict {
            dict_py.set_item(k, v)?;
        }

        let elements = vec![shape_py.into_any(), dict_py.into_any()];
        let ele_list = PyList::new(py, elements)?;

        list.push(ele_list);
    }
    let results = PyList::new(py, list)?;
    Ok(results)
}

/// out_latent to py dict
pub fn latent_rs2py<'py>(
    py: Python<'py>,
    out_latent: HashMap<String, Tensor>,
) -> Result<Bound<'py, PyDict>, Error> {
    let dict = PyDict::new(py);
    for (k, v) in out_latent {
        let tensor_py = TensorWrapper::<f32>::from_tensor(v.clone()).to_py_tensor(py)?;
        dict.set_item(k, tensor_py)?;
    }
    Ok(dict)
}
