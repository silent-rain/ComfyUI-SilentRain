//! node_helpers

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use log::error;
use pyo3::{
    types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods},
    Bound, Python,
};
use serde::{ser::SerializeSeq, Serialize};

use crate::{
    error::Error,
    wrapper::{python::isinstance_by_torch, torch::tensor::TensorWrapper},
};

/// 条件
#[derive(Debug, Clone)]
pub struct Conditioning(pub Tensor, pub HashMap<String, ConditioningEtx>);

#[derive(Debug, Clone)]
pub enum ConditioningEtx {
    Tensor(Tensor),
    Tensors(Vec<Tensor>),
    Float(f32),
}

/// 实现自定义 Serialize
impl Serialize for Conditioning {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // 将两个字段序列化为一个数组
        let mut seq = serializer.serialize_seq(Some(2))?;
        seq.serialize_element(&self.0.to_string())?;
        seq.serialize_element(&self.1)?;
        seq.end()
    }
}

/// 实现自定义 Serialize
impl Serialize for ConditioningEtx {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            ConditioningEtx::Tensor(t) => t.to_string().serialize(serializer),
            ConditioningEtx::Tensors(t) => t
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .serialize(serializer),
            ConditioningEtx::Float(f) => serde_json::to_string(&f).unwrap().serialize(serializer),
        }
    }
}

impl ConditioningEtx {
    pub fn dims(&self) -> ConditioningShapeEtx {
        match self {
            ConditioningEtx::Tensor(t) => ConditioningShapeEtx::Tensor(t.dims().to_vec()),
            ConditioningEtx::Tensors(ts) => {
                let mut t_list = Vec::new();
                for t in ts.iter() {
                    let v = t.dims().to_vec();
                    t_list.push(v);
                }
                ConditioningShapeEtx::Tensors(t_list)
            }
            ConditioningEtx::Float(v) => ConditioningShapeEtx::Float(*v),
        }
    }
}

/// 条件形状
#[derive(Debug, Clone, Serialize)]
pub enum ConditioningShapeEtx {
    Tensor(Vec<usize>),
    Tensors(Vec<Vec<usize>>),
    Float(f32),
}

/// 条件形状
#[derive(Debug, Clone, Serialize)]
pub struct ConditioningShape(
    pub ConditioningShapeEtx,
    pub HashMap<String, ConditioningShapeEtx>,
);

impl Conditioning {
    /// 条件形状
    pub fn shape(&self) -> ConditioningShape {
        let t_shape = self.0.dims().to_vec();
        let mut shape_dict = HashMap::new();
        for (k, v) in &self.1 {
            shape_dict.insert(k.to_string(), v.dims());
        }
        ConditioningShape(ConditioningShapeEtx::Tensor(t_shape), shape_dict)
    }
}

/// Set values in conditionings
pub fn conditioning_set_values(
    conditionings: Vec<Conditioning>,
    values: HashMap<String, ConditioningEtx>,
    append: bool,
) -> Result<Vec<Conditioning>, Error> {
    let mut c = Vec::new();
    for conditioning in conditionings {
        let mut n = conditioning.clone();
        for (k, val) in &values {
            let mut t_val = val.clone();
            if append {
                if let Some(old_val) = conditioning.1.get(k) {
                    t_val = match (old_val, val) {
                        (ConditioningEtx::Tensor(old_val), ConditioningEtx::Tensor(val)) => {
                            ConditioningEtx::Tensor(old_val.add(val)?)
                        }
                        (ConditioningEtx::Tensors(old_val), ConditioningEtx::Tensors(val)) => {
                            let mut old_val = old_val.clone();
                            let mut val = val.clone();
                            old_val.append(&mut val);
                            ConditioningEtx::Tensors(old_val.to_vec())
                        }
                        (ConditioningEtx::Float(old_val), ConditioningEtx::Float(val)) => {
                            ConditioningEtx::Float(old_val + val)
                        }
                        _ => {
                            error!("unknown type: {old_val:?} {val:?}");
                            return Err(Error::PyDowncastError(format!(
                                "unknown type: {old_val:?} {val:?}"
                            )));
                        }
                    };
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
            let py = v.py();

            if isinstance_by_torch(py, &v, "torch.Tensor")? {
                let v_rs = TensorWrapper::<f32>::new(&v, device)?.into_tensor();
                dict_rs.insert(k_rs, ConditioningEtx::Tensor(v_rs));
            } else if let Ok(v) = v.extract::<f32>() {
                dict_rs.insert(k_rs, ConditioningEtx::Float(v));
            } else if let Ok(list) = v.extract::<Bound<'py, PyList>>() {
                let mut list_rs = Vec::new();
                for item in list {
                    let v_rs = TensorWrapper::<f32>::new(&item, device)?.into_tensor();
                    list_rs.push(v_rs);
                }
                dict_rs.insert(k_rs, ConditioningEtx::Tensors(list_rs));
            } else {
                error!("unknown type: {v:?}");
                return Err(Error::PyDowncastError(format!("unknown type: {v:?}")));
            }
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
            match v {
                ConditioningEtx::Tensor(v) => {
                    let tensor_py =
                        TensorWrapper::<f32>::from_tensor(v.clone()).to_py_tensor(py)?;
                    dict_py.set_item(k, tensor_py)?;
                }
                ConditioningEtx::Tensors(tensors) => {
                    let mut list_py = Vec::new();
                    for tensor in tensors {
                        let tensor_py =
                            TensorWrapper::<f32>::from_tensor(tensor.clone()).to_py_tensor(py)?;
                        list_py.push(tensor_py);
                    }

                    let list = PyList::new(py, list_py)?;
                    dict_py.set_item(k, list)?;
                }
                ConditioningEtx::Float(v) => {
                    dict_py.set_item(k, *v)?;
                }
            }
        }

        let elements = vec![tensor_py, dict_py.into_any()];
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
