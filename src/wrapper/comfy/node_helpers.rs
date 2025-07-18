//! node_helpers

use std::collections::HashMap;

use candle_core::Tensor;

/// 条件
#[derive(Debug, Clone)]
pub struct Conditioning(pub Tensor, pub HashMap<String, Tensor>);

pub fn conditioning_set_values(
    conditionings: Vec<Conditioning>,
    values: HashMap<String, Tensor>,
    append: bool,
) -> Vec<Conditioning> {
    let mut c = Vec::new();
    for conditioning in conditionings {
        let mut n = conditioning.clone();
        for (k, val) in &values {
            let mut t_val = val.clone();
            if append {
                if let Some(old_val) = conditioning.1.get(k) {
                    t_val = old_val.add(val).unwrap();
                }
            }
            n.1.insert(k.to_string(), t_val);
        }
        c.push(n);
    }
    c
}
