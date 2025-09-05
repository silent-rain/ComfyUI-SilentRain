//! 数学运算

use pyo3::{
    types::{PyModule, PyModuleMethods},
    Bound, PyResult, Python,
};

use crate::core::node::NodeRegister;

mod int_constant;
pub use int_constant::IntConstant;

mod float_constant;
pub use float_constant::FloatConstant;

mod int_to_float;
pub use int_to_float::IntToFloat;

mod float_to_int;
pub use float_to_int::FloatToInt;

/// 逻辑模块
pub fn submodule(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let submodule = PyModule::new(py, "math")?;
    submodule.add_class::<IntConstant>()?;
    submodule.add_class::<FloatConstant>()?;
    submodule.add_class::<IntToFloat>()?;
    submodule.add_class::<FloatToInt>()?;
    Ok(submodule)
}

/// Logic node register
pub fn node_register(py: Python<'_>) -> PyResult<Vec<NodeRegister<'_>>> {
    let nodes: Vec<NodeRegister> = vec![
        NodeRegister(
            "IntConstant",
            py.get_type::<IntConstant>(),
            "Sr Int Constant",
        ),
        NodeRegister(
            "FloatConstant",
            py.get_type::<FloatConstant>(),
            "Sr Float Constant",
        ),
        NodeRegister("IntToFloat", py.get_type::<IntToFloat>(), "Sr Int To Float"),
        NodeRegister("FloatToInt", py.get_type::<FloatToInt>(), "Sr Float To Int"),
    ];
    Ok(nodes)
}
