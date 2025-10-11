// torch.nn.functional 包装
mod interpolation;
pub use interpolation::{InterpolationMode, interpolate};

mod pad;
pub use pad::pad;
