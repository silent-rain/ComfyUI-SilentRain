// torch.nn.functional 包装
mod interpolation;
pub use interpolation::{interpolate, InterpolationMode};

mod pad;
pub use pad::pad;
