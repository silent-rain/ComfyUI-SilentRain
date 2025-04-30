//! 类型定义
//! 相关节点定义: ComfyUI/comfy/comfy_types/node_typing.py

/*
常见输入类型：


FLOAT = ("FLOAT", {"default": 1, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 0.01})

BOOLEAN = ("BOOLEAN", {"default": True})

BOOLEAN_FALSE = ("BOOLEAN", {"default": False})

INT = ("INT", {"default": 1,  "min": -sys.maxsize, "max": sys.maxsize, "step": 1})

STRING = ("STRING", {"default": ""})

STRING_ML = ("STRING", {"multiline": True, "default": ""})

STRING_WIDGET = ("STRING", {"forceInput": True})

JSON_WIDGET = ("JSON", {"forceInput": True})

METADATA_RAW = ("METADATA_RAW", {"forceInput": True})


"required": {
                "samples": ("LATENT", ),
                "tile_mode": (["None", "Both", "Decode(input) only", "Encode(output) only"],),
                "input_vae": ("VAE", ),
                "output_vae": ("VAE", ),
                "tile_size": ("INT", {"default": 512, "min": 320, "max": 4096, "step": 64}),
            },
"optional": {
    "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32, "tooltip": "This setting applies when 'tile_mode' is enabled."}),
}
*/

pub use super::always_equal_proxy::{any_type, AlwaysEqualProxy};

pub const NODE_INT: &str = "INT";
pub const NODE_FLOAT: &str = "FLOAT";
pub const NODE_STRING: &str = "STRING";
pub const NODE_BOOLEAN: &str = "BOOLEAN";
pub const NODE_LIST: &str = "LIST";
pub const NODE_JSON: &str = "JSON";
pub const NODE_METADATA_RAW: &str = "METADATA_RAW";

pub const NODE_INT_MAX: u64 = 0xffffffffffffffffu64;
pub const NODE_SEED_MAX: u64 = 10000000;
