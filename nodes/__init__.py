from .comfyui_silentrain import *

try:
    from comfyui_silentrain_v3 import comfy_entrypoint
    _v3_ext = comfy_entrypoint()
except ImportError:
    _v3_ext = None
