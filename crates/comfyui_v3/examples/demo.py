import sys
import os
sys.path.append("/home/one/code/ComfyUI")  # Adjust the path to where ComfyUI is located

from comfy_api.latest import io

print(type(io.NodeOutput))

print(io.NodeOutput())
print(io.NodeOutput("xxx").result)
print(io.NodeOutput("xxx",1,2,3).result)
