# 库导入
import comfyui_silentrain

# 函数导入
from comfyui_silentrain import sum_as_string

# ComfyUI 注册节点导入
from comfyui_silentrain import NODE_CLASS_MAPPINGS
from comfyui_silentrain import NODE_DISPLAY_NAME_MAPPINGS


# 子模导入, 当前仅支持该方式导入
from comfyui_silentrain import text


print("comfyui_silentrain:")
print(dir(comfyui_silentrain))

print("\n\n")
print("sum_as_string:")
print(sum_as_string(1, 2))


print("\n\n")
print("NODE_DISPLAY_NAME_MAPPINGS:", NODE_DISPLAY_NAME_MAPPINGS)
print("NODE_CLASS_MAPPINGS:", NODE_CLASS_MAPPINGS)


print("\n\n")
print(dir(text.FileScanner))
