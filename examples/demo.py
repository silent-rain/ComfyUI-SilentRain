# 库导入
import ComfyUI_SilentRain

# 函数导入
from ComfyUI_SilentRain import sum_as_string

# ComfyUI 注册节点导入
from ComfyUI_SilentRain import NODE_CLASS_MAPPINGS
from ComfyUI_SilentRain import NODE_DISPLAY_NAME_MAPPINGS


# 子模导入, 当前仅支持该方式导入
from ComfyUI_SilentRain import text


print("ComfyUI_SilentRain:")
print(dir(ComfyUI_SilentRain))

print("\n\n")
print("sum_as_string:")
print(sum_as_string(1, 2))


print("\n\n")
print("NODE_DISPLAY_NAME_MAPPINGS:", NODE_DISPLAY_NAME_MAPPINGS)
print("NODE_CLASS_MAPPINGS:", NODE_CLASS_MAPPINGS)


print("\n\n")
print(dir(text.FileScanner))
