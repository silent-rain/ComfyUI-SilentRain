# 库导入
from json import dumps
import comfyui_silentrain

# 函数导入
from comfyui_silentrain import sum_as_string

# ComfyUI 注册节点导入
from comfyui_silentrain import NODE_CLASS_MAPPINGS
from comfyui_silentrain import NODE_DISPLAY_NAME_MAPPINGS


# 子模导入, 当前仅支持该方式导入
from comfyui_silentrain import text
from comfyui_silentrain import logic
from comfyui_silentrain import utils


print("comfyui_silentrain:")
print(dir(comfyui_silentrain))

print("\n\n")
print("sum_as_string:")
print(sum_as_string(1, 2))


print("\n\n")
print("NODE_DISPLAY_NAME_MAPPINGS:", NODE_DISPLAY_NAME_MAPPINGS)
print("NODE_CLASS_MAPPINGS:", NODE_CLASS_MAPPINGS)


print("\n\n")
print("FileScanner: ", dir(text.FileScanner))
print("IndexAny: ", dir(logic.IndexAny))

print("\n\n")
print("AlwaysEqualProxy: ", dir(utils.AlwaysEqualProxy))
any_type = utils.AlwaysEqualProxy
print(type(any_type),any_type)
# print(type(any_type.__str__()),any_type.__str__())
# print(type(any_type.__repr__()),any_type.__repr__())
# print("__dict__: ", type(any_type.__dict__), any_type.__dict__)
print(any_type == "*")
print(any_type == "_*_")
print(dumps(any_type))



class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

    
print("\n\n")  
any_type = AlwaysEqualProxy("*")
print(type(any_type))
print(any_type)
print(any_type.__str__())
print(any_type.__repr__())
print("__dict__", any_type.__dict__, type(any_type.__dict__))
print(any_type == "*")
print(any_type == "_*_")
print(dumps(any_type))
