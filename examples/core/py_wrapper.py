# python 函数封装
# import torch

# 子模导入, 当前仅支持该方式导入
from comfyui_silentrain import core

def rust_isinstance():
    s = "this is a str."
    print("run_isinstance - str:", run_isinstance(s, str))
    print("run_isinstance - int:", run_isinstance(s, int))
    print("core.isinstance - str:", core.isinstance(s, "str"))
    print("core.isinstance - int:", core.isinstance(s, "int"))
    print("core.isinstance - torch.Tensor:", core.isinstance(s, "torch.Tensor"))
    print("core.isinstance2 - str:", core.isinstance2(s, str))
    print("core.isinstance2 - int:", core.isinstance2(s, int))


def main():
    print(dir(core))
    
    rust_isinstance()


def run_isinstance(obj: object, class_or_tuple: any) -> bool:
    return isinstance(obj, class_or_tuple)
                

if __name__ == '__main__':
    main()
