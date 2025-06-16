# python 函数封装
# import torch

# 子模导入, 当前仅支持该方式导入
from comfyui_silentrain import wrapper

def rust_isinstance():
    s = "this is a str."
    print("run_isinstance - str:", run_isinstance(s, str))
    print("run_isinstance - int:", run_isinstance(s, int))
    print("python.isinstance - str:", wrapper.python.isinstance(s, "str"))
    print("python.isinstance - int:", wrapper.python.isinstance(s, "int"))
    print("python.isinstance - torch.Tensor:", wrapper.python.isinstance(s, "torch.Tensor"))
    print("python.isinstance2 - str:", wrapper.python.isinstance2(s, str))
    print("python.isinstance2 - int:", wrapper.python.isinstance2(s, int))


def main():
    print(dir(wrapper.python))
    
    rust_isinstance()


def run_isinstance(obj: object, class_or_tuple: any) -> bool:
    return isinstance(obj, class_or_tuple)
                

if __name__ == '__main__':
    main()
