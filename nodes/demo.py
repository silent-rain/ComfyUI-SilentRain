#!/home/one/code/ComfyUI/.venv/bin/python
# -*- coding:utf-8 -*-
# source /home/one/code/ComfyUI/.venv/bin/activate
import asyncio
import sys
import inspect

sys.path.append("/home/one/code/ComfyUI")

from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from comfyui_silentrain import build_extension, comfy_entrypoint

# 子模导入, 当前仅支持该方式导入
from comfyui_silentrain import v3


class ExampleExtension(ComfyExtension):
    nodes: list[type[io.ComfyNode]] = []

    def __init__(self, nodes: list[type[io.ComfyNode]] = []):
        super().__init__()
        self.nodes = nodes

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return []


class Example(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        """
        Return a schema which contains all information about the node.
        Some types: "Model", "Vae", "Clip", "Conditioning", "Latent", "Image", "Int", "String", "Float", "Combo".
        For outputs the "io.Model.Output" should be used, for inputs the "io.Model.Input" can be used.
        The type can be a "Combo" - this will be a list for selection.
        """
        return io.Schema(
            node_id="Example",
            display_name="Example Node",
            category="Example",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input(
                    "int_field",
                    min=0,
                    max=4096,
                    step=64,  # Slider's step
                    display_mode=io.NumberDisplay.number,  # Cosmetic only: display as "number" or "slider"
                    lazy=True,  # Will only be evaluated if check_lazy_status requires it
                ),
                io.Float.Input(
                    "float_field",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    round=0.001,  # The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    display_mode=io.NumberDisplay.number,
                    lazy=True,
                ),
                io.Combo.Input("print_to_screen", options=["enable", "disable"]),
                io.String.Input(
                    "string_field",
                    multiline=False,  # True if you want the field to look like the one on the ClipTextEncode node
                    default="Hello world!",
                    lazy=True,
                ),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(
        cls, image, string_field, int_field, float_field, print_to_screen
    ) -> io.NodeOutput:
        if print_to_screen == "enable":
            print(f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """)
        # do some processing on the image, in this example I just invert it
        image = 1.0 - image
        return io.NodeOutput(image)



def comfy_entrypoint_py() -> (
    ExampleExtension
):  # ComfyUI calls this to load your extension and its nodes.
    return ExampleExtension()


def demo_comfy_entrypoint():
    print("=== inspect comfy_entrypoint ===")
    # 检查是否需要 await
    print(f"iscoroutinefunction: {inspect.iscoroutinefunction(comfy_entrypoint)}")
    # 检查是否是一个 coroutine 对象
    print(f"iscoroutine: {inspect.iscoroutine(comfy_entrypoint)}")
    print(f"type: {type(comfy_entrypoint)}")
    print(f"type obj: {type(comfy_entrypoint())}")

    print("=== inspect comfy_entrypoint_py ===")
    print(f"iscoroutinefunction: {inspect.iscoroutinefunction(comfy_entrypoint_py)}")
    print(f"iscoroutine: {inspect.iscoroutine(comfy_entrypoint_py)}")
    print(f"type: {type(comfy_entrypoint_py)}")
    print(f"type obj: {type(comfy_entrypoint_py())}")


def demo_extension():
    # 调用函数获取返回的对象
    print("\n=== build_extension() return value (instance) ===")
    ext_instance = build_extension()
    print(type(ext_instance))
    print(f"isinstance ComfyExtension: {isinstance(ext_instance, ComfyExtension)}")

    print("\n=== ExampleExtension() return value ===")
    example_instance = ExampleExtension()
    print(type(example_instance))
    print(f"isinstance ComfyExtension: {isinstance(example_instance, ComfyExtension)}")


def demo_extension_nodes():
    print("\n=== demo_extension_nodes ===")
    ext_instance = build_extension()
    print(type(ext_instance))
    print(f"isinstance ComfyExtension: {isinstance(ext_instance, ComfyExtension)}")
    print(asyncio.run(ext_instance.get_node_list()))


def demo_node():
    print("image: ", dir(v3))
    print("image: ", dir(v3.image))
    print("image: ", dir(v3.image.InvertImage))
    print("image: ", type(v3.image.InvertImage))
    
    node_instance = v3.image.InvertImage()
    print(type(node_instance))
    print(f"isinstance io.ComfyNode: {isinstance(node_instance, io.ComfyNode)}")
    
    print("===========================\n")
    
    example_instance = Example()
    print(type(Example))
    print(type(example_instance))
    print(f"isinstance io.ComfyNode: {isinstance(example_instance, io.ComfyNode)}")
    
    
    
    # define_schema = v3.image.InvertImage.define_schema()
    # print("define_schema: ", define_schema)
    # print(type(define_schema))
    # print(f"isinstance io.ComfyNode: {isinstance(define_schema, io.ComfyNode)}")
    

"""
(ComfyUI) ➜  nodes git:(dev) ✗ python demo.py

image:  ['__all__', '__doc__', '__loader__', '__name__', '__package__', '__spec__', 'image', 'text']
image:  ['InvertImage', '__all__', '__doc__', '__loader__', '__name__', '__package__', '__spec__']
image:  ['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'define_schema', 'execute']
image:  <class 'type'>
<class 'builtins.InvertImage'>
isinstance io.ComfyNode: False
===========================

<class 'type'>
<class '__main__.Example'>
isinstance io.ComfyNode: True
"""

def main():
    # demo_comfy_entrypoint()
    # demo_extension()
    # demo_extension_nodes()
    demo_node()
    pass


if __name__ == "__main__":
    main()
