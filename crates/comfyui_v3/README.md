# ComfyUI V3

## 单元测试配置

- 配置 `.cargo/config.toml`

```toml
# .cargo/config.toml

[build]
rustflags = [
    "-C",
    "link-arg=-lpython3.12",
    "-C",
    "link-arg=-L/home/one/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib",
]
```

- 配置 VsCode 设置

```json
// .vscode/settings.json

{
    "terminal.integrated.env.linux": {
        "LIBCLANG_PATH": "/usr/lib",
        "PYTHONHOME": "/home/one/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu",
        "PYO3_PYTHON": "/home/one/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/bin/python3.12",
        "LD_LIBRARY_PATH": "/home/one/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib"
    }
}
```

- 运行单元测试

```sh
# 启用 ComfyUI 虚拟环境
source .venv/bin/activate


# 手动设置临时环境变量
# export PYO3_PYTHON="/home/one/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/bin/python3.12"
# export LD_LIBRARY_PATH=/home/one/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib

cargo test -p comfyui_v3 --example example_node_v3 -- tests::test_execute --nocapture
```
