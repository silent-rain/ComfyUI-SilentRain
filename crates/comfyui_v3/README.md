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
