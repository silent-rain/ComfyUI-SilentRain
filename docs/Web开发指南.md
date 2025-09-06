# Web开发指南

## 系统环境

- 系统: Arch Linux x86_64
- Rust: 1.89.0
- Node: 24.7.0
- Pnpm: 9.15.0

## 环境搭建

### 安装 Node

```shell
sudo pacman -S nodejs npm

sudo npm install -g pnpm
```

### 安装wasm-pack

```shell
cargo install wasm-pack
```

## 开发与部署

### 编译

```shell
wasm-pack build --target web --out-dir ../../nodes/web/pkg
```

## 参考文档

- [ComfyUI Javascript 扩展开发文档](https://docs.comfy.org/zh-CN/custom-nodes/js/javascript_overview)
- [Rust 官方文档](https://www.rust-lang.org/zh-TW/learn/get-started)
- [UV 官方文档](https://astral.sh/uv)
- [Maturin 用户指南](https://www.maturin.rs/)
