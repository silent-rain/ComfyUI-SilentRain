# 项目搭建

## 安装

```shell
# cargo
cargo install maturin
# pipx
pipx install maturin
# uv
uv tool install maturin
```

## 初始化项目

```shell
maturin init
```

## 构建python包

```shell
maturin build
```

## 部署到当前环境

```shell
maturin develop
```

## 发布到pypi

```shell
maturin publish
```

## 上传python包到pypi

```shell
maturin upload
```

## 相关文档

- [maturin使用指南](https://www.maturin.rs/)
- [PyO3/maturin-action](https://github.com/PyO3/maturin-action)
- [marketplace/actions/maturin-action](https://github.com/marketplace/actions/maturin-action)
- [pypa/manylinux](https://github.com/pypa/manylinux)
