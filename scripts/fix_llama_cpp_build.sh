#!/bin/bash

# 任何命令失败后立即退出
set -e

# 删除缓存
rm -rf target/x86_64-unknown-linux-gnu/debug/build/llama-cpp-sys-2-*
rm -rf ~/.cargo/registry/cache/index.crates.io-1949cf8c6b5b557f/llama-cpp-*
rm -rf ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/llama-cpp-*
rm -rf ~/.cargo/git/checkouts/llama-cpp-*
rm -rf ~/.cargo/git/db/llama-cpp-*
