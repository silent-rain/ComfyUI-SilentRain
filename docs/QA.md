# QA

## assertion `left == right` failed: "x86_64-unknown-linux-gnu" "x86_64-unknown-linux-gnu"

```text
  cargo:rerun-if-changed=/home/one/.cargo/git/checkouts/llama-cpp-rs-274405c613038803/4063f55/llama-cpp-sys-2/llama.cpp/tools/tts/CMakeLists.txt
  cargo:rerun-if-env-changed=TARGET
  cargo:rerun-if-env-changed=BINDGEN_EXTRA_CLANG_ARGS_x86_64-unknown-linux-gnu
  cargo:rerun-if-env-changed=BINDGEN_EXTRA_CLANG_ARGS_x86_64_unknown_linux_gnu
  cargo:rerun-if-env-changed=BINDGEN_EXTRA_CLANG_ARGS
  cargo:rerun-if-changed=wrapper.h

  --- stderr

  thread 'main' panicked at /home/one/.cargo/registry/src/rsproxy.cn-e3de039b2554c837/bindgen-0.72.1/lib.rs:917:13:
  assertion `left == right` failed: "x86_64-unknown-linux-gnu" "x86_64-unknown-linux-gnu"
    left: 4
   right: 8
  note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
warning: build failed, waiting for other jobs to finish...
```

修复：

```toml
# .cargo/config.toml

[env]
BINDGEN_EXTRA_CLANG_ARGS = "--target=x86_64-unknown-linux-gnu"
```

or

```shell
export BINDGEN_EXTRA_CLANG_ARGS="--target=x86_64-unknown-linux-gnu"
cargo build
```
