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

## AlreadyExists

```text
  --- stderr
  running: cd "/home/one/code/ComfyUI-SilentRain/target/x86_64-unknown-linux-gnu/debug/build/llama-cpp-sys-2-65404c1439e17ced/out/build" && LC_ALL="C" MAKEFLAGS="-j --jobserver-fds=8,9 --jobserver-auth=8,9" "cmake" "--build" "/home/one/code/ComfyUI-SilentRain/target/x86_64-unknown-linux-gnu/debug/build/llama-cpp-sys-2-65404c1439e17ced/out/build" "--target" "install" "--config" "Release"
  make: warning: -j8 forced in submake: resetting jobserver mode.

  thread 'main' (232101) panicked at /home/one/.cargo/git/checkouts/llama-cpp-rs-274405c613038803/0763e02/llama-cpp-sys-2/build.rs:926:56:
  called `Result::unwrap()` on an `Err` value: Os { code: 17, kind: AlreadyExists, message: "File exists" }
  note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
warning: build failed, waiting for other jobs to finish...
```

修复方案：

```sh
cargo clean
cargo clippy
```
