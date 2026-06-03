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

## PyO3 单元测试配置

## PyO3 单元测试 - build 收集

```rust
// build.rs

fn main() {
    // 使用pkg-config获取Python链接参数
    match pkg_config::Config::new()
        .atleast_version("3.12")
        .probe("python3")
    {
        Ok(library) => {
            println!("cargo:warning=pkg-config found python3");
            println!("cargo:rustc-link-lib=python3.12");
            for path in library.link_paths {
                println!("cargo:warning=link path: {}", path.to_string_lossy());
                println!("cargo:rustc-link-search=native={}", path.to_string_lossy());
            }
            for lib in library.libs {
                println!("cargo:warning=link lib: {}", lib);
                println!("cargo:rustc-link-lib={}", lib);
            }
        }
        Err(e) => {
            // 回退到系统默认路径
            println!("cargo:warning=pkg-config did not find python3: {}", e);

            // 查找并链接 Python 库
            if let Some(lib_dir) = find_python_library() {
                println!("cargo:warning=fallback to lib_dir: {}", lib_dir);

                println!("cargo:rustc-link-lib=python3.12");
                println!("cargo:rustc-link-search=native={}", lib_dir);

                // 让 pyo3 自动查找 Python
                pyo3_build_config::add_extension_module_link_args();
            } else {
                println!("cargo:warning=Could not find python library");
            }
        }
    }
}

// export LD_LIBRARY_PATH=/home/one/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib
fn find_python_library() -> Option<String> {
    // 尝试常见路径
    let paths = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/lib",
        "/home/one/.local/share/uv/python/cpython-3.12.9-linux-x86_64-gnu/lib",
    ];

    for path in &paths {
        if std::path::Path::new(path).exists() {
            return Some(path.to_string());
        }
    }

    None
}
```
