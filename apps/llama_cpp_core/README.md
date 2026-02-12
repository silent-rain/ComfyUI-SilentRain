# llama_cpp_core

基于 [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs) 的 Rust 高性能推理引擎，支持文本生成和多模态（视觉）推理。

## TODO

- 参数对接，请求参数接入；
- 模型缓存， comfyui端接入；
- 历史上下文, 待完善；
- comfyui对接；

## 特性

- **纯文本推理**：支持 GGUF 格式模型的文本生成
- **多模态支持**：通过 mmproj 文件支持视觉-语言模型（如 Qwen2.5-VL）
- **异步 API**：基于 Tokio 的异步推理接口
- **OpenAI Responses API 兼容**：使用 `open-ai-rust-responses-by-sshift` 提供标准响应格式
- **模型缓存**：内置 LRU 缓存机制，避免重复加载模型
- **聊天模板**：自动检测和应用模型聊天模板
- **媒体标记自动处理**：自动补全用户提示词中的媒体标记
- **灵活采样**：支持 temperature、top_k、top_p、repeat penalty 等多种采样参数

## 架构

```sh
llama_cpp_core/
├── src/
│   ├── lib.rs              # 库入口，导出主要类型
│   ├── pipeline.rs         # 推理流水线（核心 API）
│   ├── context.rs          # 文本上下文管理
│   ├── mtmd_context.rs     # 多模态上下文管理
│   ├── model.rs            # 模型加载与管理
│   ├── backend.rs          # llama.cpp 后端初始化
│   ├── sampler.rs          # 采样器配置
│   ├── cache.rs            # 模型缓存管理
│   ├── history.rs          # 聊天历史管理
│   ├── types.rs            # 核心类型定义
│   ├── error.rs            # 错误类型定义
│   ├── multimodal.rs       # 多模态数据类型
│   └── utils/
│       ├── image.rs        # 图像处理工具
│       └── mod.rs
├── examples/
│   └── basic_usage.rs      # 基础使用示例
└── Cargo.toml
```

## 快速开始

### 依赖

```toml
[dependencies]
llama_cpp_core = { path = "apps/llama_cpp_core" }
tokio = { version = "1", features = ["full"] }
```

### 1. 纯文本推理

```rust
use llama_cpp_core::{Pipeline, PipelineConfig, GenerateRequest, pipeline::response_extract_content};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pipeline_config = PipelineConfig::new("/path/to/model.gguf".to_string());
    let pipeline = Pipeline::try_new(pipeline_config)?;

    let request = GenerateRequest::text("Hello, how are you?")
        .with_system("You are a helpful assistant.")
        .with_temperature(0.7);

    let response = pipeline.generate(&request).await?;
    println!("Response: {}", response_extract_content(&response));
    Ok(())
}
```

### 2. 视觉推理（多模态）

```rust
use llama_cpp_core::{Pipeline, PipelineConfig, GenerateRequest, pipeline::response_extract_content};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pipeline_config = PipelineConfig::new_with_mmproj(
        "/path/to/vision-model.gguf".to_string(),
        "/path/to/mmproj.gguf".to_string()
    );

    let pipeline = Pipeline::try_new(pipeline_config)?;

    let request = GenerateRequest::text("Describe this image")
        .with_system("You are a helpful assistant.")
        .with_media_marker("<__media__>")
        .with_media_file("/path/to/image.png")?;

    let response = pipeline.generate(&request).await?;
    println!("Response: {}", response_extract_content(&response));
    Ok(())
}
```

### 3. 高级配置

```rust
use llama_cpp_core::PipelineConfig;

let config = PipelineConfig::new(model_path, mmproj_path)
    // 采样参数
    .with_temperature(0.6)
    .with_top_k(40)
    .with_top_p(0.95)
    .with_seed(42)
    // 性能参数
    .with_n_ctx(4096)       // 上下文窗口大小
    .with_n_batch(512)      // 批处理大小
    .with_n_threads(4)      // 生成线程数
    // GPU 配置
    .with_n_gpu_layers(33)  // GPU 层数
    .with_disable_gpu(false)
    // 缓存配置
    .with_cache_model(true) // 缓存模型以供重用
    .with_keep_context(true); // 保持对话上下文
```

## OpenAI Responses API 兼容性

本库使用 [`open-ai-rust-responses-by-sshift`](https://crates.io/crates/open-ai-rust-responses-by-sshift) 提供与 OpenAI Responses API 兼容的响应格式。

### 响应类型

```rust
use llama_cpp_core::types::Response;

// Response 结构包含以下主要字段：
// - id: 响应唯一标识
// - object: 对象类型 ("response")
// - created_at: 创建时间戳
// - model: 使用的模型
// - status: 响应状态 ("completed" 等)
// - output: 输出项列表
// - output_text: 合并后的输出文本 (便捷字段)
```

### 流式事件

```rust
use llama_cpp_core::types::StreamEvent;

// 流式推理返回的事件类型：
// - ResponseCreated { id }: 响应创建事件
// - TextDelta { content, index }: 文本增量事件
// - Done: 流式传输完成
```

## 核心类型

### PipelineConfig

| 参数 | 类型 | 默认值 | 说明 |
| ------ | ------ | -------- | ------ |
| `model_path` | `String` | - | GGUF 模型文件路径 |
| `mmproj_path` | `Option<String>` | `None` | 多模态投影文件路径 |
| `system_prompt` | `String` | `""` | 系统提示词 |
| `user_prompt` | `String` | `""` | 用户提示词 |
| `media_marker` | `Option<String>` | `"<__media__>"` | 媒体标记 |
| `temperature` | `f32` | `0.6` | 采样温度 |
| `top_k` | `i32` | `40` | Top-K 采样 |
| `top_p` | `f32` | `0.95` | Top-P (nucleus) 采样 |
| `n_ctx` | `u32` | `4096` | 上下文窗口大小 |
| `n_batch` | `u32` | `512` | 批处理大小 |
| `n_ubatch` | `u32` | `1024` | 微批处理大小（视觉模型需要较大值） |
| `n_predict` | `i32` | `2048` | 最大生成 token 数 |
| `n_gpu_layers` | `u32` | `0` | GPU 卸载层数 |

**`n_gpu_layers` 推荐值**：

| 模型大小 | 层数 | 推荐 n_gpu_layers |
| --------- | ------ | ------------------ |
| 2B | ~24 | 25-30 |
| 7B | ~29-33 | 35-40 |
| 13B | ~40 | 45-50 |
| 70B | ~80 | 85-100 |

注意：设置过大的值（如 10000）不会提升性能，反而会增加模型加载时间。

### GenerateRequest

生成请求结构体，用于配置推理参数：

```rust
use llama_cpp_core::GenerateRequest;

let request = GenerateRequest::text("Hello, how are you?")
    .with_system("You are a helpful assistant.")
    .with_model("gpt-4")
    .with_temperature(0.7)
    .with_max_completion_tokens(500);
```

| 参数 | 类型 | 说明 |
| ------ | ------ | ------ |
| `model` | `String` | 模型名称 |
| `temperature` | `f32` | 采样温度 (默认 0.7) |
| `max_completion_tokens` | `u32` | 最大完成令牌数 |
| `stream` | `bool` | 是否流式输出 |
| `session_id` | `String` | 会话ID（用于历史管理） |
| `keep_context` | `bool` | 是否保留上下文 |
| `medias` | `Vec<MediaData>` | 多模态媒体数据 |
| `media_marker` | `String` | 媒体占位符标记 |

### Response

OpenAI Responses API 兼容的响应结构：

```rust
use llama_cpp_core::{Response, pipeline::response_extract_content};

// 从响应中提取文本内容
let text = response_extract_content(&response);
```

## 媒体标记说明

当使用多模态模型时，用户提示词需要包含媒体标记（如 `<__media__>`）来指示图像插入位置。

**自动处理**：如果提示词中媒体标记数量少于加载的图像数量，系统会自动在提示词末尾补全标记。

```rust
// 用户提示词: "描述这张图片"
// 加载图像: 1 张
// 实际使用: "描述这张图片<__media__>"
```

## 大图像处理

处理高分辨率图像时，需要设置足够大的 `n_ubatch`（微批处理大小），否则会导致 `GGML_ASSERT` 失败。

```rust
let config = PipelineConfig::new(model_path, mmproj_path)
    .with_n_batch(1024)
    .with_n_ubatch(2048)  // 对于大图像需要 >= 图像 token 数量
    .with_user_prompt("描述这张图片".to_string());
```

**建议值**：

- 小图像（< 512x512）：`n_ubatch = 1024`
- 中等图像（512x512 ~ 1024x1024）：`n_ubatch = 2048`
- 大图像（> 1024x1024）：`n_ubatch = 4096` 或更大

## 示例

运行示例：

```bash
# 文本生成
cargo run --package llama_cpp_core --example text_generation

# 运行测试
cargo test --package llama_cpp_core --lib


# GPU
cargo run --package llama_cpp_core --example check_gpu --features vulkan
cargo run --package llama_cpp_core --example text_generation --features vulkan
cargo run --package llama_cpp_core --example vision_generation --features vulkan
```

## 编译

### Rust 编译

确保你已经安装了 Rust 和 Cargo。然后可以使用以下命令编译项目：

```bash
cargo build -p llama_cpp_core
```

### 安卓编译

- 通过脚本进行编译

```sh
cd apps/llama_cpp_core

# install: cargo install cargo-make

cargo make dev-android
```

- 设置环境变量

```sh
# 设置 Android NDK 路径
export ANDROID_NDK=$NDK_HOME
export NDK_ROOT=$NDK_HOME
export ANDROID_NDK_ROOT=$NDK_HOME
 
# C 编译器
export CC_aarch64_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang
export CC_armv7_linux_androideabi=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi35-clang
export CC_x86_64_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android35-clang
export CC_i686_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android35-clang

# C++ 编译器
export CXX_aarch64_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang++
export CXX_armv7_linux_androideabi=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi35-clang++
export CXX_x86_64_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android35-clang++
export CXX_i686_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android35-clang++

cargo build -p llama_cpp_core --target aarch64-linux-android
# cargo build -p llama_cpp_core --target armv7-linux-androideabi # 32 位架构（armv7）支持不完整。bindgen 生成的绑定代码包含硬编码的 64 位结构体大小断言，在 32 位系统上会失败。
cargo build -p llama_cpp_core --target x86_64-linux-android
cargo build -p llama_cpp_core --target i686-linux-android
```

- 变量无法透传，失败的示例

```sh
# 设置 Android NDK 路径
export ANDROID_NDK=$NDK_HOME
export NDK_ROOT=$NDK_HOME
export ANDROID_NDK_ROOT=$NDK_HOME
export API_LEVEL=35


# C 编译器
CC_aarch64_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang \
CC_armv7_linux_androideabi=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi35-clang \
CC_x86_64_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android35-clang \
CC_i686_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android35-clang \
# C++ 编译器
CXX_aarch64_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android35-clang++ \
CXX_armv7_linux_androideabi=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi35-clang++ \
CXX_x86_64_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android35-clang++ \
CXX_i686_linux_android=$NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android35-clang++ \
cargo build -p llama_cpp_core --target=aarch64-linux-android
```
