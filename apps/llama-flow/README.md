# llama-flow

`llama-flow` 是一个基于 [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs) 构建的 Rust 高性能本地推理引擎，专为生产环境设计。它提供了完整的 OpenAI Chat Completion API 兼容接口，支持纯文本和多模态（视觉）推理，让你能够在本地设备上运行大型语言模型（LLM）和视觉-语言模型（VLM），无需依赖云服务。

**核心优势：**

- 🚀 **原生性能**：利用 Rust 的零成本抽象和 llama.cpp 的 C++ 优化，提供接近原生的推理速度
- 🔌 **即插即用**：完全兼容 OpenAI API 标准
- 🎯 **生产就绪**：内置模型缓存、会话管理、错误处理等特性
- 🌐 **多模态支持**：支持视觉-语言模型
- ⚡ **GPU 加速**：支持 CUDA（NVIDIA）和 Vulkan（AMD/Intel/移动设备）
- 🔄 **流式响应**：支持 Server-Sent Events (SSE) 流式输出
- 🛡️ **类型安全**：利用 Rust 的类型系统提供编译时安全保障
- 📦 **轻量部署**：支持 Android、iOS、Linux、Windows、macOS 等多平台

## 特性

- **纯文本推理**：支持 GGUF 格式模型的文本生成
- **多模态支持**：通过 mmproj 文件支持视觉-语言模型（如 Qwen3-VL）
- **异步 API**：基于 Tokio 的异步推理接口
- **OpenAI API 兼容**：完全兼容 OpenAI Chat Completion API 标准（`async-openai`）
- **模型缓存**：内置全局缓存管理器（基于 DashMap），支持多会话模型复用
- **会话管理**：支持多会话上下文隔离和历史消息管理
- **钩子系统**：灵活的推理生命周期钩子（消息验证、历史加载、错误处理等）
- **流式响应**：支持流式和非流式两种输出模式
- **灵活采样**：支持 temperature、top_k、top_p、presence_penalty、frequency_penalty 等参数
- **跨平台 GPU 加速**：支持 CUDA（NVIDIA）和 Vulkan（AMD/Intel/移动设备）

## 架构

```sh
llama-flow/
├── src/
│   ├── lib.rs                 # 库入口，导出主要类型
│   ├── pipeline/              # 推理流水线（核心 API）
│   │   ├── mod.rs
│   │   ├── pipeline_config.rs # 流水线配置
│   │   └── pipeline_impl.rs   # 推理实现
│   ├── context.rs             # 文本上下文管理
│   ├── mtmd_context.rs        # 多模态上下文管理
│   ├── model.rs               # 模型加载与管理
│   ├── backend.rs             # llama.cpp 后端初始化
│   ├── sampler.rs             # 采样器配置
│   ├── cache.rs               # 全局模型缓存管理
│   ├── history/               # 聊天历史管理
│   │   ├── manager.rs         # 历史管理器
│   │   ├── session.rs         # 会话上下文
│   │   └── mod.rs
│   ├── hooks/                 # 推理钩子系统
│   │   ├── traits.rs          # 钩子接口定义
│   │   ├── registry.rs        # 钩子注册器
│   │   ├── context.rs         # 钩子上下文
│   │   ├── pipeline_state.rs  # 流水线状态管理
│   │   └── builtin/           # 内置钩子
│   │       ├── validate.rs    # 请求验证
│   │       ├── normalize.rs   # 消息归一化
│   │       ├── system_prompt.rs # 系统提示词处理
│   │       ├── load_history.rs  # 加载历史消息
│   │       ├── assemble_messages.rs # 消息组装
│   │       ├── save_history.rs  # 保存历史消息
│   │       └── error_log.rs     # 错误日志
│   ├── request.rs             # OpenAI 请求封装
│   ├── response.rs            # OpenAI 响应封装
│   ├── unified_message.rs     # 统一消息格式（文本/多模态转换）
│   ├── types.rs               # 核心类型定义
│   ├── error.rs               # 错误类型定义
│   └── utils/
│       ├── image.rs           # 图像处理工具（Base64/URL/本地文件）
│       ├── log.rs             # 日志初始化
│       └── mod.rs
├── examples/
│   ├── text_generation.rs             # 文本生成示例
│   ├── vision_generation.rs           # 视觉推理示例
│   ├── vision_generation_stream.rs    # 流式视觉推理
│   ├── vision_parallel_generation.rs  # 并发推理示例
│   └── check_gpu.rs                   # GPU 检测
└── Cargo.toml
```

## 快速开始

### 依赖

```toml
[dependencies]
llama_flow = { path = "apps/llama-flow" }
tokio = { version = "1", features = ["full"] }
```

### 1. 纯文本推理

```rust
use llama_flow::{
    Pipeline, PipelineConfig,
    request::{ChatMessagesBuilder, UserMessageBuilder, CreateChatCompletionRequestArgs},
    response::response_extract_content,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 配置流水线
    let pipeline_config = PipelineConfig::new("/path/to/model.gguf")
        .with_n_gpu_layers(10)      // GPU 层数
        .with_n_ctx(4096)            // 上下文窗口
        .with_temperature(0.7);      // 采样温度

    let pipeline = Pipeline::try_new(pipeline_config)?;

    // 构建消息（OpenAI 兼容）
    let messages = ChatMessagesBuilder::new()
        .system("You are a helpful assistant.")
        .user("Who won the world series in 2020?")
        .assistant("The Los Angeles Dodgers won the World Series in 2020.")
        .user("Where was it played?")
        .build();

    // 创建请求
    let request = CreateChatCompletionRequestArgs::default()
        .max_completion_tokens(2048u32)
        .model("model-name")
        .messages(messages)
        .build()?;

    // 执行推理
    let response = pipeline.generate(&request).await?;
    println!("Response: {}", response_extract_content(&response));
    Ok(())
}
```

### 2. 视觉推理（多模态）

```rust
use base64::Engine;
use llama_flow::{
    Pipeline, PipelineConfig,
    request::{ChatMessagesBuilder, UserMessageBuilder, Metadata, CreateChatCompletionRequestArgs},
    response::response_extract_content,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 配置多模态流水线
    let pipeline_config = PipelineConfig::new_with_mmproj(
        "/path/to/vision-model.gguf",
        "/path/to/mmproj.gguf"
    )
    .with_n_gpu_layers(10)
    .with_media_marker("<start_of_image>")  // 媒体标记
    .with_image_max_resolution(768);         // 图像最大分辨率

    let pipeline = Pipeline::try_new(pipeline_config)?;

    // 读取图片并编码为 Base64
    let image_data = std::fs::read("/path/to/image.png")?;
    let base64_data = base64::engine::general_purpose::STANDARD.encode(&image_data);
    
    // 自动检测 MIME 类型
    let mime_type = infer::get_from_path("/path/to/image.png")
        .ok()
        .flatten()
        .map(|t| t.mime_type().to_string())
        .unwrap_or_else(|| "image/png".to_string());

    // 构建多模态消息（文本 + 图片）
    let messages = ChatMessagesBuilder::new()
        .system("You are a helpful assistant.")
        .users(
            UserMessageBuilder::new()
                .text("Describe this image")
                .image_base64(mime_type, base64_data)  // Base64 图片
                // 或使用 .image_url("https://...")   // 远程 URL
        )
        .build();

    // 会话管理（可选）
    let metadata = Metadata {
        session_id: Some("12345".to_string()),
        ..Default::default()
    };

    let request = CreateChatCompletionRequestArgs::default()
        .max_completion_tokens(2048u32)
        .model("vision-model")
        .metadata(metadata)
        .messages(messages)
        .build()?;

    let response = pipeline.generate(&request).await?;
    println!("Response: {}", response_extract_content(&response));
    Ok(())
}
```

### 3. 流式推理

```rust
use futures::StreamExt;
use llama_flow::Pipeline;

// 流式生成（返回 Stream<CreateChatCompletionStreamResponse>）
let mut stream = pipeline.generate_stream(&request).await?;

while let Some(chunk) = stream.next().await {
    match chunk {
        Ok(response) => {
            if let Some(choice) = response.choices.first() {
                if let Some(content) = &choice.delta.content {
                    print!("{}", content);
                }
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### 4. 高级配置

```rust
use llama_flow::PipelineConfig;

let config = PipelineConfig::new("model.gguf")
    // 模型配置
    .with_mmproj_path("mmproj.gguf")    // 多模态投影文件
    .with_n_gpu_layers(33)               // GPU 层数（0 = 纯 CPU）
    .with_main_gpu(0)                    // 主 GPU 设备 ID
    .with_threads(8)                     // CPU 线程数
    
    // 上下文配置
    .with_n_ctx(8192)                    // 上下文窗口大小
    .with_n_batch(512)                   // 批处理大小
    .with_max_completion_tokens(2048)    // 最大生成 token 数
    .with_max_history(20)                // 最大历史消息数
    
    // 采样参数
    .with_temperature(0.7)               // 温度（0.0-2.0）
    .with_top_k(40)                      // Top-K 采样
    .with_top_p(0.95)                    // Top-P（nucleus）采样
    .with_seed(42)                       // 随机种子
    
    // 多模态配置
    .with_media_marker("<image>")        // 媒体标记
    .with_image_max_resolution(1024)     // 图像最大分辨率
    
    // 调试选项
    .with_verbose(false);                // 详细日志
```

## 核心概念

### 1. Pipeline（推理流水线）

`Pipeline` 是推理的核心入口，封装了完整的推理流程：

- **模型加载**：支持模型缓存和复用
- **上下文管理**：自动管理文本/多模态上下文
- **钩子执行**：在推理各阶段执行自定义逻辑
- **响应生成**：支持流式和非流式输出

```rust
// Pipeline 可以被 Arc 包装，安全地在多个异步任务间共享
let pipeline = Arc::new(Pipeline::try_new(config)?);

// 非流式推理
let response = pipeline.generate(&request).await?;

// 流式推理
let stream = pipeline.generate_stream(&request).await?;
```

### 2. 请求构建器（OpenAI 兼容）

使用 `ChatMessagesBuilder` 和 `UserMessageBuilder` 构建符合 OpenAI 标准的请求：

```rust
// 纯文本消息
let messages = ChatMessagesBuilder::new()
    .system("You are a helpful assistant.")
    .user("Hello!")
    .build();

// 多模态消息（文本 + 图片）
let messages = ChatMessagesBuilder::new()
    .system("You are a helpful assistant.")
    .users(
        UserMessageBuilder::new()
            .text("Describe this image")
            .image_url("https://example.com/image.jpg")
            .image_base64("image/png", base64_data)
    )
    .build();
```

### 3. 会话管理

通过 `Metadata.session_id` 实现多会话隔离：

```rust
use llama_flow::request::Metadata;

let metadata = Metadata {
    session_id: Some("user-123".to_string()),
    ..Default::default()
};

let request = CreateChatCompletionRequestArgs::default()
    .metadata(metadata)
    .messages(messages)
    .build()?;
```

每个 session 独立维护：

- 历史消息记录
- 上下文状态

### 4. 钩子系统

钩子系统提供灵活的扩展点，内置钩子包括：

- **validate**：请求参数验证
- **normalize**：消息格式归一化
- **system_prompt**：系统提示词处理
- **load_history**：从会话加载历史消息
- **assemble_messages**：组装最终输入消息
- **save_history**：保存推理结果到历史
- **error_log**：错误日志记录

钩子按优先级顺序执行，支持自定义扩展。

### 5. 模型缓存

全局缓存管理器自动管理模型实例：

```rust
use llama_flow::cache::global_cache;

// 自动缓存和复用
let model = global_cache().get_or_load(&config)?;

// 手动清理缓存
global_cache().clear();
```

缓存 Key 基于：

- 模型路径
- mmproj 路径
- 主要配置参数（GPU 层数、线程数等）

## 示例

运行示例：

```bash
# 文本生成
cargo run --package llama-flow --example text_generation

# 视觉推理
cargo run --package llama-flow --example vision_generation

# 流式视觉推理
cargo run --package llama-flow --example vision_generation_stream

# 并发推理
cargo run --package llama-flow --example vision_parallel_generation

# GPU 检测
cargo run --package llama-flow --example check_gpu --features vulkan

# 运行测试
cargo test --package llama-flow --lib

# GPU 模式运行（Vulkan）
cargo run --package llama-flow --example text_generation --features vulkan
cargo run --package llama-flow --example vision_generation --features vulkan
```

## 编译

### Rust 编译

确保你已经安装了 Rust 和 Cargo。然后可以使用以下命令编译项目：

```bash
# 基础编译（CPU 模式）
cargo build -p llama-flow

# 发布编译（优化）
cargo build -p llama-flow --release

# Vulkan GPU 加速
cargo build -p llama-flow --features vulkan --release

# CUDA GPU 加速（需要 NVIDIA GPU 和 CUDA 工具链）
cargo build -p llama-flow --features cuda --release
```

### 功能特性（Features）

- **默认**：CPU 模式，动态链接 llama.cpp
- **`vulkan`**：启用 Vulkan GPU 加速（跨平台，支持 AMD/Intel/移动设备）
- **`cuda`**：启用 CUDA GPU 加速（仅 NVIDIA）

```toml
[dependencies]
llama_flow = { path = "apps/llama-flow", features = ["vulkan"] }
```

### 安卓编译

- **方法一：使用脚本编译**

```sh
cd apps/llama-flow

# 安装 cargo-make
cargo install cargo-make

# 编译 Android 版本
cargo make dev-android
```

- **方法二：手动设置环境变量编译**

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

# 编译各架构
cargo build -p llama-flow --target aarch64-linux-android
cargo build -p llama-flow --target x86_64-linux-android
cargo build -p llama_flow --target i686-linux-android

# 注意：32 位 armv7 架构支持不完整，bindgen 在 32 位系统上有问题
# cargo build -p llama-flow --target armv7-linux-androideabi
```

**注意**：环境变量需要单独 export，不能在命令行中一次性传递（会失败）。

## API 参考

### PipelineConfig

流水线配置主要方法：

| 方法 | 说明 | 默认值 |
| ------ | ------ | -------- |
| `new(model_path)` | 创建配置（文本模式） | - |
| `new_with_mmproj(model, mmproj)` | 创建配置（多模态） | - |
| `with_n_gpu_layers(n)` | GPU 层数（0=CPU） | 0 |
| `with_n_ctx(size)` | 上下文窗口大小 | 4096 |
| `with_n_batch(size)` | 批处理大小 | 512 |
| `with_temperature(t)` | 采样温度 | 0.6 |
| `with_top_k(k)` | Top-K 采样 | 40 |
| `with_top_p(p)` | Top-P 采样 | 0.95 |
| `with_threads(n)` | CPU 线程数 | 自动 |
| `with_max_completion_tokens(n)` | 最大生成 token 数 | 512 |
| `with_media_marker(marker)` | 媒体标记（多模态） | `<image>` |
| `with_image_max_resolution(px)` | 图像最大分辨率 | 1024 |
| `with_verbose(bool)` | 详细日志 | false |

### Pipeline

推理接口：

```rust
impl Pipeline {
    /// 创建流水线实例
    pub fn try_new(config: PipelineConfig) -> Result<Self, Error>;
    
    /// 非流式推理
    pub async fn generate(
        &self, 
        request: &CreateChatCompletionRequest
    ) -> Result<CreateChatCompletionResponse, Error>;
    
    /// 流式推理
    pub async fn generate_stream(
        &self, 
        request: &CreateChatCompletionRequest
    ) -> Result<impl Stream<Item = Result<CreateChatCompletionStreamResponse, Error>>, Error>;
}
```

### 请求构建

```rust
// ChatMessagesBuilder
impl ChatMessagesBuilder {
    pub fn new() -> Self;
    pub fn system(self, message: impl Into<String>) -> Self;
    pub fn user(self, message: impl Into<String>) -> Self;
    pub fn users(self, builder: UserMessageBuilder) -> Self;
    pub fn assistant(self, message: impl Into<String>) -> Self;
    pub fn build(self) -> Vec<ChatCompletionRequestMessage>;
}

// UserMessageBuilder（多模态消息）
impl UserMessageBuilder {
    pub fn new() -> Self;
    pub fn text(self, text: impl Into<String>) -> Self;
    pub fn image_url(self, url: impl Into<String>) -> Self;
    pub fn image_base64(self, mime_type: impl Into<String>, data: impl Into<String>) -> Self;
    pub fn image_file(self, path: impl AsRef<Path>) -> Result<Self, Error>;
    pub fn build(self) -> ChatCompletionRequestUserMessage;
}
```

## 最佳实践

### 1. 性能优化

**GPU 加速**：

- 根据显存调整 `n_gpu_layers`（建议从 10-20 开始测试）
- 使用 Vulkan 特性可在更多设备上加速

**批处理**：

- 增大 `n_batch` 可提升吞吐量（推荐 512-2048）
- 注意显存占用

**线程配置**：

- CPU 模式下，`n_threads` 建议设为物理核心数
- GPU 模式下，线程数影响较小

### 2. 上下文管理

**上下文窗口**：

```rust
// 长文本场景
.with_n_ctx(8192)
.with_max_history(50)

// 实时对话场景
.with_n_ctx(4096)
.with_max_history(10)
```

**会话隔离**：

- 始终为不同用户设置不同的 `session_id`
- 会话历史自动管理，支持上下文连续对话

### 3. 多模态推理

**图像处理**：

```rust
// 调整分辨率以平衡质量和性能
.with_image_max_resolution(768)  // 低显存
.with_image_max_resolution(1024) // 平衡
.with_image_max_resolution(1536) // 高质量
```

**媒体标记**：

- 确保 `media_marker` 与模型训练时一致
- 常见标记：`<image>`、`<start_of_image>`、`<|image|>`

### 4. 并发推理

Pipeline 支持并发安全：

```rust
let pipeline = Arc::new(Pipeline::try_new(config)?);

// 并发处理多个请求
let tasks: Vec<_> = requests.into_iter().map(|req| {
    let pipeline = Arc::clone(&pipeline);
    tokio::spawn(async move {
        pipeline.generate(&req).await
    })
}).collect();

let results = futures::future::join_all(tasks).await;
```

## 注意事项

1. **模型文件**：
   - 确保模型文件为 GGUF 格式
   - 多模态需要同时提供模型和 mmproj 文件

2. **显存管理**：
   - GPU 层数越多，显存占用越大
   - 多会话场景注意显存溢出

3. **线程安全**：
   - Pipeline 可安全地在多线程间共享（使用 Arc）
   - 模型缓存是全局线程安全的

4. **动态链接**：
   - 默认使用动态链接 llama.cpp
   - 确保运行时环境有对应的共享库（libllama.so/dll）

5. **会话持久化**：
   - 历史消息目前存储在内存中
   - 需要持久化可扩展 `ChatHistoryManager`

## 许可证

本项目基于 [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs) 构建，遵循相关开源许可。
