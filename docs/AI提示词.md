# AI提示词

## 提示词框架

### RTF框架

RTF(Role-Task-Format)框架是一个非常简单通用的Prompt提示框架，我们和任意大模型对话场景下都可以使用该规范进行改进输出。

```text
● R-Role(角色)：指定大模型担当固定角色(程序员、数据分析师、讲解员、记者等等)

● T-Task(任务): 任务，告诉大模型需要为我们做的事情

● F-Format(格式)：大模型最终结果的返回格式(比如：表格、Markdown、英文等等)

主要优点：

● 简单、方便。

● 指定Role角色，可以让大模型在当前的角色范围内回答知识，这在一些特定的领域中非常有效。

● 指定Role角色也能让工程上检索知识能够确定边界范围，配合元数据所发挥的威力会更强。

● 如果结合RAG知识内容检索，那么上下文回答的内容会让用户感觉更加是顺畅。
```

### RISEN框架

```text
RISEN框架
● R-Role:大模型扮演的角色

● I-Instructions: 指示命令，和Task-任务差不多

● S-Steps: 步骤

● E-End Goal: 最终目标

● N-Narrowing(Constraints): 缩小范围(约束条件)，和RTF框架中的Format有异曲同工之妙，一个是格式的约束，而这里的约束可以是任意方面，比如回答的内容(特定领域)、字数限制等等方面

该框架主要适合：

● 撰写具有特定约束的任务(例如博客文章)

● 有明确指导方针的任务（例如商业计划）
```

### RODES框架

```text
● R-Role: 角色

● O - Objective: 目标

● D - Details: 详细的细节

● E - Examples: 示例

● S - Sense Check: 感官检查
```

### 思考链模式

```text
让我们逐步思考
```

### 自定义框架

```text
1.角色定义（# Role）：明确AI角色，增强特定领域的信息输出。
2.作者信息（## Profile）：包括作者、版本和描述，增加信息的可信度。
3.目标设定（## Goals）：一句话明确Prompt的目的。
4.限制条件（## Constrains）：帮助AI“剪枝”，避免无效的信息分支。
5.技能描述（## Skills）：强化AI在特定领域的知识和能力。
6.工作流程（## Workflow）：指导AI如何交流和输出信息。
7.初始化对话（## Initialization）：开始时的对话，重申关注的重点。
```

#### 模板

```md
# Role
您是一位同时精通PyTorch和Rust的AI工程师，精通以下技术栈：

- ​​PyTorch模型解析​​：熟悉PyTorch模型结构、张量操作和动态计算图
- ​​​Candle框架开发​​：掌握candle-core的模型定义、层实现和GPU加速
​- ​​PyO3互操作​​：能通过#[pyfunction]和#[pymodule]实现Rust-Python无缝交互
​- ​​性能优化​​：擅长利用Rust零成本抽象和SIMD指令提升推理效率

## Instructions
核心任务​​：将PyTorch代码转换为高性能Rust实现，需满足：

- ​​功能等价性​​：输出张量与原模型误差≤1e-5
​​- 部署兼容性​​：支持WASM浏览器环境和Python扩展调用
​​- 代码规范性​​：符合Rust所有权模型，禁用unsafe代码块

## Steps
1. 环境准备
"""python
import torch
"""

"""rust
pyo3 = { version = "0.25", features = [
    "extension-module",
    # "multiple-pymethods",
    "macros",
    "auto-initialize",
] }
numpy = "0.25" # Rust bindings for the NumPy C-API.


serde = { version = "1.0", features = ["derive"] }
serde_json = { version = "1.0" }
strum = "0.27"
strum_macros = "0.27"
walkdir = "2.5"
chardet = "0.2"
encoding = "0.2"
rand = "0.9"
rand_chacha = "0.9"

anyhow = "1.0"
thiserror = "2.0"
log = "0.4"
env_logger = "0.11"
tracing = { version = "0.1" }
tracing-subscriber = { version = "0.3", features = [] }

candle-core = "0.9"
candle-nn = "0.9"
candle-onnx = "0.9"
candle-transformers = "0.9"
hf-hub = "0.4"              # 与 huggingface hub 集成
tokenizers = "0.21"         # 标记文本
parquet = "55.0"            # 列式存储数据文件格式
image = "0.25"
ndarray = "0.16"
"""

2. Candle等效实现​

## Goal
将Python代码转换成Rust语言的代码。

## Constraints
兼容性要求​​：
- 支持PyTorch 2.0+的模型格式
```

#### 示例

```python
import numpy as np
from PIL import Image
import torch

def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)
```

```rust
[dependencies]
pyo3 = { version = "0.25", features = [
    "extension-module",
    # "multiple-pymethods",
    "macros",
    "auto-initialize",
] }
numpy = "0.25" # Rust bindings for the NumPy C-API.
candle-core = "0.9"
candle-nn = "0.9"
candle-onnx = "0.9"
candle-transformers = "0.9"
hf-hub = "0.4"              # 与 huggingface hub 集成
tokenizers = "0.21"         # 标记文本
parquet = "55.0"            # 列式存储数据文件格式
image = "0.25"
```
