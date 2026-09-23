# 05. Operator / Kernel / Hardware：一次模型计算最终怎样在设备上执行

这一章要解决的是大模型推理中一个非常关键、也最容易产生概念混淆的问题：

> **模型代码里写下的一段计算，究竟经过了哪些软件层级，最后怎样变成 GPU / NPU 上真实发生的计算和数据搬运？**

在模型架构层，我们看到的是 `Linear`、`Attention`、`RMSNorm`、`MLP`、`MoE` 等具有明确数学语义的模块；进入 PyTorch、TensorRT、CANN 等框架和编译运行系统之后，它们会被进一步表示为 MatMul、Softmax、Reduce、TopK、SiLU 等 Operator，再经过图优化、算子融合、Kernel 选择或 Kernel 生成，最终变成设备上的矩阵计算、向量计算、Load/Store、DMA/Data Movement 和同步指令。

因此，可以先建立这样一条主线：

```text
Model Semantics
模型语义
    ↓
Framework / Graph Operator
框架或计算图算子
    ↓
Graph Optimization / Fusion
图优化、算子融合
    ↓
Kernel Implementation
设备 Kernel
    ↓
Instruction + Data Movement
设备指令 + 数据搬运
    ↓
Hardware Execution Unit
GPU SM / Tensor Core
或 Ascend AI Core / Cube / Vector
```

需要特别强调的是，这条链路表达的是**抽象层级关系，而不是严格的一一映射关系**。一个模型 Module 可以展开成多个 Operator；多个 Operator 又可能被 Fusion 成一个 Kernel；一个 Operator 在不同 Shape、数据类型和硬件上可能选择完全不同的 Kernel；一个 Kernel 内部又会执行大量计算、Load/Store、同步和控制指令。

所以真正应该建立的认知不是：

```text
Operator = Kernel = Hardware Instruction
```

而是：

```text
Operator 描述“算什么”
Kernel 描述“在某种设备上具体怎么计算”
Instruction 描述“硬件执行什么基本动作”
Hardware 决定这些动作能以多大的并行度、带宽和吞吐执行
```

---

# 05.1 先建立四个层级：Module、Operator、Kernel、Instruction

## 05.1.1 Model Module：模型作者眼中的计算语义

以一个 Decoder-only Transformer Layer 为例，在模型代码中通常会看到：

```text
RMSNorm
   ↓
Self Attention
   ↓
Residual Add
   ↓
RMSNorm
   ↓
MLP / MoE
   ↓
Residual Add
```

继续展开 Attention，可能得到：

```text
Hidden States
      │
      ├── Q Projection
      ├── K Projection
      └── V Projection
             ↓
        QKᵀ MatMul
             ↓
          Scale
             ↓
           Mask
             ↓
         Softmax
             ↓
          × V
             ↓
       O Projection
```

这一层仍然属于**模型语义层**。

例如模型源码中的：

```python
self.q_proj = nn.Linear(...)
self.k_proj = nn.Linear(...)
self.v_proj = nn.Linear(...)
```

告诉我们模型需要进行三个线性变换，但没有规定：

* GPU 使用哪个 CUDA Kernel；
* NPU 使用多少个 AI Core；
* 矩阵如何切 Tile；
* 数据先进入哪一级片上存储；
* MatMul 是否和 Bias、Activation 融合；
* 是否调用 cuBLAS / CUTLASS / TensorRT Kernel；
* 是否由 CANN / Ascend C 实现；
* 最终生成多少次 Kernel Launch。

因此：

> **模型源码定义计算语义，而不是最终设备执行计划。**

---

# 05.2 Operator：把模型语义变成计算图节点

模型进入框架之后，会逐渐转化为更标准的计算 Operator。

以一个 Linear 为例：

[
Y=XW^T+b
]

从模型角度它只是：

```text
Linear
```

进一步展开可能是：

```text
MatMul
  ↓
Bias Add
```

如果接一个 SiLU：

```text
Linear
  ↓
SiLU
```

进一步可能是：

```text
MatMul
  ↓
Add
  ↓
SiLU
```

TensorRT 官方 Operator Catalog 中就分别存在 `MatrixMultiply`、`ElementWise`、`SoftMax`、`TopK`、`Normalization`、`Attention`、`MoE` 等不同 Operator。也就是说，在推理编译器视角中，模型模块通常会继续拆成具有更明确计算语义的 Operator。

这里可以建立一个重要层级：

| 模型语义       | 典型 Operator                              |
| ---------- | ---------------------------------------- |
| Linear     | MatMul / GEMM + Bias                     |
| RMSNorm    | Square / Reduce / Rsqrt / Mul            |
| Attention  | MatMul + Scale + Mask + Softmax + MatMul |
| SwiGLU     | Linear + Split + SiLU + Mul              |
| MoE Router | Linear + Softmax + TopK                  |
| Expert FFN | Grouped GEMM + Activation + GEMM         |
| Residual   | Elementwise Add                          |

但实际执行时，这个表不会保持原样，因为后面还有非常重要的一步：

# **Fusion。**

---

# 05.3 为什么 Operator 和 Kernel 不会一一对应

Kernel 可以简单理解为：

> **真正被提交给 GPU / NPU 执行的一段设备程序。**

NVIDIA 对 TensorRT 的描述很直接：TensorRT Builder 会在构建 Engine 时针对目标 GPU 为网络 Layer 选择合适的低层 Kernel，然后 Runtime 执行这些 Kernel。

但是下面三种情况都可能发生：

```text
一个 Operator
    ↓
多个 Kernel
```

```text
多个 Operator
    ↓
一个 Fused Kernel
```

```text
一个 Operator
    ↓
根据 Shape / dtype / Hardware
选择不同 Kernel
```

因此：

> **Framework Graph 是逻辑计算图，而 Profile 中的 Kernel Timeline 是经过编译、Fusion 和 Kernel Selection 之后的执行图。**

两者本来就不应该期待完全一致。

---

# 05.4 一个 Operator 为什么可能变成多个 Kernel

以 RMSNorm 为例：

[
y_i =
\frac{x_i}
{\sqrt{\frac{1}{d}\sum_jx_j^2+\epsilon}}
\cdot\gamma_i
]

从数学语义看，它只是一个 RMSNorm。

但是朴素实现可能拆成：

```text
x
│
├─ Square
│
├─ Reduce Sum
│
├─ Divide
│
├─ Add eps
│
├─ Rsqrt
│
├─ Multiply x
│
└─ Multiply gamma
```

如果每一步分别 materialize 一个 Tensor，再启动独立 Kernel，就可能出现类似：

```text
Kernel 1: square
Kernel 2: reduction
Kernel 3: rsqrt
Kernel 4: scale
Kernel 5: multiply
...
```

每一步之间都可能涉及：

```text
读取 HBM
↓
执行计算
↓
写回 HBM
↓
下一个 Kernel 再读取
```

真正昂贵的部分有时并不是 `Mul` 或 `Rsqrt` 本身，而是中间 Tensor 不断在片外显存中读写，以及多次 Kernel Launch 和同步。

因此高性能实现往往会尝试变成：

```text
RMSNorm
     ↓
Fused RMSNorm Kernel
```

Kernel 内部完成：

```text
Load x
 ↓
Square
 ↓
Reduction
 ↓
Rsqrt
 ↓
Scale
 ↓
Store y
```

大量中间结果只停留在 Register、Shared Memory、UB 等片上存储中。

这就是 Operator Fusion 的核心价值之一。

---

# 05.5 Fusion 到底优化了什么

Fusion 经常被简单理解成“减少算子数量”，这个说法不够准确。

真正重要的是：

> **减少不必要的中间 Tensor Materialization、片外内存访问、Kernel Launch 和同步。**

TensorRT 官方文档明确说明，其 Layer Fusion 会把符合模式的多个 Layer 合并成一个优化 Kernel，从而减少 Kernel Launch，并避免把中间 Tensor materialize 到 DRAM；Attention Fusion 同样可以减少 Memory Traffic、Kernel Launch 与同步，并改善硬件利用率。

例如：

```text
MatMul
 ↓
Bias
 ↓
SiLU
```

朴素执行：

```text
HBM → MatMul → HBM
              ↓
HBM → Bias ─→ HBM
              ↓
HBM → SiLU ─→ HBM
```

Fusion 后则可能接近：

```text
HBM
 ↓
MatMul
 ↓
Bias
 ↓
SiLU
 ↓
HBM
```

中间数据尽可能保留在片上。

因此 Fusion 最核心的性能逻辑可以写成：

```text
减少 Launch
+
减少 Intermediate Tensor
+
减少 HBM / GM Traffic
+
提高 Data Locality
+
增加 Compute / Memory Overlap 的机会
```

---

# 05.6 Attention 是理解 Fusion 最好的例子

标准 Attention：

[
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt d}+Mask\right)V
]

如果完全按照数学计算图展开：

```text
Q ─────┐
       ├─ QKᵀ MatMul
K ─────┘
            ↓
          Scale
            ↓
           Mask
            ↓
         Softmax
            ↓
            ├──── MatMul ─── Output
V ──────────┘
```

最朴素的实现可能产生一个大小接近：

[
S\times S
]

的 Attention Score 中间 Tensor。

当 Sequence Length 很大时，这个中间 Tensor 的读写就会非常昂贵。

因此现代 Attention Kernel 的重要优化方向是：

> **把 QK MatMul、Softmax、与 V 的计算按照 Tile 分块组织，使 Attention Score 尽可能不完整落到 HBM。**

TensorRT 官方对 Fused Attention 的解释也指出，Attention Fusion 可以把长序列场景中的 memory footprint 从 (O(S^2)) 降到 (O(S))，同时减少 memory traffic、kernel launch 和 synchronization overhead。

所以模型图里虽然看到：

```text
MatMul
Softmax
MatMul
```

Profile 中却可能只看到少数几个高度优化的 Attention Kernel。

---

# 05.7 从 Kernel 再往下一层：数据究竟放在哪里

理解 Kernel 性能必须开始关注：

```text
数据在哪里
→
搬到哪里
→
在哪里计算
→
计算完写到哪里
```

对于现代 AI 加速器，一个非常重要的共性是：

```text
大容量片外 Memory
       ↓
较小但更快的片上 Memory
       ↓
Compute Unit
```

因为：

> **计算单元本身很快，但只有数据及时送到计算单元，峰值算力才有意义。**

这也是理解 Compute-bound / Memory-bound 的起点。

---

# 05.8 NVIDIA GPU：HBM → Cache / Shared Memory → Register → Compute

CUDA 官方给出的 GPU Memory Hierarchy 包含：

```text
Global Memory
      ↓
     L2
      ↓
SM
├── L1 / Shared Memory
├── Register File
└── Execution Units
```

CUDA Programming Guide 明确区分了 Global Memory、Shared Memory、Local Memory、Constant / Texture Memory 等不同空间，其中 Shared Memory 属于 Thread Block，可由 Block 内线程共享，而 Global Memory 可以被整个 Grid 中的线程访问。

对于 GEMM，可以把典型数据路径抽象成：

```text
HBM / Global Memory
        ↓
        L2
        ↓
 Shared Memory / Tile
        ↓
      Register
        ↓
 Tensor Core / CUDA Core
        ↓
      Register
        ↓
 Shared / Global Memory
```

这里最关键的问题不是：

> GPU 能不能做 MatMul？

答案当然是能。

真正的问题是：

> **数据能否以足够高的效率持续送入矩阵计算单元。**

---

# 05.9 Ascend：GM → L1 / UB / L0 → Cube / Vector

Ascend AI Core 的结构与 GPU 的抽象不完全相同，因此不应该简单把 GPU 名词逐一翻译过去。

Ascend 官方将 AI Core 的核心硬件划分为：

```text
Compute Units
├── Cube Unit
├── Vector Unit
└── Scalar Unit

Storage Units
├── L1
├── L0A
├── L0B
├── L0C
├── Unified Buffer
└── 其他 Buffer

Movement Units
├── MTE1
├── MTE2
├── MTE3
└── FixPipe
```

其中 Cube Unit 主要承担矩阵计算，Vector Unit 负责向量运算，Scalar Unit 负责地址计算、循环控制以及指令发射；MTE 等单元负责不同存储层级之间的数据搬运。

因此对于 Ascend，需要分别理解两条典型数据路径。

### Vector 计算

```text
Global Memory
      ↓
     MTE
      ↓
Unified Buffer
      ↓
 Vector Unit
      ↓
Unified Buffer
      ↓
     MTE
      ↓
Global Memory
```

Ascend 官方明确指出，Vector Unit 的源数据和目标数据需要位于 UB 中。

### Cube / MatMul 计算

典型路径则更接近：

```text
Global Memory
      ↓
      L1
      ↓
L0A       L0B
  \       /
   Cube Unit
       ↓
      L0C
       ↓
   FixPipe
       ↓
GM / L1
```

Ascend 官方给出的典型 Cube Data Flow 正是：

```text
GM → L1 → L0A/L0B → Cube → L0C → FixPipe → GM/L1
```

因此，如果把 NVIDIA 与 Ascend 放在一起理解，可以形成一个更抽象的统一模型：

```text
Off-chip Memory
HBM / GM
      ↓
On-chip Buffer / Cache
Shared Memory / L1 / UB / L0
      ↓
Compute Unit
Tensor Core / CUDA Core
Cube / Vector
```

但是具体硬件组织、编程模型以及数据路径不能直接等同。

---

# 05.10 MatMul / GEMM 为什么是 LLM 最重要的 Kernel

Transformer 中大量主要参数都位于 Linear Layer 中：

```text
Q Projection
K Projection
V Projection
O Projection

MLP Up Projection
Gate Projection
Down Projection

MoE Expert Linear
LM Head
```

其核心最终都可以归结为：

[
C=A\times B
]

或者：

[
C=A\times B+C
]

也就是 GEMM / GEMV 类问题。

但 MatMul 性能并不只由 FLOPs 数量决定，还取决于：

```text
M / N / K Shape
Batch Size
Sequence Length
dtype
Tensor Layout
Tile Size
Memory Alignment
Cache / Buffer Reuse
并行度
Tensor Core / Cube 利用率
```

同样一个：

```text
MatMul
```

在不同 Prefill / Decode 场景下，硬件行为可能差异很大。

---

# 05.11 Prefill 与 Decode 为什么会让同一个 Linear 表现不同

Prefill 阶段通常一次处理很多 Token。

例如：

```text
X:
[Batch × Sequence, Hidden]

W:
[Hidden, Output]
```

Linear 更接近：

```text
Large GEMM
```

矩阵维度较大，因此容易拥有较高的数据复用率和较好的矩阵计算单元利用率。

Decode 阶段，每一步通常只新增少量 Token：

```text
X:
[Batch, Hidden]
```

此时很多 Linear 更接近：

```text
GEMV / Small-M GEMM
```

矩阵计算中的一个维度显著缩小。

结果是：

```text
Prefill
→ 通常更容易形成高 Arithmetic Intensity
→ 更容易充分利用矩阵计算单元

Decode
→ 每 token 都要读取大量 Weight / KV Cache
→ 数据复用降低
→ 更容易受 Memory Bandwidth 限制
```

因此同一个模型：

```text
同样的 Linear
```

在 Prefill 和 Decode 阶段可能表现出完全不同的硬件瓶颈。

这也是为什么后面做性能分析时，不能只问：

> MatMul 快不快？

而应该问：

> **哪个 Shape 下、哪个阶段、什么 Batch、什么 dtype、什么 Kernel 的 MatMul 快不快？**

---

# 05.12 Compute-bound 与 Memory-bound

理解 Kernel 性能最重要的两个概念之一就是：

```text
Compute-bound
```

和：

```text
Memory-bound
```

假设 Kernel 需要进行很多计算，但是只读取少量数据：

```text
少量 Memory Traffic
        ↓
大量 FLOPs
```

那么瓶颈可能出现在计算单元：

```text
Compute-bound
```

反过来，如果：

```text
读很多数据
   ↓
只做很少计算
```

那么计算单元经常处于等待数据状态：

```text
Memory-bound
```

因此：

> **Kernel 快慢不能只看 FLOPs，还必须看为了这些 FLOPs 搬运了多少 Byte。**

---

# 05.13 Arithmetic Intensity：把 Compute 和 Memory 联系起来

Roofline Model 中最重要的概念是 Arithmetic Intensity：

[
AI=
\frac{\text{Operations}}
{\text{Memory Traffic}}
]

单位通常可以理解为：

```text
FLOP / Byte
```

例如：

### Kernel A

```text
1000 FLOPs
1000 Bytes
```

Arithmetic Intensity：

```text
1 FLOP / Byte
```

### Kernel B

```text
10000 FLOPs
1000 Bytes
```

Arithmetic Intensity：

```text
10 FLOP / Byte
```

Kernel B 每搬一个 Byte 的数据做了更多计算，因此更有机会充分利用计算单元。

NVIDIA Nsight Compute 官方 Roofline 文档就是使用 Arithmetic Intensity、Memory Bandwidth 和 Peak Compute Performance 来判断 Kernel 更接近 Memory-bound 还是 Compute-bound。

其基本关系可以抽象成：

[
Performance
\le
\min(
PeakCompute,
MemoryBandwidth\times ArithmeticIntensity
)
]

因此 Roofline 的核心不是画图本身，而是在回答：

> **当前 Kernel 的上限到底被算力还是带宽限制？**

---

# 05.14 为什么 Elementwise Operator 常常 Memory-bound

例如：

```text
Residual Add
SiLU
Mul
Scale
```

以 Add 为例：

[
C=A+B
]

每个元素需要：

```text
Load A
Load B
Add
Store C
```

但是数学计算只有：

```text
1 Add
```

也就是说：

```text
Memory Traffic 很大
Compute 很少
```

Arithmetic Intensity 很低。

因此这类 Operator 通常很容易受到 Memory Bandwidth 限制。

这解释了为什么：

```text
Add
Mul
SiLU
RMSNorm
```

单独看计算量很小，但在 LLM 中并不能简单认为它们“免费”。

大量小 Kernel 会带来：

```text
HBM Traffic
+
Kernel Launch
+
Synchronization
```

所以：

> **Fusion 对 Elementwise / Reduction 类 Operator 往往尤其重要。**

---

# 05.15 为什么 MatMul 更容易 Compute-bound

矩阵乘：

[
C_{M\times N}
=============

A_{M\times K}
B_{K\times N}
]

每个元素可以被多次复用。

如果一个 Tile 被搬进片上 Memory：

```text
A Tile
B Tile
```

就可以参与大量 Multiply-Accumulate。

因此可以实现：

```text
一次搬运
↓
重复计算很多次
```

Arithmetic Intensity 比 Elementwise Operator 高得多。

这也是 Tensor Core、Ascend Cube 等矩阵专用计算单元能够发挥价值的场景。

---

# 05.16 Tensor Core / Cube Unit：为什么矩阵乘需要专用硬件

现代 AI 芯片并不是简单依靠传统 Scalar ALU 一项项执行：

```text
a × b
+
c
```

而是提供专用矩阵计算单元。

NVIDIA 在 CUDA 编程模型中提供 Tile / MMA 等矩阵乘加抽象，并映射到底层矩阵加速硬件，包括 Tensor Core。

Ascend Cube Unit 则直接针对矩阵运算进行硬件加速。以官方文档中的 FP16 示例为例，Cube Unit 可面向 16×16 数据块执行矩阵运算，并通过 L0A、L0B、L0C 保存输入和中间结果。

因此大模型的矩阵计算通常追求：

```text
让数据 Shape / Layout
        ↓
适配 Tile
        ↓
充分利用 Tensor Core / Cube
```

如果 Shape 很小、不规则或对齐不好，即使理论 FLOPs 很少，也可能无法充分利用硬件。

---

# 05.17 Tiling：为什么不能把整个 Tensor 一次搬进计算单元

模型里的 Tensor 可能有：

```text
几 MB
几十 MB
几百 MB
甚至 GB
```

但是单个计算 Core 内部的高速存储空间要小得多。

所以无法采用：

```text
整个 Tensor
↓
一次搬入片上
↓
一次算完
```

而需要：

```text
Tensor
↓
Tile 0
Tile 1
Tile 2
Tile 3
...
```

每一 Tile：

```text
Load
 ↓
Compute
 ↓
Store
```

这就是 Tiling。

---

# 05.18 Ascend 为什么特别强调 Tiling

Ascend C 官方把一个典型 Operator Implementation 明确拆成两部分：

```text
Host
Tiling Implementation

Device
Kernel Implementation
```

因为 AI Core 片上存储无法一次容纳完整 Operator 输入输出，所以 Host 根据 Shape 和硬件信息确定：

```text
每个 Core 处理多少数据
每个 Tile 多大
循环多少次
如何处理 Tail
使用多少 Core
```

然后 Kernel 读取这些 Tiling 参数，控制 CopyIn、Compute、CopyOut。

因此 Ascend 上可以把 Operator 执行理解为：

```text
Host
│
├─ Shape
├─ dtype
├─ AI Core Num
├─ UB / L1 Size
│
↓
Tiling Strategy
│
├─ usedCoreNum
├─ block size
├─ loop count
├─ singleCoreM/N
└─ tail handling
      ↓
Device Kernel
      ↓
CopyIn → Compute → CopyOut
```

Ascend 官方 API 甚至提供获取 UB / L1 Size、AIC/AIV Core 数量等平台信息的能力，供 Tiling Function 使用。

这说明：

> **Tiling 本质上是“把全局 Tensor 计算问题映射到有限片上资源和多个 AI Core”的过程。**

---

# 05.19 一个 MatMul 的 Ascend 执行可以怎样理解

假设：

[
C=A\times B
]

完整矩阵太大，不能全部进入片上 Memory。

Host 首先计算：

```text
M / N / K
↓
如何切成 Tile
```

例如：

```text
A

┌────┬────┬────┐
│ A00│ A01│ A02│
├────┼────┼────┤
│ A10│ A11│ A12│
└────┴────┴────┘
```

再给不同 AI Core 分配计算区域。

设备侧可能反复执行：

```text
GM
 ↓
L1
 ↓
L0A / L0B
 ↓
Cube
 ↓
L0C
 ↓
结果写回
```

Ascend 的 MatMul Tiling 结构中甚至包含：

```text
usedCoreNum
M
N
Ka
Kb
singleCoreM
singleCoreN
...
```

用于描述 MatMul 如何映射到多个 Core 和 Tile。

所以：

> **MatMul Operator 并不等于“一次矩阵乘指令”，而是一个包含切分、搬运、矩阵计算、累加和写回的数据处理过程。**

---

# 05.20 GPU 上也存在同样的 Tile 思想

GPU GEMM 同样会把矩阵切成很多 Tile：

```text
Global Memory
      ↓
A Tile / B Tile
      ↓
Shared Memory
      ↓
Register Tile
      ↓
Tensor Core MMA
```

不断重复：

```text
Load next tile
Compute current tile
```

核心目标仍然是：

> **尽量让已经搬进片上存储的数据被重复利用，而不是反复从 HBM 读取。**

因此 GPU 与 NPU 虽然具体架构不同，但高性能矩阵 Kernel 的基本问题非常相似：

```text
Tile 多大？
数据放哪里？
如何并行？
如何复用？
如何隐藏 Memory Latency？
如何让矩阵计算单元持续工作？
```

---

# 05.21 Compute-Memory Overlap：不要等数据搬完才开始计算

最简单的 Kernel：

```text
Load Tile 0
     ↓
Compute Tile 0
     ↓
Load Tile 1
     ↓
Compute Tile 1
```

时间线：

```text
Memory |████|    |████|
Compute|    |████|    |████|
```

硬件资源轮流等待。

更高效的 Pipeline 是：

```text
Load Tile 1
      ↘
       Compute Tile 0

Load Tile 2
      ↘
       Compute Tile 1
```

于是：

```text
Memory |████|████|████|████|
Compute|    |████|████|████|
```

数据搬运与计算重叠。

CUDA 官方文档明确提供 asynchronous data copy 机制，将 Global Memory → Shared Memory 的搬运与计算重叠，以减少等待并提高资源利用率。较新的架构还提供 TMA 等硬件机制实现更高效的数据搬运。

Ascend AI Core 同样具有独立的 MTE、Cube、Vector 等 Pipeline，官方编程模型指出 Scalar 可以向不同执行单元发射指令，由这些单元异步并行执行，同时通过同步信号维护依赖关系。

所以对两类硬件，都可以用一个共同概念理解：

```text
理想状态：

搬下一块数据
      +
计算当前数据
      +
写回上一块结果

尽可能同时发生
```

---

# 05.22 Kernel Launch 为什么也是性能成本

一个 Kernel 不会凭空开始执行。

Host Runtime 需要：

```text
准备参数
 ↓
提交 Kernel
 ↓
Runtime / Driver 调度
 ↓
GPU / NPU 执行
```

如果一次 Transformer Layer 被拆成大量极小 Kernel：

```text
Kernel 1   3 μs
Kernel 2   4 μs
Kernel 3   2 μs
Kernel 4   5 μs
...
```

Kernel 本身很短，此时 Host Launch、调度、同步等固定成本占比就会变大。

TensorRT 官方把减少 Kernel Launch 明确列为 Layer Fusion 的主要收益之一。

因此：

```text
100 个很小的 Kernel
```

未必比：

```text
10 个较大的 Fused Kernel
```

更高效。

这也是 CUDA Graph、Fusion、Persistent Kernel 等技术存在的重要背景。

---

# 05.23 Softmax 为什么和 MatMul 的优化逻辑不同

Softmax：

[
Softmax(x_i)=
\frac{e^{x_i}}
{\sum_je^{x_j}}
]

内部大致包含：

```text
Max Reduction
 ↓
Subtract
 ↓
Exp
 ↓
Sum Reduction
 ↓
Divide
```

这里既有：

```text
Elementwise
```

又有：

```text
Reduction
```

不像 GEMM 那样具有非常规则的大规模矩阵乘加结构。

因此：

```text
MatMul
→ 重点利用 Tensor Core / Cube
→ 数据 Tile Reuse
→ Compute Throughput

Softmax
→ Reduction
→ Vector Compute
→ Memory Traffic
→ Synchronization
```

它们虽然同样叫 Operator，但硬件执行特征完全不同。

---

# 05.24 RMSNorm 的硬件特征

RMSNorm：

```text
Square
↓
Reduction Mean
↓
Rsqrt
↓
Scale
↓
Multiply Weight
```

主要由：

```text
Vector
+
Reduction
```

构成。

因此其性能关注点通常包括：

```text
Memory Bandwidth
Reduction Efficiency
Vectorization
Fusion
中间 Tensor 是否写回 HBM / GM
```

而不是 Tensor Core / Cube 峰值算力。

---

# 05.25 SiLU 为什么通常适合 Fusion

SiLU：

[
SiLU(x)=x\sigma(x)
]

计算量很小。

如果独立执行：

```text
HBM → x
↓
SiLU
↓
HBM → y
```

数据搬运成本可能远高于实际数学运算。

因此：

```text
Linear
↓
SiLU
```

如果条件允许，更理想的是：

```text
Linear + SiLU
Fused Kernel
```

让 Linear 输出直接进入 Activation，而不是完整写回片外 Memory 后再次读取。

---

# 05.26 MoE 为什么会出现 Grouped GEMM

MoE Layer 中，一个 Token 通常只会被 Router 分给少量 Expert。

假设：

```text
256 Experts
Top-K = 8
```

每个 Expert 实际收到的 Token 数不同：

```text
Expert 0: 35 tokens
Expert 1: 2 tokens
Expert 2: 18 tokens
...
```

如果为每个 Expert 单独启动：

```text
Expert 0 GEMM
Expert 1 GEMM
Expert 2 GEMM
...
```

会产生大量 Small GEMM：

```text
小矩阵
+
低利用率
+
大量 Kernel Launch
```

因此 MoE 高性能实现通常会尽量：

```text
多个 Expert GEMM
        ↓
Grouped GEMM
```

统一调度多个不同矩阵计算。

从硬件角度看，MoE 不只是：

```text
Router + Experts
```

而是：

```text
Routing
↓
Token Rearrangement
↓
Grouped GEMM
↓
Activation
↓
Grouped GEMM
↓
Combine
```

所以 MoE 的性能问题往往同时包含：

```text
Matrix Compute
Memory Movement
Irregular Shape
Load Balance
Communication
```

后面进入 Distributed 章节之后，还会进一步加入：

```text
Expert Parallel
All-to-All
```

---

# 05.27 Vectorization 到底是什么

Vectorization 可以粗略理解为：

> **一次指令同时处理多个数据元素。**

例如：

```text
普通标量：
a0 + b0
a1 + b1
a2 + b2
a3 + b3
```

Vector：

```text
[a0 a1 a2 a3]
      +
[b0 b1 b2 b3]
      ↓
一条 Vector Instruction
```

Ascend 官方将 Vector Unit 描述为类似 SIMD 的执行方式，一条向量指令可以对多个 Operand 执行相同操作。

因此：

```text
Add
Mul
Exp
Rsqrt
SiLU
Norm
```

等操作往往需要关注：

```text
Vector Width
Alignment
Continuous Memory Access
Tail Processing
```

例如 Ascend Vector 数据需要进入 UB，而且官方文档对起始地址和长度存在 32-byte alignment 等要求。

所以“只是一个 Add”也可能因为：

```text
Alignment 不好
Shape 不规则
Tail 太多
访存不连续
```

而利用率很低。

---

# 05.28 Memory Access Pattern 为什么会影响性能

假设读取：

```text
A[0]
A[1]
A[2]
A[3]
...
```

属于连续访问。

而：

```text
A[0]
A[1024]
A[2048]
A[3072]
...
```

属于大 Stride 访问。

虽然两者读取的数据量相同，但对：

```text
Cache Line
Memory Transaction
Bandwidth Utilization
```

的利用效率可能完全不同。

因此 Kernel 优化经常不只是减少 FLOPs，而是改变：

```text
Layout
Stride
Tile
Alignment
Reorder
Transpose
```

使硬件能够更高效地读取数据。

CUDA Best Practices Guide 也明确把使用 Shared Memory 改善 Global Memory Coalescing、减少冗余 Load 和浪费的带宽作为重要优化手段。

---

# 05.29 “Kernel 慢”实际上可能是很多不同的问题

当 Profile 中发现：

```text
某个 Kernel latency 很高
```

不能直接得出：

> 算法计算量太大。

真正应该继续判断：

```text
Kernel 慢
│
├─ Compute-bound？
│   ├─ Tensor Core / Cube 利用率不足？
│   ├─ Shape 太小？
│   ├─ dtype 没走高性能路径？
│   └─ Instruction throughput 不足？
│
├─ Memory-bound？
│   ├─ HBM / GM bandwidth？
│   ├─ Cache miss？
│   ├─ 数据重复 Load？
│   ├─ Layout / stride？
│   └─ 中间 Tensor 太多？
│
├─ Parallelism 不够？
│   ├─ Grid/Core 数不足？
│   ├─ Occupancy 低？
│   └─ Workload imbalance？
│
├─ Tiling 不合理？
│   ├─ Tile 太大？
│   ├─ Tile 太小？
│   ├─ UB / Shared Memory 利用不合理？
│   └─ Tail 太多？
│
├─ Pipeline 没重叠？
│   ├─ Memory 等 Compute？
│   └─ Compute 等 Memory？
│
└─ Launch / Synchronization Overhead？
```

这才是 Operator / Kernel 性能分析真正需要建立的思维方式。

---

# 05.30 一个完整案例：Linear 最终怎样运行

模型代码：

```python
y = self.linear(x)
```

### Level 1：Model Module

```text
Linear
```

数学：

[
Y=XW^T+b
]

### Level 2：Graph Operator

可能表示成：

```text
MatMul
↓
Add Bias
```

### Level 3：Graph Optimization

可能变成：

```text
MatMul + Bias
```

甚至：

```text
MatMul + Bias + SiLU
```

### Level 4：Kernel Selection / Generation

根据：

```text
M
N
K
dtype
layout
hardware
```

选择具体 Kernel。

### Level 5：Tiling

矩阵拆成：

```text
A Tiles
B Tiles
```

分配给不同 GPU Thread Block / Ascend AI Core。

### Level 6：Memory Movement

GPU：

```text
HBM
↓
L2
↓
Shared Memory
↓
Registers
```

Ascend：

```text
GM
↓
L1
↓
L0A / L0B
```

### Level 7：Compute

GPU：

```text
Tensor Core MMA
```

Ascend：

```text
Cube
```

### Level 8：Accumulation

部分结果在：

```text
Register / Accumulator
```

或：

```text
L0C
```

累加。

### Level 9：Epilogue

可能继续执行：

```text
Bias
Activation
Quantization
```

### Level 10：Store

最后写回：

```text
HBM / GM
```

于是原来模型代码中的：

```python
self.linear(x)
```

真正执行时实际上意味着：

```text
Kernel Selection
+
Tile Partition
+
Memory Movement
+
Matrix Instructions
+
Accumulation
+
可能的 Epilogue Fusion
+
Write Back
```

---

# 05.31 再看一个完整案例：RMSNorm

模型：

```text
RMSNorm
```

数学：

[
y=
x\cdot
\frac{1}
{\sqrt{mean(x^2)+\epsilon}}
\cdot\gamma
]

逻辑 Operator：

```text
Mul
↓
Reduce
↓
Add
↓
Rsqrt
↓
Mul
↓
Mul
```

低效实现：

```text
Kernel
↓
HBM
↓
Kernel
↓
HBM
↓
Kernel
↓
HBM
...
```

高性能实现：

```text
Fused RMSNorm Kernel
│
├─ Load
├─ Vector Square
├─ Reduction
├─ Rsqrt
├─ Scale
└─ Store
```

硬件瓶颈更可能集中在：

```text
Memory Bandwidth
+
Reduction
+
Vector Pipeline
```

而不是矩阵计算能力。

因此你不能因为它出现在模型图中和 Linear 一样都是一个“Block”，就认为两者属于同一种硬件工作负载。

---

# 05.32 再看一个完整案例：Attention

Model：

```text
Self Attention
```

Graph：

```text
QKV Projection
↓
QKᵀ
↓
Scale
↓
Mask
↓
Softmax
↓
×V
↓
O Projection
```

如果朴素执行：

```text
多个 GEMM Kernel
+
多个 Elementwise Kernel
+
Softmax Kernel
+
大量 intermediate Tensor
```

经过优化后：

```text
QKV Projection Kernel

Fused Attention Kernel
├─ QK Tile
├─ Online Softmax
├─ V Accumulation
└─ Tile Writeback

O Projection Kernel
```

因此：

```text
模型里看到 10 个逻辑步骤
```

并不代表：

```text
Profiler 一定看到 10 个 Kernel
```

反过来也一样。

---

# 05.33 Profiler 为什么和模型结构图长得完全不同

现在就可以解释很多初学者第一次看 Profile 时的困惑。

模型图：

```text
Attention
RMSNorm
MLP
Residual
```

Profiler：

```text
gemm_xxx
flash_attention_xxx
elementwise_xxx
reduce_xxx
memcpy
nccl_xxx
...
```

原因是两张图回答的是完全不同的问题。

模型图回答：

> **模型在数学上做什么？**

Profiler 回答：

> **Runtime 最终在设备上执行了什么？**

中间经过了：

```text
Framework Lowering
↓
Graph Transformation
↓
Fusion
↓
Kernel Selection
↓
Code Generation
↓
Runtime Scheduling
```

因此更准确的关系应该画成：

```text
              Model Architecture
                     │
        Attention / MLP / RMSNorm
                     │
                     ↓
              Operator Graph
                     │
     MatMul / Softmax / Add / Reduce
                     │
                     ↓
            Compiler / Runtime
      Fusion / Scheduling / Tiling
                     │
                     ↓
               Kernel Graph
                     │
      GEMM / FlashAttn / FusedNorm
                     │
                     ↓
             Hardware Execution
       Compute + Memory + Control
```

这也是为什么做 AI Infra 可视化时，**模型架构图、计算图和 Profile Timeline 应该被视为三个不同语义层，而不是试图强制合并成一张图。**

---

# 05.34 NVIDIA 与 Ascend 的统一理解框架

虽然 GPU 与 Ascend NPU 的实现不同，但可以建立一个对照框架：

| 抽象问题    | NVIDIA GPU                    | Ascend NPU             |
| ------- | ----------------------------- | ---------------------- |
| 设备程序    | CUDA Kernel                   | Ascend C Kernel        |
| 大容量设备内存 | Global Memory / HBM           | Global Memory / GM     |
| 片上高速存储  | L1 / Shared Memory / Register | UB / L1 / L0A/B/C      |
| 矩阵计算    | Tensor Core 等                 | Cube Unit              |
| 向量计算    | CUDA Core / Vector execution  | Vector Unit            |
| 控制      | Warp / SM scheduling 等        | Scalar Unit            |
| 数据搬运    | Load/Store、Async Copy、TMA 等   | MTE1/2/3、FixPipe 等     |
| 数据分块    | Tile / Block                  | Tiling                 |
| 异步流水    | Async Copy / Pipeline         | MTE + Compute Pipeline |
| 性能分析    | Nsight Compute / Systems      | CANN Profiling 工具链     |

但这个表只能用于**建立概念对应关系**，不能理解成硬件结构严格同构。

---

# 05.35 最终要建立的性能分析链路

这一章之后，当你看到：

```text
某个 Operator 很慢
```

应该逐层往下问。

第一层：

```text
它是什么模型语义？
```

例如：

```text
Attention
MLP
RMSNorm
MoE Router
```

第二层：

```text
它展开成什么 Operator？
```

例如：

```text
MatMul
Softmax
Reduce
Add
TopK
```

第三层：

```text
这些 Operator 有没有 Fusion？
```

例如：

```text
MatMul + Bias + Activation
Fused RMSNorm
Fused Attention
```

第四层：

```text
实际执行什么 Kernel？
```

并查看：

```text
Kernel Count
Kernel Latency
Shape
dtype
```

第五层：

```text
Kernel 是 Compute-bound
还是 Memory-bound？
```

用：

```text
Arithmetic Intensity
Memory Bandwidth
Compute Throughput
Roofline
```

判断。

第六层：

```text
数据在哪里移动？
```

GPU：

```text
HBM → L2 → Shared → Register
```

Ascend：

```text
GM → L1 / UB / L0
```

第七层：

```text
Tiling / Parallelism 是否合理？
```

看：

```text
Tile Size
Core / Thread Block 数
Alignment
Tail
Reuse
```

第八层：

```text
Compute 和 Memory 是否充分 Overlap？
```

最终才回答：

> **这段模型语义为什么在当前设备上快或慢。**

---

# 05.36 一张完整 Knowledge Map

这一章可以最终压缩成下面这张图：

```text
                    Model Semantics
                          │
            Attention / MLP / RMSNorm / MoE
                          │
                          ▼
                    Operator Graph
                          │
       MatMul / Softmax / Add / Reduce / TopK
                          │
                          ▼
                Compiler / Graph Optimize
                          │
        Fusion / Layout / Precision / Scheduling
                          │
                          ▼
                     Kernel
                          │
       GEMM / Fused Attention / Fused Norm
                          │
                          ▼
                      Tiling
                          │
        Shape → Tile → Core / Thread Block
                          │
                          ▼
                  Memory Movement
                          │
         ┌────────────────┴───────────────┐
         │                                │
       NVIDIA                           Ascend
         │                                │
 HBM → L2 → Shared                 GM → L1 / UB
      → Register                   → L0A/B/C
         │                                │
         ▼                                ▼
   Tensor Core / CUDA Core           Cube / Vector
         │                                │
         └────────────────┬───────────────┘
                          │
                          ▼
                  Hardware Execution
                          │
            Compute ↔ Memory Pipeline
                          │
                          ▼
                 Performance Result
                          │
       ┌──────────────────┼────────────────┐
       │                  │                │
 Compute-bound       Memory-bound      Launch /
                                        Sync-bound
```

---

# 05.37 本章最重要的十个判断

1. **Model Module、Operator、Kernel、Instruction 是不同抽象层级，不能混为一谈。**

2. **Operator 描述计算语义，Kernel 描述某种具体硬件上的实现方式。**

3. **Operator 与 Kernel 不存在稳定的一一对应关系。**

4. **Fusion 的主要价值是减少 Intermediate Tensor、Memory Traffic、Kernel Launch 和同步，而不只是“减少算子数量”。**

5. **Kernel 性能同时取决于 Compute 和 Memory，因此 FLOPs 不能单独解释性能。**

6. **Arithmetic Intensity 是连接 FLOPs 与 Memory Traffic 的关键指标，Roofline 可以用于判断 Compute-bound 与 Memory-bound。** NVIDIA Nsight Compute 官方正是采用这一模型进行 Kernel 性能分析。

7. **HBM / GM 容量大但距离 Compute Unit 远，片上 Buffer 小但速度高，因此高性能 Kernel 的核心之一是数据复用。**

8. **Tiling 是把大 Tensor 映射到有限片上存储和多个计算 Core 的基本方法；Ascend C 更是将 Host Tiling 与 Device Kernel 明确作为算子实现的两个组成部分。**

9. **Compute-Memory Overlap 的目标是让搬运下一块数据与计算当前数据同时进行，而不是让计算单元等待 Memory。** CUDA 和 Ascend 都提供了相应的异步数据搬运与多 Pipeline 执行机制。

10. **最终应该分析的是“模型语义 → Operator → Kernel → Memory / Compute → Hardware”的完整因果链，而不是单独记住某个 Kernel 名称。**

---

## 本章与后续章节的边界

到这里，我们只讨论：

```text
一段模型计算
如何在一块设备内部执行
```

还没有讨论：

```text
一个模型放不下一块卡怎么办？
多块卡怎样共同完成一个 Layer？
TP 为什么需要 AllReduce？
EP 为什么产生 All-to-All？
PP 为什么需要 Send / Recv？
```

这些问题会进入下一层：

```text
Model
  ↓
Operator / Kernel / Hardware
  ↓
Distributed Execution
```

也就是说，本章解决的是：

> **一块设备内部，Compute 和 Memory 如何完成一次模型计算。**

下一章开始解决：

> **当一次模型计算需要跨多个 Device 时，Compute、Memory 和 Communication 如何共同决定推理行为。**

### 官方资料依据

本章的硬件和执行模型主要依据 NVIDIA CUDA Programming Guide、CUDA Best Practices Guide、Nsight Compute Profiling Guide、TensorRT 官方文档，以及华为昇腾 CANN / Ascend C Operator Development 官方文档。NVIDIA 官方明确描述了 CUDA Memory Hierarchy、异步数据搬运、Roofline 与 TensorRT Fusion；Ascend 官方则明确描述了 AI Core 的 Cube / Vector / Scalar、UB/L1/L0、MTE、Host Tiling 与 Device Kernel 等执行机制。
