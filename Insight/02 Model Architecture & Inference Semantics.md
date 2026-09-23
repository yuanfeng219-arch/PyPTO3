# 02. Model Architecture & Inference Semantics：模型结构为什么会改变推理行为

这一章研究的重点不是重新学习一遍 Transformer，而是建立一个面向推理工程的因果链：

> **Model Architecture → Tensor Shape / Data Flow → Compute / Memory / Communication → Inference Behavior**

模型结构不是一张只用于解释算法的静态框图。对于推理系统来说，`hidden_size`、Layer 数量、Attention 类型、KV Head 数量、FFN 宽度、Expert 数量、Top-K、是否存在 Shared Expert 等架构选择，会直接决定每个 token 需要执行哪些计算、读写多少数据、保存多少状态，以及模型切到多卡以后需要发生什么通信。

因此，这一章最终要培养的能力是：**拿到一个公开模型的 `config.json` 和 `model.py`，在完全没有 profiling 数据的情况下，就能够先推断出它主要的计算结构、显存压力来源、KV Cache 特征、可能的通信模式，以及 Prefill / Decode 哪一阶段更容易成为瓶颈。**

Profiling 的作用是随后验证“实际运行得怎么样”，而不是等 Profile 出来以后才知道“模型在做什么”。

---

## 02.1 从模型结构到推理行为：先建立三个层级

分析模型时，首先要区分三个经常被混在一起的层级。

```text
Model Architecture
模型语义层
Attention / MLP / MoE / RMSNorm
        │
        ▼
Mathematical / Framework Ops
计算操作层
Linear / MatMul / Softmax / SiLU / Mul
        │
        ▼
Runtime Kernels
硬件执行层
FlashAttention Kernel / GroupedMatmul /
MoeDispatch Kernel / ...
```

例如，一个典型的 gated MLP 在模型源码中可能是：

```text
MLP
├── gate_proj
├── up_proj
├── activation
└── down_proj
```

它的数学关系可以表示为：

[
y = W_{down}(\operatorname{SiLU}(W_{gate}x)\odot W_{up}x)
]

因此展开到计算操作层是：

```text
                 x
              ┌──┴──┐
              ▼     ▼
           MatMul  MatMul
              │     │
             SiLU   │
              └──┬──┘
                 ▼
                Mul
                 │
                 ▼
              MatMul
```

但这仍然不是 Kernel 图。编译器可能把 `MatMul + SiLU + Mul` 中的一部分融合，也可能因为张量 shape、并行方式、量化方案和硬件实现而拆成多个 Kernel。

因此：

> **模型节点描述“模型要完成什么语义计算”；Op 描述数学运算；Kernel 描述某一次具体部署如何在硬件上执行这些运算。**

静态模型架构应该由 `config.json + model.py` 确定，而不是由某块 GPU/NPU 上产生的 Kernel 反推。

---

# 02.2 Decoder-only 模型的一次 Forward 到底发生了什么

现代生成式大语言模型大量采用 Decoder-only Transformer。虽然不同模型在 Attention、FFN、位置编码、Norm 等细节上差别很大，但仍然可以先建立一个标准执行骨架：

```text
Token IDs
   │
   ▼
Embedding
   │
   ▼
Hidden States
   │
   ├─────────────────────────────┐
   ▼                             │
RMSNorm                          │ residual
   │                             │
   ▼                             │
Self Attention                   │
   │                             │
   └──────────── Add ◀───────────┘
                  │
                  ├────────────────────────┐
                  ▼                        │
               RMSNorm                    │ residual
                  │                        │
                  ▼                        │
             MLP / MoE                    │
                  │                        │
                  └────── Add ◀────────────┘
                           │
                           ▼
                      Next Layer
                           │
                          ...
                           │
                           ▼
                      Final Norm
                           │
                           ▼
                        LM Head
                           │
                           ▼
                         Logits
```

Transformer 原始工作已经建立了 Self-Attention、Position-wise Feed-Forward、Residual 和 Layer Normalization 等基本结构；现代 Decoder-only LLM 通常会调整 Norm 位置、激活函数和 Attention 实现，但“Attention + FFN + Residual”的主干仍然存在。

从推理工程角度，需要关注的并不是每个模块的数学证明，而是**每一个模块正在处理什么 Tensor，以及这些 Tensor 会产生什么成本**。

| 模块                  | 主要输入/输出                                | 推理意义                                     |
| ------------------- | -------------------------------------- | ---------------------------------------- |
| Embedding           | Token ID → Hidden State                | 将离散 token 映射为连续向量                        |
| RMSNorm / LayerNorm | Hidden State → Normalized Hidden State | 数值归一化，通常属于 bandwidth-sensitive 的逐元素/归约计算 |
| Attention           | Hidden State → Contextual Hidden State | 建立当前 token 与历史 token 的依赖，产生 KV Cache     |
| Residual Add        | 两路 Hidden State 相加                     | 保留主干信息流                                  |
| MLP / FFN           | Hidden → Intermediate → Hidden         | 通常拥有大量模型参数和矩阵乘计算                         |
| MoE                 | Hidden → Router → Expert → Hidden      | 将 FFN 计算稀疏化，并引入 Expert routing           |
| LM Head             | Hidden → Vocabulary Logits             | 将最后 Hidden State 映射到词表维度                 |

因此从第一天开始就应该把模型理解成一条 **Hidden State 主干**：

```text
Token
 ↓
Embedding
 ↓
Hidden State
 ↓
Layer 0
 ↓
Hidden State
 ↓
Layer 1
 ↓
...
 ↓
Layer N-1
 ↓
Hidden State
 ↓
LM Head
 ↓
Logits
```

Attention、MLP、MoE 等模块，本质上都在不断读取并更新这条 Hidden State。

---

# 02.3 为什么生成模型必须区分 Prefill 和 Decode

一次用户请求进入 LLM 后，通常可以从计算语义上分为两个阶段：

```text
Prompt:
"Explain why the sky is blue"

             Prefill
                 │
     一次处理整个 Prompt
                 │
                 ▼
           KV Cache 建立
                 │
                 ▼
              Decode
                 │
          生成第 1 个 token
                 │
          生成第 2 个 token
                 │
                 ...
```

假设 Prompt 有 1000 个 token。

## Prefill

Prefill 时，这 1000 个 token 可以一起经过模型，因此矩阵乘通常具有较大的 token dimension：

```text
[1000 tokens × hidden]
          │
          ▼
      Attention
          │
          ▼
         MLP
```

这种阶段拥有较高的并行度，通常能比较充分地利用大型矩阵乘单元。

同时，每层 Attention 会为 Prompt token 产生需要用于后续生成的 K/V 状态：

```text
Layer 0 KV
Layer 1 KV
Layer 2 KV
...
Layer N KV
```

这些状态组成 KV Cache。

## Decode

开始生成以后，逻辑发生变化。

生成第一个 token：

```text
Prompt
  ↓
token 1001
```

生成第二个 token 时，第一个新 token 已经成为历史：

```text
Prompt + token1001
        ↓
     token1002
```

所以自回归语言模型存在天然的数据依赖：

[
x_{t+1}=f(x_1,\ldots,x_t)
]

**token (t+1) 必须等 token (t) 产生以后才能确定输入。**

因此单条序列的 Decode 无法像 Prefill 那样一次计算未来几十个 token。

这也是 Prefill 和 Decode 性能行为差异的根源之一：

```text
Prefill
很多 token × 一次 Forward
→ 大矩阵
→ 高并行度

Decode
1 个新 token × 重复 Forward
→ 小矩阵
→ 大量状态读取
```

---

# 02.4 KV Cache：为什么不需要每次重算所有历史 token

如果没有 KV Cache，生成第 1001 个 token 后，再生成第 1002 个 token 时，就可能需要重新计算：

```text
token 1
token 2
...
token 1000
token 1001
```

但在因果 Self-Attention 中，历史 token 的 K 和 V 在模型权重不变时已经确定。

因此第一次处理：

```text
token 1 → K1 V1
token 2 → K2 V2
...
token 1000 → K1000 V1000
```

可以直接保存：

```text
KV Cache
[K1 ... K1000]
[V1 ... V1000]
```

产生 token 1001 时，只需要计算新的：

```text
Q1001
K1001
V1001
```

然后：

```text
Q1001
  │
  ▼
Attention against
K1 ... K1001
  │
  ▼
weighted sum of
V1 ... V1001
```

接着把 `K1001/V1001` append 到 Cache。

因此 KV Cache 是一种典型的：

> **用 Memory 换 Compute。**

它避免历史 token 的 K/V projection 被不断重新计算，但代价是随着：

```text
Batch Size ↑
Sequence Length ↑
Number of Layers ↑
KV Representation Size ↑
```

KV Cache 显存会快速增长。

这也是为什么现代 Attention 架构大量创新并不是围绕“Attention 会不会算”，而是围绕：

> **到底需要保存多少 KV，以及每生成一个 token 需要从显存搬多少历史 KV。**

---

# 02.5 MHA → MQA → GQA：真正变化的是 KV Representation

理解 Attention 架构时，不应该只记“多少个 Attention Heads”。

更重要的是分别看：

```text
Query Heads
Key Heads
Value Heads
```

---

## 02.5.1 MHA：每个 Query Head 都拥有自己的 K/V Head

标准 Multi-Head Attention 可以抽象为：

```text
              Hidden State
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
        Q          K          V
        │          │          │
   Q0 Q1 ...   K0 K1 ...   V0 V1 ...
```

假设：

```text
Query Heads = 32
KV Heads    = 32
```

那么每一层、每一个历史 token 都需要保存对应的 32 组 K/V。

因此 KV Cache 规模近似与：

[
N_{layers}\times N_{tokens}\times N_{kvheads}\times HeadDim
]

成正比，K 和 V 又分别存在，所以还需要乘以 2。

---

# 02.5.2 MQA：所有 Query Head 共用一组 K/V

Multi-Query Attention 保留多个 Query Heads，但只保留：

```text
KV Heads = 1
```

可以理解为：

```text
Q0 ─┐
Q1 ─┤
Q2 ─┤
... │──→ shared K / V
Q31─┘
```

MQA 最初就是针对 incremental decoding 的内存带宽问题提出的，因为 Decode 阶段需要不断读取历史 K/V；让所有 Query Head 共用 K/V 可以显著减少需要缓存和读取的数据。

所以它主要改变的并不是：

> “Attention 有几个头。”

而是：

> **每个历史 token 到底要保存多少份 K/V representation。**

---

# 02.5.3 GQA：MHA 和 MQA 之间的折中

Grouped Query Attention 位于两者之间：

```text
Query Heads = 32
KV Heads    = 8
```

意味着：

```text
Q0 Q1 Q2 Q3       → KV0
Q4 Q5 Q6 Q7       → KV1
...
Q28 Q29 Q30 Q31  → KV7
```

GQA 论文将它定义为介于 MHA 与 MQA 之间的方案：使用少于 Query Heads、但多于一个的 KV Heads，以接近 MQA 的推理效率，同时保留接近 MHA 的模型质量。

Hugging Face 的 Qwen2 配置甚至直接把这个关系编码进了配置语义：

```text
num_key_value_heads == num_attention_heads
→ MHA

num_key_value_heads == 1
→ MQA

1 < num_key_value_heads < num_attention_heads
→ GQA
```

因此看到公开模型：

```json
{
  "num_attention_heads": 32,
  "num_key_value_heads": 8
}
```

即使还没有运行模型，你已经能判断：

```text
Attention Type ≈ GQA

相对于 32-head MHA：
KV representation 数量约下降到 1/4

→ KV Cache 更小
→ Decode KV memory traffic 更低
→ 长序列 / 高 batch 推理更友好
```

这就是“从 Architecture 推断 Inference Behavior”。

---

# 02.6 MLA：进一步改变“缓存什么”

MHA、MQA、GQA 的变化，可以看成仍然围绕：

```text
到底保存多少 K Heads
到底保存多少 V Heads
```

MLA（Multi-head Latent Attention）的变化更深一层：

> **不再直接把完整 K/V representation 作为主要缓存对象，而是保存压缩后的 latent representation。**

DeepSeek-V2 正式引入 MLA，其核心是对 Key 和 Value 做低秩联合压缩；论文明确指出，MLA 的设计目的之一就是大幅降低 KV Cache，并报告 DeepSeek-V2 相对于对照架构实现了显著的 KV Cache 缩减。

可以简化理解成：

### MHA / GQA

```text
Hidden State
    │
    ├── K ──→ Cache K
    │
    └── V ──→ Cache V
```

### MLA

```text
Hidden State
    │
    ▼
KV Down Projection
    │
    ▼
Compressed KV Latent
    │
    └────────────→ Cache
```

论文中的核心关系可以抽象成：

[
c_t^{KV}=W^{DKV}h_t
]

其中：

```text
h_t
= 当前 token hidden state

c_t^KV
= 压缩后的 KV latent
```

后续 K/V 信息可以通过投影关系参与 Attention。

因此 MLA 改变的已经不仅是：

```text
多少个 KV Head
```

而是：

```text
KV Cache 中保存的数据表示本身
```

这也是为什么 DeepSeek-V3 的公开配置会出现传统 Transformer 不常见的字段，例如：

```text
kv_lora_rank = 512
q_lora_rank = 1536

qk_rope_head_dim = 64
qk_nope_head_dim = 128
v_head_dim = 128
```

这些字段直接对应 MLA 内部不同 representation 的维度。Hugging Face 当前的 DeepSeek-V3 配置同时暴露了这些参数。

因此 MLA 对推理的影响可以总结为：

```text
Architecture
MLA / KV latent compression
        ↓
Representation
Cache compressed latent rather than full KV
        ↓
Memory
Smaller KV Cache
        ↓
Bandwidth
Less historical state to load during Decode
        ↓
Inference
更适合长上下文 / 高并发 Decode
```

DeepSeek-V3 继续采用 MLA，并明确把它定位为提升推理效率的核心架构之一。

---

# 02.7 Attention Architecture 最终改变哪三件事

因此分析 Attention 时，真正应该形成下面这套框架：

| Architecture | KV Representation      | Memory  | Decode 行为                      |
| ------------ | ---------------------- | ------- | ------------------------------ |
| MHA          | 每个 Query Head 对应独立 K/V | 最大      | KV Cache / bandwidth 压力较高      |
| MQA          | 所有 Query Head 共用一组 K/V | 很小      | Decode memory traffic 显著降低     |
| GQA          | 多个 Query Head 共用一组 K/V | 中间      | 质量与效率折中                        |
| MLA          | 缓存压缩 KV latent         | 更特殊、更紧凑 | 显著改变 KV Cache 与 Attention 数据路径 |

所以看到：

```text
num_attention_heads
```

还远远不够。

至少要继续问：

```text
num_key_value_heads 是多少？
head_dim 是多少？
KV 保存什么 representation？
是否存在 latent compression？
是否存在 sliding/window/local attention？
最大 Context Length 是多少？
```

这些才是真正影响推理系统的字段。

---

# 02.8 Dense FFN：为什么模型大量计算集中在 Linear

Attention 之外，Decoder Layer 的另一个核心模块是 FFN / MLP。

以现代 LLM 常见的 gated MLP 为例：

[
MLP(x)=W_{down}\left(
\operatorname{SiLU}(W_{gate}x)
\odot
W_{up}x
\right)
]

对应：

```text
                    x
                ┌───┴───┐
                ▼       ▼
           gate_proj   up_proj
                │       │
               SiLU     │
                └───┬───┘
                    ▼
                   Mul
                    │
                    ▼
                down_proj
```

所以一个 Dense FFN 的核心参数规模主要由：

```text
hidden_size
intermediate_size
```

决定。

例如：

```text
hidden_size       = 4096
intermediate_size = 11008
```

意味着 Hidden State 会：

```text
4096
 ↓
约 11008 intermediate representation
 ↓
4096
```

由于 gate 和 up 通常拥有两套投影，所以 gated FFN 往往包含三个大型 Linear。

因此看到：

```json
{
  "hidden_size": 4096,
  "intermediate_size": 14336
}
```

你已经可以判断：

> 这是一个较宽的 FFN；大量权重存储和矩阵乘计算会集中在 FFN projection。

这就是为什么分析推理 FLOPs 时，不能只盯着 Attention。

---

# 02.9 Dense → MoE：改变的是“参数规模”和“每 token 计算量”的关系

Dense 模型里，每个 token 都经过同一套 FFN：

```text
token A ─┐
token B ─┼─→ Same FFN
token C ─┘
```

如果 FFN 有：

```text
gate_proj
up_proj
down_proj
```

所有 token 都执行它们。

MoE 则将单个 FFN 替换成多个 Expert：

```text
                  Router
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
     Expert 0    Expert 1 ... Expert N
```

但是每个 token 通常只选择其中少数 Expert：

```text
Token A → Expert 3 + Expert 19
Token B → Expert 2 + Expert 7
Token C → Expert 19 + Expert 31
```

Mixtral 8×7B 就是经典例子：每个 MoE 层拥有 8 个 Feed-Forward Expert，而每个 token 由 Router 选择其中 2 个 Expert。

因此 MoE 引入了必须区分的两个概念。

---

## 02.9.1 Total Parameters

模型实际拥有的所有参数：

```text
Expert 0
Expert 1
Expert 2
...
Expert 255
```

全部都属于：

```text
Total Parameters
```

这些参数需要被存储。

---

## 02.9.2 Activated Parameters

一个 token 真正执行 Forward 时，只会访问其中一部分 Expert。

例如：

```text
Total Experts = 256
Top-K = 8
```

则：

```text
一个 token
→ Routed Experts 中只激活 8 个
```

因此：

> **MoE 可以拥有巨大的 Total Parameters，同时让单 token 的 Activated Parameters 明显更小。**

DeepSeek-V3 就是一个典型例子：技术报告给出的总参数约为 671B，但每个 token 激活约 37B 参数。

所以比较 Dense 与 MoE 时，不能简单说：

```text
671B 模型一定比 70B 模型每 token 多算约 10 倍
```

因为对于 Sparse MoE：

```text
Total Parameters
≠
Activated Parameters
```

---

# 02.10 Router、Top-K、Shared Expert 与 Routed Expert

一个更完整的 MoE Forward 可以写成：

```text
Hidden State
     │
     ▼
   Router
     │
     ▼
Routing Scores
     │
     ▼
   Top-K
     │
     ▼
 ┌───┼─────────┐
 ▼   ▼         ▼
E3   E17       E91
 │    │         │
 └────┼─────────┘
      ▼
Weighted Combine
      │
      ▼
Hidden State
```

这里至少有四个需要理解的概念。

### Routed Expert

由 Router 动态选择的 Expert。

不同 token 可以选择不同 Expert。

### Top-K

决定每个 token 激活多少个 Routed Expert。

例如：

```text
num_experts_per_tok = 8
```

通常意味着：

```text
Top-K = 8
```

DeepSeek-V3 的 Hugging Face 配置公开了：

```text
n_routed_experts = 256
num_experts_per_tok = 8
```

即 256 个 Routed Expert 中，每个 token 路由至 8 个。

### Shared Expert

Shared Expert 不依赖 Router 的 Top-K 选择，而是承担更通用的 FFN 表达。

因此模型可能同时存在：

```text
Shared Expert
+
Selected Routed Experts
```

DeepSeek-V3 的公开配置包含：

```text
n_shared_experts = 1
n_routed_experts = 256
```

### Load Balance

如果所有 token 都倾向于几个 Expert：

```text
Expert 3   ███████████████████
Expert 4   █████████████████
Expert 5   ██
Expert 6   █
```

就会造成负载不均。

在单设备语义里，这意味着 Expert workload 不平均；在 Expert Parallel 多设备部署中，它甚至会进一步形成：

```text
某些 Rank 很忙
某些 Rank 空闲
```

因此 Router 不只是模型算法问题，也直接影响系统利用率。

---

# 02.11 为什么 MoE 会改变 Communication

这一点是 Dense → MoE 对推理基础设施最重要的改变之一。

Dense FFN 如果被切到 Tensor Parallel，可以出现 TP collective；但 MoE 在采用 Expert Parallel 时，还会产生一种新的数据运动需求。

假设：

```text
EP = 4

Rank 0: Expert 0–63
Rank 1: Expert 64–127
Rank 2: Expert 128–191
Rank 3: Expert 192–255
```

某个 token 当前在 Rank 0，但 Router 选择：

```text
Expert 7
Expert 92
Expert 171
Expert 230
```

那么 token representation 就必须被送往这些 Expert 所在设备：

```text
                Router
                  │
                  ▼
          token representation
        ┌─────────┼─────────┐
        ▼         ▼         ▼
      Rank1     Rank2      Rank3
```

Expert 执行结束以后，再把结果送回来组合：

```text
Expert outputs
      │
      ▼
   Combine
      │
      ▼
Hidden State
```

于是模型语义中的：

```text
Router
→ Dispatch
→ Experts
→ Combine
```

到了分布式部署里，就会自然对应为：

```text
Routing
→ Communication
→ Expert Compute
→ Communication
```

这就是 **Architecture → Communication** 最直接的例子。

需要注意：仅仅看到模型是 MoE，并不能断言一定发生跨卡 Dispatch；如果所有 Expert 都能放在一个设备，通信自然不存在。

更准确的因果关系是：

```text
MoE Architecture
        +
Expert Parallel Deployment
        ↓
Cross-device Dispatch / Combine
```

因此 Architecture 决定了**这种通信模式存在的可能性和语义需求**，具体是否发生以及发生多少，则由部署策略决定。

---

# 02.12 模型结构如何分别决定 Compute、Memory、Communication

到这里，可以把本章核心关系收敛成一张表。

| Architecture Feature | Compute                    | Memory                 | Communication             |
| -------------------- | -------------------------- | ---------------------- | ------------------------- |
| Layers ↑             | Forward 计算近似线性增加           | Weight / KV Cache 增加   | PP 时可能增加 Stage 边界         |
| hidden_size ↑        | Linear / Attention 计算增加    | Weight / Activation 增加 | TP tensor size 增大         |
| intermediate_size ↑  | FFN MatMul 增加              | FFN Weight 增加          | TP FFN communication 可能增加 |
| Context Length ↑     | Attention 工作量增加            | KV Cache 增加            | CP 等策略下影响通信               |
| KV Heads ↓           | Attention部分计算/表示改变         | KV Cache 减小            | KV 分片需求变化                 |
| MHA → GQA/MQA        | Q 基本保留，KV projection 减少    | KV Cache 显著降低          | Attention 并行特征改变          |
| MLA                  | Attention 数据路径改变           | KV latent 大幅压缩         | Attention 并行实现改变          |
| Dense → MoE          | 每 token 只激活部分 FFN          | Total Weight 很大        | EP 下产生 Dispatch/Combine   |
| Top-K ↑              | Activated Expert Compute ↑ | 临时 Expert workload ↑   | EP traffic 通常增加           |
| Expert Count ↑       | 不代表单 token 计算同比增加          | Total Weight ↑         | EP placement 空间增加         |

这里有一个非常重要的认识：

> **Architecture 通常决定“需要什么资源”，Parallel Strategy 决定“这些资源怎样跨设备展开”。**

因此本 Knowledge Map 后面的 Distributed 章节才能自然接上这一章：

```text
02 Model Architecture
模型想算什么、保存什么
          │
          ▼
03 Distributed Inference
这些计算和数据如何放到多设备
          │
          ▼
04 Serving
多个请求如何被调度执行
          │
          ▼
05 Cluster
大量实例如何使用整个集群
```

---

# 02.13 从 `config.json` 能提前读出什么

拿到一个陌生模型，第一步不应该立刻 Profile。

先读 config。

建议建立下面这套扫描顺序。

## 第一组：模型规模

```text
num_hidden_layers
hidden_size
intermediate_size
vocab_size
```

这些字段帮助判断：

```text
模型深度
Hidden State 宽度
FFN 宽度
LM Head 大小
```

进而粗略判断 Weight 与 Compute 分布。

---

## 第二组：Attention

```text
num_attention_heads
num_key_value_heads
head_dim

max_position_embeddings

sliding_window
layer_types
```

如果：

```text
num_attention_heads = 32
num_key_value_heads = 32
```

可以初步判断 MHA。

如果：

```text
num_attention_heads = 32
num_key_value_heads = 8
```

则是 GQA。

如果：

```text
num_key_value_heads = 1
```

则是 MQA。Hugging Face 的 Qwen2 配置就是按照这个语义定义 `num_key_value_heads`。

如果又出现：

```text
kv_lora_rank
q_lora_rank
qk_rope_head_dim
qk_nope_head_dim
```

则意味着已经不是传统 MHA/GQA 的简单 K/V representation，需要进一步检查 MLA 或其他低秩 Attention 实现。DeepSeek-V3 就属于这种情况。

---

# 02.14 第三组：FFN / MoE

Dense 模型重点看：

```text
hidden_size
intermediate_size
hidden_act
```

MoE 则继续寻找：

```text
num_experts
n_routed_experts

num_experts_per_tok
top_k

n_shared_experts

moe_intermediate_size

first_k_dense_replace
moe_layer_freq
```

例如看到：

```text
n_routed_experts = 256
num_experts_per_tok = 8
n_shared_experts = 1
first_k_dense_replace = 3
```

已经可以推断：

```text
存在 MoE
│
├── Routed Experts = 256
├── Top-K = 8
├── Shared Expert = 1
└── 前若干层仍为 Dense
```

DeepSeek-V3 的公开配置正包含这些字段。

此时即使没有 profiling，也应该马上产生三个推理假设：

```text
1. Total Parameters 很大，
   但单 token Activated Parameters 明显更小。

2. 如果使用 Expert Parallel，
   Router 后会出现 Dispatch / Combine 通信。

3. Expert load balance
   会成为性能分析的重要变量。
```

这些假设之后再由运行数据确认。

---

# 02.15 `config.json` 仍然不等于完整模型结构

不过要特别注意：

> **config 适合做 architecture fingerprint，但不能完全替代 `model.py`。**

例如 config 告诉你：

```text
hidden_act = silu
intermediate_size = ...
```

但要确定真正的 MLP 数据流：

```text
gate_proj
up_proj
SiLU
Mul
down_proj
```

仍然需要查看源码。

同样，看到：

```text
n_routed_experts = 256
```

只能证明存在 Expert 配置。

要知道具体执行：

```text
Router
→ Group Selection
→ Top-K
→ Dispatch
→ Expert
→ Combine
```

需要进一步沿 `forward()` 和 Module 调用链解析。

因此一个完整的静态分析应该是：

```text
config.json
    │
    │ 参数化事实
    ▼
model.py
    │
    │ 模块 + forward 数据流
    ▼
Semantic Architecture
```

这也是前面讨论的 model architecture skill 应该承担的核心职责。

---

# 02.16 config、model.py 与 profiling 的职责边界

可以最终明确成三个不同的问题。

### Config

回答：

> **模型被配置成什么规格？**

例如：

```text
61 Layers
256 Routed Experts
Top-K = 8
KV latent rank = 512
```

### model.py

回答：

> **这些规格怎样组成真实计算流程？**

例如：

```text
Decoder
├── RMSNorm
├── MLA
├── Residual
├── RMSNorm
├── Router
├── Shared Expert
├── Routed Experts
└── Residual
```

### Profiling

回答：

> **这次具体部署运行得怎么样？**

例如：

```text
MLA       0.83 ms
Router    0.04 ms
Dispatch  0.31 ms
Expert    1.22 ms
Combine   0.28 ms
```

因此不要把三者写成：

```text
源码
 ↓
Kernel
 ↓
验证模型架构
```

更准确的是：

```text
config + model.py
        │
        ▼
Static Semantic Architecture
        │
        ├───────────────┐
        │               │
        ▼               ▼
Architecture      Runtime Profiling
Knowledge              │
                       ▼
                Kernel / Communication
                       │
                       ▼
                 映射回模型语义
```

**Kernel 不负责证明 Attention、MLP、MoE 是否存在；Kernel 用来说明这些模型语义在某次具体硬件和软件栈上是怎样被执行的。**

---

# 02.17 用 DeepSeek-V3 做一次完整的“读 Config 推理”

假设第一次看到 DeepSeek-V3 的配置：

```text
hidden_size = 7168
num_hidden_layers = 61

num_attention_heads = 128

kv_lora_rank = 512
q_lora_rank = 1536
qk_rope_head_dim = 64
qk_nope_head_dim = 128
v_head_dim = 128

n_routed_experts = 256
n_shared_experts = 1
num_experts_per_tok = 8
moe_intermediate_size = 2048

first_k_dense_replace = 3
```

这些字段在 Hugging Face 当前 DeepSeek-V3 配置中均有对应定义。

即使完全没有 Profile，现在也可以开始形成推理画像。

### Architecture

```text
61 层 Decoder

前 3 层
→ Dense FFN

之后大量 Layer
→ MoE

Attention
→ MLA family

FFN
→ Shared Expert + Routed Experts
```

### Compute

```text
Hidden width = 7168
→ projection MatMul 较大

MoE Top-K = 8
→ 每 token 不执行 256 个 Expert
→ 只执行选中的 Routed Expert
   + Shared Expert
```

### Memory

```text
Total Expert Parameters 很大

但 MLA：
KV 使用低秩 latent representation

→ KV Cache 不应按传统
  128-head MHA 直接估算
```

### Communication

仅从 Architecture 不能断言通信方式，但已经知道：

```text
如果采用 EP
→ Routed Expert placement 跨设备
→ 必须解决 Dispatch / Combine

如果采用 TP
→ Attention / Expert Linear
   还可能产生 Tensor Parallel 通信
```

### Performance Hypothesis

在没有任何 Kernel trace 的情况下，已经可以提出：

```text
Prefill：
较大的 MatMul workload，
Compute 利用率非常重要。

Decode：
KV memory traffic、Expert routing、
小 batch GEMM 和通信更值得关注。

MoE 大规模部署：
Expert load balance 与 EP communication
可能成为重要瓶颈。
```

接下来 Profile 的作用，就是判断：

> **这些架构推断中，究竟哪一个在当前硬件、并行策略、Batch 和 Sequence Length 下成为了真正瓶颈。**

---

# 02.18 本章最终应该形成的分析方法

以后拿到任何公开模型，不要先问：

> “它有多少 B 参数？”

而应该按下面这条路径理解：

```text
① Model Scale
Layers / Hidden / FFN
        │
        ▼
每层大概有多少 Weight 和 Compute

② Attention Architecture
MHA / MQA / GQA / MLA / Local Attention
        │
        ▼
KV Cache 保存什么、保存多少

③ FFN Architecture
Dense / MoE
        │
        ▼
Total Parameters 和 Activated Parameters

④ Routing
Experts / Top-K / Shared Experts
        │
        ▼
一个 token 实际经过哪些计算

⑤ Context
Max Length / Window
        │
        ▼
Attention 和 KV Cache 如何随长度增长

⑥ Deployment Implication
TP / EP / PP / CP 等
        │
        ▼
架构产生的数据流如何跨设备展开
```

最后得到的应该不是简单的“模型介绍”，而是一张 **Inference Architecture Fingerprint**：

```text
Model
│
├── Compute
│   ├── Layer × 61
│   ├── Hidden = 7168
│   ├── MLA Projection
│   └── Sparse MoE Top-8
│
├── Memory
│   ├── Large Expert Weights
│   ├── Activated Parameters << Total Parameters
│   └── Compressed KV Latent
│
└── Communication Potential
    ├── TP → projection communication
    ├── EP → expert dispatch/combine
    ├── PP → hidden-state stage transfer
    └── CP → sequence/attention communication
```

这就是这一章真正需要建立的能力：

> **在看到 Profiling、Kernel Timeline 和集群拓扑之前，先从模型结构推导出“为什么它可能这样运行”；随后再进入 Distributed、Serving 和 Observability 章节，研究“它实际是怎样被部署和执行的”。**

换句话说，Model Architecture 是整个推理 Knowledge Map 的第一层约束。后面的并行策略、Kernel 优化、KV Cache 管理、通信优化、Serving 调度乃至 Capacity Planning，都必须建立在“这个模型本身要求执行什么计算、持有什么状态、产生什么数据流”的基础之上。
