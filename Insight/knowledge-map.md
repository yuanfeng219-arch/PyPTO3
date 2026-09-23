# 大模型推理 Knowledge Map

## 0. Knowledge Map 的组织原则

这套知识库的目标不是整理一份“大模型推理名词百科”，而是建立一个能够解释完整推理系统的知识结构：从一个模型为什么具有某种推理特征开始，沿着模型适配、Runtime、Operator、Kernel、分布式执行一路进入在线 Serving 和生产集群；与此同时，用 Correctness、Performance、Observability、Benchmark、Reliability、Capacity 等横向能力去观察这条执行链；最后再通过真实模型和硬件平台 Case，把抽象知识映射到 NVIDIA、Ascend 等实际工程体系中。

因此，整个知识体系可以概括成：

```text
                    大模型推理 Knowledge Map
                              │
             ┌────────────────┼────────────────┐
             │                │                │
        技术执行主干       横向工程能力       应用与案例
             │                │                │
        Model → Device     Correctness      Roles / Tasks
             ↓            Performance       Tools
        Distributed       Observability     Case Studies
             ↓            Benchmark
         Serving          Reliability
             ↓            Capacity / Cost
         Cluster
```

知识库中的技术事实原则上优先使用模型官方资料、NVIDIA TensorRT-LLM / Triton、Ascend MindIE / CANN / HCCL 等官方文档以及原始论文作为依据。例如 TensorRT-LLM 当前官方架构已经明确包含 Model Engine、Scheduler、KV Cache Manager 等组件，并进一步提供 Paged KV Cache、Chunked Prefill 和 Disaggregated Serving；MindIE 2.3 则明确将模型能力、TP/DP/EP/CP/SP 等并行能力与 Continuous Batching、PageAttention 等 Serving 能力分开描述。这些公开架构也验证了本 Knowledge Map 将“模型执行”和“Serving”拆成不同层级的合理性。

---

# Part I｜技术执行主干

## 01. Scenario & Core Objects：先定义“我们到底在研究哪一次推理”

推理知识的入口不应该直接是 Transformer 或 Kernel，而应该首先建立 Scenario，因为任何性能数字都依赖上下文。同一个模型在不同输入长度、输出长度、并发、硬件、精度和 Serving 配置下，会表现出完全不同的计算、内存和通信特征。

一个完整的推理 Scenario 至少可以表达为：

```text
Inference Scenario
=
Model
× Workload
× Hardware
× Configuration
```

其中 Model 描述模型版本、架构、参数规模、精度和 Context Length；Workload 描述 Chat、RAG、Summarization、Agent 等业务类型，以及 ISL、OSL、Request Rate、Concurrency 等负载特征；Hardware 描述 GPU/NPU 型号、显存、服务器和互联拓扑；Configuration 描述 TP、EP、DP、Batch、KV Cache、P/D 等运行参数。

在这个基础上建立统一对象模型，包括 Model、Request、Session、Run、Endpoint、Instance、Rank、Batch、Token、KV Cache、Metric、Trace、Profile 和 Experiment。以后知识库中的任何性能结论，都应该能够回到这些对象上，而不是留下一个没有上下文的“tokens/s”。

---

## 02. Model Architecture & Inference Semantics：模型结构为什么会改变推理行为

这一部分研究的重点不是完整复述 Transformer，而是理解：

> **Model Architecture 如何决定 Compute、Memory 和 Communication。**

首先需要建立标准 Decoder-only Transformer 的推理结构，理解 Embedding、Attention、MLP、Residual、RMSNorm 和 LM Head 在一次 Forward 中分别处理什么数据；随后进入 Prefill 和 Decode，理解为什么自回归生成需要逐 token 运行，以及 KV Cache 为什么能够避免重复计算历史 token 的 K/V。

在 Attention 方向，需要建立 MHA → MQA → GQA → MLA 的演进关系，重点观察不同结构如何改变 KV representation、KV Cache footprint、Memory Bandwidth 和 Attention Kernel，而不能只停留在“有多少 Attention Heads”。

在 FFN 方向，需要建立 Dense → MoE 的演进关系。Dense 模型的 token 会执行完整 FFN，而 MoE 会经过 Router，只选择部分 Expert，因此需要同时理解 Total Parameters、Activated Parameters、Top-K Routing、Shared Expert、Routed Expert、Dispatch / Combine 和 Load Balance。

这一层最终应该形成一种能力：拿到一个公开模型的 `config.json`，就能开始判断它大致具有怎样的推理特征，而不需要等到 Profile 以后才知道发生了什么。

---

## 03. Inference Adaptation：训练模型如何成为可部署的推理模型

训练产物并不能天然等价于某个平台上已经优化好的推理程序，中间还存在一整层 Inference Adaptation。

典型链路可以表示为：

```text
Checkpoint / Config / Tokenizer
             ↓
        Model Loading
             ↓
Format / Weight Conversion
             ↓
Precision / Quantization
             ↓
Layout / Sharding
             ↓
Unsupported Op Adaptation
             ↓
Graph Transformation / Fusion
             ↓
Parallel Planning
             ↓
Executable Model / Engine
```

这一部分需要理解模型格式、权重转换、FP16/BF16/FP8/INT8/INT4 等精度形式、Quantization、Tensor Layout、Graph Rewrite、Operator Support 和硬件适配，同时区分“模型源码可以加载”“计算图可以编译”“模型可以正确执行”“模型精度达到要求”几个不同状态。

这一层也解释了为什么同一份开源模型，在 Transformers、TensorRT-LLM、vLLM、MindIE 等 Runtime 中最终看到的执行图和 Kernel 并不会完全相同。

---

## 04. Engine / Compiler / Runtime：谁把模型计算真正组织起来

这一层重点理解 Framework、Compiler、Inference Engine 和 Runtime 的职责边界。

可以建立如下抽象：

```text
Model Definition
      ↓
Framework
      ↓
Graph / IR
      ↓
Compiler / Graph Optimizer
      ↓
Inference Engine
      ↓
Runtime Executor
      ↓
Device
```

这里需要理解 Graph Capture、Operator Fusion、Kernel Selection、Memory Planning、Execution Plan，以及 Runtime 如何组织每一次 Forward。进一步进入 TensorRT-LLM、vLLM、MindIE 时，再观察不同系统如何实现这些抽象，而不是把某个框架自己的术语误认为通用推理原理。

NVIDIA 当前 TensorRT-LLM 的 PyExecutor 就明确拆成 Model Engine、Decoder、Scheduler、Resource Manager / KVCacheManager 等组件，其中 Scheduler 决定当前 step 哪些 Request 可以获得资源并执行 Model Forward，这说明 Engine 执行与在线请求调度实际上已经开始在 Runtime 层交汇。

---

## 05. Operator / Kernel / Hardware：一次模型计算最终怎样在设备上执行

这一部分沿着模型语义继续向底层拆解：

```text
Model Module
    ↓
Graph Operator
    ↓
Fused Operator
    ↓
Kernel
    ↓
Instruction / Memory Access
    ↓
GPU / NPU Execution Unit
```

这里需要系统理解 Linear / MatMul、Attention、RMSNorm、Softmax、SiLU、MoE Grouped GEMM 等典型 Operator，以及一个 Framework Operator 为什么可能对应一个或多个 Kernel；同时理解 Fusion 为什么会让静态模型图里的 Operator 与 Profile 中看到的 Kernel 不再一一对应。

硬件层继续进入 Compute-bound / Memory-bound、Roofline、HBM、片上 Cache/SRAM、Ascend UB/L1、Tiling、Vectorization、Tensor Core / AI Core、Kernel Launch、Memory Access 和 Compute-Memory Overlap。

这一章最终要回答的问题不是“这个 Kernel 叫什么”，而是：

> **这段模型语义经过什么计算和数据搬运，最终为什么在当前硬件上快或慢。**

---

## 06. Distributed Model Execution：一个 Request 如何跨多个计算设备执行

当一个 Model Instance 由多个 GPU/NPU 协同执行时，就进入 Distributed Model Execution。

这一层首先需要严格区分各种 Parallelism 的计算语义，包括 Tensor Parallelism、Expert Parallelism、Data Parallelism、Context Parallelism、Sequence Parallelism，以及具体 Runtime 支持时的 Pipeline Parallelism。不同平台支持范围不能混为一谈，例如 MindIE 2.3 当前官方能力表支持 TP、DP、EP、CP 和 SP，而 PP 标记为不支持。

随后建立一条非常重要的关联链：

```text
Parallel Strategy
        ↓
Tensor / Expert Placement
        ↓
Communication Semantics
        ↓
Collective / P2P
        ↓
Communication Group
        ↓
Rank Mapping
        ↓
Physical Topology
```

TP 需要理解为什么 Tensor Sharding 会产生 AllReduce、AllGather、ReduceScatter 等通信；EP 需要理解 Router、Token Dispatch、All-to-All、Expert Computation、Combine 和 Expert Load Imbalance；DP 则主要研究不同 Request / Batch 如何分布到不同计算设备。MindIE 官方对 DP 的定义就是将 inference requests 划分到不同计算设备并行处理，并允许与 TP 组合，例如 `tp × dp = worldSize`。

这一章还必须系统解释 Rank、World Size、Parallel Group、Node、Server、SuperNode，以及 NVLink/NVSwitch、HCCS/RoCE 等物理通信域。Ascend HCCL 当前已经为 Atlas A3 推理产品明确列出 inter-SuperNode Collective Communication 支持，因此 SuperNode 应理解为重要的物理通信域，而不能简单理解成推理绝对不能跨越的边界。

---

## 07. Serving Runtime：从“执行模型”进入“调度请求”

进入这一层以后，问题从“模型怎么算”转变成：

> **大量 Request 到达之后，当前这个 step 应该计算哪些 Request。**

核心对象包括 Request Queue、Scheduler、Continuous Batching、Batch Formation、Paged KV Cache、Prefix Cache、Chunked Prefill、Preemption、Speculative Decoding 和 Sampling。

可以把 Serving Runtime 抽象为：

```text
Request Queue
      ↓
   Scheduler
      ↓
Continuous Batch
      ↓
Model Executor
      ↕
KV Cache Manager
      ↓
   Decoder
```

KV Cache 在这一层已经不只是一个 Tensor，而成为需要动态分配、复用、回收甚至跨实例传输的系统资源。TensorRT-LLM 当前官方 KV Cache System 使用 Block Pool，根据 Request 动态分配 KV Blocks，并支持跨请求 reuse、offloading 和 prioritized eviction，这正是为什么 KV Cache 应该同时出现在“推理原理”和“Serving Runtime”两个层级。

---

## 08. Production / Distributed Serving：从一个 Model Instance 到在线推理系统

这一层讨论完整生产系统如何组织多个 Model Instance，并持续满足请求吞吐、延迟和稳定性目标。

典型关系可以抽象为：

```text
                  Request Stream
                        ↓
                Gateway / Router
                        ↓
                    Scheduler
               ┌────────┼────────┐
               ↓        ↓        ↓
           Instance A Instance B Instance C
               ↓        ↓        ↓
        Distributed Model Execution
```

这里需要区分 **Scale-up** 与 **Scale-out**：前者扩大一个 Model Instance 的模型并行域，例如增加 TP/EP 规模；后者增加 Instance / Replica，让更多独立请求能够并行处理。

Prefill / Decode Disaggregation 是这一层的重要专题。Prefill 计算 Prompt 并产生 KV Cache，Decode 利用已有 KV Cache 逐 token 生成；由于二者计算特征不同，可以被部署到不同计算资源池，并在两者之间传输 KV Cache。TensorRT-LLM 当前官方明确区分 Aggregated Serving 与 Disaggregated Serving，并允许 Context 与 Generation 使用不同 GPU 池和不同 Parallelism Configuration，从而分别优化 TTFT 与 TPOT。

这一章还需要覆盖 Request Routing、Load Balancing、Replica、Autoscaling、Admission Control、Multi-node Serving、Failure Isolation 和 Multi-tenant Serving。

---

## 09. Cluster / Topology / Infrastructure：生产推理最终运行在哪里

最后进入物理基础设施，把前面的逻辑结构映射到实际集群。

这一层需要理解 Device → Server → Node → SuperNode / NVLink Domain → Cluster 的层级，以及 PCIe、NVLink、NVSwitch、HCCS、RoCE、RDMA、Ethernet 等不同通信链路的性能和适用范围。

重点不是背硬件参数，而是理解：

```text
Logical Parallel Group
        ↓
Rank Placement
        ↓
Physical Topology
        ↓
Communication Path
        ↓
Bandwidth / Latency / Contention
        ↓
Inference Performance
```

到这一层以后，才可以正确讨论“TP 是否跨服务器”“EP 是否跨 SuperNode”“增加并行规模还是增加 Replica”这类生产部署问题。

---

# Part II｜横向工程能力

技术执行主干描述“系统是怎样工作的”，但真正的推理工程还需要六条横向能力。这些能力都跨越多个技术层，因此不应该被塞进某一个执行章节。

## A. Correctness & Quality

这一轴建立：

```text
Load Success
≠ Compile Success
≠ Runtime Success
≠ Numerical Correctness
≠ Model Quality Acceptable
```

需要覆盖 Reference Output、Tensor Comparison、Tolerance、NaN/Inf、Quantization Error、Layer-wise Error、End-to-End Evaluation 和 Model Quality Regression。最终回答的是：一次优化虽然更快，但结果是否仍然正确、模型能力是否仍然满足业务要求。

## B. Performance & Trade-off

这一轴建立统一性能语言，包括 TTFT、TPOT/ITL、E2E Latency、Request Throughput、Token Throughput、Goodput、Concurrency、MFU/Compute Utilization、Memory Bandwidth、Communication Ratio 和 P50/P95/P99 等指标。

更重要的是理解这些指标之间存在 Trade-off，例如提高 Batch 可能增加 Throughput，却恶化单请求延迟；扩大 EP 可能增加 Expert 并行能力，却引入更大的通信域；P/D 分离能够分别优化 Prefill 和 Decode，却产生 KV Transfer 成本。

## C. Observability & Evidence

这一轴区分 Monitoring、Logging、Tracing、Profiling 和 Hardware Counter，并建立从低开销长期观察到高开销深度诊断的证据层级。

最终形成：

```text
Request
  ↓
Serving Trace
  ↓
Instance / Batch
  ↓
Rank
  ↓
Communication / Compute
  ↓
Operator
  ↓
Kernel
  ↓
Hardware Counter
```

一个成熟的诊断系统应该允许从异常 Request 一路下钻到真正的执行根因，而不是把 Serving Dashboard 和 Kernel Profiler 做成两个完全割裂的工具。

## D. Benchmark & Experiment

这一轴负责回答“一个性能结论是否可信”。任何对比都需要控制 Model、Precision、Engine、Hardware、ISL、OSL、Batch/Concurrency、Request Rate、Sampling、Warm-up、Duration 等实验变量，并区分 Microbenchmark、Offline Benchmark、Controlled Serving Benchmark 和 Production Observation。

Performance Optimization 本质上是一种实验过程：

```text
Hypothesis
    ↓
Controlled Experiment
    ↓
Measurement
    ↓
Evidence
    ↓
Root Cause
    ↓
Optimization
    ↓
Regression Validation
```

## E. Reliability & Operations

这一轴覆盖 Health Check、Failure Detection、Request Retry、Replica Failover、Node Failure、Network Failure、Graceful Degradation、Rolling Upgrade、Capacity Headroom、SLO/SLA 和 Incident Diagnosis。

生产推理的目标不是单次执行成功，而是在设备、网络和流量不断变化的情况下持续提供可预测服务。

## F. Capacity & Cost

这一轴必须拆成两个不同问题。

第一部分是 **Model Instance Sizing**：

```text
Instance Memory
=
Weights
+ KV Cache
+ Runtime Workspace
+ Temporary Tensor
+ Communication Buffer
+ Runtime Overhead
```

这里研究参数量、Precision、Context Length、Batch、TP/EP 等变量怎样决定一个 Model Instance 的资源需求。

第二部分是 **Serving Capacity Planning**：

```text
Traffic
× Workload
× SLA
        ↓
Required Throughput
        ↓
Instance Performance
        ↓
Replica / Resource Pool
        ↓
Cluster Capacity
        ↓
Cost
```

因此，“模型实例需要多少设备”和“生产业务需要多少计算资源”必须始终作为两个独立问题。

---

# Part III｜产品与工程应用层

## 10. Roles / Tasks / Decisions

同一套推理系统，不同用户面对的问题不同，因此知识库应该继续映射到角色和任务。

模型开发者主要关心模型适配、精度和算子支持；推理性能工程师更关注 Operator、Kernel、Memory、Parallelism 和 Communication；Serving 工程师主要关注 Scheduler、KV Cache、Routing、P/D 和 Capacity；运维人员关注集群健康、SLA 和故障；UX/Product 则需要理解每种角色在什么问题出现时，需要做什么判断、查看什么证据以及执行什么操作。

因此产品设计最终关注的不是“如何展示更多指标”，而是：

> **用户正在做什么决策，当前缺少哪一层证据，下一步应该下钻到哪里。**

## 11. Tool Ecosystem

工具生态应该按照解决的问题分类，而不是简单罗列工具名称。

例如 NVIDIA TensorRT-LLM、vLLM、SGLang、Ascend MindIE 属于不同形态的 Inference Runtime / Serving Engine；NCCL/HCCL 属于 Communication Library；Nsight Systems、Nsight Compute、Ascend Profiling 等属于不同层级的性能分析工具；Dynamo 等则进一步处理分布式 Serving 和资源编排。

每个工具都应该映射回 Knowledge Map 中的层级，这样才能判断两个工具究竟是竞争关系、上下游关系，还是解决完全不同的问题。

---

# Part IV｜Case Study：用真实模型贯穿整套 Knowledge Map

知识库不应该为每个模型重复建立一套理论，而应该选择少量具有代表性的模型，让同一个 Case 在不同章节反复出现，逐渐增加复杂度。

## Case A｜Qwen3-32B：Dense 基础主 Case

Qwen3-32B 作为整套知识库的 **Dense Baseline**。官方模型卡显示其为 32.8B 参数、64 层，采用 GQA，Q Heads 为 64、KV Heads 为 8，因此结构足够典型，同时规模又适合用于解释实际推理部署。

它负责贯穿：

```text
Decoder-only Transformer
→ GQA
→ Prefill / Decode
→ KV Cache
→ Dense MLP
→ Operator / Kernel
→ Quantization
→ TP
→ Serving Runtime
→ Benchmark
→ Capacity
```

所有基础概念原则上先在 Qwen3-32B 上解释清楚，再进入大型 MoE。

---

## Case B｜DeepSeek V3.2：大型 MoE 工程主 Case

DeepSeek V3.2 负责承载 **大型 MoE + 复杂分布式推理**。官方配置可以直接观察到 61 个 Layer、256 个 Routed Experts、1 个 Shared Expert、每 token 激活 8 个 Routed Experts，同时包含 MLA 相关的低秩 KV 表示参数，因此能够把模型架构、内存优化、MoE Routing 和分布式通信同时串起来。

它负责贯穿：

```text
MLA
→ MoE
→ Router / Top-K
→ Routed / Shared Expert
→ Active vs Total Parameters
→ EP
→ Token Dispatch / Combine
→ All-to-All
→ Expert Load Balance
→ Communication-Compute Overlap
→ Multi-device / Multi-node
→ Production Serving
```

因此 Qwen3-32B 与 DeepSeek V3.2 的关系不是简单的“两种模型介绍”，而是从基础 Dense Inference 逐渐进入大型 MoE Distributed Inference。

---

## Case C｜Qwen3-32B vs Qwen3-235B-A22B：Dense → MoE 控制变量专题

如果要专门回答“MoE 相比 Dense 到底改变了什么”，不应该直接比较 Qwen 与 DeepSeek，因为两者除了 FFN 结构不同，Attention 等模型设计也不同。

因此增加一个控制变量 Case：

```text
Qwen3-32B
Dense + GQA

        VS

Qwen3-235B-A22B
MoE + GQA
```

Qwen 官方给出的 Qwen3-235B-A22B 配置为 235B Total、22B Activated、94 层、128 Experts、每 token 激活 8 Experts，并且仍然采用 GQA。

这一组 Case 专门研究：

```text
Dense MLP
   ↓
改为 MoE
   ↓
Router
Expert Placement
Active Parameters
EP
Dispatch / Combine
Load Balance
Communication
Memory Distribution
```

这样能够更干净地观察 **Dense FFN → MoE FFN** 对推理系统带来的变化。

---

# 最终 Knowledge Map

把所有部分压缩后，完整结构为：

```text
LLM Inference Knowledge Base

Ⅰ. 技术执行主干
│
├─ 01 Scenario & Core Objects
├─ 02 Model Architecture & Inference Semantics
├─ 03 Inference Adaptation
├─ 04 Engine / Compiler / Runtime
├─ 05 Operator / Kernel / Hardware
├─ 06 Distributed Model Execution
├─ 07 Serving Runtime
├─ 08 Production / Distributed Serving
└─ 09 Cluster / Topology / Infrastructure


Ⅱ. 横向工程能力
│
├─ A Correctness & Quality
├─ B Performance & Trade-off
├─ C Observability & Evidence
├─ D Benchmark & Experiment
├─ E Reliability & Operations
└─ F Capacity & Cost


Ⅲ. 产品与工程应用
│
├─ 10 Roles / Tasks / Decisions
└─ 11 Tool Ecosystem


Ⅳ. Case Study
│
├─ Case A：Qwen3-32B
│          Dense 基础主 Case
│
├─ Case B：DeepSeek V3.2
│          大型 MoE 工程主 Case
│
└─ Case C：Qwen3-32B vs Qwen3-235B-A22B
           Dense → MoE 控制变量 Case
```

这套结构的关键价值在于，它同时保留了两种观察推理的方法：纵向可以沿着 **Model → Runtime → Kernel → Distributed Execution → Serving → Cluster** 追踪一次推理到底怎样执行；横向则可以从 **Correctness、Performance、Observability、Benchmark、Reliability、Capacity** 任一工程问题切入，跨越多个执行层寻找证据。

最终整个知识库应该能够回答三类不同问题：**“系统是怎么工作的”“系统为什么出现这个问题”“为了做出工程或产品决策，我应该观察什么证据”**。Case Study 则负责把这些抽象知识持续落到真实模型和真实硬件平台上，避免知识库最终退化成一套彼此孤立的术语解释。
