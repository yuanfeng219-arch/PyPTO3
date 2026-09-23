# PTO 调度与执行：开发者作业辅助内容规划

> 文档版本：v1.0  
> 更新时间：2026-08-17  
> 适用范围：PyPTO / PTO 算子 Coding、编译、调试与性能调优  
> 产品形态：嵌入代码编辑器的上下文型 Agent 与右侧可视化面板

---

## 1. 文档目标

本文从开发者实际作业过程出发，梳理 PTO 调度与执行知识中适合产品化呈现的内容，回答以下问题：

1. 开发者编写算子代码时，哪些调度与执行信息能够帮助其正确 Coding？
2. 编译期间，哪些中间结构和变换结果值得呈现？
3. 运行异常时，怎样解释任务为什么没有执行或为什么执行错误？
4. 性能调优时，怎样把调度、硬件和耗时关联起来？
5. 如何在右侧面板中聚合这些内容，避免拆分出过多页签？

产品设计的核心原则是：

> Coding 时解释“代码将如何执行”，调试时解释“为什么这样执行”，调优时解释“时间为什么花在这里”。

PTO 的全部编译器和 Runtime 知识不应被原样平铺给开发者。Agent 应围绕当前文件、当前 `@pl.jit` 入口、当前源码位置、当前选中任务和当前 Run，有选择地组织内容。

---

## 2. 开发者需要持续回答的核心问题

调度与执行相关内容应优先回答以下问题：

```text
我正在开发哪个执行入口？
代码会生成哪些任务？
任务为什么按这个顺序执行？
哪些任务可以并行？
数据经过哪些内存和硬件？
当前任务为什么没有运行？
实际执行和静态推断是否一致？
性能时间花在哪里？
```

对应到产品中的核心对象包括：

| 对象 | 开发者需要理解的内容 |
|---|---|
| JIT Entry | 当前执行入口、多个入口之间的独立关系、入口的特化版本 |
| Function | Orchestration、InCore、Inline、Opaque、AIC、AIV、Group、SPMD 等层级 |
| Task | Call/Submit、TaskId、状态、目标核、输入输出、前后继 |
| Tensor | Shape、DType、方向、Region、Producer、Consumer、生命周期 |
| Dependency | RAW、WAR、WAW、自动依赖、显式 TaskId、Barrier、Fan-in |
| Scope | AUTO、MANUAL、自动跟踪边界、依赖治理规则 |
| Runtime | Ready Queue、Task Ring、Dependency Pool、Dispatch、Resolve |
| Hardware | AIC、AIV、Group、SPMD Block、内存层级、TPUSH/TPOP |
| Pipeline | CopyIn、Compute、CopyOut、Stage、多缓冲、跨核流水 |
| Evidence | 源码事实、静态推断、编译事实、运行实测、性能采样 |

---

## 3. 信息可信度与证据分层

调度可视化必须区分“预计怎样执行”和“实际怎样执行”，避免把静态推断包装成运行事实。

| 证据级别 | 来源 | 适合支持的结论 |
|---|---|---|
| 源码事实 | 当前 Python 源码及函数装饰器 | JIT 入口、函数类型、调用关系、显式 deps、Scope |
| 静态推断 | Shape/DType、参数方向、Region 和硬件规则 | 预计任务图、潜在 RAW/WAR/WAW、预计硬件路径 |
| 编译事实 | IR、Pass Dump、Codegen 产物 | 实际任务、实际依赖、Scope 物化、内存推断、生成代码 |
| Runtime 实测 | Task Trace、依赖图、状态、时间戳 | Ready/Blocked/Running/Complete、实际派发、实际等待 |
| 硬件实测 | PMU、Timeline、Tensor Dump、Profiler | 核利用率、搬运耗时、流水气泡、性能瓶颈、精度误差 |

界面中的结论应带有简洁的证据标记，例如：

```text
预计依赖 · 源码静态分析
确认依赖 · AutoDeriveTaskDependencies 后 IR
实际等待 · Runtime Trace
```

---

## 4. 作业阶段总览

| 作业阶段 | 开发者主要问题 | 首要呈现内容 | 推荐可视化 |
|---|---|---|---|
| 算子 Coding | 当前代码表达了什么，将如何执行？ | 入口、函数层级、任务图、数据依赖、硬件映射 | 多入口拓扑、任务计算图、数据流、硬件路径 |
| 编译与 Lowering | 源码被转换成了什么，为什么编译失败？ | 关键 Pass、任务生成、依赖推导、Codegen 映射 | Pass 前后 Diff、源码—IR—产物链路 |
| 调度调试 | 任务为什么没有运行，为什么结果错误？ | Task 状态、未满足依赖、Tensor 生命周期、跨核状态 | 动态任务图、阻塞链、SPMD/Group 展开图 |
| 性能调优 | 时间花在哪里，什么限制了并发？ | Timeline、关键路径、并发度、核利用率、流水效率 | Swimlane、关键路径、核内流水、Runtime 容量 |

---

## 5. Coding 阶段：解释“代码将如何执行”

### 5.1 JIT 入口与函数层级

Coding Agent 首先应识别并区分：

| 源码声明 | 调度含义 |
|---|---|
| `@pl.jit` | 芯片级 Orchestration 入口 |
| `@pl.jit.host` | Host Orchestration 入口 |
| `@pl.jit.incore` | 独立 InCore Kernel |
| `@pl.jit.inline` | 在调用点展开的辅助函数 |
| `@pl.jit.opaque` | 独立 IR 函数 |
| `@pl.jit.extern` | 外部 Kernel |
| `FunctionType.AIC/AIV/Group` | 具体硬件核或混合核函数 |

推荐呈现为多入口拓扑：

```text
当前文件
├─ qwen3_decode                 @pl.jit
│  ├─ input_rmsnorm             inline
│  ├─ attention                 inline
│  │  └─ paged_attention        incore
│  └─ mlp                       inline
└─ warmup_decode                @pl.jit
   └─ ...
```

当文件中存在多个普通 `@pl.jit` 时，必须明确提示：

- 它们是多个独立 Orchestration 入口，不按源码顺序自动连续执行；
- 当前计算图以哪个入口为根；
- 每个入口依赖哪些 InCore、Inline、Opaque 或 Extern 子函数；
- 哪些子函数被多个入口共享；
- 不同顶层入口之间不会自动建立 Tensor 依赖；
- 每个入口拥有独立的 JIT 特化与编译缓存。

普通 `@pl.jit` 入口不会自动发现另一个普通 `@pl.jit` 入口。若需要在更高层组织多个芯片级入口，应由 `@pl.jit.host` 承担调度；若希望多个阶段进入同一个任务图，应使用一个顶层 `@pl.jit`，并将阶段实现为 InCore、Inline、Opaque 或 Extern 子函数。

### 5.2 任务图预览

Agent 应从当前 Orchestration 源码推导可能生成的 Task：

```text
RMSNorm
   ↓ RAW(hidden_norm)
QKV Projection
   ├─→ RoPE
   └─→ KV Cache Update
            ↓
      Paged Attention
            ↓
       Output Projection
```

任务节点应按需呈现：

- 任务名称与对应源码函数；
- FunctionType；
- Call 或 Submit；
- 输入、输出 Tensor；
- 是否生成 TaskId；
- 预计执行硬件；
- 是否为 SPMD、Group 或普通 Kernel；
- 可并行的兄弟任务；
- 所属 AUTO/MANUAL Scope；
- 编译期可确认程度。

任务图不应只表达数学关系，还应表达执行含义。例如：

```text
Paged Attention
AIV · SPMD × 20 Blocks
Input: query, key_cache, value_cache
Output: attention_out
Dependency: KV Cache Update
```

### 5.3 数据流与依赖叠加

任务图上应支持叠加以下信息：

- Tensor 名称、Shape、DType；
- `Input`、`Out`、`InOut` 方向；
- Producer 和 Consumer；
- Slice/Region 与重叠关系；
- RAW、WAR、WAW；
- 读—读无依赖；
- 自动依赖或显式 TaskId；
- 编译期确定或 Runtime 判断。

示例：

```text
KV Cache Update
      │
      │ key_cache
      │ RAW · Runtime OverlapMap
      ▼
Paged Attention
```

连线选中后应解释依赖原因：

```text
依赖类型：RAW
原因：KV Cache Update 写入 key_cache，
      Paged Attention 随后读取相同 Region。
来源：Runtime 自动依赖
```

默认只显示 Tensor 名称和依赖类型；Shape、DType、Region 和证据在节点选中或缩放后展开，避免计算图过载。

### 5.4 AUTO 与 MANUAL Scope

源码和任务图应同时标记依赖治理边界：

```text
AUTO Scope
├─ task_a
├─ task_b
└─ task_c

MANUAL Scope
├─ task_d deps=[task_b]
└─ task_e deps=[barrier]
```

需要呈现：

- 当前任务属于哪个 Scope；
- Scope 是 AUTO 还是 MANUAL；
- AUTO 中最终依赖为自动依赖与显式依赖的并集；
- MANUAL 中只使用显式 TaskId；
- `no_dep`、`manual_dep` 等局部退出机制；
- Scope 的进入和退出位置；
- 是否存在未覆盖依赖或过度串行风险。

即时风险示例：

```text
潜在缺失依赖

当前任务位于 MANUAL Scope，读取了 stage_a 写入的 temp，
但 deps 中未发现 stage_a_tid。
```

### 5.5 硬件执行映射

Coding 阶段应提前呈现任务的预计硬件位置：

```text
Orchestration      AICPU
MatMul             AIC / Cube
RMSNorm            AIV / Vector
Mixed Attention    Group：1C2V
SPMD Kernel        N Blocks
```

计算图节点可叠加：

```text
QK MatMul
AIC · L1 → L0A/L0B → L0C

Softmax
AIV · GM → UB → GM
```

选中节点后再展开：

- 核类型；
- `core_num` 与 Block 数；
- `sync_start`；
- `allow_early_resolve`；
- 是否属于 Group；
- Group 内 AIC/AIV 的生产消费关系；
- 是否使用 TPUSH/TPOP；
- 当前结论是静态映射还是设备实测。

### 5.6 核内 Tile 流水

选中 InCore/AIC/AIV 节点后，应支持下钻到 Tile 级数据路径：

```text
GM
 ↓ CopyIn
L1 / UB
 ↓ Move
L0A / L0B
 ↓ Compute
L0C
 ↓ CopyOut
GM
```

当存在 `pl.pipeline(stage=F)` 时，可视化不同迭代的重叠：

```text
时间 →
迭代 0  CopyIn ─ Compute ─ CopyOut
迭代 1           CopyIn ─ Compute ─ CopyOut
迭代 2                    CopyIn ─ Compute ─ CopyOut
```

适合展示：

- Tile Shape 和 MemorySpace；
- CopyIn、Compute、CopyOut；
- Pipeline Stage；
- Buffer 数与 Ping-Pong；
- `range`、`parallel`、`unroll`、`pipeline` 的循环语义；
- 搬运与计算能否重叠；
- 潜在生命周期和容量冲突。

---

## 6. 编译阶段：解释“源码被转换成了什么”

### 6.1 源码到产物的对应关系

默认展示关键链路，而不是全部中间 IR：

```text
Python 源码
  ↓
Orchestration IR
  ↓
Call / Submit / RuntimeScope
  ↓
AICPU Orchestration C++
  ↓
Simpler Runtime Task
```

选中一个任务后，可显示：

```text
源码：qwen3_decode.py:128
IR：Submit(paged_attention)
Codegen：rt_submit_aiv_task(...)
产物：orchestration/qwen3_decode.cpp
```

### 6.2 关键 Pass 变化

适合开发者理解的关键 Pass 包括：

- `OutlineHierarchy`；
- `OutlineIncoreScopes`；
- `ConvertTensorToTile`；
- `InferMemory`；
- `ExpandMixedKernel`；
- `SkewCrossCorePipeline`；
- `LowerPipelineLoops`；
- `DeriveCallDirections`；
- `AutoDeriveTaskDependencies`；
- `ExpandManualPhaseFence`；
- `MaterializeRuntimeScopes`。

界面应展示 Pass 带来的结构变化，而不是仅列出 Pass 名称：

```text
AutoDeriveTaskDependencies

变换前：
task_b deps=[]

变换后：
task_b deps=[task_a]

原因：
task_a 写 temp，task_b 读取 temp，形成 RAW。
```

### 6.3 JIT 特化与缓存

当开发者修改 Shape、DType 或编译配置时，展示当前使用的编译版本：

```text
JIT Entry：paged_attention
Specialization：
  query       [4, 32, 128] FP16
  block_size  128
  platform    Ascend 910B
  strategy    Default

Cache：Hit
```

以下变化可能触发重新编译：

- 当前入口或其依赖源码变化；
- 静态 Shape 变化；
- DType 变化；
- 编译期标量变化；
- Platform 变化；
- 优化策略变化；
- 内存规划变化；
- 动态维声明和相关编译选项变化。

多个 `@pl.jit` 入口的缓存状态必须分别呈现。

---

## 7. 调试阶段：解释“为什么这样执行”

### 7.1 动态任务状态

运行后，静态任务图应切换为动态状态图：

```text
Resolved → Ready → Running → Complete
              ↑
           Blocked
```

每个任务节点应显示：

- TaskId；
- Runtime 状态；
- 未完成依赖数量；
- Ready、Dispatch、Start、Complete 时间；
- 实际执行核；
- SPMD Block 信息；
- 运行异常或超时标记。

选择 Blocked 节点时，应直接解释：

```text
Paged Attention 尚未运行

未满足依赖：2
├─ task_17：KV Cache Update，仍在 Running
└─ task_19：RoPE，尚未 Ready
```

### 7.2 最终依赖 Fan-in

动态任务图应区分：

- Tensor 自动依赖；
- 显式 TaskId；
- 编译和调用关系；
- TPUSH/TPOP 跨核关系。

选中任务时给出最终依赖构成：

```text
最终依赖 = 自动依赖 ∪ 显式依赖

自动依赖：
- task_3，RAW(query)
- task_5，WAW(cache)

显式依赖：
- task_7，deps=[barrier_tid]
```

MANUAL Scope 中应明确提示：

```text
自动依赖：关闭
最终依赖：仅显式 TaskId
```

### 7.3 Tensor 生命周期与精度

Tensor 级调试视图应呈现：

```text
query
Create → Write → Read by RoPE → Read by Attention → Release
```

重点信息包括：

- 首次定义位置；
- 写入任务和消费任务；
- Shape/DType 演变；
- Slice/Region；
- 所属内存层级；
- Dump 是否启用；
- 实际值与参考值的误差；
- 首个出现数值偏差的任务。

精度结果可直接叠加到任务图节点：

```text
RMSNorm
max_abs_error = 2.1e-4
cosine = 0.99998
状态：通过
```

### 7.4 SPMD 与跨核调试

SPMD 节点应支持展开：

```text
SPMD Grid TaskId
├─ Block 0  Complete
├─ Block 1  Complete
├─ Block 2  Running
└─ Block 3  Blocked
```

需要呈现：

- `core_num`；
- Block 到数据分片的映射；
- `block_idx`；
- `sync_start`；
- Grid TaskId；
- `allow_early_resolve`；
- 最慢 Block；
- Block 间负载差异。

Group/Cluster 节点展开后可呈现：

```text
Group Task
├─ AIC Producer
│    └─ TPUSH queue 0
├─ AIV Consumer 0
└─ AIV Consumer 1
```

应补充 FIFO 深度、生产消费进度和等待对象。

---

## 8. 性能调优阶段：解释“时间花在哪里”

### 8.1 调度时间线

使用 Runtime Swimlane 呈现不同资源上的实际执行：

```text
时间 →
AICPU   Submit A ─ Submit B ─ Submit C
AIC     [ MatMul A      ] [ MatMul C ]
AIV-0       [ RMSNorm ]     [ Softmax ]
AIV-1       [ RoPE    ]     [ Vector  ]
```

时间线应区分：

- Submit；
- 依赖等待；
- Ready Queue 等待；
- Runtime 派发；
- 硬件执行；
- 同步等待；
- TPUSH/TPOP 等待；
- CopyIn、Compute、CopyOut。

### 8.2 关键路径

在任务图上高亮决定总时延的链路：

```text
RMSNorm → QKV → RoPE → Attention → Projection
```

同时显示：

- 关键路径总时长；
- 关键节点耗时；
- 非关键并行分支；
- 可并发但被依赖限制的任务；
- 关键路径上的主要等待来源。

建议必须指向证据和原因：

```text
Attention 不是计算瓶颈。

其 41% 时间处于 Ready Queue 等待，
原因是前置 KV Cache Update 形成 WAW 依赖。
```

### 8.3 并发度与核利用率

适合展示：

- 理论可运行任务数与实际运行任务数；
- 平均和峰值并发度；
- AIC/AIV 活跃核数；
- AIC/AIV 负载不均；
- SPMD Block 利用率；
- Group 内 Cube/Vector 等待比例；
- 保守依赖造成的串行化。

示例：

```text
理论可并发任务：6
实际峰值并发：3

限制因素：
- 2 个任务被保守 Tensor 重叠判断串行化
- 1 个任务等待 TPUSH 数据
```

### 8.4 核内流水效率

针对 InCore 节点提供：

```text
CopyIn          28%
Compute         46%
CopyOut         18%
Pipeline Bubble  8%
```

分析内容包括：

- 搬运是否覆盖计算；
- Stage 数是否合适；
- Buffer 是否足够；
- Tile 是否过大或过小；
- UB/L1/L0 占用；
- Pipeline Prologue/Epilogue 比例；
- 生命周期是否阻碍内存复用。

### 8.5 Runtime 容量与调度压力

高级性能诊断中可提供：

- Task Ring 使用率；
- 最大在途任务数；
- Dependency Pool 使用率；
- Ready Queue 长度；
- AICPU 提交速度；
- Runtime 消费速度；
- Dummy Barrier 数量；
- Fan-in/Fan-out 规模。

示例：

```text
Task Ring 峰值：126 / 128

风险：
编排层仍有 37 个可提交任务，Ring Window 接近上限，
可能限制并发暴露。
```

这类信息不应默认占据主视图，应作为性能页中的高级诊断按需展开。

---

## 9. 右侧面板信息架构

为避免页签过多，建议聚合为三个核心页签。

### 9.1 执行图

默认页签，承载 Coding 阶段最常用的内容：

- 多 JIT 入口选择；
- 函数层级；
- 任务计算图；
- Tensor 数据流；
- RAW/WAR/WAW；
- AUTO/MANUAL Scope；
- Shape、DType；
- 硬件映射；
- 节点展开后的核内流水。

通过图层开关叠加信息，而不是拆成更多页签：

```text
[数据] [依赖] [硬件] [精度] [运行状态]
```

### 9.2 运行与调试

承载动态证据：

- TaskId；
- Ready/Running/Blocked/Complete；
- 未满足依赖；
- Tensor Dump；
- 精度误差；
- SPMD Block；
- TPUSH/TPOP；
- 源码—IR—Runtime 对应关系；
- 错误与异常定位。

### 9.3 性能

承载调优信息：

- 调度 Swimlane；
- 关键路径；
- 并发度；
- 核利用率；
- 等待原因；
- 核内流水；
- 内存层级流量；
- Ring/Dependency Pool；
- 优化建议与验证结果。

---

## 10. 核心交互原则

### 10.1 统一选择上下文

同一时刻应维护统一的当前对象：

```text
当前 JIT Entry
  → 当前 Function
    → 当前 Task
      → 当前 Tensor / Dependency
```

源码、任务图、硬件图、Timeline 和详情面板均围绕该对象联动。

### 10.2 双向定位

- 点击源码函数，高亮对应任务图节点；
- 点击任务节点，定位源码定义和调用位置；
- 点击依赖边，定位 Producer、Consumer 和 Tensor；
- 点击 Timeline 片段，定位 Task 和源码；
- 点击 Tensor，显示完整 Producer/Consumer 路径。

### 10.3 节点逐级展开

```text
Orchestration Entry
  ↓ 展开
Runtime Task Graph
  ↓ 展开
SPMD Blocks / Group Members
  ↓ 展开
InCore Tile Pipeline
```

展开后不应重复保留一个含义相同的上层节点，以免任务数量被重复计算。

### 10.4 默认降噪

- 默认显示入口、关键任务和关键依赖；
- Shape、DType、硬件、精度等采用叠加图层；
- TaskId、Region、IR 属性在选中后展示；
- 完整 Pass、代码生成和 Runtime 配置按需下钻；
- 静态预计与运行实测使用明确标识。

---

## 11. 内容优先级

### P0：Coding 默认必须提供

- 当前 `@pl.jit` 入口；
- 多入口关系与当前根入口；
- 函数层级；
- 任务计算图；
- Tensor Producer/Consumer；
- RAW/WAR/WAW；
- AUTO/MANUAL Scope；
- Shape、DType；
- 硬件映射；
- 源码节点定位；
- 静态推断与编译事实标识。

### P1：编译或运行后提供

- 实际 TaskId 和任务状态；
- Blocked 原因；
- 实际依赖 Fan-in；
- 动态执行时间线；
- SPMD Block 状态；
- Tensor 生命周期；
- 精度叠加；
- 关键路径；
- 核利用率；
- JIT Cache 命中状态。

### P2：按需展开

- 完整 Pass 前后差异；
- 生成的 AICPU C++；
- Runtime API 参数；
- Task Ring 和 Dependency Pool；
- 跨核 FIFO 细节；
- 内存规划结果；
- 完整 JIT Cache Key；
- 全量 IR；
- 高开销 PMU 和细粒度 Trace。

---

## 12. Agent 输出建议

Agent 的结论应采用统一结构：

```text
结论
当前任务被 task_17 阻塞。

原因
task_17 写入 key_cache，当前任务读取相同 Region，形成 RAW。

证据
Runtime Trace + TensorMap/OverlapMap。

影响
Paged Attention 的启动延迟 42 μs，位于关键路径。

建议
确认两个任务是否确实访问同一 Region；若 Region 独立，完善切片信息以减少保守依赖。

验证
重新编译并比较依赖图、关键路径和端到端耗时。
```

Agent 不应仅给出“优化 Scope”“增加并行度”一类泛化建议，而应说明：

- 作用对象；
- 判断证据；
- 预期收益；
- 正确性风险；
- 验证方法。

---

## 13. 产品验收标准

### 13.1 Coding 阶段

- 能识别一个文件中的所有 `@pl.jit` 及其变体；
- 能明确多个普通 `@pl.jit` 是独立入口；
- 能以当前入口为根生成函数与任务拓扑；
- 能显示 Tensor 方向、Shape、DType 和 Producer/Consumer；
- 能解释主要依赖的 RAW/WAR/WAW 原因；
- 能区分 AUTO/MANUAL Scope；
- 能将任务映射到 AICPU、AIC、AIV、Group 或 SPMD。

### 13.2 编译阶段

- 能关联源码、IR Task、Codegen 调用和编译产物；
- 能显示关键 Pass 对任务、依赖、Scope 和硬件结构的改变；
- 能显示当前 JIT 特化和缓存状态；
- 能标记静态推断与编译确认结果的差异。

### 13.3 调试阶段

- 能显示 Task 实际状态和未满足依赖；
- 能从 Blocked Task 追踪到根因任务；
- 能关联 Task、Tensor、源码和运行证据；
- 能展开 SPMD Block 和 Group 内部状态；
- 能把精度误差定位到最早异常节点。

### 13.4 性能阶段

- 能生成跨 AICPU/AIC/AIV 的调度 Timeline；
- 能识别关键路径和主要等待来源；
- 能区分依赖等待、Runtime 排队和硬件执行；
- 能显示核内 CopyIn/Compute/CopyOut 流水；
- 能用可复验数据支撑优化建议。

---

## 14. 总结

PTO 调度与执行知识适合以“当前入口和当前任务”为中心嵌入开发过程，而不是作为独立知识库展示。

最有价值的产品闭环是：

```text
源码入口
  → 静态任务图
    → 编译确认任务图
      → Runtime 动态状态
        → 性能与精度证据
          → Agent 结论和可验证建议
```

右侧面板应聚合为“执行图、运行与调试、性能”三个核心页签；数据、依赖、硬件、精度和状态通过图层叠加，避免信息被拆散。这样既能在 Coding 阶段建立正确的执行心智模型，也能在调试和调优阶段复用同一套对象与证据链。
