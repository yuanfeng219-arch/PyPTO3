# PTO3 源码、调度与硬件执行概念术语手册

> **目的**：建立从 PyPTO 源码到编译产物、运行时调度和设备实测的共同语言。本文特别区分“用户写下的静态结构”与“一次运行中实际出现的动态对象”。
>
> **适用范围**：PyPTO / PTO 的算子开发、编译诊断、运行时调试和性能调优。本文不把产品页面中的“适用范围（scope）”或一般 Python 词法作用域混入执行模型。

---

## 1. 一句话总览

PTO3 的执行对象沿三条线演进：

```text
用户源码线：       scope / 函数 / Tile 操作 / SPMD 意图
                       ↓ 编译、外提、Lowering
系统与调度线：     Function / kernel / Submit / Task / 依赖 / logical block
                       ↓ 运行时派发
真实设备执行线：   AIC/AIV core group / block 实例 / 片上内存 / trace event
```

这三条线的关键关系是：

```text
源码中的一个 InCore scope
  → 编译后成为一个函数
  → 成为一个 kernel，或混合时成为 AIC + AIV 两个 kernel
  → 每次运行被提交为一个或多个 task
  → SPMD task 被展开为 N 个 logical block
  → 每个 block 在某一时刻被派发到可用的物理 core / core group
  → trace 记录该次实际执行
```

它们不是普遍的一一对应关系。尤其是：**scope 不等于 kernel，kernel 不等于 task，task 不等于 block，block 也不固定等于某一个物理核。**

---

## 2. 三条线：谁创建对象，谁解释对象

| 线 | 主要创建者 | 主要对象 | 它回答的问题 | 证据性质 |
|---|---|---|---|---|
| 用户源码线 | 算子开发者 | `pl.at`、`pl.spmd`、`pl.scope`、函数、Tile 操作、显式 `deps` | “我希望如何切分计算、在哪一层运行、数据如何流动？” | 静态事实 / 意图 |
| 系统与调度线 | Parser、编译 Pass、Runtime、Scheduler | IR Function、AIC/AIV/Group kernel、`Submit`、TaskId、依赖、ready queue、logical block | “代码被编成什么，哪些工作已可运行，调度器准备派发什么？” | 编译事实 + 运行时状态 |
| 真实设备执行线 | 设备硬件与运行时派发结果 | AIC/AIV core group、block 实例、UB/L1/L0、实际时间戳、Worker/Scheduler trace | “最终在哪个资源上跑、何时开始结束、为何空转或等待？” | 实测事实 |

同一对象跨线时要保留**血缘**，而不要假设同名即同一实例。例如 `qk` 是源码 `name_hint`，而 `task(r0t17)` 是某次运行产生的 TaskId；前者稳定、后者随每次运行变化。

---

## 3. 核心关系图（文本版）

```text
源码
────
@pl.function / @pl.jit 入口
  └─ with pl.spmd(48)                         ← 指定 48 个逻辑分片
       └─ with pl.at(CORE_GROUP, name_hint="qk")
            └─ tile.load / tile.move / tile.matmul

编译与调度
──────────
SpmdScope(core_num=48)  ──外提──→ Function(Spmd)
InCoreScope("qk")      ──外提──→ Function(InCore, "qk")
                                          └─纯 Cube→ AIC kernel
                                          └─纯 Vector→ AIV kernel
                                          └─混合    → AIC kernel + AIV kernel + Group
Function(Spmd) / kernel ──一次调用──→ TaskId
TaskId(core_num=48)     ──fan-out──→ block[0] ... block[47]

真实设备
────────
block[i] ──等待依赖和资源──→ scheduler ──派发──→ 可用 AIC/AIV core group
                                                    └─实际执行 event
Worker trace：block 在哪条核泳道、何时执行、持续多久
Scheduler trace：ready / dispatch / finish 等调度阶段
```

---

## 4. 概念术语

### 4.1 用户源码线

| 术语 | 源码形态 | 含义 | 不是 |
|---|---|---|---|
| **入口（Entry）** | `@pl.jit`、`@pl.function(...)` | 一次编译、调用或编排的起点；决定从哪里建立调用与依赖图 | 自动连续执行的整个文件 |
| **InCore scope** | `with pl.at(level=pl.Level.CORE_GROUP, ...)` | 用户声明的 AICore 计算区域；范围内的 Tile 操作作为同一个设备侧子图处理 | 内存分配区域、一次 task |
| **`name_hint`** | `name_hint="qk"` | 用户给 scope 的可读名称；用于外提函数命名、诊断和源码—运行证据关联 | 永远唯一的运行时 ID |
| **SPMD scope** | `with pl.spmd(N, sync_start=...)` | 声明该区域以 `N` 个逻辑 block 执行 | 生成 `N` 份不同的源代码或 `N` 个不同 kernel |
| **Runtime scope** | `with pl.scope()`、`with pl.manual_scope()` | 编排侧的自动/手工依赖追踪及 ring 资源管理边界 | InCore 计算 scope 或 Tile MemorySpace |
| **Tile 操作** | `tile.load`、`tile.move`、`tile.matmul`、`tile.store` | kernel 内部的数据搬运、计算与写回语义 | 一个独立 task 的保证 |
| **显式依赖** | `deps=[task_id]` | 开发者指定后继任务必须等待的 producer TaskId | 自动依赖的替代品；在 AUTO scope 中二者可叠加 |

#### 如何理解一个源码 InCore scope

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="qk"):
    q_11 = pl.load(q, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
    k_11 = pl.load(k, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
    qa = pl.move(q_11, target_memory=pl.Mem.Left)
    kb = pl.move(k_11, target_memory=pl.Mem.Right)
    s = pl.matmul(qa, kb, out_dtype=pl.FP32)
```

这段代码是在说：“请把这一段命名为 `qk`，并作为 Core Group 上的计算子图编译。”它同时定义：

1. **边界**：哪些 Tile 操作属于 `qk`；
2. **执行位置意图**：这是设备侧的 Core Group 计算，而不是 Host 编排；
3. **编译候选单元**：后续会外提为 InCore Function；
4. **源码身份**：`qk` 可被诊断、编译产物和性能分析引用。

`target_memory=Mat/Left/Right/Acc` 描述的是 Tile 的**物理内存空间**，与 `qk` 这个 scope 是两件事。

### 4.2 编译与调度线

| 术语 | 创建阶段 | 含义 | 关键关系 |
|---|---|---|---|
| **IR scope** | Parser | 源码 scope 在 IR 中的结构节点，例如 `InCoreScopeStmt`、`SpmdScopeStmt`、`RuntimeScopeStmt` | 保存源码意图，尚不是设备执行实例 |
| **Function** | Scope 外提后 | 被命名、带参数和返回值的可调用 IR 单元 | InCore scope 可外提为 `Function(InCore)`；SPMD scope 可外提为 `Function(Spmd)` |
| **kernel** | 后端编译后 | 可在设备侧执行的代码单元 | 纯 InCore 通常变成一个 AIC 或 AIV kernel；mixed InCore 可拆为两个 kernel |
| **Group** | 混合核展开后 | 协调 AIC 与 AIV kernel 的函数/调度组 | 一个逻辑计算区域可能对应 AIC + AIV 两个 kernel |
| **Call** | IR / 编译期 | 普通函数调用语义 | 不一定产生可观察的独立 TaskId |
| **Submit** | 编排 IR / Runtime | 一次可调度的任务提交，可携带 `deps`、`core_num`、`sync_start` | Runtime 将其建立为 task 并纳入依赖图 |
| **task** | 每次运行 | 调度器管理的一次工作单元，有动态 TaskId 和依赖状态 | 同一 kernel 在不同运行、循环迭代或调用点会形成不同 task |
| **TaskId** | 每次运行 | 某次 task 的唯一运行时标识 | 不是 `name_hint`，不应跨 Run 对比其数值 |
| **依赖（fanin）** | 编译/Runtime | task 可开始前必须完成的前驱集合 | 可来自 AUTO 跟踪、显式 `deps` 或其他运行时机制 |
| **ready queue** | Runtime | 依赖满足、但尚未取得设备执行资源的 task 队列 | 不等于正在设备核上运行 |
| **logical block** | SPMD task 展开后 | 同一 SPMD kernel 的一份逻辑执行实例，携带 `block_idx` 与 `block_num` | 固定绑定的物理核 |

### 4.3 真实设备执行线

| 术语 | 含义 | 与上层对象的关系 |
|---|---|---|
| **AIC / Cube** | 执行矩阵类 Cube 指令的核心类型 | AIC kernel 的实际执行资源 |
| **AIV / Vector** | 执行 Vector 指令的核心类型 | AIV kernel 的实际执行资源 |
| **core group** | 同一计算组中的物理计算资源组合 | `CORE_GROUP` 是源码/IR 的执行层级意图；实际哪个 group 执行由调度器决定 |
| **block 实例** | 某 logical block 在某次设备派发中实际执行的实例 | 会占用一个合适的 core 或 core group 一段时间 |
| **wave（波次）** | 因 block 数超过可用资源而分批执行的一组 block | `spmd(48)` 可能一波完成，也可能分多波 |
| **MemorySpace** | Tile 在 DDR、Vec/UB、Mat/L1、Left/L0A、Right/L0B、Acc/L0C 等位置的放置 | 由 Tile 数据流与后端约束推导，不由 task 或 scope 名称决定 |
| **MemRef** | Tile 的 buffer 元数据：地址、大小、标识及共享关系 | 在 memory inference、reuse 和地址分配后确定 |
| **Worker trace event** | block 在某条 AIC/AIV 泳道上的实际执行记录 | 是 block 的设备侧证据 |
| **Scheduler trace event** | submit、ready、dispatch、finish 等控制侧记录 | 用于分辨依赖等待、调度开销和核上执行 |
| **device wall** | 从设备执行窗口角度观察的总时长 | 看重叠后的时间包络，不是全部 block 时长相加 |
| **core-time** | 所有 block 的核上持续时间之和 | 衡量资源消耗；可大于 device wall，因为 block 可以重叠 |

---

## 5. 最常用的基数关系

| 上游对象 | 下游对象 | 常见基数 | 为什么不是固定 1:1 |
|---|---|---|---|
| 源码 InCore scope | 外提 Function | 通常 1:1 | 同名冲突会追加后缀；编译器也可能调整边界 |
| Function(InCore) | device kernel | 1:1 或 1:2 | 纯 Cube / Vector 走单 kernel；mixed 会拆为 AIC + AIV |
| kernel | task | 1:N | 每次 Run、循环迭代、不同调用点均可能产生新 task |
| task | logical block | 1:1 或 1:N | 普通 task 常为单 block；SPMD task 展开为 `core_num=N` 个 block |
| logical block | physical core/core group | N:M（按时间） | block 不永久占某个核；同一核会在不同时间执行多个 block |
| block | Worker trace event | 通常 1:1 | mixed 或采集视图差异可能形成多条相关事件 |
| task | Scheduler trace event | 通常 1:N | 一个 task 可能经过多个阶段、等待或重复调度观察 |

---

## 6. 以 `pl.spmd(48)` 和 `qk` 为例

### 6.1 源码表达的内容

```text
pl.spmd(48)     ：这段计算按 48 个逻辑分片执行
scope "qk"      ：其中的 QK 计算构成一个设备侧 InCore 区域
tile 操作       ：每个分片的 load → move → matmul 数据路径
```

`48` 是逻辑执行规模。每个 block 应通过 `tile.get_block_idx()` 或由其推导出的偏移，访问不同的数据切片。若有效计算路径完全不使用 block 身份，48 个 block 可能处理同一数据，通常不符合并行切分的预期。

### 6.2 编译与运行时如何承接

```text
pl.spmd(48)
  → SpmdScope(core_num=48)
  → Function(Spmd)
  → 一次 runtime dispatch / task
  → task 的 48 个 block：block_idx = 0 ... 47，block_num = 48

scope "qk"
  → Function(InCore, "qk")
  → 本例含 matmul，通常归为 AIC/Cube kernel
```

如果 `qk` 同时含有不可归为单一核心类型的 Cube 与 Vector 工作，它会被展开为 AIC kernel、AIV kernel 和协调二者的 Group；此时一个逻辑 block 可能对应两个硬件侧 kernel half。

### 6.3 设备上实际发生什么

```text
block 0  ─┐
block 1  ─┼─ 依赖满足后进入 ready 状态
...      ┤
block 47 ─┘
             ↓
调度器按当时可用的 AIC/AIV core group 分配
             ↓
资源足够：一波并发；资源不足或被占用：分波 / 排队 / 穿插执行
             ↓
每个实际 block 在 Worker trace 留下时间区间和核泳道
```

因此，`pl.spmd(48)` **不保证 48 个 block 同时开始**。只有 `sync_start=True` 才提出原子化同步启动要求；即便如此，是否可满足仍受硬件资源、核心类型和运行时约束限制。

---

## 7. Scope 的多种含义，必须分开说

| 说法 | 正确含义 | 典型写法或位置 | 主要作用 |
|---|---|---|---|
| **InCore scope** | 设备侧计算区域 | `pl.at(level=CORE_GROUP, name_hint="qk")` | 划定编译子图与执行层级 |
| **SPMD scope** | 数据并行启动区域 | `pl.spmd(N)` | 指定 logical block 数和启动语义 |
| **Runtime scope** | 依赖追踪与 ring 管理区域 | `pl.scope()` / `pl.manual_scope()` | AUTO/MANUAL 依赖治理及资源回收边界 |
| **CommDomain scope** | 分布式通信 window buffer 区域 | 编译器物化的 `CommDomainScopeStmt` | 管理 HCCL 通信域及对应 buffer |
| **词法/SSA scope** | 变量可见性范围 | 函数、循环、分支、`with` | 保证变量定义、引用和 yield 合法 |
| **产品适用 scope** | 一条结论适用的条件 | 芯片、dtype、shape、并发等 | 防止把局部结论泛化 |

除非有明确限定，性能页面中的“scope 排行”应写为“**按 callable / 源 scope 名称聚合的统计范围**”，不能让读者误解为 MemorySpace 或 RuntimeScope。

---

## 8. 如何从任意对象跨线定位

| 你手里有什么 | 向上追溯到源码 | 向下确认运行与硬件 | 需要的证据 |
|---|---|---|---|
| `name_hint="qk"` | 找 `pl.at` / `pl.spmd` 的源码行与调用链 | 查外提 Function、kernel 名称和 callable 映射 | 源码、IR、name map |
| kernel / callable 名称 | 找外提 scope 的 `name_hint` 或编译器生成名 | 查其 TaskId、block 数、AIC/AIV 类型 | 生成代码、trace、name map |
| TaskId | 查 task 的提交点、callee、输入输出与前驱 | 展开 block，查看 ready/dispatch/finish | deps、Scheduler trace、Worker trace |
| block event | 通过 TaskId / FuncId 找到 task 和 callable | 看所占核心、持续时间、相邻等待 | Worker trace、Scheduler trace |
| MemorySpace / MemRef | 找产生和消费该 Tile 的源码操作 | 看推导结果、复用、地址和硬件搬运 | Pass dump、PTOAS/生成代码 |

推荐始终带上以下四元组，避免跨层串错对象：

```text
Run ID + TaskId + FuncId/callable + block_idx
```

其中，源码定位还应补充文件与行号；跨 Run 对比则应使用 stable 的源码/编译产物身份，而不是比较 TaskId 数字。

---

## 9. 性能指标该归属到哪一层

| 指标或现象 | 首先归属的线 | 正确解释 |
|---|---|---|
| `core_num=48` | 用户源码 / 编译 | 逻辑 fan-out 配置，不是已实现的并发度 |
| `block_num=48` | Runtime | 某次 SPMD task 实际携带的逻辑 block 数 |
| AIC/AIV core 利用率 | 真实设备 | 某段时间内硬件实际忙碌比例 |
| ready queue 非空 | Runtime | 任务依赖已满足但尚未被派发；不能单独证明核在忙或调度器是根因 |
| task 的 `aicpu-duration` | Runtime | 提交、领取、依赖处理到完成的控制面观察，不等于纯核上计算 |
| Worker block `duration` | 真实设备 | 某 block 的设备执行持续时间；需确认是否包含 local setup |
| `core-time` | 真实设备聚合 | 多 block 时长之和，表达资源使用量 |
| `device wall` | 真实设备聚合 | 端到端设备窗口；表达用户感知的时间，不可用 core-time 直接替代 |
| Tile 的 `Mat/Left/Right/Acc` | 编译到设备 | 内存空间和数据路径，不是 task 状态或 scope 类型 |

---

## 10. 常见误解与改写方式

| 容易说错的话 | 应改为 |
|---|---|
| “这个 scope 就是一个 kernel。” | “这个 InCore scope 通常会编译成一个 kernel；mixed scope 可能拆成 AIC 与 AIV 两个 kernel。” |
| “`spmd(48)` 就是同时启动 48 个核。” | “`spmd(48)` 产生 48 个 logical block；实际是否同波执行取决于可用资源和调度。” |
| “一个 kernel 只有一个 task。” | “一次 kernel 调用通常产生一个 task；同一 kernel 在不同 Run 或调用点可产生多个 task。” |
| “block 就是物理核。” | “block 是逻辑工作分片；调度器在运行时把它映射到物理核或 core group。” |
| “scope 控制 Tile 放到 L0 或 UB。” | “Tile 的 MemorySpace 和算子约束决定其放置；scope 主要描述计算区域或运行时依赖边界。” |
| “ready 代表正在运行。” | “ready 代表依赖已满足、可被派发；它仍可能等待调度资源。” |
| “core-time 就是总延迟。” | “core-time 是所有核上时长之和；总延迟应看 device wall 或相应的关键路径。” |
| “同名就可以在 trace 中精确回溯源码。” | “应同时核对 Run、TaskId、FuncId/callable、name map 和源码 span；同名可能冲突或带后缀。” |

---

## 11. 排查时的最小提问顺序

1. **这是什么线上的对象？** 源码声明、编译/Runtime 对象，还是设备实测？
2. **它的稳定身份和动态身份分别是什么？** 例如 `name_hint="qk"` 与 `TaskId`。
3. **它是一对一还是一对多关系？** 特别检查 mixed kernel 与 SPMD fan-out。
4. **本次运行实际发生了什么？** 查 block 数、依赖、ready/dispatch、核泳道和时间。
5. **性能结论来自哪种证据？** 源码预期、编译事实、Runtime trace 或硬件实测不能互相替代。

---

## 12. 相关实现与文档

- [Scope IR 和 FunctionType](../repo/pto/docs/zh-cn/dev/ir/01-hierarchy.md)
- [InCore scope 外提](../repo/pto/docs/zh-cn/dev/passes/08-outline_incore_scopes.md)
- [SPMD / Cluster scope 外提](../repo/pto/docs/zh-cn/dev/passes/09-outline_cluster_scopes.md)
- [混合 InCore 到 AIC/AIV/Group](../repo/pto/docs/zh-cn/dev/passes/19-expand_mixed_kernel.md)
- [Runtime scope：AUTO / MANUAL](../repo/pto/docs/zh-cn/dev/passes/41-materialize_runtime_scopes.md)
- [SPMD block 身份进入生成 kernel](../repo/pto/docs/zh-cn/dev/codegen/00-pto_codegen.md)
- [Tile MemorySpace 推导](../repo/pto/docs/zh-cn/dev/passes/16-infer_tile_memory_space.md)
- [内存复用与地址分配](../repo/pto/docs/zh-cn/dev/passes/30-memory_reuse.md) / [AllocateMemoryAddr](../repo/pto/docs/zh-cn/dev/passes/31-allocate_memory_addr.md)
- [调度与执行的产品化说明](PTO_调度与执行_开发者作业辅助内容规划.md)
