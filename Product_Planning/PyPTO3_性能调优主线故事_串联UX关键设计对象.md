# PyPTO3 性能调优主线故事：用两个真实案例串联 UX 关键设计对象

> 整理日期：2026-09-23
> 素材来源：[PyPTO 性能调优典型用户案例](../Insight/PyPTO性能调优典型用户案例_基于GitHub_20260923.md) 中的 GitHub Issue/PR、社区调优日志，以及本地编译产物 `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/`
> 对象体系：沿用 [算子作业可视化基础对象与原语封装规划](./算子作业可视化基础对象与原语封装规划.md)（6 族 32 类对象、4 个横切契约、3 个载体、17 个图元），以及 [PyPTO3 算子开发可视化基础作业对象](./PyPTO3算子开发可视化基础作业对象.md)

---

## 一、阅读说明

### 1.1 这份文档要解决什么

规划文档已经定义了对象和图元，但评审时常被问到：**这些对象在一次真实工作里是怎么先后出场、互相引用的？**本文用两个故事回答这个问题。每个故事都从一个真实问题出发，依次经过以下关键体验环节：

| 体验环节 | 故事一 | 故事二 |
| --- | --- | --- |
| 性能分析（从模型到核内） | ●●● | ●● |
| 编译过程可视（Pass、IR、生成代码、编译提示） | ●● | ●●● |
| 算子代码与依赖结构 | ●●● | ● |
| 硬件约束（容量、搬运、核间同步） | ●●● | ●● |
| 实验设计与对比验证 | ●●● | ●● |
| 口径校准与跨团队定界 | ● | ●●● |
| 交付、沉淀与回归看护 | ●● | ●● |

- **故事一：DeepSeek-V4 解码压缩链路调优。**主角是模型开发者，闭环完整，最终合入 PR，重点讲“怎么找到瓶颈、怎么改、怎么证明有效”。
- **故事二：Attention 的“2.3 倍差距”。**主角是算子开发者，需要与编译器和 ISA 专家协作。故事的结局并不是“调快了”，而是判断差距到底出在哪一层、应该交给谁；重点讲“数字能不能比、问题属于哪一层、如何交接”。

### 1.2 叙事边界

遵循 [体验设计汇报叙事与案例 V2](./PyPTO3.0_体验设计项目材料包/PyPTO3.0_体验设计汇报叙事与案例_V2.md) 的约定：**不同案例来自不同问题和版本快照，不能讲成同一次开发的连续经历。**因此：

- 每个故事中的“主角”是**复合角色**，由多位真实作者的工作拼接而成。每个节点都标注来源，读者可以逐条追溯。
- 标注约定：**【实录】**表示数据和做法出自公开来源；**【拼接】**表示事实都是真实的，但来自同一主题下的不同文件或时间点，本文把它们放在同一条线上；**【Mock】**表示为补全体验而虚构的环节，不含虚构数据；**【设计】**表示产品方案建议，不是已有能力。
- 性能数字绑定 a2a3（910B）、特定 shape 和特定版本，只用于说明方法，不构成性能结论。

---

## 二、故事一：DeepSeek-V4 解码压缩链路，从“核很空”到“搬运太碎”

> 按任务展开的作业过程、原始数据和 UX 重点设计，见 [故事一扩展：DeepSeek-V4 压缩链路调优作业全景](./PyPTO3_故事一扩展_DeepSeek-V4压缩链路调优作业全景.md)。

### 2.0 故事梗概

一位模型开发者负责 DeepSeek-V4 decode 路径上的 compressor 家族（`compressor_ratio128`、`decode_indexer_compressor`、`decode_compressor_ratio4` 等）。本文把这些文件的优化串成一条线【拼接】，依次遇到四类问题：

1. **派发开销**：task 过碎；
2. **伪并行**：task 多但核空闲；
3. **访存事务过碎**：加载被拆成大量小笔搬运；
4. **容量墙与编译器的静默行为**：想做的优化受限于片上容量，或被编译器悄悄忽略。

每一步都要用证据推翻或确认假设。最终累计多次两位数的收益，并把经验沉淀为带适用条件的法则。

```text
基线与目标 → 泳道发现派发开销 → 窗口占用率发现伪并行 → 源码依赖结构修正
    → 编译视角：静默忽略、缺 codegen、编译提示 → 硬件约束：Mat/UB/L0C 容量墙
    → 核内下钻：MTE2 散读 → 多轮实验与噪声判定 → 交付：跨文件签名传播、法则沉淀、回归看护
```

### 节点 1：立基线，定测量协议

**发生了什么**【实录，pypto-lib#314】
- 开发者没有先改代码，而是先写出一份基线：commit `8f88c8f`，a2a3，B=64/S=1。
  - Total Test Time 394.84 µs，共 94 个 task；
  - 执行时间 1787.4 µs，Tail OH（task 完成后等待被调度器发现的时间）727.8 µs，占 latency 的 28.3%，P50/P95 分别为 9.5 / 13.3 µs；
  - 附有 12 个 kernel 的 Exec/Latency 排行。
- 同时写明**测量协议**：每次改动后跑 `--enable-l2-swimlane`，以 Total Test Time 为主指标；精度不过或编译失败时，记录阻塞原因并回退。

**用户卡点**：基线数据散落在终端输出和泳道 JSON 里，需要手工整理成表格。后续每一步也要手工对照这张表。

**设计对象与图元**

| 对象 | 在这一步的角色 | 图元 |
| --- | --- | --- |
| `Workload`、`Objective` | B=64/S=1 的 decode 形态；以 Total Test Time 为主指标 | C1 MetricView（KPI 行） |
| `Run`、`Environment` 指纹 | 绑定 commit、平台和 ptoas 版本 | L2 FindingCard 角标 |
| `Baseline` | 基线本身是一个对象，后续所有对比都引用它 | B4 TrendView 基线带 |
| `Task` 聚合 → `Kernel` | 12 个 kernel 的 Exec/Latency | D2 TableView 排行态 |
| `Plan` | 4 条优化原则，每条带风险和成功判据 | D2 TableView 明细态 |

**【设计】**“新建调优任务”时自动生成基线卡，内容包括环境指纹、主指标、kernel 排行和开销构成，并要求用户填写测量协议（主指标、重复次数、精度门禁）。此后每次运行都自动与这张基线卡比较，**没有基线就不显示“提升 x%”**。

### 节点 2：泳道显示“执行很短，等待很长”

**发生了什么**【实录，pypto-lib#314】
- 排行中 `kv_cache_write` 被派发 64 次，每次只执行 2.06 µs，latency 却是 13.24 µs，latency/exec 达到 6.4 倍。原因是按 batch 一行派发一次。
- 改为 `pl.parallel(B, chunk=16)` 后派发次数从 64 降到 4，483.64 → 395.58 µs（−18.2%），是整轮最大的单步收益。
- 之后逐步加 pipeline、合并 scope、按 batch 并行，累计 485.48 → 379.88 µs（−21.8%），task 数 94 → ~34。
- 再把 4 处 `pl.parallel + pl.at` 改为 `pl.spmd`，Head OH 从 2.29 µs 降到 0.57 µs，又降 9%。这一步依赖编译器修复 pypto#1414（`pl.spmd` 在 if/else 分支内的 SSA 问题）。

**用户卡点**：
- “开销”要拆成 Head OH 和 Tail OH 分别看，需要自己理解调度器的派发和完成语义；
- `pl.spmd` 这条路被编译器 bug 挡住过，用户要自己判断是写法问题还是编译器问题。

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Task`/`Event` | 每个 task 的派发、开始、结束、被发现四个时刻 | B1 TimelineView（按等待原因着色） |
| `Bottleneck`（类型：派发开销） | 由 latency/exec 比值和 Tail OH 分布判定 | B3 BreakdownView：执行、Head OH、Tail OH 的构成 |
| `Scope` → `SourceMap` | 从 `kv_cache_write` 跳回 `pl.parallel(B)` 所在源码行 | L1 TextCanvas 行高亮 |
| `Experiment`/`SweepAxis` | chunk 取值扫描 | C4 ScatterView 单参数敏感性曲线 |
| `Diagnostic`（外部阻塞） | pypto#1414 让 spmd 改写暂时不可行 | L2 FindingCard，状态为 blocked，并链接到 Issue |

### 节点 3：窗口占用率揭示“task 多不等于并行”

**发生了什么**【实录，调优日志 §8】
- 泳道中段全是细碎的核。按时间窗口统计：中段 1500 µs（占 72%）里 **AIC 只有 1.9%、AIV 只有 5.8% 在忙**；每个 scope 有几百个 task，却只落在 1~6 个核上。
- 开发者先排除了“padding 造成空算”的假设，然后做了三次真机实验定位：

| 版本 | Total | 现象 |
| --- | --- | --- |
| 原始，全部融合在一个循环里 | 1815 µs | 整个循环只用 5~7 个核 |
| 拆成 3 个 pass | 1866 µs | 计算 scope 解放出来，但计算本来就不是瓶颈 |
| 拆成 4 个 pass | 2593 µs | scatter 单独成一个 pass 后只剩 1 个核 |
| **3 个 pass + 折叠 scatter** | **~1025 µs（−43%）** | scatter 的 task 数 512 → 128，占用核数 5 → 42 |

**反例**【实录，§6】：另一处 scope 曾被判断为 HBM 带宽瓶颈。开发者减少了重读，单个 task 确实更快，但 Total 从 582 µs 退化到 621 µs，由此推翻带宽假设。窗口内 Cube 利用率只有约 21%，真正的原因是依赖和调度停顿。

**用户卡点**：泳道图上能看到“碎”和“空”，但**看不出为什么空**。依赖关系藏在源码的写法里。

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Counter`（窗口占用率） | 对任意时间窗口计算 AIC/AIV 忙碌比例 | B1 TimelineView 区间刷选，得到 C1 占用率读数 |
| `Bottleneck`（类型：串行化/停顿） | 规则：窗口利用率低于约 30% 就不是计算瓶颈 | L2 FindingCard，附判定依据 |
| `DepEdge` | 跨迭代的 WAW 边 | A1 GraphView 依赖图，把关键链描边 |
| `Finding`（被推翻的假设） | “带宽瓶颈”被实验推翻后保留在记录中 | D1 DiffView 基线因果 Diff |

**【设计】**用户在泳道上框选一段时间窗口，系统立即给出 AIC/AIV 利用率，并按规则给出“这不是计算瓶颈”的提示，同时建议做 split test（把 tile 砍半，看单个 task 是否也减半），用来区分计算瓶颈和停顿瓶颈。这相当于把调优日志里的法则变成工具内置的诊断规则。

### 节点 4：回到源码，一行缩进决定并行度

**发生了什么**【实录，§8】
- 根因在源码结构：`for o0` 写在 `pl.at` 外面，导致每个 batch 生成 4 个独立 scope，每个 scope 都对同一个 GM 句柄 `compress_state_flat` 做 `pl.assemble` 重新赋值。跨迭代的 WAW 链由此形成，把 64 个 batch 串在一起执行。
- 修改方法是把 `pl.at` 提到外层，包住循环：

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_paged"):
    for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):      # 原来写在 pl.at 外面
        for s in pl.range(S):
            compress_state_flat = pl.assemble(compress_state_flat, kv_tile, [state_row, o0])
```

**另一个“反直觉”的源码结构**【实录，§14】：一个 matmul + dequant 的融合 scope，把 dequant 固定在关键路径的时间窗口内，与关键路径上的 `qr_proj_aiv` 争抢 AIV 核。拆开融合后，dequant 被调度器推迟到 AIV 空闲的窗口执行，整图 −69 µs。**融合并不总是更快。**

**用户卡点**：源码里的一处缩进、一次重新赋值，会在编译后变成依赖边，再在运行时变成核空闲。用户需要在三层之间来回切换，才能把它们对上。

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Scope` | `pl.at`、`pl.spmd` 的边界，以及每个 scope 实际展开出的 task 数 | A2 TreeView：Scope 树，带 task 数和核数列 |
| `Tensor`（GM 句柄） | 被重新赋值的共享张量是依赖链的源头 | A1 GraphView 中高亮该张量的读写边 |
| `SourceMap` | 源码行 → scope → task → 泳道条 | L3 EvidenceChain（源码 → IR → Task → 症状） |
| `Candidate` | 原写法与修改后写法两个候选 | D1 DiffView 源码 Diff，加每个 scope 的核数变化 |

**【设计】**在编辑器里选中一个 `pl.at` 块时，侧栏直接显示它上一次运行展开出的 task 数、实际占用核数，以及对哪些共享张量有写依赖。如果“task 数很多但占用核数很少”，就标记为伪并行风险，并定位到造成依赖链的那一行 `pl.assemble`。

### 节点 5：编译过程可视——编译器做了什么、没做什么

这一节的三件事都说明：**用户写下的意图，不一定被编译器执行。**

**(a) 静默忽略**【实录，§19】
- 开发者想让 matmul 的权重以 NZ 格式预先打包，于是在输入张量上声明了 `pl.Tensor[..., pl.NZ]`。
- 用探针脚本对比后发现，NZ 和 ND 两种声明生成的 `.pto` **逐字节相同**：codegen 静默忽略了这个声明，没有任何报错或提示。

**(b) API 存在但没有 codegen**【实录，§20】
- `pl.col_expand_sub` / `pl.col_expand_div` 在 Python API 中存在，编译时却报 `No codegen registered`。
- 开发者只能绕道：减法用 `col_expand_expdif`，除法用 `col_expand_mul(x, recip(sum))`。

**(c) 编译提示已经很多，但用户难以消化**【实录，本地数据；与上文不是同一版本快照】
- 本地 DeepSeek-V4 DSpark 的 `decode_csa` 构建（2026-09-03）产出了 **52 个 pass 的 IR dump**（`00_frontend` → `…_after_MemoryReuse` → `…_after_AllocateMemoryAddr` …），以及 **230 条性能提示**：
  - `PH001` 197 条：tile 最内维字节数低于 512B 的 L2 cache line；
  - `PH-MR-001` 33 条：请求的流水深度放不下，相邻 stage 被迫共用存储并串行执行。
- 同一个 compressor 家族的文件也在提示范围内，例如：
  - `decode_compressor_ratio4.py:110`：Right buffer 请求深度 2，但只能放下 1 份（每份 32768 B，空闲 65536 B）；
  - 同一文件中多处 `fp32[64]` 的加载和存储，最内维只有 256B（低于 512B）。
- 这类提示所描述的，正是节点 7 中“`[1,64]` 散读”那一类问题。
- 按文件统计，提示最多的是 `qkv_proj_rope.py`（62 条）和 `hc_pre.py`（41 条）。一次构建中，没有排序和聚合的提示有两百多条。

**用户卡点**：
- 编译器的“没做”没有任何信号（a、b）；
- 编译器的“提示”又太多，缺少优先级（c）；
- 用户不知道哪条提示与当前瓶颈有关。

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Pass`、`IRNode` | 52 个 pass 的顺序、每步 IR 的增量 | B2 SequenceView，无变化的 pass 可折叠 |
| `CodegenArtifact` | `.pto` 与生成的 C++ | D1 DiffView：两种声明生成的产物“完全相同”也是一条证据 |
| `Diagnostic`（静默忽略 / 缺 codegen） | 声明未生效、API 不可用 | L2 FindingCard，Provenance 标为 `compiled` |
| `Diagnostic`（perf hint） | `PH001` / `PH-MR-001`，带源码位置 | D2 TableView 排行态：按运行时热点排序，而不是按出现顺序 |
| `SourceMap` | 提示 → 源码行 → 该行在泳道上的耗时 | L3 EvidenceChain |

**【设计】**
- **意图核对**：对用户显式声明的 layout、pipeline 深度、融合等意图，编译后逐项对比“请求值”与“实际生效值”，未生效的项必须显示出来。
- **提示分诊**：把 230 条提示与上一次运行的泳道 join 起来，只把“落在热点 scope 上”的提示排到前面；其余提示折叠，按规则类型聚合。

### 节点 6：硬件约束——每个“再大一点”都会撞墙

**发生了什么**【实录，§2、§19、§20，pypto-lib#665】

| 想做的事 | 撞到的墙 | 数据 | 最终处理 |
| --- | --- | --- | --- |
| `qr_proj` 输出 tile 256 → 512 | UB | `qr_acc [T,512] INT32 = 256KB > 192KB` | 上限定为 256 |
| `qr_hadamard` 跟随 rope 一起加大 | L0C 累加器 | `[HEAD_ROWS,128] FP32` 在 GRP=4 时已占满 128KB | 拆成独立参数组 |
| `kv_score_proj` 流水加深到 stage=3 | L1（Mat） | `576KB > 512KB` | stage=2 已是上限 |
| `softmax_pool` 头维加宽到 256 | UB | `3×[128,256] FP32 > 192KB` | 上限定为 128 |
| Qwen3 MLP `TN=512`（旁证） | L1（Mat） | `655360 > 524288` | 编译失败，改走其他路线 |

- 容量墙之外还有**搬运形态**的约束：`MOV_OUT_TO_L1_MULTI_ND2NZ` 把按行存储的 `[K,N]` 权重拼成 16×16 分形块，产生大量短而跨步的 burst。每条通道只有约 21 GB/s，远低于 HBM 上限，所以瓶颈是 transaction 笔数，不是带宽。
- 把权重转置存储，并调用 `matmul(..., b_trans=True)`，加载方式从 ND2NZ 变为 DN2ZN，短 burst 变成长 burst：busy −15.1%，CSA 整图 wall −12.9%（3 次运行取中位数）。

**用户卡点**：
- 容量上限只能**试出来**：改参数 → 编译失败 → 再改；
- 各级存储之间互相牵连：UP_DOWN 切分会缩小 L0C 行，但不会缩小某些 `create_tensor`；
- 搬运形态藏在指令名里，用户需要自己理解 ND/NZ 分形。

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Constraint`（带 Scope 契约） | UB 192KB、L1 512KB、L0C 容量，绑定平台和版本 | L2 FindingCard，附“适用于 a2a3”标签 |
| `MemBlock` | 每个 tile 在哪一级存储、占多少、存活多长时间 | C5 OccupancyView（地址 × 生命周期），叠加容量包络线 |
| `Tile`/`Chunk` | tile 的切分方式和最内维字节数 | A4 TilingView |
| `SweepAxis` | 可调参数与上限 | C3 HeatmapView 参数扫描（合法区、越界区） |
| `Counter`（每条 MTE 通道） | 通道带宽和 transaction 笔数 | A3 TopologyView 芯片内底图，标注搬运路径 |

**【设计】**
- **编译前预算**：调参数时实时显示各级存储的占用与上限，“再加大一档就会越界”要在编译前给出提示。可复用现有的 Memory Inspector / Ascend Memory Studio 原型。
- **搬运形态解释**：把 `ND2NZ` 这类指令翻译成“每笔多长、共多少笔”，并与理论带宽对照。这样用户在下“带宽墙”的结论之前，先检查是不是 transaction 过碎。

### 节点 7：核内下钻——Vector 忙，但忙的是搬运

**发生了什么**【实录，§20】
- 整图排名中，`softmax_pool` 的 core-time 为 9495 µs，是第 2 忙的 scope。
- 开发者早先在 ratio4 上得出过“这个 scope 不在关键路径”的判断，这次在 ratio128/HCA 形态下被自己推翻。
- 用 incore op-sim 下钻单个 task：

| 观察项 | 数据 |
| --- | --- |
| veccore span | 20.58 µs |
| MTE2 占比 | 69%，由 **256 笔 `[1,64]` 散读**组成 |
| VECTOR 占比 | 63%，大部分是逐行 staging 和两次转置 |
| 真正的 softmax 计算 | 合计 < 1 µs |

- **陷阱**：kernel 带有数据门控，自动生成的 golden 把 `position_ids` 清零，导致循环跑 0 轮，trace 看起来“很快”。开发者手动把输入写成 127 后重跑 op-sim。

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Counter`（按指令类型） | MTE2 / VECTOR / CUBE 的周期和笔数 | B3 BreakdownView：按“搬运 / 计算 / 转置”重新归类 |
| `Kernel` → `IRNode` → `SourceMap` | 256 笔散读对应源码中的哪一行切片 | L3 EvidenceChain |
| `TensorDump`（控制张量） | 用来判断 trace 是否退化 | L2 FindingCard 警告：“本次 trace 循环 0 轮，数据不可信” |

**【设计】**核内视图的默认分组应该回答“时间花在搬运还是计算上”，而不是按硬件单元罗列周期。如果循环轮数为 0，或远低于预期，要把 trace 标记为不可用，防止用户基于它下结论。

### 节点 8：实验与验证——三道防线

**发生了什么**【实录，§6、§19、§20】

1. **扫描拐点**：`kv_score_proj` 的 task 数取 2、4、8 三个值，对应 Total 621、582、598 µs，4 是拐点。数量更多或更少都会变差。
2. **可复现的逐位正确性**：R1 改为按块读取（MTE2 cycles −90.3%，事务笔数 256 → 32）；R2 用列归约替代转置、tile 从 64 加宽到 128。两轮都做到 `max_error_ratio=0.0`。
3. **用无关 scope 判断噪声**：

| 版本 | 整图 wall | softmax_pool core-time | `qk_pv`（无关，作噪声锚点） |
| --- | --- | --- | --- |
| 原始 | 1303.6 µs | 9495 µs | 22301 µs |
| R1 | 1276.2 µs | 3203 µs | **19754 µs**（这次运行偏快） |
| R2 | **1190.1 µs** | **2792 µs（−71%）** | 21918 µs |

- R1 的 wall 收益大半来自那次运行偶然偏快，所以只认原始版本与 R2 之间的 **−8.7%**。§19 也记录过单次运行出现 +20% 假象的情况。

**用户卡点**：判断噪声完全靠个人经验。选哪个 scope 当锚点、跑几次、看 wall 还是 busy，每个人的做法都不一样。

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Experiment`/`Candidate` | 原始、R1、R2 三个候选，每个都有版本号 | D2 TableView 明细态 |
| `Baseline`（带比较资格） | 同 session、多次运行、同一口径 | C1 MetricView 门禁徽标：“可比 / 不可比” |
| `Gate`（精度） | 逐位一致或容差内 | C1 门禁 |
| 噪声锚点（`Counter` 派生） | 自动挑选与改动无关、耗时稳定的大 scope | D1 DiffView 基线因果 Diff，增加“锚点漂移”列 |
| `Distribution` | 多次运行的分布 | C2 DistributionView |

**【设计】**每个实验对比自动附上“比较资格检查”：是否同 session、次数是否足够、锚点是否漂移、精度是否通过。任一项不满足，就把提升数字置灰，并说明缺哪一步。

### 节点 9：交付——改动的影响范围，以及经验的适用边界

**发生了什么**
- 【实录，§19】`b_trans` 改变了权重签名，需要从 `decode_compressor_ratio4` 一路改到 `decode_attention_csa`、`decode_layer`、`decode_fwd` 四处；而共用 compressor 名字的另外两条路径**不能改**。漏改时，运行时报的是 507018，而不是 shape 错误。CI 上还出现过一个与本次改动无关的报错，原因是分支基于旧 base，需要 rebase 解决。
- 【实录】PR pypto-lib#628（6/29）和 #641（6/30）合入，PR 描述里写明了改前改后的 build 目录。
- 【实录，pypto#2309】同一个 `b_trans`，在 Qwen3-14B 浅 K tile（TK=64）的场景反而慢了 16~23%。**一条经验如果不写明适用条件，就会被误用。**
- 【还原】pypto-lib 在 9 月接入每日算子性能 CI（#1274、#1284）。本故事中的 scope 是否被纳入监控，来源没有说明；按流程，下一步应当纳入。

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Lineage` | 改一个签名会波及哪些调用方，哪些同名路径不受影响 | A1 GraphView 血缘版式 |
| `Artifact`/`Manifest` | PR 附带的 build 目录、泳道、op-sim 结果 | L2 FindingCard 折叠证据 |
| `Recipe`/`Knowledge`（带 Scope 契约） | “分页池按块读取”“先试 b_trans”等法则，附 shape 和平台条件，以及反例 | D2 TableView，附“适用 / 不适用”两栏 |
| `Regression`/`Baseline` | 合入后进入每日看护 | B4 TrendView，标注门禁线和变更点 |

### 小结：故事一的对象出场顺序

```text
Workload/Objective → Run + Environment → Baseline → Task/Kernel → Bottleneck(派发)
  → Counter(窗口占用) → Bottleneck(串行) → DepEdge → Scope → SourceMap → Candidate
  → Pass/IRNode → CodegenArtifact → Diagnostic(静默/缺 codegen/perf hint)
  → Constraint → MemBlock → Tile → SweepAxis → Counter(MTE/Vector) → TensorDump
  → Experiment → Gate → Baseline(比较资格) → Lineage → Artifact → Recipe(Scope 边界) → Regression
```

---

## 三、故事二：Attention 的“2.3 倍差距”，一次跨层定界

### 3.0 故事梗概

一位算子开发者负责 Qwen3-14B decode 的 attention。一张对比表显示 attention 明显落后，一个“2.3 倍差距”被写进 Issue，并被转到编译器团队；同期团队决定用外部 CANN 算子替换原生实现。随后发现这个数字的统计范围不一致，真实差距接近 0。但在继续追查余量的过程中，确实找到了编译 pass 和硬件路径上的问题，这些问题需要交给编译器和 ISA 专家处理【拼接：pypto-lib#465、#622、#607，pypto#1475、#2040，pto-isa#172；最后一个节点使用的是 pto-isa 仓库自带的 flash_atten 示例，与 Qwen3 的 `fa_fused` 不是同一个 kernel】。

```text
逐阶段对比表 → 被转发的 2.3× → 口径校准：0.95× → 选择分析视角（单独 vs 整层）
  → PMU：管道空闲，不是带宽墙 → 编译 pass 级定位：MemoryReuse 造成 WAR
  → 硬件路径约束：A2/A3 核间经 GM 往返 → ISA 层回归：一行 SyncPeriod → 定界与交接
```

### 节点 1：一张逐阶段对比表

**发生了什么**【实录，pypto-lib#465】
- 把 Qwen3-14B 单层 decode 拆成 9 个 stage，与 ASC 小算子逐项对比。方法写得很完整：3 次均值、各 stage 的 CoV（均小于 7%，`qk_norm` 除外）、交叉校验。
- dep_gen 在这张大图上溢出（丢失 128~320 条记录），task 名称只能通过**重放编排提交序列**还原。
- 结论：attention 为 421.7 µs，比值 34.1%（🔴），占整层约 36%，并且相对上一次快照回退了 131 µs。

**用户卡点**：一张可信的对比表需要大量方法工作；诊断工具在大图上会溢出，用户只能自己补救。

**设计对象**：`Model`/`Layer`/`ModuleNode`（stage 划分）、`Baseline`（ASC 参考列）、`Run`×3、`Distribution`（CoV）、`Diagnostic`（dep_gen 溢出）。
**图元**：D2 TableView 排行态（格内条形 + 红绿比值）、C2 DistributionView、L2 FindingCard（“名称经编排序列还原，已 100% 校验”）。

### 节点 2：一个被转发的数字

**发生了什么**【实录，pypto-lib#622、pypto#2040、pypto-lib#765】
- TraCR trace 显示 CCE 约 548 µs，pypto 约 1263 µs，差距约 2.3 倍。
- 这个数字随后被写进 pypto#2040，转到编译器团队做根因分析；同期 pypto-lib#765 把 decode attention 换成了 CANN FAI。
- 评论指出：1263 µs 是**整个 decode layer**（约 35 个 kernel）的 on-core 时间，而 CCE 只统计了 paged attention。只取 attention 部分约为 350 µs。
- 统一 runtime、各自使用最佳 tiling 后重新测量：pypto 约 326 µs，CCE 约 343 µs，**比值 0.95×**。CCE 的默认 tiling 被 `is_long_seq` 阈值误导到慢路径，约 462 µs。

**用户卡点**：一个未经校准的数字会被下游团队当作输入，影响他们的判断。比较双方至少有四个维度没有对齐：统计范围、runtime、tiling、计时方式。

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Baseline`（比较资格） | 双方的统计范围、runtime、tiling、计时方式四项对照 | D2 TableView 矩阵态：四维对齐检查 |
| `Scope` 契约 | 数字只在声明的范围内成立 | FindingCard 上的 Scope 标签 |
| `Provenance` | `measured` 与 `estimated` 分开；被转发时保持原始来源 | 角标 |
| `Finding` 的版本 | “2.3×”被修正为“0.95×”，修正过程可追溯 | B2 SequenceView：结论的演变 |

**【设计】**只有通过比较资格检查的数字才能被复制或引用到其他任务、Issue 中；被引用的结论更新后，引用方会收到通知。

### 节点 3：选择分析视角——单独运行还是整层运行

**发生了什么**【实录，pypto-lib#607】
- 同一个 attention 在单独运行时气泡明显，在整层中却被 MLP 和投影计算掩盖。
- 结论：**要优化 attention 本身，看单独运行；要衡量端到端，看整层运行。**两者回答的是不同问题。

**设计对象**：`Run`（运行上下文：standalone / in-layer）、`Objective`（当前在回答哪个问题）。
**图元**：B1 TimelineView 双视图对照，其中“被其他工作掩盖的气泡”用不同着色标出。

### 节点 4：PMU 显示余量来自管道空闲

**发生了什么**【实录，pypto-lib#622】
- 两种实现的有效带宽都只有约 0.78~0.82 TB/s（峰值的 52~55%），都没有达到 ATB 的 1.0~1.1 TB/s。
- PMU 显示 MTE2 只有 65~70% 在忙；HBM 实际读取约 304 MiB，理论值 256 MiB，没有过量读取。因此问题是**管道空闲，不是带宽上限**。
- 另外 PMU 本身会让 wall 膨胀约 15%（343 → 401 µs），所以计时和计数要分两次运行采集。

**设计对象**：`Counter`（PMU：各 MTE 通道忙碌比例、实际读取字节数）、`Estimate`（理论字节数）。
**图元**：C4 ScatterView（Roofline：实测点 vs 带宽屋顶）、B3 BreakdownView（理论字节 vs 实际字节）、L2 FindingCard 警告“本次运行开启 PMU，wall 不可用”。

### 节点 5：编译过程可视——两个 matmul 为什么没有重叠

**发生了什么**【实录，pypto#1475】
- **症状**：trace 显示 `fa_qks` 循环体里的两个 QK matmul 在 Cube 上串行执行，第二个 matmul 的 K 加载在等第一个。
- **沿 pass dump 追溯**：
  - `28_after_InitMemRef.py` 中，两个 matmul 各有独立的 Mat buffer，共 4 个 alloc；
  - `29_after_MemoryReuse.py` 中，它们被合并到同一组 buffer，只剩 2 个 alloc。
- **沿生成代码追溯**：`fa_qks_aic.cpp` 第 114 行出现 `wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1)`。第二次 `TLOAD` 必须等第一次 `TMOV` 读完同一块 buffer，形成一个**只因复用才出现的 WAR 依赖**。
- **结论**：复用在功能上是正确的，但牺牲了流水重叠。Issue 仍未关闭。
- **版本差异**【实录，本地数据】：在 2026-09-03 的 DeepSeek-V4 构建中，同一个 pass 已经排到 `35_after_MemoryReuse.py`。pass 编号会随版本变化，所以跨版本对照 dump 时，必须按 pass 名称对齐，而不是按编号。

**这个节点是 L3 EvidenceChain 最完整的一个实例**：

```text
源码：两次 matmul（fa_qks 循环体）
  → IR：28_after_InitMemRef（4 个 Mat alloc）→ 29_after_MemoryReuse（2 个 alloc）   [Pass / IRNode DiffView]
  → Kernel：fa_qks_aic.cpp:90 TMOV / :114 wait_flag / :115 TLOAD                  [CodegenArtifact + Fence]
  → Task：Cube 上两个 matmul 串行                                                   [TimelineView]
  → 症状：attention 管道空闲                                                        [Counter]
```

**用户卡点**：这条链需要人工打开 50 多个 dump 文件、生成的 C++ 和 trace 三处材料逐一对照；普通算子开发者很难独立完成。

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Pass` | 找出“哪个 pass 让 alloc 数变少” | B2 SequenceView，逐 pass 统计 alloc 数量 |
| `IRNode` | 两个版本 IR 的结构化差异 | D1 DiffView IR 结构化 Diff |
| `MemBlock` | buffer 复用前后的地址与生命周期 | C5 OccupancyView：两块 buffer 合并为一块 |
| `Task`/`Event`/`Fence` | `wait_flag` 对应的 set/wait 配对 | A1 GraphView 同步依赖 + B1 TimelineView |
| `SourceMap` | 源码 ↔ IR ↔ C++ 行号 | L1 TextCanvas 三栏联动 |

**【设计】**这一类问题适合用 Pass Transform Explorer / Pass Atlas 原型承载：用户在泳道上点击“等待”条，系统反向追溯到引入该等待的 `wait_flag`，再定位到引入依赖的 pass，最后标出该 pass 的决策：“为节省 Mat 空间复用了 buffer”。

### 节点 6：硬件路径约束——同样的融合，在不同平台收益不同

**发生了什么**【实录，pypto#2040；本地数据】
- 在 A2/A3（910B）上，融合的 cube+vector 根节点，其核间边界（C2V/V2C）仍然经过一块 GM pipe buffer。原因是 `InjectGMPipeBuffer` 在 910B 上被后端开关限制。
- 所以融合只带来了负载均衡，**没有省掉 AIC 与 AIV 之间经 GM 的往返**。在 A5 上这条路径留在片上。
- 这是一个待编译器团队确认的假设，Issue 仍未关闭。
- 本地 9 月的构建 dump 中也能看到 `24_after_InjectGMPipeBuffer.py` 这一步。

**用户卡点**：“融合”在源码里只是一个写法，它在硬件上走哪条数据路径，要看平台和后端开关。用户无法从源码判断。

**设计对象**：`Constraint`（带 Scope：platform=a2a3，backend gate）、`Pass`（InjectGMPipeBuffer 是否生效）、`Counter`（GM 往返流量）。
**图元**：A3 TopologyView 芯片内底图，把“A2/A3 路径”（AIC → GM → AIV）与“A5 路径”（片上）并排显示，并在视觉上区分静态映射与实测路径；A5 FlowView 显示核间流量的去向。

### 节点 7：ISA 层回归——一行代码，10% 性能

**发生了什么**【实录，pto-isa#172；注意这是 pto-isa 仓库自带的 flash_atten 示例，不是 Qwen3 的 `fa_fused`】
- **现象**：同类的 Cube/Vec 四阶段 FIFO 流水 attention 在 S ≥ 32768 时，从与 torch_npu 持平掉到 0.89×。
- **隔离变量**：同一张卡、同一个 ptoas、同一份 MLIR 和 C++，**只切换头文件根目录**，两棵头文件树只有 `TPush.hpp:37` 一行不同。
- **定位**：`SyncPeriod` 从 `SlotNum/2` 改成了 `SlotNum`，使 Cube 与 Vec 的同步从“每 4 个 slot 一次”变成“每 8 个一次”，同步变得粗糙而突发，打断了稳态重叠。
- **验证**：恢复后大 S 场景 +9.6%~+11.1%，误差逐字节一致。
- **处置**：不做全局回退，因为那次改动可能是为了其他配置的正确性。给出三个方案：只在 C2V 单生产者/单消费者路径上恢复；把 `SyncPeriod` 暴露为模板参数；保留新默认值，同时补回归测试。
- **同类旁证**【实录，PTOAS#643】：生成的 GEMM 只有 torch_npu 的 0.37×。原因是 DSL 中多段 if/else 让编译器把 set/wait 外提；改为单个 if/else 后达到 83%，全展开后与手写持平。**源码写法会直接决定同步插入的质量。**

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Regression` | 在哪个版本、由哪个 commit 引入 | B4 TrendView，标注变更点 |
| `Environment` 指纹 | 两次运行只差一个头文件 | D1 DiffView 环境 Diff |
| `Fence`/`CommOp`（核间 FIFO） | slot 数、同步周期、生产者与消费者的节拍 | B1 TimelineView：Cube 与 Vec 两条轨道的节拍重放（参照现有 Flash Attention Replay 原型） |
| `SweepAxis` | 不同 S 下的恢复幅度 | C4 ScatterView 敏感性曲线 |

### 节点 8：定界与交接

**发生了什么**
- 【实录】截至归档时间：pypto#1475（MemoryReuse）和 pypto#2040（GM 往返）仍未关闭；pto-isa#172 已关闭并给出三个处置方案；pypto-lib#765 选择在该 shape 上使用外部 kernel。
- 【Mock】算子开发者把定界结论整理成交接包，发给编译器团队：
  1. 已排除项：统计范围差异、带宽上限；
  2. 证据链：节点 5 的完整链；
  3. 平台条件：只在 a2a3 上成立；
  4. 最小复现：`decode_layer.py -p a2a3sim`，加上两个 pass dump 的路径；
  5. 期望：MemoryReuse 能权衡流水代价，或提供开关。

**用户卡点**：定界本身就是一项主要工作。交接时，对方需要重新搭建环境、重新建立上下文。

**设计对象与图元**

| 对象 | 角色 | 图元 |
| --- | --- | --- |
| `Finding`（带状态：待确认 / 已确认 / 已排除） | 每条假设的当前状态 | D2 TableView 矩阵态 |
| `Bottleneck`（归属层） | 算子 / 编译器 / 运行时 / ISA / 外部库 | L3 EvidenceChain，首个异常点落在哪一层 |
| `Artifact`/`Manifest` | 最小复现、dump、trace、环境指纹 | 交接包 |
| `Incident`/外部 Issue | 与 GitHub Issue 双向链接 | FindingCard 动作：“创建 Issue 并附证据” |

### 小结：故事二的对象出场顺序

```text
Layer/ModuleNode → Run×3 + Distribution → Baseline(ASC) → Diagnostic(dep_gen 溢出)
  → Finding(2.3×) → Baseline(比较资格) → Scope 契约 → Finding 修订(0.95×)
  → Run(standalone vs in-layer) → Counter(PMU) + Estimate → Bottleneck(管道空闲)
  → Pass → IRNode Diff → MemBlock → Fence → SourceMap → EvidenceChain
  → Constraint(平台/后端开关) → TopologyView 路径
  → Regression → Environment Diff → Fence 节拍 → Finding 状态 → Artifact 交接包 → 外部 Issue
```

---

## 四、两个故事如何覆盖设计对象体系

### 4.1 对象族覆盖

| 对象族 | 故事一出场 | 故事二出场 | 说明 |
| --- | --- | --- | --- |
| 族 1 意图与目标 | `Workload` `Objective` `Plan` `Constraint` | `Objective` `Estimate` `Constraint` | 故事二的 `Objective` 决定采用单独运行还是整层视角 |
| 族 2 模型与资产 | `Recipe`/`Knowledge` | `Layer`/`ModuleNode` | `Recipe` 必须带 Scope 边界（`b_trans` 正反例） |
| 族 3 源码与编译 | `Scope` `Tile` `Pass` `IRNode` `Diagnostic` `MemBlock` `CodegenArtifact` `SourceMap` | `Pass` `IRNode` `MemBlock` `CodegenArtifact` `SourceMap` | 故事一侧重“意图是否生效”，故事二侧重“哪个 pass 引入了问题” |
| 族 4 执行与证据 | `Run` `Task`/`Event` `DepEdge` `Counter` `TensorDump` | `Run` `Task`/`Fence` `Counter` `CommOp` | `Counter` 同时承载窗口占用率、MTE 笔数和 PMU |
| 族 5 判断与实验 | `Bottleneck` `Finding` `Experiment`/`Candidate`/`SweepAxis` `Gate` `Baseline` `Regression` | `Finding`（多版本）`Bottleneck`（归属层）`Baseline`（比较资格）`Regression` | 两个故事都出现结论被推翻或修正，说明 `Finding` 需要状态和版本 |
| 族 6 治理与沉淀 | `Lineage` `Artifact` `Environment` 指纹 | `Environment` Diff `Artifact`/`Manifest` `Incident` | 故事二的交接包是最典型的 `Manifest` |

### 4.2 四个横切契约的实例

| 契约 | 故事一中的实例 | 故事二中的实例 |
| --- | --- | --- |
| **Anchor** | 编译提示 `decode_compressor_ratio4.py:110` → 泳道中对应 scope | `fa_qks_aic.cpp:114` ↔ pass 29 ↔ Cube 泳道上的等待条 |
| **Provenance** | NZ 声明被忽略（`compiled` 证据：`.pto` 逐字节相同）；优化空间标为 `estimated` | 2.3× 与 0.95× 都是 `measured`，但比较资格不同；带 PMU 的 wall 不可用 |
| **Action** | 扫描 chunk、做 split test、重跑 op-sim（需确认设备预算） | 创建 Issue 并附证据、只切换头文件的 A/B |
| **Scope** | `b_trans` 在 DeepSeek-V4 上 −8%，在 Qwen3 TK=64 上 +23% | GM 往返只在 a2a3 成立；0.95× 只在 b16/s4096 且各自最佳 tiling 下成立 |

### 4.3 图元覆盖

原规划的总表实际列出 16 个图元（A1–A5、B1–B4、C1–C5、D1–D2；标题写作“17 个”，与表内数量不一致，建议在原文中核对），两个故事全部用到。其中 A5 FlowView（核间流量去向）只出现一次；B4 TrendView 只出现在基线和回归节点，没有展开。这两处是两个故事覆盖较弱的地方。3 个载体都承担了关键节点：

- **L1 TextCanvas**：源码、IR、C++ 三栏联动（故事二节点 5）；
- **L2 FindingCard**：几乎每个节点；
- **L3 EvidenceChain**：故事一节点 4、5、7，故事二节点 5，是最完整的实例。

### 4.4 与现有原型的映射

| 故事节点 | 可承载的现有原型（`launch.html`） | 需要补的能力【设计】 |
| --- | --- | --- |
| 故事一 1~3、故事二 1 | 算子调优控制台 / V2、Run Overview | 窗口占用率刷选、比较资格检查 |
| 故事一 4 | 算子开发 Copilot、算子对象工作台 | 编辑器内显示 scope 的“task 数 / 核数 / 写依赖” |
| 故事一 5、故事二 5 | Pass Transform Explorer、Pass Atlas、Pass Decision Studio | 意图核对（请求值 vs 生效值）、性能提示分诊、从等待条反查 pass |
| 故事一 6 | Memory Inspector、Ascend Memory Studio | 编译前容量预算、搬运形态解释 |
| 故事一 7 | 调试与调优工作台（Operator Lab） | 按“搬运 / 计算”归类、退化 trace 检测 |
| 故事二 6 | DeepSeek V4 · CSA Mapping Explorer、Decode Layer 计算图 | 按平台切换硬件路径 |
| 故事二 7 | Flash Attention Replay | 环境 Diff、变更点 |
| 故事一 9、故事二 8 | Toolkit Studio（Trusted Kernel Agent Workbench） | 带 Scope 的 Recipe、交接包 |

---

## 五、从两个故事提炼的设计原则

1. **先资格，后数字。**任何“提升 x%”或“差距 x 倍”都要先通过比较资格检查（统计范围、runtime、tiling、计时方式、重复次数、噪声锚点）。依据：故事一节点 8、故事二节点 2。
2. **意图与生效值并排显示。**用户声明的 layout、流水深度、融合方式，编译后要对比“请求值 / 实际生效值”，静默忽略也必须显示。依据：NZ 声明被忽略、`PH-MR-001` 流水深度放不下、MemoryReuse 合并 buffer。
3. **提示要分诊，不要堆叠。**两百多条编译提示需要与运行时热点 join，只把相关的排到前面。依据：本地 230 条性能提示。
4. **把约束放到编译之前。**容量墙应该在调参数时预判，而不是等编译失败。依据：5 处容量墙。
5. **证据链是主要的交互路径。**从症状（泳道上的等待条）到源码行的反向追溯，是两个故事里最费时、也最关键的操作。依据：故事二节点 5。
6. **结论需要状态和版本。**结论会被推翻、被修正、被别人引用，`Finding` 要有状态机和修订历史。依据：带宽误判、2.3× → 0.95×、“不在关键路径”被推翻。
7. **经验必须带适用边界。**`Recipe` 必须记录 shape、平台条件和反例，否则会被误用。依据：`b_trans` 的正反两例。
8. **定界也是交付物。**不是每个问题都能由当前用户修复。把证据整理成可交接的包，本身就是价值。依据：故事二节点 8。

---

## 附录：来源索引

| 故事节点 | 来源 | 状态（抓取时） |
| --- | --- | --- |
| 一·1、2 | [pypto-lib#314](https://github.com/hw-native-sys/pypto-lib/issues/314) | 已关闭 |
| 一·3、4、6、7、8、9 | [swimlane-tuning-log.zh.md @9be7112](https://github.com/wangqin1723-max/pypto-lib/blob/9be7112769fdfbe07a089018ee9b88159ef0156b/docs/swimlane-tuning-log.zh.md) §2、§6、§8、§14、§19、§20；索引见 [pypto-lib#828](https://github.com/hw-native-sys/pypto-lib/issues/828) | 固定版本 |
| 一·5c | `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/report/perf_hints.log`、`passes_dump/` | 本地构建产物 |
| 一·6 旁证 | [pypto-lib#665](https://github.com/hw-native-sys/pypto-lib/issues/665) | 已关闭 |
| 一·9 | [pypto-lib#628](https://github.com/hw-native-sys/pypto-lib/pull/628)、[#641](https://github.com/hw-native-sys/pypto-lib/pull/641)、[pypto#2309](https://github.com/hw-native-sys/pypto/issues/2309)、pypto-lib#1274/#1284 | 已合入 / 未关闭 / 已合入 |
| 二·1 | [pypto-lib#465](https://github.com/hw-native-sys/pypto-lib/issues/465) | 未关闭 |
| 二·2、4 | [pypto-lib#622](https://github.com/hw-native-sys/pypto-lib/issues/622)、[pypto#2040](https://github.com/hw-native-sys/pypto/issues/2040)、pypto-lib#765 | 未关闭 / 未关闭 / 已合入 |
| 二·3 | [pypto-lib#607](https://github.com/hw-native-sys/pypto-lib/issues/607) | 未关闭 |
| 二·5 | [pypto#1475](https://github.com/hw-native-sys/pypto/issues/1475) | 未关闭 |
| 二·7 | [pto-isa#172](https://github.com/hw-native-sys/pto-isa/issues/172)、[PTOAS#643](https://github.com/hw-native-sys/PTOAS/issues/643) | 已关闭 |
