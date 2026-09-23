# PyPTO 3.0 对位 NVIDIA / TileLang / Triton · 三线深度对比

> 成文日期：2026-09-17
> 己方证据：`repo/pto` @ main（本文所有【事实·内】均于 2026-09-17 重新核对）、`github_issues/pto/pypto_issues.md`、`Product_Planning/`、`Competitive_Analysis/PyPTO对位TileLang_深度对比.html`（2026-09-07）
> 外部证据：2026-09-17 联网核实，链接与性质见 §11
> 阅读对象：PyPTO 3.0 立项评审、产品规划、竞争答辩准备

---

## 0. 证据标注与口径校准

### 0.1 标注约定（沿用仓库既有方式）

- **【事实·内】** 可在本仓库文件、源码或 Issue 归档中直接核对。
- **【事实·外】** 2026-09-17 联网核实的公开信息，标注一手（官方文档/官方仓库）或二手。
- **【统计】** 由本文检索方法得出，方法在括号内给出，可复现。
- **【推断】** 基于上述证据的产品判断。
- **【建议】** 产品动作建议。

### 0.2 校准一：本文比较的 PyPTO 是哪一个

**【事实·内】** `repo/pto/pyproject.toml` 第 16 行 `version = "0.1.0"`；代码库内不存在 "3.0" 这个版本号。"3.0" 是产品与规划层的叙事标签。

因此全文严格分两栏：

| 栏位 | 含义 | 证据来源 |
|---|---|---|
| **PyPTO 已交付** | `repo/pto` 当前 main 分支真实存在的能力 | 源码与开发文档 |
| **3.0 规划** | 尚未交付的产品承诺 | `Product_Planning/PyPTO3.0_Toolkit_产品功能规划.md` |

**对比时只有"已交付"栏可以与竞品的已发布能力同台。** 把 3.0 规划与竞品已交付能力放在同一张表里比较，是本文明确禁止的做法——这是竞争答辩最容易被击穿的地方。

### 0.3 校准二："NVIDIA" 不是一个产品，而是一个闭环

用户提出的三个对比对象里，"NVIDIA" 是唯一一个不对应单个产品的。本文把它定义为**四件东西的组合**：

| 层 | NVIDIA 侧产品 | 对应 PyPTO 3.0 的哪一块 |
|---|---|---|
| 高层 Tile 语言 | cuTile Python | `pl.*` 统一层 / Intent Studio |
| 专家层 | CuTe DSL / CUTLASS 4.x | `pl.tile.*` |
| 观测与归因 | Nsight Compute / Nsight Systems | Performance Lab / Provenance Explorer |
| 正确性 | Compute Sanitizer（memcheck / racecheck / initcheck / synccheck） | Correctness Lab / Compile Guardian |

**【推断】** 这个定义方式本身就是一条结论：**PyPTO 3.0 的定位（可信证据层）在 NVIDIA 侧的对应物不在语言里，在工具里。** 如果只拿 cuTile 对 PyPTO 比语法，会完全错过真正的对标面。

### 0.4 校准三：Triton 和 TileLang 各有两个身份，不能混谈

| | 上游 | Ascend 分支 |
|---|---|---|
| **Triton** | triton-lang/triton + Gluon | Triton-Ascend |
| **TileLang** | tile-ai/tilelang | tile-ai/tilelang-ascend |

**【推断】** 两个身份回答两个不同的问题：

- **上游决定"开发者期待什么"**——它定义了心智基准与工具链的及格线。PyPTO 的开发者体验会被拿去和上游比，即使上游跑不了昇腾。
- **Ascend 分支决定"客户今天能选什么"**——它才是真正的替代路径。

把两者混成一栏（例如用上游的 7.4k star 去论证 TileLang-Ascend 的成熟度，或用 Ascend 分支的能力空白去论证 TileLang 整体落后）会同时高估和低估对手。

---

## 1. 一句话结论：四家赌的是四件不同的事

| | 赌注 | schedule 由谁产生 | 跨 kernel 编排在哪 | 主要反馈信号 | 分发面 |
|---|---|---|---|---|---|
| **PyPTO** | 编译器自己产生 schedule + 用 IR 契约保证这个过程可信 + 把编排与通信也编译进来 | 编译器（40+ pass） | **编译产物内**（Orchestration 路径 → AICPU） | 诊断（IR dump、DFX、verifier 报错） | 华为 AI 处理器（许可证钉死） |
| **TileLang** | 覆盖足够多硬件 + autotune 帮你找到好 schedule | 用户给骨架，编译器优化 | 外部框架（PyTorch / ACLGraph） | 搜索（autotune）+ 诊断（AutoDD / Pass Diff） | 多后端，MIT |
| **Triton** | 最低心智成本 + 生态引力，用 Gluon 补控制力 | 编译器（layout/调度），用户给 grid | 外部框架（PyTorch / NCCL / vLLM） | 搜索（`@triton.autotune`）+ 验证（interpreter / sanitizer） | NVIDIA + AMD，MIT，事实标准 |
| **NVIDIA** | 不赌单一语言，赌闭环——语言可以有好几个，但 profiler 与 sanitizer 只有一套，并做成基础设施 | 各语言层不同 | 运行时与服务层（CUDA Graphs / NCCL / NVSHMEM / Dynamo） | **三样全有且产品化** | 自家硬件，工具随 CUDA 分发 |

**【推断】** 由此得出本文的三条主判断，后续各节都是对它们的展开：

1. **语言层上，PyPTO 与 TileLang-Ascend 是同量级，与 Triton-Ascend 根本不在同一维度竞争。** 前者比的是编译器自动化质量，后者比的是心智迁移成本——后一场 PyPTO 结构上赢不了，也不该打。
2. **工具层上，PyPTO 3.0 的真正对标是 NVIDIA，而这是差距最具体、最可量化、也最紧迫的一条。** §5 会给出一条已经关闭的时间窗口。
3. **架构层上，MPMD + 编译期编排是三家都没有的**，但必须诚实说明：NVIDIA 不是"没有编排"，而是**把编排放在了运行时与服务层**。差异化主张要精确到这一点，否则会在答辩中被一句话反驳。

---

## 2. 维度一：schedule 的归属权（根本分歧，不是程度差异）

### 2.1 四种形态并置

**【事实·内】** PyPTO 的语言指南明确写"推荐尽量使用 `pl.*`"，即官方推荐的写法是**不写内存层次**。flash attention 示例（`repo/pto/examples/models/03_flash_attention.py`）全程 Tensor 语义：无 shared memory 声明、无 copy、无 fragment、无 `num_stages`、无 grid/threads。

**【事实·外，一手】** 四家的典型写法差异：

| | 用户必须写出来的东西 | 编译器负责的东西 |
|---|---|---|
| **PyPTO `pl.*`** | `pl.at(level=...)` 圈执行层级；tile 常量；可选 `pl.pipeline(stage=F)` | 内存空间归属、L0 分块、双缓冲、AIC/AIV 拆分、跨核流水、buffer 复用、地址分配、任务依赖 |
| **cuTile Python** | `@ct.kernel`、`ct.load/ct.store` 的 index 与 shape、block 索引 `ct.bid()` | 线程、数据搬运、Tensor Core 使用、block 级并行与异步 |
| **Triton** | `grid`、`BLOCK_SIZE` 常量、`tl.load/tl.store` 与 mask | layout、寄存器分配、向量化、部分流水与 warp specialization |
| **TileLang** | `T.Kernel(grid, threads)`、`T.alloc_shared/alloc_fragment`、`T.copy`、`T.Pipelined(num_stages=)` | layout inference、向量化、同步插入（Ascend 侧另加自动 buffer reuse 与 Cube/Vector 自动分核） |

**【事实·外，一手】** cuTile 的官方表述是"cuTile 与 Tile IR 会搞定线程、数据搬运和 Tensor Core"，并"自动化 block 级并行与异步、内存搬运及其他 GPU 编程低层细节"——**这与 PyPTO 的 `pl.*` 赌注是同一个赌注**，不是 TileLang 那个。

### 2.2 自动化程度排序

**【推断】** 按"用户不需要写出来的决策数量"排序：

```
PyPTO pl.*  ≳  cuTile Python  ≈  Triton  >  TileLang Developer  >  TileLang Expert ≈ Gluon ≈ CuTe DSL
   最高抽象                                                                      最强控制
```

这个排序推出两条判断：

**【推断】** 第一，**PyPTO 的抽象赌注在业界不孤立，NVIDIA 押的是同一注。** 这对立项是好消息：方向被最强玩家验证了。但也意味着"抽象高"不构成差异化——它是趋势，不是优势。

**【推断】** 第二，**抽象越高，编译器猜错时用户越无处下手。** 这不是理论风险：PyPTO 没有 autotuner（§4），逃生舱需要重写数据流（§3），两者叠加意味着抽象赌注**没有对冲**。cuTile 与 Triton 押同样高的抽象，但各自都配了 autotune 与连续的降级路径。**抽象的高度必须由退路的宽度来支付，这是本文对语言层的核心判断。**

---

## 3. 维度二：逃生舱——性能不达标时的退路

这是四家差别最清晰、且对 PyPTO 最不利的一个维度。

| | 降级路径 | 迁移成本 |
|---|---|---|
| **Triton** | `triton.language` → **Gluon** | 同一编译器栈内换一层 API。**【事实·外，一手】** Gluon"直接暴露 layout、shared memory、warp specialization 与目标特定特性，让高级 kernel 用便利性换控制力"；两者都是 tile-based SPMD 模型，差别只在谁决定 layout/内存/搬运/异步 |
| **TileLang** | Developer → **Expert** | **【事实·外，一手】** 同一套语法内写出 `T.Scope("C")` / `T.Scope("V")` 加手工 flag 管理，其余代码不动 |
| **NVIDIA** | cuTile Python → **CuTe DSL** → **CUTLASS C++** → PTX/SASS | 四级连续谱，**且每一级都有对应的 profiler 视图**（§5）。**【事实·外，一手】** CuTe DSL 提供 `CUTE_DSL_LINEINFO=1` 的 Python↔PTX/SASS 关联、`CUTE_DSL_KEEP_PTX/KEEP_CUBIN` 导出，以及实验性的 IKET 内核内事件追踪 |
| **PyPTO** | `pl.*` → **`pl.tile.*`** | **重写数据流**：显式 `pl.load(target_memory=pl.Mem.Mat)` → `pl.move(Mem.Left)` → `pl.matmul` → `pl.store`。**【事实·内】** 手工模式确实存在，但这是换一套写法，不是加一个标注 |

**【推断】** 三家的做法收敛到同一个共识：**降级是在同一套语法内加标注或换 API 层，而不是让用户换一套写法。** PyPTO 是四家中唯一一个降级需要重写的。

这条的严重性容易被低估，因为它只在"性能不达标"时才暴露——而那正是开发者最焦虑、最没耐心的时刻。用户此时面临的选择是：读 IR dump 理解 40+ 个 pass 的决策，或者把整个数据流重写一遍。**两条路都比对手贵一个量级。**

**【建议】** 3.0 的逃生舱应该做成**一个操作而不是一份文档**：在 `pl.*` 代码上给出"生成等价 `pl.tile.*` 骨架"的能力——把编译器已经推导出的内存空间、L0 分块、搬运序列**回写成用户可编辑的 tile 级源码**。这件事 PyPTO 有独占的可行性（编译器内部已经有全部答案），且它同时解决了逃生舱与"编译器决策不可见"两个问题。这是本文认为投入产出比最高的三件事之一。

---

## 4. 维度三：反馈信号的三种类型（本文的核心分析）

前两个维度是语言设计问题。这一维度是**产品问题**，也是 PyPTO 3.0 定位成立与否的关键。

### 4.1 先把"工具"拆成三类信号

开发者在"东西不对/不够快"时需要的信号有三种，它们不能互相替代：

| 信号类型 | 回答的问题 | 典型形态 |
|---|---|---|
| **诊断** | 为什么会这样？ | IR dump、pass 决策解释、timeline、source 映射、错误字典 |
| **搜索** | 换几组参数试试 | autotuner、参数扫、实验数据库 |
| **验证** | 结果到底对不对？ | 数值 oracle、内存/竞态 sanitizer、断言、解释器模拟 |

**【推断】** 关键在于：**诊断降低理解成本，搜索降低试错成本，验证降低信任成本。** 一个只有诊断的工具，会把每一次失败都变成一次学习任务——而开发者在 deadline 前要的不是学习，是出路。

### 4.2 四家在三类信号上的实况

**PyPTO（已交付）**

- **诊断：强。** **【事实·内】** 48 篇 pass 开发文档（`repo/pto/docs/zh-cn/dev/passes/*.md`，本日计数）；`dump_passes=True` 的每 pass IR 快照；五个正交 DFX 开关（L2 swimlane / args dump / PMU / dep gen / scope stats，1:1 映射 simpler `CallConfig`，产物路径契约固定）；Span 从架构起点贯穿每个 IR 节点。
- **搜索：零。** **【统计】** 2026-09-17 在 `repo/pto` 全仓库对 `autotun` 做不分大小写递归检索（`.py/.md/.h/.cc`），**零命中**。调优完全依赖用户给出的 tile 常量、`pl.pipeline(stage=F)`、`pl.split(...)` 加编译器固定策略。
- **验证：强，但只在编译期，且只对不变量。** **【统计】** `repo/pto/src/ir/verifier/` 下 23 个 `verify_*.cpp` 加 `type_check_pass.cpp`、`mixed_kernel_expanded_verifier.cpp` = **25 个正确性检查器**；同目录下**性能提示检查器只有 1 个**（`perf_hint_tile_innermost_dim.cpp`，即 PH001）。`VerificationInstrument` 在每个 pass 前检查 `required`、每个 pass 后检查 `produced` 与全部结构性属性。
- **数值验证：有资产，未成产品。** **【事实·内】** `repo/pto/python/pypto/debug/torch_codegen.py` 的模块 docstring 是"Emit executable PyTorch code from PyPTO IR for debugging and numerical verification"——**这已经是一个 oracle 生成器**：从 IR 直接产出可执行的 PyTorch 参考实现。但它对应的文档只有 `docs/zh-cn/dev/debug/00-torch_codegen.md` 一篇，位置在"开发者文档"而不是产品能力。

**Triton（上游）**

- **搜索：强。** **【事实·外，一手】** `@triton.autotune` 配合 `TRITON_PRINT_AUTOTUNING=1` 输出最佳配置与总耗时。
- **验证：四条独立路径。** **【事实·外，一手】** ① `TRITON_INTERPRET=1` 让 kernel 绕过编译、由解释器用 numpy 等价算子模拟，可以直接 `print()`、挂 `pdb`、在 kernel 里下断点（不支持 bf16 与间接访存）；② `static_print` / `static_assert` 编译期、`device_print` / `device_assert`（需 `TRITON_DEBUG=1`）运行期；③ NVIDIA 侧 `compute-sanitizer`、AMD 侧 LLVM AddressSanitizer、跨平台 `triton-viz` 访存可视化、编译器级浮点 instrumentation（FpSan）；④ **【事实·外，一手】** Triton-Sanitizer——面向 Triton 的快速、设备无关内存 sanitizer，带丰富诊断上下文，发表于 ASPLOS '26（2026-03）。
- **诊断：中等偏强。** **【事实·外，一手】** Proton：面向 Triton 的多层自适应 profiler，提供前端 API 选择性地 profile 区域、聚合结果、采集硬件计数器拿不到的自定义指标，并用类 SQL 语言查询 profile（CGO 2026 论文）；多阶段 IR dump；`knobs.py` 集中的运行时配置。

**TileLang**

- **搜索：强。** **【事实·外，一手】** 2026-03-12 起 eager-mode autotuning，2026-05-11 加入并行 autotune：pipelined compilation、grouped compilation、multi-GPU benchmarking。
- **诊断：在快速追赶，且追的正是 PyPTO 的卖点。** **【事实·外，一手】** 官方仓库列出 layout 可视化与 fragment inspection、IR dump 与 pass 可视化、**自动 delta debugging（AutoDD，支持 frozen regions）**、**Pass Diff（跨 pass 的 IR 比较）**、compiler pass timing、IKET profiler 的 CUDA timeline instrumentation；v0.1.13（2026-08-03）把 "source-location compiler diagnostics" 作为新特性发布。
- **验证：弱。** Ascend 侧的调试能力是 `T.printf` 与 `T.dump_tensor`（设备侧打印与 dump），没有 sanitizer 或解释器模拟。

**NVIDIA**

- **三样全有，且都是随工具链分发的产品，不是脚本。**
- 搜索：**【事实·外，一手】** cuTile 的 experimental 命名空间提供 `autotune_launch` 与 `clear_autotune_cache`；CUDA 13.x 侧另有编译器 autotuning。
- 验证：**【事实·外，一手】** Compute Sanitizer 是 CUDA toolkit 内置的功能正确性检查套件，含 memcheck（精确定位并归因越界与非对齐访存）、racecheck（报告 shared memory 数据竞争）、initcheck、synccheck。
- 诊断：Nsight Compute + Nsight Systems，详见 §5。

### 4.3 三条判断

**【推断】判断一：PyPTO 的"验证"与三家的"验证"是两件不同的事，而用户问的是后者。**

PyPTO 的 25 个 verifier 保证的是"**编译器没有把 IR 改坏**"——TypeChecked、UseAfterDef、PipelineLoopValid、ManualDepsOnSubmitOnly 这些都是编译器内部不变量。这是极高质量的工程资产，**但它不回答用户的问题。** 用户的问题是"我的输出为什么不对"。

三家的验证工具全部指向用户的问题：compute-sanitizer 说你越界了，TRITON_INTERPRET 让你 print 中间结果，Triton-Sanitizer 给你带上下文的访存诊断。

**这正是 3.0 Correctness Lab 的真正立足点，也是 `torch_codegen` 这个已有资产最该被变现的地方。** PyPTO 能做一件三家都做不到的事：**从 IR 自动生成一份可读、可执行的 PyTorch 参考实现，用作逐层 oracle。** Triton 的 interpreter 是用 numpy 模拟算子语义，只能告诉你"解释器跑出来是这样"；PyPTO 的 torch_codegen 产出的是一份**用户可以读、可以改、可以单步调试的参考程序**。这个差异在"首个分歧定位"这个场景下是决定性的。

**【推断】判断二：搜索维度上 PyPTO 是零，而三家都有——搜索是最便宜的开发者出路。**

这条已在既有材料中被标为最高优先级缺口，本文补充一个论证角度：**没有 autotuner，不只是"少一个功能"，而是让 §2 的抽象赌注失去对冲。** 抽象越高、编译器决策越多，启发式失手的场景就越多；而失手时用户唯一的出路是最贵的那条（读 IR 或重写）。cuTile 押了同样高的抽象，但它配了 autotune——**NVIDIA 用产品设计承认了"高抽象必须配搜索"这件事。**

**【推断】判断三：诊断这条 PyPTO 的领先项，正在被两侧同时侵蚀。**

TileLang 的 AutoDD（自动 delta debugging）、Pass Diff（跨 pass IR 比较）、source-location diagnostics，与 PyPTO 的 `dump_passes` + Span 是同一类能力，而且 **AutoDD 与 Pass Diff 在形态上比 PyPTO 的"IR 快照 + 人工比对"更产品化**——它们是工具，PyPTO 的是产物。Nsight 那侧见 §5。

**【推断】** 综合起来：**PyPTO 在三类信号上是"一强、一空、一错位"——诊断强，搜索空，验证强但强在用户不问的那一半。** 这比"缺一个 autotuner"是一个严重得多的判断，也更准确地解释了为什么内部评估中 PyPTO 的能力密度高而用户感知弱。

---

## 5. 维度四：源码映射——窗口不是正在关闭，是已经关了三个版本

### 5.1 时间线并置

**【事实·外，一手】** Nsight Compute 在 Tile 工作负载上的交付节奏：

| 版本 | 交付内容 |
|---|---|
| **2025.4** | 新增 **Tile section**（汇总 tile 维度与流水线利用率）；Source 页支持 **SASS ↔ 高层 Tile 源码关联**（限 cuTile Python） |
| **2026.2** | 改进 Source 页的 CUDA Tile 支持；Function Statistics 显示**按高层源码行的数据与所代表的时间区间**；支持 CUDA 13.3 |
| **2026.3** | Source 页新增 **Tile IR 视图**，含 **source→Tile IR、Tile IR→PTX、Tile IR→SASS 三向关联**；Source Comparison 扩展到 Tile IR 代码；CLI 输出 Tile IR；新增 **Compute Triage Guide**（自上而下的瓶颈定位工作流）；支持 CUDA 13.4 与 Rubin 架构 |

**【事实·内】** PyPTO 侧对应的状态：`hw-native-sys/pypto#1305`，标题为 "[Feature] Improve PH001 TileInnermostDimGranularity verifier: memory-space awareness, source mapping, dedup, and report clarity"，**创建于 2026-05-07，状态 open，至归档时无更新**。

这个 issue 的正文值得完整引用其五条，因为它是**唯一一条性能提示在真实模型上的实测报告**（W8A8C16 量化的 DeepSeek V4 decode MoE expert kernel，把所有可控 chunk 常量推到预算上限后仍有约 24 条 PH001 提示）：

1. **verifier 不看 `target_memory`**——只按 `shape.back() × dtype.bits` 算，于是对 `Mem.Right / Mem.Acc / Mem.Mat`（cube 私有 L0/L1，根本不过 L2）也报"L2 cache line = 512B"。
2. **`<string>:LINE:COL` 回不到用户源码**——span 落在 pipeline 后的 IR 文本 dump 上，而该 dump 不以干净形式落盘；最接近的 `passes_dump/33_after_Simplify.py` 因 AIC/AIV 子函数被拆出而行号不匹配。**用户无法把 `<string>:122:12` 映射回自己 Python 源码里的 `pl.at` 块。**
3. **不说哪个 DSL 常量控制内层维度**——用户要反推 `K_CHUNK` 还是 `INTER_CHUNK`。
4. **同一处源码语义重复报 4 条**（每次循环展开/每个 matmul_acc tile 报一次），淹没信号。
5. **报出的字节数与用户能从 IR 算出的不一致**（`[32,256] INT32` 按公式是 1024B，报的是 256B）。

### 5.2 一条必须写进立项材料的更正

**【推断】** 既有材料《PyPTO 3.0 建设必要性与核心价值分析》§5.5 把 Nsight 的 Tile 源码关联记为 **2026.2** 的新增，并据此判断"窗口正在关闭"。本次核实的结果更严重：

> **SASS ↔ 高层 Tile 源码关联在 2025.4 就已交付，2026.3 已经做成 source → Tile IR → PTX → SASS 的三向关联，并把 Tile IR 纳入 Source Comparison。**
>
> 也就是说：**窗口不是正在关闭，是已经关闭三个版本了。** 同期 PyPTO 的状态是"唯一一条性能提示的 span 回不到用户源码，且该问题 open 四个月"。

这条更正改变的是紧迫性的量级，不是方向，所以必须进材料。

### 5.3 但差异化仍然成立——只是主张要换

**【推断】** 不能由此推出"PyPTO 在源码映射上已经输了"。真实的格局是：

| | 关联链路 | 覆盖范围 |
|---|---|---|
| Nsight（cuTile） | source ↔ Tile IR ↔ PTX ↔ SASS | 单 kernel 内，编译产物到指令 |
| MindStudio Insight | runtime timeline ↔ 源码映射 ↔ 内存快照 | 运行事实，含百卡千卡 |
| TileLang v0.1.13 | source-location compiler diagnostics | 编译诊断 |
| **PyPTO（架构上可达）** | request/session → model node → **source span → DSL op → IR op/pass → PTOAS op/sync → ISA instruction → runtime task/event/fence → tensor/KV → metric/oracle** | 从服务请求到指令到跨卡 |

**【推断】** PyPTO 的链路确实更长，而且长在对手结构上拿不到的两端：**上游的模型/服务语义**与**下游的跨卡 runtime 任务图**。所以正确的主张是：

> 不是"我们有源码映射"（那已是入场券），而是"**我们的证据链能从一次服务请求走到一条 ISA 指令再走到另一张卡上**"。

但这个主张有一个前置条件：**末端必须能接回用户源码。** #1305 恰好是这个末端。**在 #1305 关闭之前，这条链路在用户处是断的，整个"可信证据层"定位无法兑现。** 这是本文认为必须列为 P0 的第二件事。

---

## 6. 维度五：编排与分布式——唯一的结构性护城河，以及必须诚实的那一句

### 6.1 事实并置

**【事实·内】** PyPTO 侧：

- `pypto` 有**两条并行 codegen 路径**，出自同一个程序、同一棵 IR：InCore 路径编译"算子里面"（→ `.pto` MLIR 方言 → PTOAS → pto-isa C++ → AICore），Orchestration 路径编译"算子之间"（→ C++ PTO2 runtime API → AICPU 任务图执行 → simpler）。
- `pl.submit` / `pl.spmd_submit` 提交任务，`Submit` 是与 `Call` 并列的一等 IR kind，有独立的 `deps_` / `core_num_` / `sync_start_` 字段，贯穿全流水线不下沉，并由 `verify_manual_deps_on_submit_only.cpp` 保证依赖 attr 永不落到普通 `Call` 上。
- `AutoDeriveTaskDependencies` pass 自动推导任务依赖边，并改写 call-site direction 以省去运行时的 overlap 查询。
- 12 个 `pld.*` 算子作用于窗口绑定的 `DistributedTensorType`，verifier 用严格 kind-trait 拒绝普通 `TensorType` 进入跨 rank 槽位——**非窗口绑定的 tensor 永远不会被误传入跨 rank 槽位，这是类型系统保证的，不是文档约定。**
- **【事实·内】** 分布式 comm op 的 ST 覆盖相对完整：`test_l3_allreduce`（含 ring / host / parallel / intrinsic 四种实现）、`test_l3_broadcast`、`test_l3_allgather`、`test_l3_reduce_scatter`、`test_l3_get`、`test_l3_put`、`test_l3_notify_wait` 均有覆盖；`tget_async` / `tput_async` / `ttest` / `build_async_session` 无 ST。

**【事实·外，一手】** 三家侧：

| | 并行模型 | 跨 kernel / 跨卡编排在哪 |
|---|---|---|
| **Triton** | 纯 SPMD | 语言层无；交给 PyTorch / NCCL / vLLM。Triton-Ascend 的多卡内容出现在"从 GPU 迁移"的指南里，不是语言能力 |
| **TileLang** | 纯 SPMD | 上游有 cluster launch / cluster 同步 / TMA multicast / SM-to-SM cluster 传输——注意这些是**SM 级而非卡级**；**Ascend 侧文档无任何多卡或模型级编排声明** |
| **NVIDIA** | 各语言层 SPMD | **不在语言层，在运行时与服务层**：CUDA Graphs、NCCL、NVSHMEM、Dynamo / TensorRT-LLM |

### 6.2 必须诚实的那一句

**【推断】** "对手没有 MPMD"这个说法对 Triton 与 TileLang 成立（它们的语言与编译器里确实没有这个位置），但**对 NVIDIA 不成立**。NVIDIA 有完整的编排答案，只是放在了另一层。在立项答辩中，一句"NVIDIA 没有 MPMD"会被"CUDA Graphs 加 NCCL 加 Dynamo 不算编排吗"直接击穿。

**【推断】** 精确的差异化主张只有一句：

> **PyPTO 把编排决策放进了编译产物，因此编排可以被验证、被归因；对手把编排放在运行时，结构上拿不到编译期证据。**

这句话是可防御的，因为它不声称对手缺功能，而是指出**证据的可得性差异**：`AutoDeriveTaskDependencies` 推导出的依赖边是编译期产物，可以和 IR、Span、tensor 关联；CUDA Graph 的依赖是运行时构造的，Nsight 能看到它执行了什么，但看不到"它为什么是这个依赖"。

**【推断】** 而且这条价值有明确的兑现时机：**它只在万亿参数、超长上下文、超节点部署时才真正值钱**——而昇腾恰好正在进入这个场景。这是必要性与核心价值的交汇点。

---

## 7. 维度六：分发面（结构性劣势，先摆事实，再给一条反向事实）

### 7.1 事实表

**【事实·外，一手，核实于 2026-09-17】** / **【事实·内】**

| | PyPTO | Triton（上游） | Triton-Ascend | TileLang（上游） | TileLang-Ascend | cuTile Python |
|---|---|---|---|---|---|---|
| 许可 | **CANN OSL 2.0**（衍生软件仅可用于华为 AI 处理器） | MIT | MIT | 开放 | MIT | **Apache 2.0** |
| 社区 | 组织内为主（公开仓约 102★，2026-08-05 抓取） | **20.2k★ / 3.2k forks** | 官方托管在 GitCode，GitHub 为镜像 | 7.4k★ / 743 forks | 369★ / 166 forks / 1,463 commits | GitHub 公开 |
| 最新版本 | 0.1.0（内部） | pip `triton`，CPython 3.10–3.14 | **3.2.1（2026-04-30，需 CANN 9.0.0）**，2026 计划升到 Triton 3.5 | v0.1.13（2026-08-03；PyPI 已有 0.1.14） | 需 CANN 8.3.RC1+ 与 torch-npu 2.6.0.RC1+ | pip `cuda-tile` |
| 安装 | 源码构建（CMake + nanobind + scikit-build-core） | `pip install triton` | pip | `pip install tilelang` | 预构建 wheel 或脚本源码构建 | pip |
| 硬件 | Ascend A2/A3（A5 覆盖更弱） | NVIDIA CC 8.0+、AMD ROCm 6.2+、CPU 开发中 | Atlas A2/A3 训练与推理全系 | CUDA SM70–SM120、ROCm CDNA/RDNA、Metal；LLVM CPU / CuTe DSL / WebGPU 实验性 | **A2 / A3 实测，未提 A5** | Blackwell + Ampere/Ada，**Hopper 尚未支持**；需驱动 R580+、CUDA 13.1+ |
| API 覆盖声明 | — | — | **约 85% Triton Python API** | — | GEMM/BatchGEMM/FlashAttn/SparseFA/linear attn/softmax/norm/reduce/sort/**conv**/CE loss/LightningIndexer/TopK/Dispatch&Combine | — |
| 框架集成 | ChipWorker / DeviceTensor；**无 torch.compile / custom_op 集成**（本日检索 `repo/pto/python` 对 `torch.compile`/`custom_op`/`torch.library` 零命中） | PyTorch Inductor 原生后端 | 适配 vLLM / SGLang 算子 | — | PyTorch / ACLGraph（`torch_tl_ascend` 示例） | CuPy；PyTorch/JAX 见测试依赖 |

### 7.2 算子覆盖的自评对照

**【事实·内】** `repo/pto/docs/zh-cn/dev/ptoas-op-status.md`（快照 2026-06-23）的统计原文：**148 个 PTOAS op 行；PTOAS 提供接口 143；pypto 前端已写好 98；有 ST 测试 67。** 该文档的判定原则写得很硬——"一个 op 是否'完成'以是否有 ST 测试为准，没有 ST 的 op 一律视为未完成"。明确缺口包括**卷积整族**（TIMG2COL / TSETFMATRIX / TSET_IMG2COL_*）PTOAS 未实现、`tpartargmax/min` 等标 MISSING、a2a3 已知 ISA 缺陷（`pto.tsubc` 误算 `a-b-c`）导致部分 ST 下架。

**【推断】** 对照 TileLang-Ascend 的算子清单里**明确含 conv、sort、CE loss**，PyPTO 在"能跑的 op 数量"这个最直白的维度上落后。45%（67/148）的真机 ST 覆盖率意味着接近一半的 op 写了但未在真机验证——**这份自评的诚实很宝贵，但它同时是一张明牌。**

### 7.3 一条可以用在答辩上的反向事实

**【事实·外，一手】** cuTile Python 官方 README 的硬件支持表述是：当前支持 **Blackwell 与 Ampere/Ada**，**"Hopper GPU 将在后续版本支持"**；另需驱动 R580+、CUDA Toolkit 13.1+、Python 3.10+。（其早期要求为 compute capability 10.x / 12.x，CUDA 13.2（2026-03）起把 Tile 支持扩展到 CC 8.x 的 Ampere/Ada。）

**【推断】** 这条很有用：**NVIDIA 最新的 Tile 语言层，在自家最主流的数据中心架构（Hopper）上都还没跑通。** 它说明"新抽象层要在成熟硬件上补齐覆盖"是这一代 Tile 语言的共同学费，不是 PyPTO 独有的窘境。PyPTO 的 A5 覆盖偏弱、ST 覆盖 45%，在这个对照下是**行业常态而非特例**。

同时必须承认差别：cuTile 的硬件覆盖缺口是"新语言追旧硬件"，PyPTO 的缺口是"语言追新硬件（A5）+ 算子追自己的 ISA"，后者叠加了 §2.3 的代际成本问题。**这条反向事实可以用来化解"你们成熟度不行"的质疑，但不能用来论证"我们没问题"。**

---

## 8. 三张总表

### 8.1 能力矩阵

图例：**●** 明确领先 / **◐** 势均力敌 / **○** 落后 / **—** 不在该赛道

| 能力 | PyPTO（已交付） | Triton + Gluon | TileLang（+Ascend） | NVIDIA 闭环 |
|---|:---:|:---:|:---:|:---:|
| 高层 Tile 抽象（不写内存层次） | ● | ◐ | ○ | ◐ |
| 专家级手工控制 | ◐ | ● | ● | ● |
| **逃生舱连续性** | **○** | ● | ● | ● |
| **Autotuner** | **○（零）** | ● | ● | ● |
| 编译器 IR 工程与 pass 契约 | ● | ◐ | ◐ | ◐ |
| 编译期不变量验证 | ● | ○ | ○ | ○ |
| **运行期数值/访存验证** | **○** | ● | ○ | ● |
| pass 级诊断（IR dump / diff / delta debugging） | ◐ | ◐ | ● | ◐ |
| **源码 ↔ 指令关联（末端接回用户源码）** | **○（#1305 open）** | ◐ | ◐ | ● |
| 运行时 timeline 与 PMU | ◐ | ◐ | ◐ | ● |
| **MPMD 任务编排（编译期）** | **●** | — | — | — |
| **分布式通信原语（语言一等）** | **●** | — | — | — |
| 模型/框架集成 | ○ | ● | ◐ | ● |
| 服务层闭环（请求级证据） | ○（3.0 规划） | — | — | ● |
| 多后端可移植 | ○（法律约束） | ● | ● | — |
| 社区与上手门槛 | ○ | ● | ● | ● |

### 8.2 反馈信号矩阵（§4 的结论落表）

| | 诊断（为什么） | 搜索（试更多） | 验证（对不对） |
|---|---|---|---|
| **PyPTO** | 48 篇 pass 文档、每 pass IR 快照、5 个 DFX 开关、Span 全程 | **零** | 25 个编译期不变量 verifier；`torch_codegen` 有资产未成产品；**1 个性能提示且 span 断裂** |
| **Triton** | Proton 多层自适应 profiler（可 SQL 查询）、多阶段 IR dump | `@triton.autotune` + `TRITON_PRINT_AUTOTUNING` | `TRITON_INTERPRET` numpy 模拟 + pdb、static/device assert、compute-sanitizer、triton-viz、FpSan、Triton-Sanitizer(ASPLOS'26) |
| **TileLang** | AutoDD 自动 delta debugging、Pass Diff、layout 可视化、pass timing、IKET | eager + 并行 autotune（pipelined / grouped / multi-GPU bench） | `T.printf` / `T.dump_tensor`（Ascend 侧） |
| **NVIDIA** | Nsight Compute（Tile section、四层源码关联、Compute Triage Guide）+ Nsight Systems | `autotune_launch` + CUDA 编译器 autotuning | Compute Sanitizer 四工具（memcheck/racecheck/initcheck/synccheck） |

**【推断】** 这张表是本文最应该被带进评审会的一页。它给出的结论不是"PyPTO 弱"，而是**"PyPTO 强在一类信号的一半上"**：诊断产物丰富但工具化不足，验证深但只验编译器自己，搜索完全空缺。

### 8.3 时间窗口表

| 能力 | 对手交付时间 | PyPTO 当前状态 |
|---|---|---|
| SASS ↔ 高层 Tile 源码关联 | Nsight Compute **2025.4** | #1305 open（2026-05-07 至今） |
| source → Tile IR → PTX → SASS 三向关联 + Tile IR 源码比较 | Nsight Compute **2026.3** | 无对应能力 |
| source-location compiler diagnostics | TileLang **v0.1.13（2026-08-03）** | Span 从 day 1 就有（**领先项，但已被追上**） |
| eager / 并行 autotune | TileLang **2026-03 / 2026-05** | 零 |
| 面向 DSL 的内存 sanitizer | Triton-Sanitizer **ASPLOS'26（2026-03）** | 无 |
| 自动 delta debugging + Pass Diff | TileLang（官方仓库已列） | 有 IR 快照，无 diff/缩小工具 |
| Ascend 上的 DeepSeek V4 kernels | TileLang-Ascend **2026-04-24** | 内部已有 V4 真实作业数据（`Data/DeepseekV4/`） |

---

## 9. 对 PyPTO 3.0 的六条产品结论

每条给出：为什么、对标谁、可验收的指标。

### P0-1　建 autotuner——不是为了跑分，是为了给抽象赌注买保险

- **为什么**：§2 的抽象赌注目前没有对冲；§4 的搜索信号为零；用户在性能不达标时唯一的出路是最贵的那条。
- **对标**：TileLang 的并行 autotune（pipelined / grouped compilation）、cuTile 的 `autotune_launch`。
- **验收**：给定 kernel 与 shape 集合，能在无人工干预下扫出候选配置并给出正确性联合验收；"到达性能目标的实验次数"进入北极星指标的分解项。

### P0-2　关掉 #1305——让 span 接回用户源码

- **为什么**：这是"可信证据层"定位的地基。链路再长，末端断在 `<string>:122:12` 就等于没有。且这条已被对手交付三个版本（§5）。
- **对标**：Nsight 2026.3 的 source→Tile IR→PTX→SASS；TileLang v0.1.13 的 source-location diagnostics。
- **验收**：任一 verifier / perf hint 的诊断，100% 携带用户 Python 文件的 `file:line:col`，并指明控制该维度的 DSL 常量名；同一源码位置的重复提示折叠为一条带计数。

### P0-3　把 `torch_codegen` 产品化为 Correctness Lab 的 oracle 生成器

- **为什么**：这是**已经存在但未变现的最高 ROI 资产**（§4.3 判断一）。它能做到三家都做不到的事——产出用户可读可改可单步的参考实现，而非解释器的黑盒模拟。
- **对标**：`TRITON_INTERPRET` + numpy（Triton）、compute-sanitizer（NVIDIA）。
- **验收**：从"输出不对"到"首个分歧 tensor"的中位时间；每个 ST 失败自动附带一份可执行的 torch 参考与逐层比对结果。

### P1-1　逃生舱重做为"同语法降级 + 一键生成 tile 级骨架"

- **为什么**：§3。业界共识是同语法降级，PyPTO 是唯一需要重写的。而编译器内部已有全部答案（内存空间、L0 分块、搬运序列），把它回写成用户可编辑源码是可行的。
- **对标**：Gluon、`T.Scope("C")`、cuTile→CuTe DSL。
- **验收**：从 `pl.*` 生成的 `pl.tile.*` 骨架数值等价；从"性能不达标"到"拿到可编辑的手工版本"不超过一次命令。

### P1-2　MPMD 叙事精确化为"编排可被验证与归因"

- **为什么**：§6.2。"对手没有编排"这个说法对 NVIDIA 不成立，会在答辩中被击穿。
- **对标**：CUDA Graphs + NCCL + Dynamo。
- **验收**：能对一次多卡运行给出"这条依赖边是哪个 pass 依据哪条规则推出来的"，并关联到用户源码与 runtime 事实。

### P2　把 Triton-Ascend 当漏斗上游，不当竞品

- **为什么**：§1 判断一。Triton-Ascend 的优势是不用学新东西（85% API 覆盖、适配 vLLM/SGLang），这场 PyPTO 结构上赢不了。
- **对标**：Triton-Ascend 3.2.1。
- **验收**：提供"Triton-Ascend 快速原型 → PyPTO 高性能实现"的迁移路径与收益评估（迁移后性能增益的预估与实测对照）。

---

## 10. 三件明确不该做的事

**【建议】**

1. **不要在"tile DSL 好不好用"上与 TileLang 正面比。** 该战场上 PyPTO 抽象更高但逃生舱更窄、无 autotuner、算子覆盖落后，赢面不大。
2. **不要在生态与上手成本上与 Triton 正面比。** CANN OSL 2.0 把分发面钉死在华为硬件上，这是法律约束不是产品选择；20.2k★ 与 `pip install triton` 不可能被追上。正确做法是把对手的入口变成自己的漏斗上游。
3. **不要把 Nsight 或 MindStudio 描述成落后工具。** 它们的弱点是分层位置（事后观测 vs 写码期契约），不是能力缺失。把对手说弱，会让本方所有量化结论的可信度一起下降。

---

## 11. 外部来源清单（2026-09-17 联网核实）

### NVIDIA
- cuTile Python 官方仓库：https://github.com/NVIDIA/cutile-python
- CUDA Tile 产品页：https://developer.nvidia.com/cuda/tile
- `cuda-tile` PyPI：https://pypi.org/project/cuda-tile/
- Nsight Compute 2025.4 更新：https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/updates-2025-4.html
- Nsight Compute 2026.2 更新：https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/updates-2026-2.html
- Nsight Compute 2026.3 新特性：https://developer.nvidia.com/nsight-compute-2026_3-new-features
- Compute Sanitizer：https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/
- CUTLASS / CuTe DSL 调试文档：https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html
- CUTLASS 4.x CHANGELOG（IKET）：https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md

### Triton
- 官方仓库：https://github.com/triton-lang/triton
- Gluon 总览：https://triton-lang.org/main/gluon/index.html
- Gluon 教程（layouts / warp specialization / multi-CTA）：https://triton-lang.org/main/getting-started/tutorials/gluon/intro.html
- 调试文档：https://triton-lang.org/main/programming-guide/chapter-3/debugging.html
- Proton profiler：https://github.com/triton-lang/triton/tree/main/third_party/proton ；论文 https://deep-learning-profiling-tools.github.io/CAT-Lab/publication/proton_cgo2026/
- Triton-Sanitizer（ASPLOS '26）：https://dl.acm.org/doi/pdf/10.1145/3779212.3790241
- Triton-Ascend（GitCode 官方，GitHub 为镜像）：https://gitcode.com/Ascend/triton-ascend ；文档 https://triton-ascend.readthedocs.io

### TileLang
- 上游仓库：https://github.com/tile-ai/tilelang
- Ascend 适配仓库：https://github.com/tile-ai/tilelang-ascend
- PyPI：https://pypi.org/project/tilelang/

### 内部来源
- `repo/pto`（源码与 `docs/zh-cn/dev/`，48 篇 pass 文档、`ptoas-op-status.md`、`03-runtime-dfx.md`、`debug/00-torch_codegen.md`）
- `github_issues/pto/pypto_issues.md`（#1305 原文）
- `Competitive_Analysis/PyPTO对位TileLang_深度对比.html`（2026-09-07）
- `Product_Planning/PyPTO3.0_Toolkit_产品功能规划.md`、`PyPTO3.0_建设必要性与核心价值分析.md`

---

## 12. 待补证据（明确缺口，不假装已有）

以下是本文**无法**从公开信息或仓库内证据得出结论的部分，需实测或内部确认：

1. **同机同 shape 同精度的性能对照**：PyPTO vs TileLang-Ascend vs Triton-Ascend 在 A2/A3 上跑 GEMM 与 FlashAttention。既有材料引用的 "TileLang GEMM ≈0.98× AscendC、融合算子 ≈1.0×" 是对方自述，不是同机对照。
2. **首个 kernel 成功时间**：四家从零环境到 vector add 正确运行的步骤数与耗时。
3. **错值定位时间**：注入同一个 layout / shape / sync 错误，比较 PyPTO（IR dump + DFX）与 Triton（interpreter + sanitizer）定位到首个分歧点的耗时。这是 §4.3 判断一的关键实证，目前只有逻辑推理没有数据。
4. **Triton-Ascend 的 85% API 覆盖在复杂融合算子上的实际可用性**：覆盖率是 API 计数，不等于能写出生产级 MoE / paged attention。
5. **cuTile 的 autotune 成熟度**：目前仅确认它在 experimental 命名空间（`autotune_launch` / `clear_autotune_cache`），实际搜索空间与效果未核实。
6. **PyPTO A5 覆盖的具体缺口清单**：`ptoas-op-status.md` 的快照是 2026-06-23，且未按平台拆分 A2/A3 与 A5 的 ST 覆盖。

---

> 本文的判断部分（【推断】/【建议】）未经评审，不得直接进入对外材料。事实部分（【事实·内】/【事实·外】/【统计】）的核对方法均已在正文给出，可复现。
