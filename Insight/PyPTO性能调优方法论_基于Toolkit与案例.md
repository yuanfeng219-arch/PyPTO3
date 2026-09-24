# PyPTO 性能调优方法论：从证据采集到可复现收益

> 版本：2026-09-24  
> 适用对象：PyPTO 算子、模型子图、Runtime 与 Serving 性能问题。  
> 核心原则：**先确认数字可比，再定位真实阻塞；一次只验证一个假设；收益必须带适用边界。**

---

## 1. 方法论概览

性能调优不是“找最长的 Kernel 然后改代码”。端到端延迟可能来自服务进程不均衡、Host 数据搬运、任务图串行化、调度延迟、核内访存、编译 Pass 或 ISA 同步策略。正确流程是用同一条可追溯证据链逐层排除。

```text
目标与测量契约
  → 干净基线
  → Serving / Host / Device 定界
  → 函数与任务实例定位
  → Swimlane + 关键路径 + 依赖图归因
  → 调度 / 图结构 / 核内 / 编译内存分支下钻
  → 最小修改
  → 成对 A/B、精度与回归验证
  → Recipe 或跨团队交接包
```

调优的成功标准不是“某条指标下降”，而是同时满足：

- 目标计时口径确实改善；
- 改善来自声明的机制，而不是测量噪声或统计范围变化；
- 正确性、精度、资源容量和关键 shape 没有不可接受的回归；
- 结论写明平台、版本、shape、并行度和数据条件。

## 2. 证据层与工具分工

PyPTO3 Toolkit 是一个**读取、关联和可视化已有产物**的工具，不负责采集运行记录；`CPM_static*.json` / `CPM_observed*.json` 也需由关键路径工具预先生成。

| 证据层 | 主要输入 | 回答的问题 | 不能单独证明什么 |
| --- | --- | --- | --- |
| 服务层 | `serving-strace-swimlane.json`、端到端 benchmark | 哪个 WorkerProcess 负载或尾耗时异常？ | AICore 内某个 Kernel 为什么慢 |
| Host / Device | `BenchmarkStats`、独立 benchmark | 延迟是 Host、Device，还是两者共同贡献？ | Device 内的依赖和 pipe 根因 |
| 函数汇总 | `name_map*.json` + Swimlane | 慢来自单次慢、次数多，还是波动？ | 是否影响 wall-clock |
| 运行时执行 | `chip_swimlane_records.json` | Task 何时 dispatch、start、finish；Core 是否空转？ | 图的最短延迟下界 |
| 关键路径 | `CPM_static*.json`、`CPM_observed*.json` | 哪些 Task 决定 makespan，等待来自哪里？ | 内核内部哪条硬件 pipe 受限 |
| 依赖图 | `deps.json`、`name_map*.json` | 哪些边造成串行、哪些边可审计为冗余？ | 去边后一定有可见收益 |
| 编译过程 | `passes_dump*`、Pass 日志 | 哪个 Pass 改了 IR、产生了 warning？ | 该变化一定是性能根因 |
| 片上内存 | `*after_AllocateMemoryAddr.py` | Tile 的地址、生命周期、复用和容量约束 | MemoryReuse 本身就是错误 |
| 核内执行 | InCore simulator trace、instruction metrics | MTE / Vector / Cube / Scalar 哪条 pipe 受限？ | 该单核合成结果必然等于端到端收益 |

### 2.1 推荐的产物目录

关联文件应放在同一轮、同一配置的输出目录中：

```text
<case>/
├── dfx_outputs/
│   ├── chip_swimlane_records.json
│   ├── deps.json
│   ├── name_map_<timestamp>.json
│   ├── CPM_static.json              # 可选，关键路径工具生成
│   └── CPM_observed.json            # 可选，关键路径工具生成
├── serving-strace-swimlane.json     # 若分析 Serving
└── passes_dump_<timestamp>/
    ├── 00_frontend.py
    ├── NN_<PassName>.py
    ├── NN_<PassName>.log            # 可选
    └── *_after_AllocateMemoryAddr.py
```

缺少 `deps.json` 时泳道不能恢复依赖连线；缺少 `name_map*.json` 时函数名会降级为 ID；缺少 CPM 文件时不能用关键路径高亮。不要把不同运行、不同 shape 或不同版本的文件混放后一起解读。

---

## 3. 标准调优流程

### 步骤 0：冻结目标与测量契约

**需要看到**

- 业务目标：TTFT、TPOT、单算子时延、吞吐、P99 或成本中的哪一个；
- case manifest：shape、dtype、batch、序列长度、并行配置、输入分布；
- 环境指纹：设备/平台、Runtime、CANN、PyPTO、编译选项、代码版本；
- 正确性门槛：golden、`rtol` / `atol`、容许的确定性变化；
- 计时定义、warm-up、重复次数、统计量和基线样本。

**为什么看**

不同范围的数字不能比较。例如“attention 对 attention”与“attention 对整层”、含 Host 复制与只测 device、不同 tiling 或不同 warm-up，都可能制造虚假的倍数差。

**如何分析**

逐项比对 A/B：被测工作是否相同、参考是否相同、计时窗口是否相同、输入是否固定、统计规则是否一致。若任一项不同，结果只能是探索性观察，不能写为性能收益。

**下一步**

冻结 manifest 和 baseline。调优过程中若必须增加 shape 或改计时范围，应新建版本而非覆盖原基线。

### 步骤 1：先把问题定界到 Serving、Host 或 Device

**需要看到**

- Serving Strace 中各 `WorkerProcess` 的任务数、平均、最大和最小耗时；
- 独立 benchmark 中的 `host_wall_us`、`device_wall_us` 与端到端时间；
- 是否存在每轮重复 bind/H2D、编译/注册、结果拷回或不均衡 worker。

**为什么看**

Device 侧调优不能解决 Host 的重复上传或服务调度失衡。反之，端到端指标中 Host 很小，优先优化服务只会分散注意力。

**如何分析与下一步**

| 看到的信号 | 判断 | 下一步 |
| --- | --- | --- |
| `host_wall_us` 主导 | 数据搬运、注册、Host 编排是主要成本 | 常驻 weights/KV cache/workspace，register-once、dispatch-many，再测 Host/Device 是否按预期分离 |
| 个别 WorkerProcess 的 Count 或 Max 明显异常 | Serving 工作分配或尾部阻塞 | 排查请求分配、队列和对应 worker 的下游依赖 |
| `device_wall_us` 主导 | 问题在设备执行路径 | 采集 Chip Swimlane 与依赖图 |

> 不要用开启 Swimlane 的 DFX 运行作为正式 wall-clock 基线。Runtime 文档说明该采集会引入额外过程与观测扰动；正式 A/B 应来自独立的干净 benchmark。

### 步骤 2：用函数性能表筛选候选，而不是直接下结论

**需要看到**

- Function Name、Count、Max Duration、Min Duration、Avg Duration；
- 对应任务的 Setup 占比和实际实例所在 Core；
- 同一函数在不同 shape/round 的分布。

**为什么看**

函数汇总排除本地 Setup，只统计 Kernel 执行时间，适合建立候选池；但函数总耗时不等于端到端贡献。

**如何分析与下一步**

| 看到的信号 | 初步解释 | 下一步 |
| --- | --- | --- |
| Avg 高、Count 高 | 累积执行量大 | 在 Swimlane 搜索函数，确认是否在关键路径 |
| Avg 高、Count 低 | 少数大任务 | 看单任务 pipe、tiling 或依赖前置间隙 |
| Count 高、每次很短 | 任务粒度可能过细 | 看 `dispatch→start`、连续单依赖链与 Scheduler phase |
| Max 远高于 Avg | 存在抖动、条件路径或资源竞争 | 定位最慢实例，查看输入、依赖、Core 和前驱 |

### 步骤 3：在 Chip Swimlane 中还原“实际发生了什么”

**需要看到**

- Worker View：AIC/AIV 任务的 `start/end`、条宽、Core 空闲、Setup、SPMD 边界、fan-in/fan-out；
- Scheduler View：`dispatch→finish`，尤其是 `dispatch→start` 间隔；
- AICPU Scheduler：`dispatch`、`early_dispatch`、`resolve`、`release`、`drain` 等阶段；
- AICPU Orchestrator：提交 envelope 与 Scheduler 的衔接；
- 性能面板：Kernel 统计、连续单依赖链、关键路径、Gap/Blocker、Early Dispatch。

**为什么看**

编译期 `perf_hint` 说明的是编译器怀疑什么；泳道说明该轮运行中真正执行、等待和空转了什么。

**如何分析与下一步**

| 看到的信号 | 解释 | 下一步 |
| --- | --- | --- |
| 一个宽任务、前后几乎无空隙 | 可能是核内瓶颈 | 进入 InCore / roofline / 访存分析 |
| 许多短任务串成台阶 | 图或粒度限制并行度 | 看连续单依赖和 `deps.json` |
| Core 空闲，仍有 ready 任务未启动 | Scheduler 未及时派发 | 看 Scheduler phase、任务数量和可合并链 |
| `dispatch→start` 大 | pickup、调度或资源等待 | 结合 CPM 区分 core-wait / front-gap |
| Setup 区域占比高 | 本地 receive-to-start 准备成本高 | 优先检查任务粒度、局部准备和重复初始化 |
| 同名 SPMD 整体边界很长且 Core 不均衡 | SPMD 规模或工作分配不均 | 按 SPMD 组检查 block 数和尾部 Core |

用搜索、依赖高亮和时间参考线把“某个统计条目”变成“某次具体执行”。没有回到具体实例的函数级结论，仍然只是候选假设。

### 步骤 4：以关键路径决定优化优先级

**需要看到**

- Static CPM：假设 Core 无限时，由依赖本身决定的最长路径；
- Observed CPM：从最后完成任务反向追溯的实际执行路径；
- makespan、compute/stall 占比、`data-wait`、`core-wait`、`front-gap`；
- 验证信息：采集完整性、`tiling check: exact`、函数名解析、rank 覆盖、独立 wall 对照。

**为什么看**

只有路径上的缩短才直接缩短该 rank 的 makespan。函数很热但完全在路径之外时，可能提升吞吐或 Core 利用率，却不改变当前 wall。

**如何分析与下一步**

| 关键路径信号 | 结论 | 处置 |
| --- | --- | --- |
| Static CPM 接近 makespan | 依赖受限，图就是时延下界 | 缩短链、拆长任务、减少 fan-in 串行 |
| Static CPM 明显低于 makespan 且 stall 高 | 运行时执行损失超过依赖下界 | 按 stall 类型继续归因 |
| `data-wait` 主导 | 当前任务在等上游生产者 | 优化命名出的上游任务，而非等待者 |
| `core-wait` 主导 | 资源序列化或 Core 分配不均 | 扩宽工作、重平衡、降低竞争 |
| `front-gap` 大 | 首任务发射/Host/编排延迟 | 排查 Host、注册、提交与 orchestration |
| compute 高、stall 低 | 真正 compute-bound | 优化路径上 compute share 高的 Kernel family |

Static 与 Observed 路径服务于不同问题：前者说明如何降低理论依赖下界，后者说明当前运行被什么实际阻塞。报告必须标明改动作用于哪一条路径。

### 步骤 5：审计依赖，解除伪串行但不破坏正确性

**需要看到**

- `deps.json` 的任务、Tensor、前驱/后继、`block_num`、shape/stride/offset；
- Full、Reduced、Omitted、Reduced DF、Omitted DF 五种图模式；
- 边来源 `explicit`、`tensormap`、`creator`，以及 DAG 深度；
- 连续单依赖分析和 Early Dispatch 状态。

**为什么看**

共享 GM 句柄、WAW、循环携带依赖和保守 dataflow 推断，可能让“很多任务”在很少的 Core 上串行执行。但删掉真实边会造成 race 或数据错误。

**如何分析**

- 先看 Full，确认原始图和任务类型；再看 Reduced/Reduced DF 聚焦可审计边；Omitted 模式用于查看被省略的边。
- 深度为 1 时不存在两跳路径，不能有传递冗余边，结束此项审计。
- `creator` 是生命周期保持边；有任何 `creator` 时，普通 Reduced 的 `0` 不是“图已最小”的证据，必须再看 Reduced DF。
- 图上的 `🔥` / `⭐` 表示提前派发的结构资格；它**不等于**本次运行已经提前派发成功。实际情况应回到 Scheduler View 的阶段记录验证。

**处置顺序**

1. 连续单入度/单出度链：评估融合、循环内化或 mixed kernel；
2. 同一 buffer 但写入区域实际不重叠：以最窄粒度使用 `no_dep_args`、`manual_dep=True` 或显式依赖；
3. 读任务被错误建模为 `INOUT`：恢复真实方向，改由 writer 显式控制依赖；
4. 需要同一图但更早派发：确认所有前驱条件后使用 `allow_early_resolve`；
5. 每次解除依赖后都重跑正确性和 CPM，绝不只看图更“漂亮”。

### 步骤 6：把核内热点拆成硬件 pipe 问题

**需要看到**

- InCore profile 的 `summary.txt`、`manifest_export.csv`、`instr_metrics.json`、清洗后的 trace；
- MTE2、MTE1、CUBE、VECTOR、FIXPIPE、MTE3、Scalar/Synchronization 的 cycle；
- L2 命中、访存事务形状、tile 大小、指令数量和资源冲突；
- 计算下界、带宽下界与实际利用率。

**为什么看**

“Vector 算子”不代表 Vector 是瓶颈；大量 Vector cycle 也可能只是搬运和转置。单 Kernel 时延需要知道哪条 pipe 受限，才能选择正确杠杆。

| 看到的信号 | 常见机制 | 优先处置 |
| --- | --- | --- |
| MTE2 / MTE3 主导、短 burst 多 | 访存碎片、非连续内维、搬运无法重叠 | 连续化布局、增大连续维 tile、块读取、双缓冲 |
| VECTOR 主导且利用率低 | tile/并行度不匹配或指令冗余 | 调整 tile、并行粒度、消除循环内重复控制指令 |
| CUBE 主导 | 算术或矩阵形状受限 | 调整 M/N/K tile、考虑 split-K 或算法变体 |
| Scalar / sync 主导 | mask、循环控制、同步过密 | 提升循环不变量、减少同步、检查自动插入的 fence |
| pipeline 有明显空洞 | load/compute/store 未重叠 | 检查至少两项可流水工作，增加合法双缓冲 |

采集成功不等于采到了预期工作：matmul 的 CUBE cycle 为零、混合核只出现单类 pipe、指令 CSV 几乎全是同步/Scalar 时，应先检查 profile 输入和控制参数，而不是宣布 Kernel 很快。

### 步骤 7：用 IR Pass Trace 与 Memory Map 验证“意图是否生效”

**需要看到**

- IR Pass Trace 中 Changed / No-op Pass、warning、前后快照的函数级 diff；
- `*after_AllocateMemoryAddr.py` 的 Memory Map：地址横轴、生命周期纵轴、Tile 创建/使用位置；
- Swimlane 中对应等待、任务和时间区间。

**为什么看**

源码写法到 Runtime 行为之间会经过多个 Pass。一个优化意图可能被忽略、被重写，或被内存复用和同步插入抵消。

**如何分析与下一步**

1. 在 IR Trace 先过滤 Changed，定位可疑函数和有 warning 的 Pass；
2. 将 Swimlane 中的等待/任务锚定到函数或源码位置，比较异常前后 Pass；
3. 在 Memory Map 看相关 Tile 是否地址复用、生命周期是否过长、增加 pipeline stage 是否有容量；
4. 若首个异常出现在编译 Pass、后端或 ISA，停止在上层猜测，整理最小复现、环境指纹、pass diff、泳道和已排除项交接。

MemoryReuse 不是天然错误：它可能降低空间占用，也可能引入 WAR/同步并切断流水。结论必须同时包含时序证据与地址/生命周期证据。

### 步骤 8：测量资源上限并谨慎调容量

**需要看到**

- `scope_stats.jsonl` 的 `task_window`、heap、dep-pool、tensormap 峰值；
- 每个 ring 的容量与峰值；
- Memory Map 中的片上 Tile 生命周期和重叠。

**为什么看**

更深流水、更大 tile、更多 in-flight task 都会消耗有限的片上或 Runtime 资源。直接“把 ring 调大”可能掩盖 scope 放置失衡，并放大内存成本。

**如何分析与下一步**

- 峰值紧贴容量：该 ring 是实际约束，才考虑调 `ring_task_window`、`ring_heap` 或 `ring_dep_pool`；
- 仅单 ring 低效/高压：先重平衡 scope，再扩大容量；
- Tile 存活跨越不必要语句：缩短生命周期或重新排序；
- 流式、单次读取的权重污染缓存：在满足一致性前提下评估 `CachePolicy.BYPASS`。

所有容量调整必须在新的 `scope_stats` 中验证“原来顶到上限的峰值不再顶住”，并同时测量性能与内存代价。

### 步骤 9：以成对实验完成验证与交付

**需要看到**

- 同环境、同 manifest、同计时规则的 baseline / candidate 多次样本；
- 目标指标、Host/Device 拆分、关键路径变化、核心 busy 或 pipe cycle；
- 全量正确性和至少关键相邻 shape 的回归；
- 不受改动影响的大 scope 或指标作为噪声锚点。

**为什么看**

单次 DFX/benchmark 是单样本。只看最好的 wall，容易把随机波动、PMU 干扰或统计范围变化写成收益。

**验收规则**

1. 同 session 或等价环境成对运行 A/B，采用预先约定的中位数或稳健统计；
2. 目标指标改善，同时机制指标与假设一致，例如 MTE2 cycle 降低、关键路径缩短或 Host wall 降低；
3. 正确性通过，且没有未声明的 shape、精度、内存或吞吐回归；
4. 结果写清“对什么有效、对什么无效”，小 shape 无收益或回归不是失败数据，应保留；
5. 无稳定收益时，明确结论为“当前配置下未证实”，不要把候选手段写成 Recipe。

---

## 4. 常见症状到行动的速查表

| 症状 | 首选证据 | 典型根因 | 优先行动 |
| --- | --- | --- | --- |
| Serving P99 高且部分 worker 异常 | Serving Strace | 请求分配不均、下游 worker 阻塞 | 分析 WorkerProcess 分布与尾任务 |
| 端到端高、Device 很低 | Host/Device benchmark | H2D、bind、注册、D2H | 常驻数据、注册一次、多次 dispatch |
| 函数 Count 高、单次极短 | 函数表 + Scheduler | 任务过细、派发成本 | 合并、循环内化、SPMD |
| Core 空闲但任务看似很多 | Worker + deps + CPM | WAW、保守依赖、资源串行 | 审计图、明确真实依赖、重平衡 |
| 调度前间隙大 | Scheduler + CPM | front-gap、pickup、Host 编排 | 优化提交路径或 early resolve 条件 |
| 关键路径在等上游 | Observed CPM | `data-wait` | 优化生产者而非等待者 |
| Kernel 很宽 | InCore trace | 计算、访存或同步 pipe 受限 | 按 pipe 选择 tile/布局/算法/同步方案 |
| MTE2 多个短事务 | InCore + perf hints | 内维不连续、短 burst、散读 | 连续化、增大连续维、块读取 |
| 试图加深流水但编译失败/变慢 | Memory Map + scope stats | Tile 空间或 Runtime ring 容量不足 | 先缩短生命周期/重平衡，再调容量 |
| 源码改动未带来预期行为 | IR Trace + Swimlane | Pass 重写、提示未处理、后端限制 | 追溯首个异常 Pass，构造最小复现 |

## 5. 案例提炼出的规则

### 5.1 HCA `softmax_pool`：热点必须先成为路径热点

仓库案例中，`softmax_pool` 在 HCA 形态处于高占用并影响整图，而在另一个压缩比例场景中不在关键路径。核内 trace 进一步表明主要耗时是 MTE2 散读和转置搬运，不是 softmax 算术。

**规则**：函数排名只用于筛选；是否优化、优化谁，必须由当前 workload 的关键路径确认。

### 5.2 Softmax mask lifting：收益必须带 shape 条件

A5 softmax 案例中，将重复 mask 更新提升出寄存器循环后，大 shape 的任务时长下降，最小 shape 却轻微回归。

**规则**：每条 Recipe 都要记录正例、反例、形状阈值、平台和原因；“对大 shape 有效”不是“全局有效”。

### 5.3 “2.3× 差距”校准：比较资格先于结论

案例中的巨大差距经统计范围校准后接近持平。

**规则**：在代码、图或 Kernel 修改前，先完成“同输入、同范围、同计时、同环境”的比较资格检查。

### 5.4 依赖删除与提前派发：结构资格不等于实测收益

依赖图的边模式和 Early Dispatch 标记给出的是结构信息；它们不证明 Runtime 已经提前执行，也不证明去边会缩短 wall。

**规则**：任何依赖/调度修改都要同时通过正确性测试、Scheduler/Worker 复查和独立 A/B 测量。

---

## 6. 每轮调优的最小交付模板

```markdown
### 调优轮次：<名称>

- 目标：<TTFT / TPOT / device wall / Kernel duration>
- 适用范围：<model、shape、dtype、并行度、平台、版本>
- 基线：<计时口径、样本、统计方法、正确性基线>
- 证据：<dfx_outputs、CPM、deps、passes_dump、InCore trace 路径>
- 观察：<事实，不写推测>
- 判断：<依赖 / 调度 / Host / 访存 / 计算 / 编译内存>
- 假设：<一个可证伪的机制>
- 改动：<最小代码或配置改动>
- 验证：<A/B、精度、关键路径或 pipe 指标、回归 shape>
- 结论：<确认 / 否定 / 未证实 / 需上游处理>
- 边界与风险：<不适用条件、已知回归、内存/正确性义务>
- 下一步或交接对象：<具体行动>
```

## 7. 资料依据

### PyPTO3 Toolkit 中文文档

- [Toolkit 文档首页](../repo/pypto-tools/docs/zh/index.md)
- [准备分析文件](../repo/pypto-tools/docs/zh/getting-started/data-files.md)
- [Chip Swimlane](../repo/pypto-tools/docs/zh/runtime/chip-swimlane.md)
- [任务依赖图](../repo/pypto-tools/docs/zh/runtime/dependency-graph.md)
- [函数性能表](../repo/pypto-tools/docs/zh/runtime/function-performance.md)
- [Serving Strace Swimlane](../repo/pypto-tools/docs/zh/runtime/serving-strace-swimlane.md)
- [IR Pass Trace](../repo/pypto-tools/docs/zh/compiler/ir-pass-trace.md)
- [内存复用分析](../repo/pypto-tools/docs/zh/compiler/memory-reuse.md)

### Runtime 文档、技能与案例

- [Reading the Swimlane](../repo/pypto/docs/en/user/performance/00-swimlane.md)
- [Task Granularity](../repo/pypto/docs/en/user/performance/01-task-granularity.md)
- [Runtime Overhead](../repo/pypto/docs/en/user/performance/02-runtime-overhead.md)
- [Managing Dependencies](../repo/pypto/docs/en/user/performance/03-dependencies.md)
- [Tuning the InCore Function](../repo/pypto/docs/en/user/performance/04-incore.md)
- [Memory](../repo/pypto/docs/en/user/performance/05-memory.md)
- [Host](../repo/pypto/docs/en/user/performance/06-host.md)
- [critical-path-analysis skill](../repo/pypto-skills/plugins/pypto-user/skills/critical-path-analysis/SKILL.md)
- [dependency-redundancy skill](../repo/pypto-skills/plugins/pypto-user/skills/dependency-redundancy/SKILL.md)
- [incore-profiling skill](../repo/pypto-skills/plugins/pypto-user/skills/incore-profiling/SKILL.md)
- [generate-ir-trace skill](../repo/pypto-skills/plugins/pypto-user/skills/generate-ir-trace/SKILL.md)
- [PyPTO 性能调优典型用户案例](PyPTO性能调优典型用户案例_基于GitHub_20260923.md)
- [softmax 性能报告](<../Data/softmax 新研pto pro agent产物数据/PERFORMANCE_REPORT.md>)

> 本文的案例数字仅用于说明方法。性能结论始终绑定具体的硬件、软件版本、shape、输入分布和测量口径，不构成通用性能承诺。
