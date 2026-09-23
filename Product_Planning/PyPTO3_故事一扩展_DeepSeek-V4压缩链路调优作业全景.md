# 故事一扩展：DeepSeek-V4 压缩链路调优作业全景

> 整理日期：2026-09-23
> 上位文档：[PyPTO3 性能调优主线故事](./PyPTO3_性能调优主线故事_串联UX关键设计对象.md) 中的“故事一”
> 本文目的：把故事一拆成 9 个关键任务。每个任务说明用户**实际怎么做**（用什么工具、看什么内容）、**已有哪些原始数据**（字段和本地实测值）、**工具和界面应当提供什么**，以及 **UX 需要重点设计什么**。

---

## 一、阅读说明

### 1.1 标注约定

| 标注 | 含义 |
| --- | --- |
| **【GitHub】** | 来自公开 Issue/PR 或社区调优日志，出处写在同一行 |
| **【本地】** | 来自仓库 `Data/` 下的真实构建产物，数值由本次统计得出 |
| **【推断】** | 根据多个来源推断出的做法，来源没有明说 |
| **【设计】** | 产品方案建议，不是已有能力 |

### 1.2 两份本地样本

| 样本 | 路径 | 内容 | 与故事一的关系 |
| --- | --- | --- | --- |
| **S1：CSA decode 构建** | `Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/`（72 MB，391 个文件） | 2 个 rank 的泳道、依赖图、52 个 pass dump、124 个 `.pto`/`.cpp`、230 条性能提示、编排代码、replay 脚本 | 包含故事一中的 `kv_score_proj`、`scatter_softmax_pool`、`compress_state_commit` 等 kernel。它是 **9 月 DSpark 版本**，比 6 月的调优晚，因此可以看到调优**之后**的状态 |
| **S2：decode_fwd 构建** | `Data/_jit_decode_fwd_layers_20260625_184941/`（33 MB） | 泳道记录、依赖图、42 个 pass dump、**内存分配报告**、性能提示 | 提供 S1 缺少的内存报告格式；已有分析见 [实际检测报告](../Insight/_jit_decode_fwd_layers_20260625_184941_实际检测报告.md) |
| 模型源码 | `Data/DeepseekV4/deepseek_v4_flash_dspark/*.py` | 包括 `decode_compressor_ratio4.py` 等 | 源码中保留了调优成果和原因注释（见 T9） |

### 1.3 主角与目标

- **主角**：模型开发者（复合角色）。熟悉 PyPTO DSL 和模型结构，不是编译器或运行时专家。
- **调优对象**：DeepSeek-V4 decode 的 compressor 家族，包括 `compressor_ratio128`、`decode_indexer_compressor`、`decode_compressor_ratio4`，以及它们所在的 CSA/HCA 注意力子图。
- **目标形态**【GitHub】：在不改变对外 API（分页 block table 等）、精度逐位一致或在容差内的前提下，降低 decode 子图的 wall。

---

## 二、任务全景

```text
T1 建任务、立基线 ──► T2 整图定位热点 ──► T3 诊断调度与依赖 ──► T4 修正源码结构
                                                                     │
T9 交付与沉淀 ◄── T8 实验与验证 ◄── T7 核内下钻 ◄── T6 硬件资源预算 ◄── T5 核对编译过程
        ▲                                   │
        └─────── 每次修改都会回到 T8，再回到 T2 看热点是否转移 ──┘
```

| 任务 | 用户要回答的问题 | 主要原始数据 | 主要工具（现状） | 典型耗时（推断） |
| --- | --- | --- | --- | --- |
| T1 建任务、立基线 | 现在多快？怎么测才算数？ | STRACE、泳道汇总、环境版本 | 终端、手写表格 | 半天 |
| T2 整图定位热点 | 时间花在哪个 scope？哪段时间核是空的？ | `merged_swimlane`、`chip_swimlane_records`、`name_map` | Perfetto、临时脚本 | 1~2 天，反复进行 |
| T3 诊断调度与依赖 | 是派发开销，还是依赖把并行变成了串行？ | 调度阶段记录、`deps.json`、泳道 flow 事件 | `deps_viewer`、Perfetto | 1~3 天 |
| T4 修正源码结构 | 是哪一行写法造成的？怎么改？ | 源码、`deps.json`、泳道 | 编辑器、泳道 | 数小时 |
| T5 核对编译过程 | 我的意图生效了吗？编译器提示了什么？ | `passes_dump`、`.pto`、kernel `.cpp`、`perf_hints.log` | 文本对比、grep | 数小时~1 天 |
| T6 硬件资源预算 | 再加大一档放得下吗？搬运是什么形态？ | 内存报告、`.pto` 的 `alloc_tile`、编译错误 | 编译试错 | 数小时，反复进行 |
| T7 核内下钻 | 单个 task 时间花在搬运还是计算？ | op-sim `*.clean.json`、PMU | `msprof op simulator`、TraCR | 数小时~1 天 |
| T8 实验与验证 | 改动真的有效吗？是噪声吗？精度对吗？ | 多次运行的泳道、golden 比对结果 | 手工多跑、表格 | 每轮 0.5~1 天 |
| T9 交付与沉淀 | 改动会影响谁？经验在什么条件下适用？ | PR、CI 日志、源码注释 | GitHub、CI | 数小时~数天 |

---

## 三、原始数据总表

下表列出当前真实存在的数据、字段和本地实测值，是后续设计界面的依据。

| # | 数据 | 产生方式 | 关键字段 / 结构 | 本地实测（S1 rank0，除非另注） |
| --- | --- | --- | --- | --- |
| D1 | `merged_swimlane_*.json` | `--enable-l2-swimlane` 采集后由 `swimlane_converter` 合并，格式为 Perfetto trace | 4 个进程：AICPU Orchestrator、AICPU Scheduler、Scheduler View、Worker View；`X` 事件的 `args` 包含 `event-hint`（Task/FuncId/CoreId）、`fanin-hint`、`fanout-hint`、`duration-us`、`kernel-duration-us`、`local_setup_us`；`s`/`f` 为依赖 flow；`C` 为 `shared_ready_queue` 计数（AIC/AIV/MIX） | 36,336 个事件（X 9,480、flow 12,745 对、C 1,128）；Worker View 中有 4,081 条 AICore 任务；66 个 scope；wall 4,879.8 µs |
| D2 | `chip_swimlane_records.json` | 同上，未转换的原始记录 | `chip_swimlane_level`（1~4）；`metadata.clock_freq_hz`、`num_cores`、`core_types`、`core_to_thread`；`aicore_tasks[core, task_id, seq, start, end, …]`；`aicpu_tasks`；`aicpu_scheduler_phases`（kind=dispatch/complete、loop_iter、tasks_processed、pop_hit/pop_miss）；`aicpu_orchestrator_phases`（submit_idx、task_id、start/end） | level 4；50 MHz；72 核（24 AIC + 48 AIV）；4,038 条 aicore 记录 |
| D3 | `deps.json` | dep_gen 抓图（onboard 平台上单独跑一遍） | `tasks[task_id, scope, early_dispatch, kernel_ids, block_num, args[type, tensor_id, dtype, shape, strides]]`；`tensors[tensor_id, buffer_addr, version, buffer_numel]`；`edges[pred, succ, source, flags(wait/retain), tensor_id, consumer_shape]` | 86 个 task、138 个张量、338 条边 |
| D4 | `name_map.json` | 编译时生成 | `callable_id_to_name`，把 FuncId 映射到 kernel 名 | 如 `24 → kv_score_proj`、`25 → scatter_softmax_pool` |
| D5 | `host.*.log`（STRACE） | 运行时 host 计时 | 层级 span：`chip.run` → `bind`（args/prebuilt）→ `runner_run` → `device_wall`（preamble/graph_build/orch/sched/post_orch）→ `validate`；带 `inv` 调用序号 | rank0 第 1 次调用 device_wall **45.19 ms**，第 2 次 **5.13 ms**；rank1 分别为 3.61 / 4.02 ms（第 1 次并不慢）；rank0 第 1 次 `bind.prebuilt` 187 ms |
| D6 | `passes_dump/NN_after_<Pass>.py` | 编译时逐 pass 导出 IR | 可读的 Python 形式 IR，带 `pl.MemRef(...)` 注解 | 52 个文件、35 MB，每个约 1.1 MB；`kv_score_proj` 位于 `35_after_MemoryReuse.py:2229` |
| D7 | `ptoas/*.pto` | codegen 生成的 PTO IR | `pto.alloc_tile ... loc=acc/right/mat/vec, dtype, rows, cols, blayout, slayout, fractal`；`pto.textract` 等 | 124 个文件；`kv_score_proj.pto` 50.8 KB；Right tile 为 `bf16[256,64] blayout=row_major slayout=col_major`（即 b_trans 后的 ZN 形态） |
| D8 | `kernels/aic|aiv/*.cpp` | ptoas 生成的 kernel C++ | `TLOAD/TEXTRACT/TMATMUL(_ACC)/TSTORE`、`set_flag/wait_flag/pipe_barrier` | `kv_score_proj.cpp` 33 KB：6 TLOAD、24 TEXTRACT、2 TMATMUL、10 TMATMUL_ACC、2 TSTORE；33 对 set/wait_flag、7 个 pipe_barrier |
| D9 | `report/perf_hints.log` | 编译 pass 输出的性能提示 | `[perf_hint <规则 ID>] <Pass>: <说明> at <文件>:<行>:<列>` | 230 条：`PH001` 197 条（最内维低于 512B）、`PH-MR-001` 33 条（流水深度放不下）；其中 `decode_compressor_ratio4.py:110`（即 `kv_score_proj`）有 5 条 PH-MR-001 |
| D10 | `report/memory_after_AllocateMemoryAddr.txt`（S2） | 编译 pass 输出 | 按函数列出每个存储空间的 Used / Limit / Usage / MemRefs，以及每个 buffer 的大小、地址区间、生命周期区间 | 38 个函数；**Vec 上限 184.0 KB**、Mat 512 KB、Right 64 KB；`down_proj` 的 Right 为 100% |
| D11 | `kernel_config.py` | 编译时生成 | `RUNTIME_CONFIG`（runtime=tensormap_and_ringbuffer）；`KERNELS[func_id, name, source, core_type, signature]` | `func_id 24 kv_score_proj aic [IN,IN,IN,OUT,OUT]` |
| D12 | `debug/run.py` | 编译时生成的 replay 脚本 | CLI：`--swimlane-level 1~4`、`--pmu`、`--dump-args`、`--dep-gen`、`--no-rebuild-from-pto`；`_inline_inputs()` | **输入为随机数，动态维被填为 1**（注释提示需手工修改） |
| D13 | `distributed_meta.json` | 编译时生成 | `params[name, direction, shape, dtype]` | 动态维用 `-1` 表示 |
| D14 | 模型源码 `*.py` | 用户编写 | `pl.spmd/pl.at/pl.pipeline/pl.parallel`、`name_hint`、`deps=[...]` | `decode_compressor_ratio4.py:108-133` 保留了“调度让位”和 b_trans 的原因注释 |
| D15 | op-sim `*.clean.json` | `msprof op simulator` 加 incore-profiling skill【GitHub，§19/§20】 | 按指令、按单元（MTE1/MTE2/MTE3/VECTOR/CUBE）列出周期和笔数 | **本地没有**；GitHub 案例中有数值 |
| D16 | `pmu.csv` | `--enable-pmu N` | 各 MTE 通道忙碌周期、`main_read_req` 等 | **本地没有**；pypto-lib#622 中有数值 |
| D17 | `scope_stats.jsonl` | `--enable-scope-stats` | 按 scope 聚合的统计 | **本地没有** |
| D18 | golden 比对结果 | 测试 harness | `ratio_allclose`、`max_error_ratio`、各输出 PASS/FAIL | 本地 S1 没有 `golden.py`，只有 `_user_compare` 占位 |

> **本地重要发现**【本地】：
> - D5 显示 rank0 第 1 次调用的 device_wall（45.19 ms）是第 2 次（5.13 ms）的约 9 倍，而 rank1 第 1 次并不慢；两个 rank 第 2 次调用的值也相差约 28%（5.13 ms vs 4.02 ms）。**不区分调用次序、不区分 rank 的数字不可比，而且冷启动是否出现本身也不稳定。**
> - D10 中的 Vec 上限是 184 KB，而调优日志里常说“UB 192KB”。**硬件规格与编译器实际可用上限不同**，用户容易拿错数字。

---

## 四、关键任务详解

### T1 建立调优任务与可信基线

| 项 | 内容 |
| --- | --- |
| 目标 | 明确“调什么、以什么为准”，得到一份后续每次对比都引用的基线 |
| 进入条件 | 模型子图能跑通，精度通过 |
| 完成标志 | 有一份包含环境指纹、主指标、kernel 排行、开销构成和测量协议的基线记录 |

**A. 用户作业过程（现状还原）**

| # | 用户动作 | 工具 / 界面 | 看什么 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | 选定调优对象和配置（B=64、S=1、FLASH 配置） | 编辑器、模型配置文件 | shape、batch、平台 | 【GitHub】#314 |
| 2 | 记录版本 | 终端 `git log`、`pip show` | pypto-lib / pypto / simpler / ptoas / CANN 的 commit 和版本 | 【GitHub】#314、#2040 的环境表 |
| 3 | 跑一次并采集泳道：`python models/.../compressor_ratio128.py -p a2a3 --enable-l2-swimlane` | 终端 | 终端末尾的汇总：Total Test Time、Total Tasks、Exec/Latency | 【GitHub】#314 |
| 4 | 把汇总和每个 kernel 的 Exec/Latency 手工抄进 Markdown 表 | Markdown 编辑器 | 12 个 kernel 的次数、平均 Exec、平均 Latency | 【GitHub】#314 |
| 5 | 计算调度开销分布（Head OH、Tail OH 的 P50/P75/P95） | 终端输出或临时脚本 | Tail OH 727.8 µs，占 28.3% | 【GitHub】#314；【推断】部分来自工具汇总 |
| 6 | 写测量协议：以哪个指标为准、失败时怎么办 | Issue 正文 | 主指标、回退规则 | 【GitHub】#314 |
| 7 | 判断基线是否可信：冷启动、rank 差异、设备是否被共享 | 看 STRACE 日志（很少有人这样做） | `inv=1` 与 `inv=2` 的 device_wall | 【本地】D5；【GitHub】#465 做了 3 次运行和 CoV |

**B. 现状卡点**
- 基线是一份手抄的表格，与原始数据没有链接；换一台机器或一个版本后无法复核。
- 冷启动、rank 差异、设备共享等影响可信度的因素要靠个人经验判断。本地 D5 中冷热相差 9 倍，但界面上完全不可见。
- 测量协议写在 Issue 正文里，工具不会执行，也不会检查。

**C. 可用的原始数据**：D5（STRACE 分层耗时，带调用序号）、D1/D2（泳道汇总）、D11（运行时配置）、D13（参数 shape）、环境版本（需要额外采集，构建目录里没有统一的环境指纹文件）。

**D. 工具 / 界面应提供的内容**
1. **环境指纹卡**：pypto、pypto-lib、simpler、ptoas、pto-isa、CANN 的版本，平台、设备号、runtime 类型（D11），以及与 CI pin 版本的差异。
2. **基线卡**：
   - 主指标（用户选定，例如 device_wall、Total Test Time 或子图 wall）；
   - 次指标（busy、task 数、Head/Tail OH）；
   - 采样条件（次数、是否排除冷启动、是哪个 rank）。
3. **可信度检查**：自动识别第 1 次调用（冷启动）并默认排除；多 rank 时显示 rank 间差异；显示多次运行的离散程度（CoV）。
4. **测量协议表单**：主指标、重复次数、精度门禁（`ratio_allclose` 阈值或逐位一致）、失败处理方式。协议之后由 T8 自动执行。
5. **kernel 排行快照**：进入 T2 的入口。

**E. UX 重点设计**
- **P0 基线是一个对象，而不是一张截图。**它必须引用原始数据文件，任何“提升 x%”都必须指向某个基线 ID。
- **P0 冷启动与稳态分开显示。**同一次运行里的 `inv=1` 与 `inv≥2` 默认拆开显示，冷启动数据不能进入主指标。
- **P1 协议先于数据。**首次建任务时提示填写测量协议，但允许跳过；跳过后，结论卡上标注“未定义协议”。
- **待决策**：主指标默认用 device_wall（STRACE）还是泳道 wall？两者口径不同：泳道 wall 为 4.88 ms，第 2 次调用的 device_wall 为 5.13 ms。建议两者都显示，并说明差异来自哪里。

**F. 对象与图元**：`Workload` `Objective` `Environment` `Run` `Baseline` `Plan` → C1 MetricView、D2 TableView、C2 DistributionView。

---

### T2 整图定位热点：时间花在哪里、哪里在空转

| 项 | 内容 |
| --- | --- |
| 目标 | 从整图 wall 找到值得优化的 scope，并判断它是否在关键路径上 |
| 完成标志 | 得到按 core-time 排序的 scope 列表，每项标明“计算型 / 停顿型 / 不在关键路径” |

**A. 用户作业过程（现状还原）**

| # | 用户动作 | 工具 / 界面 | 看什么 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | 打开 `merged_swimlane_*.json` | Perfetto（ui.perfetto.dev） | 72 条核轨道上的色块分布，找“碎”和“空”的区域 | 【GitHub】§8、#622 |
| 2 | 按 kernel 名求和，得到 core-time 排行 | 临时 Python 脚本，或 Perfetto SQL | Σdur、次数、平均值 | 【GitHub】§20（softmax_pool 9,495 µs，第 2 忙）；【本地】下表 |
| 3 | 泳道的 Latency/Total 列缺数据时，自己重建 wall | 脚本读 `l2_swimlane_records.json`：func key 偏移 2^32，tick 按 50 MHz 换算 | `max(finish) − min(dispatch)` | 【GitHub】§20 |
| 4 | dep_gen 溢出、task 没有名字时，重放编排提交序列来还原名字 | 脚本对照 `orchestration/*.cpp` 的 `rt_submit_*` 序列 | 100% 匹配校验 | 【GitHub】pypto-lib#465 |
| 5 | 按时间窗口计算核占用率 | 临时脚本 | 各窗口的 AIC/AIV 忙碌比例 | 【GitHub】§6、§8 |
| 6 | 判断热点是否在关键路径上 | 看泳道上的依赖箭头；或使用 `critical_path` 工具（当前工作区没有） | fanin/fanout 提示 | 【GitHub】§18 → §20 自我纠正；【本地】检测报告说明工具不可用 |

**本地实测（S1 rank0，Worker View）**【本地】

| 排名 | scope | task 数 | Σdur (µs) | 占比 | 平均 (µs) | 占用核数 | 起点 (µs) | 跨度 (µs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `qk_pv_aiv_spmd` | 48 | 31,626 | 25.2% | 658.9 | 48 | 2,634 | 886 |
| 2 | `csa_merge_pack_publish_spmd` | 48 | 18,597 | 14.8% | 387.4 | 48 | 3,019 | 711 |
| 3 | `qk_pv_aic_spmd` | 24 | 15,036 | 12.0% | 626.5 | 24 | 2,634 | 836 |
| **4** | **`kv_score_proj_spmd`** | **512** | **5,654** | 4.5% | **11.0** | 24 | 1,431 | **764** |
| 5 | `indexer_topk_group_wave_spmd` | 48 | 5,420 | 4.3% | 112.9 | 48 | 2,193 | 133 |
| 14 | `scatter_softmax_pool_spmd` | 64 | 2,140 | 1.7% | 33.4 | 48 | — | — |

**按 500 µs 窗口统计的占用率**【本地】（整图平均 AIC 32.2%、AIV 37.5%）：

| 窗口 (µs) | 0–500 | 500–1000 | 1000–1500 | 1500–2000 | 2000–2500 | 2500–3000 | 3000–3500 | 3500–4000 | 4000–4500 | 4500–4880 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AIC | 40% | 19% | **15%** | 50% | 28% | 72% | 53% | **0%** | **7%** | 30% |
| AIV | 31% | 22% | **12%** | 9% | 59% | 73% | 100% | 38% | **7%** | 15% |

> **读法**【推断】：
> - `kv_score_proj` 平均每个 task 只有 11 µs，却切成了 512 个，跨度 764 µs，而且大部分落在 1000–2000 µs 这段低占用窗口里。按故事一的法则，这是先看“停顿 / 派发”、再看计算的信号。
> - 6 月调优过的 `scatter_softmax_pool` 在 9 月版本里已退到第 14 位，热点转移到了 `qk_pv` 和 `csa_merge`。**热点会随优化和版本迁移**，排名必须与版本绑定。

**B. 现状卡点**
- Perfetto 是通用工具，不认识 scope、spmd、AIC/AIV 配对，也不会汇总 core-time 或计算窗口占用率，这些都要靠临时脚本。
- 同一个 trace 里有 Worker View 和 Scheduler View 两个视图，用脚本汇总时很容易重复计数（本次统计就踩到了这个坑）。
- 关键路径工具不可用，用户只能凭经验判断“在不在关键路径上”，因此出现了 §18 的误判。

**C. 可用的原始数据**：D1（事件和 flow）、D2（原始记录和时钟）、D3（依赖，用于关键路径）、D4（名称映射）、D5（device_wall，用于校验）。

**D. 工具 / 界面应提供的内容**
1. **scope 排行表**：Σdur、占比、task 数、平均时长、占用核数、起点、跨度、AIC/AIV 类型；支持按版本对比排名变化。
2. **占用率热带**：在泳道上方叠加一条按窗口计算的 AIC/AIV 占用率色带（上表的可视化），低占用窗口高亮显示。
3. **区间刷选**：在泳道上框选任意区间，给出该区间的占用率、在跑的 scope 构成、等待中的 task 数（D1 的 `shared_ready_queue` 计数）。
4. **关键路径**：用 D3 的依赖加上 D1 的时间计算关键路径，标出每个 scope 的 slack（最多能推迟多久而不影响 wall）。
5. **scope 身份**：统一显示 kernel 名（D4）、源码位置（`name_hint` → 源码行）、FuncId、核类型。

**E. UX 重点设计**
- **P0 以 scope 为基本单位，而不是 task 或核。**4,081 条 task 记录对人没有意义，用户思考的单位是 `pl.spmd` / `pl.at` scope。所有视图的默认聚合粒度应当是 scope，并能展开到 task。
- **P0 同时展示“多忙”和“在不在关键路径上”。**只看 Σdur 会被误导（§18、§20 的反转），排行表必须同时显示 slack。
- **P1 占用率是第一层诊断信息。**低于 30% 的窗口直接给出“停顿型”判断和原因候选，把用户引到 T3。
- **P1 视图去重。**Worker View 与 Scheduler View 分开，指标计算只用其中一个，并在界面上说明口径。
- **难点**：泳道事件量大（S1 为 3.6 万个事件、9.3 MB），整图层级需要预聚合，下钻时再加载明细。

**F. 对象与图元**：`Task`/`Event` `Scope` `Counter`（窗口占用率）`DepEdge` `Bottleneck` → B1 TimelineView（带占用率色带）、D2 TableView 排行态、A1 GraphView（关键路径）。

---

### T3 诊断调度与依赖：派发开销还是伪并行

| 项 | 内容 |
| --- | --- |
| 目标 | 区分两类停顿：派发开销（task 太碎）和依赖串行（task 多但核空），并找到源头 |
| 完成标志 | 每个停顿型热点都得到一个根因假设，并有支持证据 |

**A. 用户作业过程（现状还原）**

| # | 用户动作 | 工具 / 界面 | 看什么 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | 看每个 kernel 的 latency/exec 比值 | 泳道汇总表 | `kv_cache_write` 64 次，执行 2.06 µs，latency 13.24 µs，比值 6.4 倍 | 【GitHub】#314 |
| 2 | 看 Head OH / Tail OH 分布 | 泳道汇总；临时脚本 | Tail OH P50 9.5 µs；平均每个 task 等待约 38.8 次调度循环才被发现 | 【GitHub】#314 |
| 3 | 看某个 scope 的 task 落在几个核上 | Perfetto 按轨道逐条数 | “几百个 task，只落在 1~6 个核上” | 【GitHub】§8 |
| 4 | 查依赖：为什么不能并行 | `deps_viewer` 把 `deps.json` 渲染成 HTML（小于 500 节点用 dot，更大用力导向布局）；或看泳道 flow 箭头 | 跨迭代的 WAW 链 | 【GitHub】§8；【本地】`03-runtime-dfx.md` |
| 5 | 做拆分实验验证假设（3 pass / 4 pass） | 改代码、真机运行 | Total 与占用核数 | 【GitHub】§8 |
| 6 | 用 split test 区分计算型和停顿型 | 改 tile，看单个 task 是否也减半 | 单 task 时长 vs Total | 【GitHub】§6 |

**本地数据中已有、但当前没被充分利用的信号**【本地】
- D2 的 `aicpu_scheduler_phases` 带有 `loop_iter`、`tasks_processed`、`pop_hit`、`pop_miss`。例如某次派发阶段 `pop_hit=1, pop_miss=162`，可以用来解释“task 完成后要等很多轮才被发现”。
- D1 的 `fanin-hint` / `fanout-hint` 直接给出每个 task 的前驱和后继，例如 `kv_score_proj` 的某个 task：fanin 4 个、fanout 2 个。
- D3 的边带有 `source`（creator/explicit/tensormap）和 `flags`（wait/retain），可以区分“显式依赖”和“张量推导出的依赖”。
- D14 的源码中已经有人工调度干预：`deps=[late_dep]` 配合注释“qr_proj_matmul 是关键路径，需要先抢到核”。这说明用户已经在手工编排依赖，但工具看不到这层意图。

**B. 现状卡点**
- Head/Tail OH 的含义需要用户理解调度器内部机制。
- 依赖图与泳道分别在两个工具里，“这条依赖边让哪两个 task 串行了”要靠人工对照。
- 依赖图规模大（S2 中有 426 个 task、1,369 条边），渲染出来难以阅读。

**C. 可用的原始数据**：D1（fanin/fanout、flow）、D2（调度阶段计数）、D3（边类型）、D14（显式 `deps`）。

**D. 工具 / 界面应提供的内容**
1. **scope 调度诊断卡**，内容包括：
   - task 数、占用核数、平均执行时间、平均 latency、latency/exec 比值；
   - Head OH、Tail OH 的分布；
   - 判定结果：“派发开销型”或“依赖串行型”，并附判定依据。
2. **依赖链视图**：只画与所选 scope 相关的依赖链，关键链描边，每条边标注类型（显式 / 张量推导）和涉及的张量。
3. **从泳道反查依赖**：点击泳道上的空隙，回答“这个核在等谁”，显示前驱 task 和对应的依赖边。
4. **可做的实验建议**：chunk 扫描、改用 spmd、split test 等，每项附预期效果和风险（例如“chunk 过大时单 task 变慢”）。

**E. UX 重点设计**
- **P0 用一句话给出诊断结论。**例如：“`scatter` 有 512 个 task，只用了 5 个核，原因是 `compress_state_flat` 上的 WAW 链”。把 D1、D2、D3 三类数据合成一个判断，而不是让用户自己拼。
- **P0 依赖要能回到源码。**每条依赖边都要能跳到产生它的源码语句（`pl.assemble` 重新赋值的那一行），这是 T3 到 T4 的桥梁。
- **P1 显示用户手工写的依赖意图。**把 `deps=[...]`、`task_dummy` 这类显式依赖与自动推导的依赖区分显示，避免用户误以为是工具推导出来的。
- **难点**：依赖图的可读性。建议默认只显示“关键链加上所选 scope 的一跳邻居”，全图作为二级入口。

**F. 对象与图元**：`Task` `DepEdge` `Tensor` `Bottleneck`（派发 / 串行）`Experiment` → B3 BreakdownView（执行 / Head / Tail）、A1 GraphView（依赖链）、B1 TimelineView（等待原因着色）。

---

### T4 修正源码结构

| 项 | 内容 |
| --- | --- |
| 目标 | 把 T3 的根因落到具体的源码行，并给出修改方案 |
| 完成标志 | 有一个或多个候选改法，每个都说明预期影响的 scope 和风险 |

**A. 用户作业过程（现状还原）**

| # | 用户动作 | 工具 / 界面 | 看什么 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | 从 kernel 名找回源码 | 编辑器全局搜索 `name_hint` | `name_hint="state_scatter_paged"` 所在位置 | 【推断】 |
| 2 | 理解循环和 scope 的嵌套关系 | 读源码 | `for o0` 在 `pl.at` 外还是内 | 【GitHub】§8 |
| 3 | 找共享张量的读写 | 读源码 | `compress_state_flat = pl.assemble(compress_state_flat, …)` | 【GitHub】§8 |
| 4 | 改写：`pl.at` 外提、融合或拆分 scope、调整 chunk、改用 `pl.spmd` | 编辑器 | 源码 diff | 【GitHub】§8、§14、#314 |
| 5 | 推演改动会怎么展开 | 只能靠运行后看泳道 | task 数、核数 | 【GitHub】§8 的 3 次实验 |

**B. 现状卡点**
- 源码与运行结果之间没有直接映射。改一行代码的效果，必须完整编译、上板、采泳道后才知道，每轮成本在小时级。
- 有些写法是否可用受编译器限制，例如 `pl.spmd` 在 if/else 中不可用（pypto#1414），用户无法区分是自己写错还是编译器不支持。

**C. 可用的原始数据**：D14（源码）、D6（`08_after_OutlineHierarchyScopes`、`09_after_OutlineIncoreScopes` 等 dump 中 scope 被外提后的函数）、D3（task 对应的 scope）、D11（func_id ↔ kernel）。

**D. 工具 / 界面应提供的内容**
1. **源码侧栏**：光标位于某个 `pl.spmd` / `pl.at` / `pl.pipeline` 块时，显示：
   - 该块最近一次运行展开出的 task 数、占用核数、Σdur、所在窗口；
   - 读写的共享张量，以及其中哪些会产生跨迭代依赖；
   - 相关的编译提示（D9 中同一行的条目，例如 `:110` 的 PH-MR-001）。
2. **静态预判**：不上板，就能从 IR 推导出改动后的 task 数和依赖边数，在编辑时给出估算，并标注为 `estimated`。
3. **候选管理**：把一次改法保存为候选，自动关联 T8 的实验记录。
4. **已知限制提示**：写法命中编译器已知限制时，给出 Issue 链接和绕行方式。

**E. UX 重点设计**
- **P0 源码是主界面，不是附属信息。**模型开发者的工作起点和终点都在源码上。性能证据应以行内标注和侧栏的方式出现在源码旁边，而不是让用户切换到另一个工具。
- **P0 估算与实测严格区分。**静态预判的 task 数、依赖数标为 `estimated`，运行后变为 `measured`，两者并列显示并标出差异。
- **P1 伪并行风险提示。**满足“task 多、占用核少、存在跨迭代写”时，在对应的 `pl.assemble` 行上标出风险。
- **P2 反直觉的写法提示。**例如“融合后 vec 部分与关键路径抢核”（§14），这类提示需要关键路径信息，依赖 T2 的能力。

**F. 对象与图元**：`Scope` `Tensor` `SourceMap` `Candidate` `Diagnostic` → L1 TextCanvas（行内标注）、A2 TreeView（Scope 树）、D1 DiffView（源码 Diff + 指标变化）。

---

### T5 核对编译过程：意图是否生效、提示是否相关

| 项 | 内容 |
| --- | --- |
| 目标 | 确认源码里写的优化意图（layout、流水深度、融合、b_trans）在编译后真正生效，并从编译提示中找出与热点相关的条目 |
| 完成标志 | 每个意图都有“生效 / 未生效 / 部分生效”的结论；热点 scope 的相关提示都已处理或被判定为无关 |

**A. 用户作业过程（现状还原）**

| # | 用户动作 | 工具 / 界面 | 看什么 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | 怀疑 NZ 声明没生效，写探针脚本分别生成 NZ 与 ND 两个版本 | 编辑器 + 终端 | 两份 `.pto` | 【GitHub】§19 `_tmp_nz_weight_probe.py` |
| 2 | 对比两份 `.pto` | `diff` | **逐字节相同**，说明声明被忽略 | 【GitHub】§19 |
| 3 | 调用 `col_expand_sub` 编译失败 | 终端报错 | `No codegen registered` | 【GitHub】§20 |
| 4 | 翻 pass dump 找某个变换在哪一步发生 | 编辑器打开 52 个约 1 MB 的 `.py` 文件；grep 函数名 | `kv_score_proj` 在各 pass 中的形态 | 【本地】D6；【GitHub】pypto#1475 |
| 5 | 看生成的 C++ 里的同步 | 编辑器打开 `kernels/aic/*.cpp` | `set_flag` / `wait_flag` / `pipe_barrier` | 【GitHub】PTOAS#643、pypto#1475；【本地】D8 |
| 6 | 看性能提示 | 打开 `perf_hints.log`（230 行） | 按文件找与自己相关的条目 | 【本地】D9 |

**本地实例：同一个 scope 在 5 层中的形态**【本地】

| 层 | 文件 | `kv_score_proj` 在这一层的样子 |
| --- | --- | --- |
| 源码 | `decode_compressor_ratio4.py:110-136` | `pl.spmd(... name_hint="kv_score_proj", deps=[late_dep])`，内有 `pl.pipeline(0, D // K_TILE, stage=2)`，调用 `matmul(..., b_trans=True)` |
| IR | `passes_dump/35_after_MemoryReuse.py:2229` | 函数 `kv_score_proj(...)`，参数带有 `pl.MemRef("mem_ddr_1", ..., 8388608)` 等存储注解 |
| PTO | `ptoas/kv_score_proj.pto`（50.8 KB） | `alloc_tile loc=acc f32[16,64]` ×2（kv / score 两个累加器）；`loc=right bf16[256,64] blayout=row_major slayout=col_major`，即 b_trans 生效后的 ZN 形态 |
| Kernel | `kernels/aic/kv_score_proj.cpp`（33 KB） | 6 TLOAD、24 TEXTRACT、12 次 matmul 指令；33 对 set/wait_flag、7 个 pipe_barrier |
| 提示 | `perf_hints.log` | `decode_compressor_ratio4.py:110` 有 5 条 PH-MR-001：Right 请求深度 2，只放得下 1 份（每份 32,768 B，空闲 65,536 B），原因是与其他 buffer 共存。也就是说 **`stage=2` 在 Right 上并未完全生效** |

> 最后一行说明：6 月调优时用户以为 `pipeline(stage=2)` 已经生效；而 9 月这个版本的编译提示指出，在 L0B（Right）这一级，两个 matmul（wkv / wgate）的 Right tile 同时存在，深度 2 放不下。**意图部分生效，而用户并不知道。**【本地 + 推断】

**B. 现状卡点**
- “没生效”没有任何信号，只能靠用户起疑后自己写探针。
- pass dump 是 52 个大文本文件，找到某个函数在某一步的变化要靠 grep 和人工对比。
- 230 条提示按编译顺序输出，没有排序，也不知道哪些与当前瓶颈有关。PH001 就占了 197 条，重要的 PH-MR-001 被淹没在其中。
- pass 编号会随版本变化（MemoryReuse 从 29 号变成 35 号），跨版本对比时容易错位。

**C. 可用的原始数据**：D6、D7、D8、D9、D14；D10（内存报告，只在 S2 中有）。

**D. 工具 / 界面应提供的内容**
1. **意图核对表**：自动从源码中提取显式意图（`stage=`、`b_trans=`、layout 声明、`name_hint`、融合结构），与编译产物对照，给出“请求值 / 实际值 / 证据位置”，例如：

   | 意图 | 请求值 | 实际值 | 证据位置 |
   | --- | --- | --- | --- |
   | pipeline stage | 2 | Right 上为 1 | perf_hints + pto |

2. **单 scope 的 pass 轨迹**：只显示选中 scope 在 52 个 pass 中**有变化的那几步**，每步给出结构化 diff（alloc 数、MemRef、循环结构、同步点数）。
3. **五层对照视图**：源码 ↔ IR ↔ PTO ↔ C++ ↔ 提示，选中任一层的某行，其他层同步定位。
4. **提示分诊**：
   - 按规则和源码位置聚合，197 条 PH001 可以合并成若干组；
   - 与 T2 的热点 join，只把“落在热点 scope 上”的提示排在前面；
   - 每条提示给出一个可执行的建议动作。
5. **能力查询**：调用 API 时即时提示该 op 在当前后端有没有 codegen，以及替代写法。

**E. UX 重点设计**
- **P0 显示“意图 vs 生效值”。**这是本任务最独特的价值：用户写的、编译器真正做的、二者的差异，三者并排显示。
- **P0 提示要分诊，不要堆叠。**默认视图最多显示 5~10 条与热点相关的提示；其余按规则折叠。每条提示都能跳到源码行和泳道中的 scope。
- **P1 按变化浏览 pass。**默认隐藏没有变化的 pass；用 pass 名称而非编号作为跨版本对齐的键。
- **P1 同步点可视化。**把 C++ 中的 `set_flag` / `wait_flag` 配对画成流水之间的依赖，供 T7 与 T3 复用。
- **难点**：从源码意图到产物的对应关系需要编译器提供结构化元数据（目前只有 `loc(...)` 源码位置和提示文本），这是平台依赖项。

**F. 对象与图元**：`Pass` `IRNode` `CodegenArtifact` `Diagnostic` `SourceMap` `Fence` → B2 SequenceView（pass 轨迹）、D1 DiffView（IR/产物 Diff）、L1 TextCanvas（多栏联动）、D2 TableView（提示分诊）。

---

### T6 硬件资源预算：放得下吗，怎么搬

| 项 | 内容 |
| --- | --- |
| 目标 | 在调整 tile、pipeline 深度和融合方式之前，知道每一级存储还剩多少空间、会不会越界，以及数据搬运是什么形态 |
| 完成标志 | 每个候选参数都有资源预算结论（放得下 / 越界 / 挤占流水），并知道搬运笔数与粒度 |

**A. 用户作业过程（现状还原）**

| # | 用户动作 | 工具 / 界面 | 看什么 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | 把 tile 调大一档，编译 | 终端 | 失败时的报错：`qr_acc [T,512] INT32 = 256KB > 192KB`、`Mat buffer usage 655360 > 524288` 等 | 【GitHub】§2、#665 |
| 2 | 加深 pipeline，编译 | 终端 | `Mat buffer 576KB > 512KB` | 【GitHub】§19 |
| 3 | 心算容量：`3×[128,256] FP32 > 192KB` | 草稿、心算 | 各 tile 大小相加 | 【GitHub】§20 |
| 4 | 查看内存报告 | 打开 `memory_after_AllocateMemoryAddr.txt`（715 行） | 各函数、各空间的 Used/Limit、buffer 地址与生命周期 | 【本地】D10 |
| 5 | 看加载指令形态，判断是带宽还是事务问题 | op-sim 指令名：`MOV_OUT_TO_L1_MULTI_ND2NZ` | 每通道 GB/s、笔数 | 【GitHub】§19 |
| 6 | 理解存储之间的耦合（UP_DOWN 切分会缩小 L0C 行，但不会缩小 `create_tensor`） | 读代码、试错 | — | 【GitHub】§2 |

**本地内存报告样例**【本地，S2】

```text
--- down_proj ---
  Space  |  Used       |  Limit      |  Usage   |  MemRefs
  Mat    |   260.0 KB  |   512.0 KB  |   50.8%  |  4
  Right  |    64.0 KB  |    64.0 KB  |  100.0%  |  2
--- dcr_xgamma ---
  Vec    |   128.0 KB  |   184.0 KB  |   69.6%  |  2
    mem_vec_5  |  64.0 KB  |  [0, 65536)       |  [6, 13]
    mem_vec_6  |  64.0 KB  |  [65536, 131072)  |  [7, 11]
```

**GitHub 案例中出现过的容量墙**【GitHub】

| 存储 | 编译器上限 | 撞墙场景 |
| --- | --- | --- |
| Vec（UB） | 184 KB（本地报告）；调优日志中常写 192 KB | `qr_acc` 256 KB；`softmax_pool` 3×[128,256] FP32 |
| Mat（L1） | 512 KB | stage=3 需要 576 KB；MLP TN=512 需要 640 KB |
| Right（L0B） | 64 KB | `kv_score_proj` 深度 2 放不下（本地提示）；`down_proj` 100% |
| Acc（L0C） | 日志未给出具体上限，写作“GRP=4 已占 128KB 顶满” | `qr_hadamard` 不能跟随 rope 一起加大 |

**B. 现状卡点**
- 能不能放下只能靠编译试错；报错只给出总量，不说明是哪几个 buffer 占的。
- 硬件规格与编译器上限不一致（192 KB 与 184 KB），用户拿错数字，推算就会出错。
- 存储之间的耦合（一个参数同时影响 L1、L0B、L0C）只能靠经验。
- 搬运形态藏在指令名里，“短 burst”“事务过碎”需要深厚背景知识才能判断。

**C. 可用的原始数据**：D10（每个函数、每个空间的 buffer、地址和生命周期）、D7（`alloc_tile` 的 loc/rows/cols/dtype/layout）、D9（PH-MR-001 给出每份大小和空闲量；PH001 给出最内维字节数）、编译错误文本。

**D. 工具 / 界面应提供的内容**
1. **分级容量预算条**：Vec / Mat / Left / Right / Acc 五级，每级显示已用、上限、占用明细（按 buffer，可跳到源码中的 tile），上限取编译器实际值，并注明硬件规格值。
2. **参数推演**：用户在源码或侧栏调整 `K_TILE`、`stage`、`HEAD_TILE` 时，实时重算各级占用，标出“再加大一档会在哪一级越界”，以及“越界的是哪几个 buffer”。
3. **地址 × 生命周期图**：基于 D10 的地址区间和生命周期区间，展示哪些 buffer 同时存在、哪里可以复用，以及流水深度为什么放不下（PH-MR-001 的“与其他 buffer 共存”）。
4. **搬运形态卡**：对每个 load/store 给出每笔字节数、笔数、是否跨步、与 512B cache line 的关系（PH001），以及 ND2NZ / DN2ZN 等形态的通俗解释和建议（例如“先试 b_trans”，附适用条件）。

**E. UX 重点设计**
- **P0 在编译前看到容量墙。**把“试错—报错”的循环前移成“调参数时即时预算”，这是节省最多时间的地方。
- **P0 上限带来源和平台。**每个上限都标注来源（编译器 / 硬件手册）和平台（a2a3 / a5），不同来源的数值不一致时明确提示。
- **P1 越界要能归因到具体 buffer。**报错时指出“哪几个 tile 共存导致越界”，并给出能释放空间的候选。
- **P1 搬运用业务语言解释。**把“ND2NZ”翻译成“这次加载被拆成 256 笔 64 列宽的碎片搬运”，并与带宽上限对照，防止用户过早得出“带宽瓶颈”的结论。
- **可复用原型**：Memory Inspector（AST 推导的 tile 生命周期）、Ascend Memory Studio（地址 × 时间占用）。需要补上与编译器真实报告（D10）的对接。

**F. 对象与图元**：`Constraint`（带平台和版本）`MemBlock` `Tile` `SweepAxis` `Counter`（搬运）→ C5 OccupancyView、B3 BreakdownView 预算模式、C3 HeatmapView（参数扫描合法区）、A3 TopologyView（搬运路径）。

---

### T7 核内下钻：单个 task 的时间花在搬运还是计算

| 项 | 内容 |
| --- | --- |
| 目标 | 对 T2 中的计算型热点，看清单个 task 内各流水单元的构成 |
| 完成标志 | 明确是 MTE2（加载）、VECTOR、CUBE 中的哪一个在主导，以及主导它的是哪几条指令、对应哪一行源码 |

**A. 用户作业过程（现状还原）**

| # | 用户动作 | 工具 / 界面 | 看什么 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | 对目标 kernel 跑 op-sim | incore-profiling skill → `msprof op simulator` | 生成 `softmax_pool.clean.json` | 【GitHub】§19、§20 |
| 2 | 看各单元的占比 | clean.json / 文本汇总 | veccore span 20.58 µs；MTE2 占 69%；VECTOR 占 63% | 【GitHub】§20 |
| 3 | 看指令构成 | 同上 | 256 笔 `[1,64]` 加载；`MOV_UB_TO_UB` staging；两次 `VNCHWCONV` 转置；真正的计算不到 1 µs | 【GitHub】§20 |
| 4 | 发现 trace 退化，修正输入后重跑 | 手动把 `v1.bin` 写成 127 | 循环轮数 | 【GitHub】§20 |
| 5 | 上板用 PMU 核实（可选） | `--enable-pmu 2/4` | MTE 忙碌比例、实际读取字节数 | 【GitHub】pypto-lib#622 |
| 6 | 对照 MTE 通道并集，判断能否通过加深流水填满 | clean.json | 5 条 MTE2 子队列并集 0 gap | 【GitHub】§19 |

**B. 现状卡点**
- op-sim 需要单独的工具链和 skill，输入要从 golden 生成。**数据相关的门控会让 trace 退化**，而 replay 脚本（D12）的默认输入是随机数、动态维填 1，同样可能触发这个问题。
- 按硬件单元罗列周期，用户要自己把指令归类为“搬运 / 计算 / 转置 / staging”。
- PMU 会让 wall 膨胀约 15%，计时与计数必须分两次运行采集，这点很少有人知道。

**C. 可用的原始数据**：D15（op-sim，本地没有）、D16（PMU，本地没有）、D8（kernel C++ 中的指令序列，可做静态统计）、D12（replay 入口）。

**D. 工具 / 界面应提供的内容**
1. **一键核内分析**：从 T2 的热点 scope 直接发起，自动选取有代表性的 task 输入（包括控制张量的真实值），生成核内报告。
2. **构成视图**：默认按“搬运（GM↔片上）/ 片上搬移 / 格式转换 / 计算”四类归类，其次才按硬件单元展开。
3. **指令 ↔ 源码**：每类指令能回到产生它的源码切片，例如“256 笔加载来自 `compress_state_flat[blk:blk+1, …]`”。
4. **退化检测**：循环轮数为 0 或明显偏低、控制张量为全零时，把报告标为“不可用”并说明原因。
5. **PMU 采集提示**：开启 PMU 的运行自动标注“wall 不可用于性能对比”。

**E. UX 重点设计**
- **P0 回答“时间花在搬运还是计算上”。**默认分类以用户的决策需要为准，而不是按硬件单元列出。
- **P0 采样有效性优先显示。**trace 是否可信，要排在所有数字之前显示。
- **P1 与静态信息对照。**把 D8 的静态指令计数（例如 6 TLOAD、24 TEXTRACT）与 op-sim 实测周期并排，帮助用户理解“哪条指令贵”。
- **待决策**：op-sim 需要额外的仿真时间和环境。是在工作台中集成调用，还是只导入结果？这取决于平台依赖。

**F. 对象与图元**：`Counter`（指令、单元）`Kernel` `TensorDump`（控制张量）`SourceMap` → B3 BreakdownView、B1 TimelineView（单 task 内多流水轨道）、L3 EvidenceChain。

---

### T8 实验与对比验证

| 项 | 内容 |
| --- | --- |
| 目标 | 判断一个候选改动是否真的更快、是否正确，并排除噪声 |
| 完成标志 | 通过比较资格检查、精度门禁和噪声判定的实验结论 |

**A. 用户作业过程（现状还原）**

| # | 用户动作 | 工具 / 界面 | 看什么 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | 扫参数：`kv_score_proj` 的 task 数取 2、4、8 | 手工改常量，逐个运行 | Total 为 621、582、598 µs，4 是拐点 | 【GitHub】§6 |
| 2 | 同一 session 做 A/B，每个版本跑 2~3 次 | 终端，逐次记录 | 中位数 | 【GitHub】§19、§20 |
| 3 | 选一个无关的大 scope 当噪声锚点 | 自己判断，例如选 `qk_pv` | 锚点在各次运行之间是否漂移 | 【GitHub】§20 |
| 4 | 精度验证 | 测试 harness | `max_error_ratio=0.0`、各输出 PASS | 【GitHub】§20 |
| 5 | 记录每轮的 build 目录，以便追溯 | 手写在日志里 | `_jit_..._20260629_150217` 等路径 | 【GitHub】§20 |
| 6 | 标注失败和被推翻的尝试 | 调优日志 | 3-pass、4-pass 的反例；R1 的“假收益” | 【GitHub】§8、§20 |

**B. 现状卡点**
- 每轮都要手工改代码、手工运行、手工记录，参数扫描非常耗时。
- 噪声判定完全依赖个人经验：选哪个锚点、跑几次、看 wall 还是 busy。§19 中单次运行曾出现 +20% 的假象。
- 实验记录散落在个人日志、PR 描述和本地目录中（日志说明原始 JSON 没有随资料提交）。
- 冷启动、rank 差异（D5）进一步放大了不确定性。

**C. 可用的原始数据**：多次运行的 D1/D2/D5、golden 结果（D18）、源码版本（git）、D12 的 replay 入口（可以在不重新编译的情况下重跑）。

**D. 工具 / 界面应提供的内容**
1. **实验面板**：选择候选（源码版本或参数组合）和扫描轴，按测量协议（T1）自动安排运行，默认交错执行（例如 ABBA）以抵消漂移。
2. **结果表**：每个候选给出主指标的中位数和分布、busy、目标 scope 的 core-time、锚点 scope 的漂移、精度门禁结果。
3. **比较资格徽标**：同 session、次数足够、锚点稳定、无 PMU、非冷启动，全部满足才显示“可比”，否则显示缺少哪一项。
4. **敏感性曲线**：扫描结果画成曲线，标出拐点和不可编译的区域（来自 T6）。
5. **失败实验也保留**：记录被推翻的假设及其证据，供 T9 沉淀使用。

**E. UX 重点设计**
- **P0 自动选择噪声锚点。**从“与改动无关、时长较大、历史上稳定”的 scope 中推荐锚点，并在结果表中显示它的漂移。这是把 §20 的法则产品化。
- **P0 不合格的数字置灰。**不满足比较资格时，提升百分比置灰显示，并给出补做什么实验。
- **P1 实验要节省设备时间。**批量安排实验时显示预计占用的设备时长，扩大实验规模需要确认（与人和 Agent 的协作规划一致）。
- **P1 被推翻的结论可以回看。**实验时间线上保留“假设 → 实验 → 推翻或确认”的过程。

**F. 对象与图元**：`Experiment` `Candidate` `SweepAxis` `Baseline` `Gate` `Distribution` → D2 TableView 明细态、C2 DistributionView、C4 ScatterView（敏感性）、D1 DiffView（基线因果 Diff 加锚点列）、C1 门禁徽标。

---

### T9 交付与沉淀

| 项 | 内容 |
| --- | --- |
| 目标 | 把改动安全合入，让审阅者能复核，并把经验沉淀成带适用条件的知识 |
| 完成标志 | PR 合入且 CI 通过；经验记录包含适用条件和反例；目标 scope 进入回归看护 |

**A. 用户作业过程（现状还原）**

| # | 用户动作 | 工具 / 界面 | 看什么 | 来源 |
| --- | --- | --- | --- | --- |
| 1 | 修改权重签名后，找出所有调用方并一路修改 | 全局搜索 | 需要改 4 个文件；另外两条同名路径**不能改** | 【GitHub】§19 |
| 2 | 漏改导致运行时报 507018（而不是 shape 错误） | CI 日志 | 报错与真实原因相距很远 | 【GitHub】§19 |
| 3 | CI 报了一个与本次改动无关的错，排查后发现是分支基于旧 base | CI 日志、`git rev-list` | pypto 版本偏差 | 【GitHub】§19 |
| 4 | 写 PR 描述：改前改后数据、build 目录、精度结果 | GitHub | — | 【GitHub】pypto-lib#628、#641 |
| 5 | 在源码中留下原因注释 | 编辑器 | “b_trans 让 DN2ZN 变成长 burst，约 −14% busy” | 【本地】D14 第 122-125 行 |
| 6 | 把经验写成“法则”，发布到调优日志或 Issue | Markdown | 带条件的法则 | 【GitHub】#828 |
| 7 | 纳入每日性能看护 | CI 配置 | — | 【推断】pypto-lib 于 9 月接入每日算子性能 CI（#1274、#1284） |

**B. 现状卡点**
- 改动影响面靠全局搜索，漏改的代价是一个难以理解的运行时错误。
- 经验散落在源码注释、PR、个人日志和 Issue 中；适用条件往往只写在正文里。例如 `b_trans` 的反例（pypto#2309）出现在另一个仓库、另一个时间，与原经验之间没有任何关联。
- 本地环境与 CI 环境的版本偏差会制造假问题。

**C. 可用的原始数据**：D14（源码与注释）、D13（参数签名）、D3（调用与张量关系）、git 历史、CI 日志、PR 描述。

**D. 工具 / 界面应提供的内容**
1. **影响面分析**：修改某个 kernel 的签名或布局时，列出所有调用链上需要同步修改的位置，以及“同名但不受影响”的路径。
2. **交付包**：自动汇总基线、最终候选、实验表、精度结果、环境指纹、原始数据链接，生成 PR 描述草稿。
3. **环境一致性检查**：本地与 CI 的 pypto / ptoas 版本对比，存在偏差时提示“先 rebase 或对齐再排查”。
4. **经验卡**：法则、适用条件（平台、shape 范围、形态）、证据（实验链接）、反例（例如 pypto#2309），以及和源码位置的双向链接。
5. **回归看护**：把目标 scope 的 core-time 与整图 wall 纳入趋势监控，设置门禁线。

**E. UX 重点设计**
- **P0 经验卡必须写明适用条件。**没有条件和反例的经验卡不能发布；关联到其他地方出现的反例时，自动提示经验卡的作者。
- **P1 影响面先于修改。**在用户修改签名前显示影响面，而不是等运行时报错。
- **P1 源码注释与经验卡关联。**源码中的调优原因注释（如 D14）可以一键生成或关联经验卡，让知识留在代码旁边。
- **P2 回归看护按 scope 设置。**只盯整图 wall 太粗，调优过的 scope 应当有自己的趋势线。

**F. 对象与图元**：`Lineage` `Artifact`/`Manifest` `Recipe`/`Knowledge`（带适用范围）`Regression` `Environment` → A1 GraphView 血缘版式、B4 TrendView、L2 FindingCard。

---

## 五、跨任务汇总

### 5.1 任务 × 原始数据矩阵

| 数据 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 merged_swimlane | ○ | ● | ● | ○ |  |  |  | ● |  |
| D2 chip_swimlane_records | ○ | ● | ● |  |  |  |  | ● |  |
| D3 deps.json |  | ○ | ● | ○ |  |  |  |  | ○ |
| D4 name_map |  | ● | ○ | ○ |  |  |  |  |  |
| D5 STRACE | ● | ○ |  |  |  |  |  | ● |  |
| D6 passes_dump |  |  |  | ○ | ● | ○ |  |  |  |
| D7 .pto |  |  |  |  | ● | ● | ○ |  |  |
| D8 kernel .cpp |  |  |  |  | ● |  | ○ |  |  |
| D9 perf_hints |  |  |  | ○ | ● | ● |  |  |  |
| D10 内存报告 |  |  |  |  | ○ | ● |  |  |  |
| D11 kernel_config | ○ | ○ |  | ○ |  |  |  |  |  |
| D12 replay 脚本 |  |  |  |  |  |  | ● | ● |  |
| D14 源码 |  | ○ | ○ | ● | ● | ○ | ○ | ○ | ● |
| D15 op-sim（本地无） |  |  |  |  |  | ○ | ● |  |  |
| D16 PMU（本地无） |  |  |  |  |  | ○ | ● |  |  |
| D18 golden 结果 | ○ |  |  |  |  |  |  | ● | ○ |

● 为主要依据，○ 为辅助。

### 5.2 数据缺口（设计前需要与平台确认）

| 缺口 | 影响的任务 | 现状 | 建议 |
| --- | --- | --- | --- |
| 源码意图 → 产物的结构化映射 | T4、T5 | 只有 `loc(...)` 源码位置和提示文本 | 编译器输出“意图—生效值”元数据 |
| 关键路径计算 | T2、T3 | 检测报告指出 `critical_path` 工具在当前工作区不可用 | 在工作台内基于 D1 + D3 计算 |
| 统一的环境指纹文件 | T1、T8、T9 | 构建目录里没有 | 在构建时写出 |
| op-sim 与 PMU 结果的标准格式 | T7 | 本地没有样本，只有 GitHub 描述 | 取得一份 `*.clean.json` 与 `pmu.csv` 样本 |
| Acc（L0C）上限与各级存储的规格来源 | T6 | 内存报告中有 Vec/Mat/Right；日志中的 L0C 说法不完整 | 从编译器取得完整上限表 |
| 多次运行的实验元数据 | T8 | 只有目录名中的时间戳 | 为运行记录增加候选 ID 和协议 ID |
| 泳道 Latency/Total 列缺失的条件 | T2 | §20 提到需要 `runtime_debug_mode=1` 加 `DUMP_DEVICE_PERF` | 在界面上说明缺失原因 |

### 5.3 UX 重点设计清单（按优先级）

| 优先级 | 设计点 | 所属任务 | 为什么重要（证据） |
| --- | --- | --- | --- |
| P0 | 基线对象，冷启动与稳态分开 | T1 | 本地 D5 中冷热相差 9 倍 |
| P0 | 以 scope 为单位的排行，同时显示关键路径 slack | T2 | §18 → §20 的误判；本地热点已经迁移 |
| P0 | 窗口占用率作为第一层诊断 | T2、T3 | §6、§8：利用率低于 30% 就是停顿型；本地窗口低至 0~15% |
| P0 | 依赖边回到源码行 | T3、T4 | §8：一行缩进让 64 个 batch 串行 |
| P0 | 源码作为主界面，行内显示证据 | T4 | 模型开发者的工作起止点都在源码上 |
| P0 | 意图与生效值并排 | T5 | §19 NZ 被静默忽略；本地 `:110` 的 stage=2 部分生效 |
| P0 | 编译提示分诊 | T5 | 本地 230 条提示没有优先级 |
| P0 | 编译前容量预算，上限带来源 | T6 | 5 处容量墙；184 KB 与 192 KB 不一致 |
| P0 | 核内按“搬运 / 计算”归类，并先判断 trace 是否有效 | T7 | §20：Vector 忙的其实是搬运；trace 退化 |
| P0 | 自动选择噪声锚点、不合格数字置灰 | T8 | §20 的 R1 假收益；§19 单次运行 +20% 的假象 |
| P0 | 经验卡必须带适用条件和反例 | T9 | b_trans 在两个场景结果相反 |
| P1 | 影响面分析、环境一致性检查 | T9 | §19 的 507018 与 CI 旧 base 问题 |
| P1 | 按变化浏览 pass，用 pass 名对齐版本 | T5 | pass 编号从 29 变为 35 |
| P1 | 搬运形态的业务语言解释 | T6 | ND2NZ 被误判为带宽瓶颈 |
| P1 | 实验占用设备时长的显示与确认 | T8 | 与人—Agent 协作规划一致 |

### 5.4 界面组织建议【设计】

按任务组织为一个“调优任务工作区”，而不是按工具分页：

```text
调优任务工作区
├── 概览：基线卡、当前最优候选、比较资格、待办假设               ← T1、T8
├── 整图：泳道 + 占用率色带 + scope 排行 + 关键路径               ← T2、T3
├── Scope 详情：调度诊断卡、依赖链、源码侧栏、相关提示            ← T3、T4、T5
├── 编译：意图核对表、单 scope 的 pass 轨迹、五层对照、提示分诊   ← T5
├── 资源：分级容量预算、地址 × 生命周期、搬运形态                 ← T6
├── 核内：构成视图、指令 ↔ 源码、有效性检查                       ← T7
├── 实验：候选、扫描、结果表、敏感性曲线、失败实验                ← T8
└── 交付：影响面、交付包、经验卡、回归看护                        ← T9
```

- 所有页面共享同一个“当前选中 scope”：在整图选中后，切到编译、资源、核内页面时，自动聚焦到这个 scope。
- 任何数字都带版本、运行 ID 和比较资格角标，点击即可回到原始数据文件。
- 可复用现有原型：算子调优控制台 / V2（概览、整图、实验）、Pass Transform Explorer / Pass Atlas（编译）、Memory Inspector / Ascend Memory Studio（资源）、调试与调优工作台（核内）、算子开发 Copilot（源码侧栏）、Toolkit Studio（交付）。

---

## 附录：本次本地统计口径

- 泳道统计只使用 `merged_swimlane` 中 **Worker View**（pid=4）里线程名为 `AIC_*` / `AIV_*` 的 `X` 事件，避免与 Scheduler View 重复计数；wall = 最早开始到最晚结束；占用率 = 该类核忙碌时间之和 ÷（核数 × 窗口长度），AIC 按 24 核、AIV 按 48 核计算。
- STRACE 数值取自 `host.*.log` 中的 `chip.run.runner_run.device_wall` span，按 `inv` 区分调用次序。
- `kernels/aic/kv_score_proj.cpp` 的指令和同步计数是文本匹配计数，不代表运行时执行次数（循环内的指令会被多次执行）。
- 以上统计只基于 S1 的 rank0 和 S2 的内存报告，不代表其他版本或配置。
