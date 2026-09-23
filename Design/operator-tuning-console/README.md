# 算子调优控制台 · Tuning Console

把「端到端 → L2 调度 → L1/L0 单核流水 → 编译降级 → ISA / 布局」这套调优闭环，做成一个可操作的产品界面，
而不是一篇讲解或一个向导。对象是一次**真实上板执行**：

```
Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/
```

打开入口：`Design/operator-tuning-console/index.html`，也在 `launch.html` 的「内存与性能」分类里。

---

## 这个 Demo 是什么形态

一个 IDE 形态的工作台，两个真实 case 可切换，工作单元是**瓶颈队列里的一条条目**：

| 区域 | 内容 |
|---|---|
| Explorer | 运行树（program → rank/device → invocation → 产物）+ **瓶颈队列**（10 条，按层从 E2E 往下排，severity 为人工标注，可按层筛选） |
| 中心 | 五个层级视图，用页面级 Tab 切换：E2E / L2 调度 / L1·L0 / 编译器 / ISA·布局 |
| Inspector | 跟随当前视图的对象（Run / 任务 / 提示 / Pass / 瓶颈条目）+ **实验台账**编辑器 |
| 底部 Dock | Visualization（AICPU 调度 / Ready queue / 核占用）与 Terminal（Problems / Output / Artifacts）互斥切换 |
| 状态条 | case、rank、trace span、关键路径节点数、AIC/AIV 占用、调度器占用、PMU 状态、进行中实验数 |

闭环被做成了产品约束，而不是提示语：

- **门禁先行**：E2E 视图第一屏是四个门禁（Case 固定 / 工具链 / 迭代次数 / PMU）。本次 dump 只有 2 次调用，
  「迭代次数」门禁直接是 warn，并写明 mean/median 不成立。
- **每轮只验证一个假设**：台账同时只允许一条 `open` 实验。已有进行中实验时，点开另一条瓶颈只会看到拦截提示和
  「回到 Ex」按钮，不给第二个「开始实验」。
- **顺序是固定的**：实验步进器必须先记录正确性，才出现性能输入框；两者齐了才出现「保留 / 回退」，
  并在决定前再显示一次该条目的护栏。
- **每条结论都带护栏和复测口径**：瓶颈条目由 `证据 → 杠杆 → 护栏 → 复测` 四段组成，护栏写的是这条杠杆的代价
  （例如 prefetch 占 SDMA、提前 dispatch 以吞吐换时延、PMU-on 不能与 PMU-off 基线比较）。

---

## 瓶颈 → 屏幕：证据是怎么对上的

点一条瓶颈之后，中间不会只是「换了个页签」。**证据条**会钉在中心区顶部，把这条结论和屏幕上的具体对象绑起来。
它只有两行，没有解说：

```
┌ F1  通信等待独占关键路径 36.72%   1791.82 us / 4879.82 us   [聚焦证据 ✓] [退出] ┐
│ 证据 4  ①cp_token_allgather_payload_wait 1080.52us  ②o_group_a2a_wait 663us … │
└──────────────────────────────────────────────────────────────────────────────┘
```

三处用**同一套编号**串起来：

| 位置 | 表现 |
|---|---|
| 证据条的 chip | `① ② ③ ④`，点一下把时间轴缩放到那个对象上 |
| 中心画布 | 关键路径 ribbon 多出一条「证据」行，泳道里对应的块加白框 + 同号圆标；非证据对象在「聚焦证据」开启时压暗到 16% |
| 右侧 Inspector | 证据小节标题变成「N 项 · 已在中间标号」，每条证据行带 `标号 1 / 2 / 3`，整行可点，跳到同一个对象 |

编译器层的瓶颈（F4 / F5）没有画布，改成**表格行高亮**：`is-subject` 行加主色左条和淡底，9 / 12 行一眼可辨。
F10 这种「缺席证据」的条目不给 chip，明确写出「本页没有可标记的对象」。

`聚焦证据` 只压暗、不隐藏——被压暗的东西仍然可悬停、可点击，避免把上下文一起删掉。
`退出` 清除瓶颈上下文，回到自由浏览；切页签不会清掉它，因为一条瓶颈常常要跨两层看。

### 页面上不写解释

这是工具不是教程：中心区没有任何叙述性段落，五个页签下的 `<p>` 数量是 0 / 0 / 4 / 0 / 0
（L1 的 4 条是试算器判定，全部是带数字的短读数）。说明性内容只存在三处，且都是可执行的：

- 瓶颈条目的 `杠杆 / 护栏 / 复测` 三张卡——这是要照着做的动作，不是背景介绍
- 编译提示 Inspector 的「处理」两张卡——`减少同驻 tile，而非调大 stage` 这类一句话结论
- 实验台账的拦截消息——`E1 进行中 · 结论后才能开下一个`

页签名不重复出现在页签下面：工具条只放控件（编译器的 `Pass 轨迹 / 流水深度 / 搬运粒度` 子切换、
L2 的泳道 / 着色 / 叠加、L1 的 kernel 选择、L2 / L1 的 rank 选择），没有控件的页签（E2E、ISA / 布局）
工具条整行隐藏。

### E2E 没有 rank 切换

L2 和 L1 整页都属于一个 rank——泳道、关键路径、任务表、占用率全部换掉，所以工具条上有 rank 选择。

E2E 不是：它的职责就是**比较**。四个区块里三个本来就同时列两个 rank（`每 rank / 每次调用` 4 行、
`对账` 2 行），只有 `调用剖分` 过去跟着全局 rank 走——切一次 rank，整页只有那一块的数字变。
现在 `调用剖分` 也改成两列并排、**共用一个分母**，于是 `runner_run.device_wall` 这一行直接读出
`5132.80 / 4017.34`，F3 的 1.28x 偏斜不用切页签就在同一行里。E2E 因此不再依赖 `S.rank`，rank 选择从它的
工具条上去掉了。

`S.rank` 仍然存在，含义变成「带去 L2 / L1 的那个 rank」：在两张表里点行即可选中，被选中的 rank 在
`调用剖分` 的列头和数值上加重，状态条上也始终写着 `rank rank0 inv=2`。

其余位置一律用「标题 + 计数 / 单位」做小标题，例如 `门禁 · 4 项`、`编译 IR 全流程 · 51 / 51 个 Pass 改动了 IR`、
`缺失产物 · 本 dump 不含 PTOAS / VPTO 级记录`。原来的 ISA 空状态是一段散文加项目符号，现在是一张三列表
（产物 / 用于 / 状态），四行全部标红「缺失」。

---

## 两 rank 怎么对齐的（F3 的推导）

两份 host log 的 `ts=` 来自**同一台主机的 CLOCK_MONOTONIC**（pid 2263908 / 2263922，连号），
所以两份各自从 t=0 起算的设备 trace 可以摆到同一条绝对轴上：

```
rank0  chip.run.runner_run  ts = 1717978439221135 ns
rank1  chip.run.runner_run  ts = 1717978440374926 ns   →  rank1 晚 1153.79 us
```

把这个偏移加到 rank1 各 `*_wait` 的到达时刻上，就得到 **rank0 在该点最多能等多久的上界**
（rank0 可以在 rank1 的数据落地时就被释放，而落地不晚于 rank1 自己走到 wait，所以是上界不是等式）：

| wait | rank0 到达 | rank1 到达（对齐后） | 上界 | 实测 | 占上界 |
|---|---|---|---|---|---|
| `cp_token_allgather_payload_wait` | 268.10 | 1385.33 | 1117.23 | **1080.52** | 96.7% |
| `o_group_a2a_wait` | 3732.34 | 4450.71 | 718.37 | **663.00** | 92.3% |
| `cp_token_allgather_readback_wait` | 1371.56 | 1491.09 | 119.53 | 47.02 | 39.3% |
| `tp_o_rs_wait` | 4779.52 | 4822.25 | 42.73 | 1.28 | 3.0% |
| | | | **1997.86** | **1791.82** | 89.7% |

四个全部落在上界内，两个主导项贴到 92–97%。配合「两卡 AIC busy 只差 0.67%、任务与块数完全相同」，
负载不均的零假设被排除——**rank0 不是慢，是先到**。

这条推导有两处不是实测:

1. 跨 rank 时钟同步是从同一主机的 mono `ts` 推的，dump 里没有显式的同步记录
2. 上界成立只说明「等待可以被错峰解释」，不证明错峰是唯一成因

两条都写进了 F3 的护栏，不在正文里冒充结论。错峰本身产生在
`runner_run` 123109.06 us（host 钟）对 `device_wall` 5132.80 us（设备钟）这段未拆解的主机时间里，
dump 内没有更细的 span 可归因。

---

## Ready queue 图表

`shared_ready_queue` 是 trace 里的 counter 事件（`ph: "C"`），每个采样点给出当刻
「依赖已满足但尚未被 AICPU 派发」的任务数，按引擎分三路。rank0 有 1128 个采样点，rank1 有 1016 个。

- **三条线**：AIC（danger）/ AIV（warning）/ MIX（accent），图例里各自带本 rank 的峰值
- **阶梯而非折线**：counter 的值保持到下一个采样点，所以两点之间画成水平段再跳变。
  这跟 `build-data.cjs` 积分 `busyTime` 的口径一致——如果画成线性插值，图和数字就是两套说法
- **悬停读数**：十字准星 + 三个系列的点，tooltip 给出光标时刻 `t`、命中的采样点编号与时间戳、
  这个值**保持多久**、三路各自的值和待派发合计。tooltip 用 `swimlane-task` pattern 的
  `createTooltip / showTooltip / hideTooltip`，只有行内容是本页的
- 图表与 L2 页签共用时间窗口：在 L2 缩放 / 平移后，dock 的刻度、曲线和悬停换算一起跟着变

---

## L2 调度页：时间花在哪个 scope，哪段时间核在空转

这一页替掉的是「Perfetto 打开 36k 事件 + 临时脚本按 kernel 汇总 core-time」那套流程。
右侧面板原本显示 L1 的 kernel 详情，现在换成 L2 自己的三段。

### 统计口径先说清楚

同一个块在 trace 里出现两次(Worker View pid 4 / Scheduler View pid 3)。
面板第一段把两个总量并排列出，并直接点名相加是错的：

```
Worker View     125572 us · 4038 块      ← scope 排行只用这一份
  kernel        108940 us
  setup          16632 us
Scheduler View  194708 us · 4038 块
hand-off 差     +69135 us
⚠ 两边相加得 320280 us —— 这是重复计数，不是总量
```

### scope 排行带 slack，不只是 Σdur

按 callable 归并成 scope，每行给 core-time、占比和 **slack**：

| scope | core-time | 占 | slack |
|---|---|---|---|
| `qk_pv_aic` | 46662 | 37.2% | **0** |
| `csa_merge_pack_publish` | 18597 | 14.8% | **0** |
| `indexer_score_leaf_wave_aic` | 6494 | 5.2% | **0** |
| `kv_score_proj` | 5654 | 4.5% | 172 |
| `qr_rms_norm_quant` | 3183 | 2.5% | 517 |

slack 来自 fanin/fanout DAG 上的前推/后推(ES/EF → LF/LS，用实测 span)，
`slack = LS − ES`。**这是结构 slack，不含资源争抢** —— 两个 slack=0 的任务仍可能在抢同一个核。

没有 slack 只看 Σdur 的话，`kv_score_proj`(4.5%)和 `qr_rms_norm_quant`(2.5%)会排在
前列看着值得动；加上 slack 就能看出它们分别有 172 us 和 517 us 余量，
真正卡住总时长的是前三个 slack=0 的。关键路径上的 scope 左侧有红色条。

### 空转窗口

把 run 切成 240 个等宽窗口(rank0 每窗 20.33 us)，统计每窗 AIC / AIV 的核占用。
AIC 与 AIV **同时**低于 15% 的连续窗口合并成一段：

| 窗口 | 时长 | AIC | AIV | 实测核容量占用 |
|---|---|---|---|---|
| 3721–4392 us | **671** | 0% | 2.2% | **1.5%** |
| 1159–1423 us | 264 | 0% | 2.8% | 1.9% |
| 2501–2623 us | 122 | 0% | 2.0% | 1.3% |

rank0 共 8 段、1199.6 us，占 span 的 24.6%。最长那段 671 us 里 72 个核只用掉 1.5% 容量，
而 `o_group_a2a_wait` 用 **1 个块**占了 660 us —— 单块任务挡住全部核，
这正是 F1 在 L2 层的具体形态。

点任意一行会把泳道时间窗收到那一段。

### 泳道上的占用率色带

泳道顶部固定两条色带(AIC / AIV)，透明度跟着每窗占用率走；
被判定为空转的窗口整列打上 warning 色底、两侧虚线、上方标 `671 us 空转`。
不用读 72 条泳道就能看出机器什么时候闲着。

**校验**:把 240 个窗口的占用率取均值，必须等于泳道法算出的平均占用 ——
rank0 `32.22% / 37.50%`，decode_fwd_layers `70.92% / 20.43%`，两边逐位相等。

---

## 两个 case

顶栏的 case chip 打开切换菜单。两份 dump 不是同一个程序,也不带同样的产物 ——
菜单里每一行直接标出它能回答哪几层:

| | decode_csa | decode_fwd_layers |
|---|---|---|
| 模型 | DeepSeek V4 flash_dspark | Qwen3 14B decode_layer |
| 层级 | L3,2 rank | L2,1 device |
| 采集 | 2026-09-03 | 2026-06-25 |
| span | 4879.82 / 3774.04 us | 993.24 us |
| 任务 / 块 | 84 / 4038(每 rank) | 426 / 579 |
| AIC / AIV 占用 | 32.2% / 37.5% | 70.9% / 20.4% |
| **E2E** | ✅ host STRACE,2 次调用 | ❌ **无 host log** |
| L2 调度 | ✅ | ✅ |
| L1 / L0 | ✅ | ✅ |
| 编译器 | ✅ 52 pass,PH001 + PH-MR-001 | ✅ 42 pass,**仅 PH001** |
| **ISA / 布局** | ❌ 无 PTOAS 产物 | ✅ **38 个 .pto + .cpp** |
| 瓶颈条目 | 10 条(F1–F10) | 7 条(F2、F5–F10) |

两份正好互补,但**不能拼在一起读**:不同模型、不同卡数、不同采集时间。

### 缺席是一种状态,不是错误

切到 decode_fwd_layers 时:

- **E2E 页签仍然可点**,显示的是一张缺失清单 —— `host.*.log` 缺失 →
  `inv=`(迭代次数)、`device_wall`、`bind.prebuilt` 全部不可得,下面跟一块"仍然可测"的设备侧读数
- **瓶颈队列少 3 条**:F1(无 `*_wait`,单卡)、F3(无 host 钟,无法对齐)、F4(无 PH-MR-001)。
  少的条目直接不生成,不占位、不补零
- **流水深度页签**不画空表,而是列出"PH-MR-001 × 0 / pl.pipeline × 38 / Left-Right 可用字节缺失"三条事实
- **片上预算试算器**的 depth 判定变成 `无法判定` —— 没有实测 free 字节可以对账
- **rank 选择器消失**(只有一个执行单元),状态条写 `device0 · 无 host log`
- **ISA 页签**反过来有真内容:38 个 PTOAS 单元的 `.pto → .cpp` 行数与展开比

生成器里这是一张 `CAN` 表,每条瓶颈只有在本 dump 真能支撑时才生成:

```js
const CAN = {
  F1: waitTasks.length > 0,      F2: !!worstHandoff,
  F3: !!launchSkew,              F4: depthDegraded.length > 0,
  ...
};
const findings = [ CAN.F1 && {...}, CAN.F2 && {...}, ... ].filter(Boolean);
```

### 两份 trace 的格式差异

同一个 `merged_swimlane`,两次采集的 swimlane level 不同,generator 里做了归一:

| | decode_csa | decode_fwd_layers |
|---|---|---|
| setup 时间 | 块事件上的 `local_setup_us` | **独立的 `setup` 事件** |
| `duration-us` | setup + kernel | **kernel only** |
| `kernel-duration-us` | 有 | 无 |
| ready 计数器 | `shared_ready_queue` | 多一组 `local_ready_buf_T{0,1,2}`(已过滤) |
| 调度相位 | dispatch / complete / release / early_dispatch / resolve / dummy | wire / dispatch / complete / release / resolve |

两边都归一成 `{dur = setup + kernel, kdur = kernel, setup}`,所以"一个块的三种口径"
在两个 case 下是同一个意思。

---

## 数据来源：每个数字都来自这次执行

`data.js` 由 `build-data.cjs` 从 dump 目录直接生成，没有任何建模、估算或造数。

```bash
node Design/operator-tuning-console/build-data.cjs
```

| 产物 | 提取出来的东西 |
|---|---|
| `distributed_meta.json` | 55 个绑定参数的 shape / dtype / 方向 → Case fingerprint |
| `dfx_outputs/rank{0,1}/d0/host.*.log` | STRACE host span（bind / runner_run / device_wall / graph_build / sched / orch）→ E2E 剖分；`ts=` 还用来对齐两 rank，见下 |
| `dfx_outputs/rank{0,1}/d0/merged_swimlane_*.json` | Worker View（pid 4）每块的 `duration-us` / `kernel-duration-us` / `local_setup_us` / CoreId；Scheduler View（pid 3）同一块的 `dispatch-time-us → finish-time-us`；AICPU scheduler phase；`shared_ready_queue` 计数器；dependency / hb_violation flow |
| `dfx_outputs/rank*/d0/deps.json` | `block_num` / `scope` / `early_dispatch` / 每个任务的绑定张量 |
| `dfx_outputs/rank*/d0/name_map.json` | 64 个 callable id → 名字 |
| `report/perf_hints.log` | 230 条 perf hint：197 条 PH001（搬运末维粒度，累计 261 次命中）+ 33 条 PH-MR-001（软流水深度回退） |
| `passes_dump/` | 52 份 IR dump 的行数、DSL / 内存空间计数；每个相邻快照的真实增删行、受影响函数和最多 6 个改写片段；AutoTileMatmulL0 的真实 L0 tile 形状与 55 个 pipeline 站点 |
| `next_levels/.../binary_context.json` | platform、pto-isa revision、runtime 名与 revision → 工具链指纹 |

### 两个视角，不要混

Trace 里同一个块有两份记录，工具把它们分开显示，因为它们回答的是不同问题：

- **Worker View（pid 4）**：核上发生了什么。`duration − kernel_duration = local_setup`。
- **Scheduler View（pid 3）**：AICPU 看到的 `dispatch → finish`。它减去核上时长就是领取与依赖等待。

L1 视图的「一个块的三种口径」就是这三层：`kernel` → `+ setup` → `+ hand-off`。

### 关键路径怎么算的

用每个任务的 `fanin-hint` 建图，按实测 start 排序做 DP，取加权最长链（权重是任务自身的 span）。
rank0 得到 33 个节点、链上 span 合计 4985.0 us；链上正向间隙 378.3 us，重叠 539.7 us
（重叠为负间隙，说明后继在前驱最后一块结束前就起来了）。

### 对账

E2E 视图会把 host 报的 `device_wall.sched` 与设备 trace 的跨度对齐，确认这份 trace 属于哪一次调用。
两个 rank 都命中 `inv=2`，偏差 < 1%。不对账就无法保证「看的 trace 和读的数字是同一次运行」。

---

## 瓶颈条目（全部由数据推导，非人工填写）

下表是 **decode_csa** 的 10 条。decode_fwd_layers 只生成得出 7 条（缺 F1 / F3 / F4），
理由见上面的「两个 case」。

| ID | 层 | 结论 | 实测依据 |
|---|---|---|---|
| F1 | L2 | 通信等待独占关键路径 36.72% | 4 个 `*_wait` 任务合计 1791.8 / 4879.8 us，全部单块单核 |
| F2 | L1 | `csa_merge_pack_publish` hand-off 比核上计算还贵 | AICPU 1032.1 us vs 核上 387.4 us（+644.6）；核上 57.8% 是 setup |
| F3 | E2E | rank1 晚发 1153.79 us，rank0 在集合点替它等 | 两卡 AIC busy 差 0.67%，4 个 `*_wait` 全部落在错峰上界内（1792 / 1998 us） |
| F4 | 编译器 | 9 处软流水深度被降到 1 | 33 条 PH-MR-001；Left/Right 32–64 KB/stage vs 64 KB free |
| F5 | 编译器 | 261 次搬运末维 < 512B cache line | 最小 4B，覆盖 9 个算子文件、192 个源码点 |
| F6 | L2 | AICPU 调度器平均占用 42.3% | 3 线程合计 busy 6143.9 us；complete 3452.6 us / 4038 次 = 0.855 us/次 |
| F7 | L2 | AIC ready-but-undispatched 占窗口 26.6% | `shared_ready_queue` avg 0.433、peak 6，同期 AIC 核占用仅 32.2% |
| F8 | L1 | `qr_hadamard_matmul` 块时长离散 7.82x | 256 块 / 24 核，max 19.7 vs med 2.52 us |
| F9 | L1 | `qk_pv_aic` 混合核 span ≈ 单块时长 | 72 块 / 72 核，span 886.4 us，单块 648.1 us |
| F10 | L2 | 全程 0 处 `pl.prefetch` | 前端 IR 计数；10 个 `w*` 权重参数存在但无静态预取 |

---

## 片上预算试算器

L1 视图里的试算器不是通用公式演示，它对着**本 run 的实测上限**校验：

- `Left = M × K × bytes(AB)`，`Right = K × N × bytes(AB)`，`Acc = M × N × bytes(Acc) × live`
- Left / Right 的可用空间取本 run MemoryReuse 报告里的 `65536 B`；Vec 侧是 `188416 B`
- Acc 不报上限，所以只给「本 run 出现过的最大 Acc tile = 128 KB」作为**下界**，并明确说明超过它属于待验证
- 末维检查：`N × bytes(AB)` 对 512B cache line；不足时给出该 dtype 的元素倍数（INT8 512 / BF16·FP16 256 / FP32 128）

默认值就是 dump 里真实存在的那条 K 循环：`Left INT8[16,64]` + `Right INT8[64,512]` + `Acc INT32[16,512]`，`stage=2`。
在这个配置下 `stage × max(L,R) = 65536 B` 正好等于 free，但本 run 的 `qkv_proj_rope.py:375` 仍然只放下 1 个 buffer
——同驻 tile 会分走这块空间。试算器因此把「刚好等于上限」判为**放不下**，而不是刚好通过。这是用实测反例校准的判据，
不是理论公式。

点右侧「AutoTileMatmulL0 dump 里真实出现的 L0 tile」任意一行，可以把该形状回填到试算器。

---

## 诚实的空缺

- **没有 kernel → 源码映射**。perf hint 挂在源码位置上，IR 只保留 outline 后的 incore scope 名，两者之间
  这份 dump 不提供可验证的对应关系。L1 视图因此不做自动归因，只提供模块选择器 + 明确说明「对应关系需人工确认」。
- **没有 PTOAS / VPTO 级产物**。ISA 视图只给这份 dump 能支撑的结论（工具链指纹、布局与内存空间分配、L0 tile 清单、
  512B 约束），并列出需要补齐哪些产物（TileLib 模板选择记录、VPTO 指令排布报告、cycle cost model 预测、PMU counter）
  才能把结论推进到指令层。
- **没有 PMU**。trace 里没有硬件 counter，门禁标为 `off`，并注明 PMU 打开会改变调度，不能与本基线直接比较。
- **只有 2 次调用**。所以工具不显示 mean/median，只显示每次调用的值，并在门禁里把这点标成 warn。

---

## 设计系统落地

页面是 `vendor/pto-design-system` 的消费者，不自建视觉语言。

- Shell：`patterns/ide-frame`（standalone host、activity rail 四键、explorer/inspector toggle、bottom dock 互斥、status strip），
  resize 交给它委派的 `patterns/workbench-shell`。保留共享渐变 / aura / pane 磨砂皮肤，只放宽 4:3 为全视口
  （与仓库里其它全屏 Demo 一致）。
- 所有计时任务条与 hover tooltip 走 `patterns/swimlane-task`：`drawTaskBar`、`createTaskColormap`、
  `initHoverTooltip` + `showTooltip` / `hideTooltip`。没有本地重写任务条几何、配色哈希或 tooltip 行为。
- 不使用播放条：这个页面没有 demo 时间轴 / step / scrubber 语义。
- 颜色：`styles.css` 里 0 个硬编码色值，全部走 token；`app.js` 里 0 个私有调色板，lane/任务配色全部由
  `createTaskColormap()` 决定。
- 默认 dark（新建 standalone PTO 页面的默认），右上角可切 light。

### 一条对齐基线

所有容器标题与内容区文字落在同一条竖线上：pane body 取与 `.pto-ide-frame__pane-header` 相同的
`--tc-gutter: 10px`，而带内边距的表面（表格滚动区、canvas、gate / tile 卡、台账条目、树行、瓶颈卡）
用 `--tc-bleed` 反向外溢自己的 cell padding，于是表面边缘压进 gutter、表面内的文字正好回到 gutter 上。
这是 VS Code 侧栏的老做法：行的高亮满幅，文字对齐。

### 已知例外

1. **全视口**：`.tc-frame` 放宽 `aspect-ratio` 为 `auto`、去掉圆角与投影。渐变、aura、pane fill、blur 全部保留，
   没有用页面私有面板替换共享皮肤。与 `deepseek-gap-investigator`、`pass-decision-studio` 的处理一致。
2. **Canvas 内 data-viz 标注低于 11px**：72 泳道的 lane label 用 10px，时间轴刻度用 11px。行高只有 8px，
   放大字号会互相压盖；这属于设计系统允许的「可缩放 data-viz 内部标注」例外，且所有信息都有 hover tooltip
   与 Inspector 作为 fallback。`swimlane-task` pattern 自身的条内文字也是 8–9px。
3. **inspector-section 家族在本页实现**：`.inspector-rail / .inspector-section / -head / -title / -kicker`
   与 `.inspector-soft-card` 写在 `references/quick-reference.md` 里、token 定义在 `tokens/components.css`，
   但设计系统没有发布对应的 CSS 规则（仓库里每个 Demo 都各自实现）。本页按那组 token 实现，不引入新值。
   在此之前它们完全没有样式，`<h3>` 回落到浏览器默认的 16px 粗体 block，导致「瓶颈队列 / 10 / 10」换行。
4. **封面是卡片不是截图**：`Design/assets/launch-previews/tuning-console.svg` 是一张说明性封面卡，
   里面内联了 token 色值（`<img>` 加载的 SVG 拿不到宿主页面的 CSS 变量）。它没有伪装成产品截图。

### 审计

```bash
cd vendor/pto-design-system
node scripts/audit-typography.mjs ../../Design/operator-tuning-console/styles.css
node scripts/audit-theme.mjs      ../../Design/operator-tuning-console/styles.css
```

两项均无告警。

---

## 交互速查

| 操作 | 结果 |
|---|---|
| 点瓶颈条目 | 跳到证据所在层级，钉出证据条，给证据对象编号并压暗其余部分 |
| 点证据 chip / Inspector 证据行 | 跳到同一个编号对象；时间轴自动缩放过去，Inspector 保持停在这条瓶颈上 |
| 证据条「聚焦证据」 | 压暗非证据对象（不隐藏，仍可悬停点击）；再点一次恢复 |
| 证据条「退出」 | 清除瓶颈上下文；切页签不会清除，方便跨层追同一条 |
| `1`–`5` | 切换五个层级视图 |
| `/` | 聚焦搜索（kernel / 任务 / 编译提示 / Pass，回车跳第一条） |
| L2 画布点击 | 选中任务 → Inspector 显示绑定张量、依赖链、关联瓶颈、所在核占用 |
| L2 画布 shift + 拖动 | 水平平移；工具栏 `+` / `−` / `Fit` 缩放，Dock 时间轴跟随同一窗口 |
| Dock「核占用」 | 72 条泳道的占用、空洞数、最大空洞、首末块 |
| Terminal「Problems」 | 230 条编译提示当作 IDE 问题列表，点击跳到对应源码点 |
| Inspector 依赖 chip | 沿 fanin / fanout 在任务图里走 |
| 试算器里点 L0 tile 行 | 把该真实形状回填进试算器 |

---

## 文件

```
Design/operator-tuning-console/
├── index.html        ide-frame shell 与槽位
├── styles.css        仅页面级布局；颜色全部走 token
├── app.js            状态机、五个视图、canvas 渲染、台账逻辑
├── build-data.cjs    从 dump 目录生成 data.js（可重跑）
├── data.js           生成产物，勿手改
└── README.md
```
