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

## E2E 页：证据层阶梯

E2E 页以前是一堆表（门禁 / 每 rank 每次调用 / 调用剖分 / 对账），
它们各自成立，但读者不知道**这一层能回答什么、不能回答什么**。
现在整页以一张证据层阶梯开头，每层三件事：输入、回答的问题、**不能单独证明什么**。

| 层 | 输入 | 回答 | 不能单独证明 | 本 dump |
|---|---|---|---|---|
| 服务层 | `serving-strace-swimlane.json`、端到端 benchmark | 哪个 WorkerProcess 负载或尾耗时异常 | AICore 内某个 kernel 为什么慢 | **缺** |
| Host / Device | `BenchmarkStats`、独立 benchmark | 延迟是 Host、Device，还是两者共同贡献 | Device 内的依赖与 pipe 根因 | **部分** |
| 函数汇总 | `name_map*.json` + Swimlane | 慢来自单次慢、次数多，还是波动 | 是否影响 wall-clock | **齐** |

三层里只有一层齐，这本身就是结论：

- **服务层整层缺席** —— 这份 dump 是单进程 JIT run，没有 serving 侧采集，
  多 WorkerProcess 的负载与尾耗时无从谈起。
- **Host / Device 只有一半** —— 有 STRACE host span（bind / runner_run / device_wall / sched，
  两种时钟已对齐），但没有 `BenchmarkStats`：没有 rounds / warmup 统计量，
  本 case 只有 2 次调用，给不出 mean / median。decode_fwd_layers 连 STRACE 都没有，整层缺。
- **函数汇总齐** —— 62 个 scope / 64 个 kernel 名，每块时长齐全。

### 「不能单独证明」是按钮，不是脚注

每层那一格是可点的：点「AICore 内某个 kernel 为什么慢」跳 L1，
点「Device 内的依赖与 pipe 根因」和「是否影响 wall-clock」跳 L2。
把一句承认的局限变成一条去路，比写在角落里的免责声明有用。

## 函数汇总：Σ = 重复 × 宽度 × 均值

「慢来自单次慢、次数多，还是波动」这三项要能分开，Σ 就得先拆对。

第一版我拆成 `块数 × 均值`，错了：**块数把两件事混在一起**。

```
fa_fused    72 块  =  重复 1  × 宽 72 核     ← 一次铺开占满 72 核，不是调用了 72 次
up_proj     85 块  =  重复 85 × 宽  1 核     ← 真的是 85 次 launch
```

按块数算，两个都是「次数多 ×72 / ×85」，而第一个其实是**单次慢**。
所以拆成三项：

| 项 | 定义 | 对应问题 |
|---|---|---|
| 重复 | launch 次数 × 波数（同一批核跑了几轮） | 次数多 |
| 宽度 | 一次铺开占几个核 | 并行度，不是成本 |
| 均值 | 单块平均时长 | 单次慢 |
| p90 / 中位 | 分布的宽窄 | 波动 |

两个恒等式都精确成立（最大偏差 0.17 us，来自波数的 r2 舍入）：

```
块数 = 重复 × 宽度
Σ    = 块数 × 均值
```

还有一列 `偏离中位` = `Σ − 块数 × 中位`，**带符号**。
为负说明中位高于均值，是少数快块把均值拉低了，不是长尾问题 ——
`qk_pv` 就是 −9667（中位 782 vs 均值 648）。第一版我把它 clamp 到 0，
恰好把更有意思的那种情况藏了。

「主因」列给的是**相对本 run 所有 scope 中位数的倍数**，不是绝对阈值：

```
qk_pv               Σ46662  重复   1  宽72核  均值 648.08  单次慢 ×69     依赖关键路径
kv_score_proj       Σ 5654  重复21.3 宽24核  均值  11.04  次数多 ×21     观测路径
tp_o_a              Σ 2540  重复   8  宽16核  均值  19.85  次数多 ×8.0 + 波动  观测路径
qr_hadamard_quant   Σ 2265  重复 5.3 宽48核  均值   8.85  次数多 ×5.3 + 波动  都不在 · slack 640
qproj_matmul        Σ 2218  重复 2.7 宽24核  均值  34.65  两者兼有       都不在 · slack 1794
```

### 末列就是这一层证明不了的那件事

`在路径上` 一列来自「路径归责」那一层。这不是装饰 ——
`qr_hadamard_quant` 是个 2265 us、波动 2.35 倍的 scope，看 Σ 排行会想去动它，
但它**两条路径都不在，slack 640 us**：把它优化掉不缩短墙钟。

把「不能单独证明」直接做成表里的一列，比只写一句话更难被忽略。

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

### 下钻可返回

scope 行和空转行都会改写时间窗，所以两种下钻都留了回路：

| 动作 | 改了什么 | 面包屑 | 返回方式 |
|---|---|---|---|
| 点 scope 行 | 面板 → kernel 详情 + 时间窗 | `← scope 排行 / csa_merge_pack_publish` | 点面包屑，或 Esc |
| 点空转行 | 只改时间窗，面板不动 | `← 恢复时间窗 / 3587–4526 us` | 同上 |

返回时时间窗恢复到下钻前的值（不是重置成全量）。连续下钻不会覆盖最初的返回点。

### 泳道上的占用率色带

工具条上 `着色` 是一个开关。**关掉之后所有任务条变中性灰**（`--surface-4`，
深色 `#313131` / 浅色 `#CCCCCC`），对比度全部让给占用率色带、空转底色和关键路径，
图例也跟着换成「任务（配色已关）/ 空转窗口 N 段 / 关键路径 N 节点」。

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

## 关键路径归责：照着 simpler 的算法重写了一遍

原来这里只有**结构 slack** —— 在 `deps.json` 上前推 ES/EF、后推 LS/LF。
它能说「这个 scope 动了能缩短总时长」，说不了「这段时间到底在等什么」。
面板里那句「不含资源争抢」就是在认这个账。

`repo/simpler/simpler_setup/tools/critical_path.py` 正好补这一块。把它的算法搬了过来。

### 两条路径，不是一条

**静态 CPM** —— 依赖决定的延迟下界（无限核）。

依赖边先按实测时间戳过滤，只有**真的先完成**的才算 happens-before：

```js
if (en[ptag] <= st[t.tag] + TOL && st[ptag] < st[t.tag]) keep.push(ptag);
```

`start(前驱) < start(本节点)` 这个严格条件让边保留是反对称的 ——
**即使时间戳打平，保留下来的图也可证明无环**，后面的 DP 和反向走查才安全。
decode_csa rank0 上 133 条边留 126、弃 7。

**观测路径** —— 从最后完成的任务往回「归责」走，每步在两类前驱里挑卡得最紧的：

| 前驱 | 记为 |
|---|---|
| 数据依赖（happens-before 边） | `data-wait` |
| 同核资源（这条泳道上此前被释放的最晚时刻，running max） | `core-wait` |
| 一个都没有 | `front-gap` |

同核前驱用 running max 而不是「前一块」，所以流水重叠的切片也算得对。

### 第一版关键路径是错的，这次修了

demo 从第一次提交（2026-09-21）起就有一条关键路径：在 **原始 deps 图**上走最长链。
它把每条依赖边都当成 happens-before，于是消费者实际先于生产者结束就开始时
（早发、或者只表达所有权不表达顺序的 lifetime 边），两段时长被相加而不是取其一。

```
decode_csa / rank0
  老：33 节点，chainSpan 4985.02 us  =  makespan 的 102.2%
```

**一个「依赖决定的延迟下界」不可能超过它所界定的墙钟时间。** 查下去，那条链上有
5 条边的后继在前驱结束之前就开始了，重叠合计 539.74 us：

```
r2t53 qk_pv          end=3520.66  →  r2t55 csa_merge_pack_publish start=3018.50   重叠 502.16
r2t12 allgather_push end=288.16   →  r2t13 payload_wait          start=268.10   重叠  20.06
r2t11 rms_norm       end=255.28   →  r2t12 allgather_push        start=245.60   重叠   9.68
...
```

过滤后的 CPM 是 14 节点 / 3066.8 us / 62.85%，而且这 14 个是老的 33 个的**严格子集**
（新独有 0 个）——老的不是算岔，是多吞了 19 个本不该串起来的节点。

现在 `critical` 直接由 `cpath.cpm` 派生，全 demo 只有一条关键路径定义。

### slack 也在同一张图上跑了

修完第一处，第二处不一致立刻暴露：关键路径 14 节点，而 `slack == 0` 的任务有 33 个。
因为 slack 的前推/后推还在用原始 `fanin/fanout` 边——没有真正卡住消费者的边照样把
ES 往前推，slack 因此偏小，太多任务读成零 slack。

改成同一张过滤图后：

```
rank0    路径 14 节点   slack==0  14   onCrit  14
rank1    路径 11 节点   slack==0  11   onCrit  11
device0  路径 14 节点   slack==0  14   onCrit  14
```

**这改变了实际建议。** 第二大 scope `csa_merge_pack_publish`（18597 us，14.8%）
原来标 `slack 0 / 在关键路径上`，现在是 `slack 1341.94 us / 不在` ——
从「动它直接缩短总时长」变成「它胖但不急，先看并行度」。
scope 排行里被误标红边框的一共 19 个。

### 下游读数也跟着改了（这一步之前漏了）

改完数据层不等于改完。过滤后 CPM 从 33 掉到 14 节点，而观测路径是 32 节点 ——
页面上两个东西都叫「关键路径」，于是直接自相矛盾：

```
L2 路径归责表   第一行    q_rope_prepare   stall 88.1 us（全路径最大）
L1 页签         同一任务   「不在关键路径上」
```

观测路径 32 个节点里有 23 个不在 CPM 上，所以这不是偶发。现在两条分别命名：

| 名字 | 是什么 | 动它得到什么 |
|---|---|---|
| **依赖关键路径**（静态 CPM） | 依赖决定的延迟下界 | 降下界 |
| **观测路径**（反向归责） | 计算 + stall 精确铺满 makespan | 去掉 stall |

逐项改动：

- **泳道顶部的色带**改画观测路径 —— 它本来就画在真实时间轴上，就该显示铺满这根轴的那条。
  原来画 CPM，表头却写「走完 4879.8 us」，而那条链只有 3066.8 + 185.8 = 3252.6 us。
  现在表头是 `计算 4597.5 us + stall 282.3 us = 4879.8 us`，
  CPM 节点在条上用红色 tick 标出，行标签从 `CRIT PATH / GAP` 改成 `OBS PATH / STALL`。
- **L1 任务头和 Inspector** 报两条的归属：`在 依赖关键路径 1/14 · 观测路径 6/32`，
  或者 `两条路径都不在`。
- **工具栏「只看关键路径」开关**变成三选：全部任务 / 观测路径 / 依赖关键路径。
  原来那个开关在修复后只剩 14 个任务，而且没说是哪一条。
- **F1 标题改了**。原来是「通信等待独占关键路径 36.72%」—— 那个 36.72% 是
  wait span 占 **makespan**，走的是观测路径；4 个 `*_wait` 里只有 2 个在 CPM 上。
  现在是「通信等待占 makespan 36.72%」，claim 里把两条的归属分别说清。
- **F1 的证据来源改名**：不再写 `critical path (fanin/fanout hints)` ——
  图已经按实测时间戳过滤过，不是原始 hint 图了。证据里同时列出两条路径各自的数。
- **F9 的路径证据**从「在/不在关键路径上」改成 `依赖关键路径第 14 节点 · 观测路径第 16 节点`。
- E2E 对比、ISA、状态栏、scope 排行 tooltip 里所有裸的「关键路径」都加了限定词。
  回归脚本里加了一条检查：页面文本中不允许出现不带限定词的「关键路径」。

### 两条不变量，现在是算出来摆在界面上的

```
✓ 归责闭合   compute + stall = makespan（差 0.00 us）
✓ 依赖下界   CPM ≤ makespan（3067 ≤ 4880 us）
```

第二条就是这次被违反的那条。它一直成立不了，只是**之前没有人检查**。
过滤生效的另一个证据：路径上的残留重叠从 539.74 us 变成 **精确 0**。

### 归责闭合检查

正向 frontier sweep 保证 compute + stall **精确铺满** makespan：

```js
const gap = Math.max(0, a - frontier);
const eff = Math.max(0, b - Math.max(a, frontier));
frontier = Math.max(frontier, b);
```

三个 rank 全部 `delta 0.00`。**这个检查不过，下面的逐节点归责就不成立** ——
所以它是算出来摆在面板上的，不是拿文字声称的。

```
decode_csa / rank0   4879.82 = 4879.82   ✓
decode_csa / rank1   3774.04 = 3774.04   ✓
decode_fwd_layers    993.24  = 993.24    ✓
```

### 照搬会错的一处：模型把等待算成了 compute

`critical_path.py` 的 `dur = end - start` 是节点的 **wall span**。
路径上的 `*_wait` 是通信等待，它的 span 占着路径但根本不是计算。直接套判据会得到：

```
rank0  compute 94.21%  stall 5.79%   →  "compute-bound"
```

而这 4597.5 us「compute」里有 **1791.82 us（39%）是 4 个 `*_wait`**。
所以这里把 compute 拆成**真正计算**和**通信等待**，并让它参与判据：

```
rank0   真算 57.5%   等待 36.7%(4 个)   stall 5.8%   →  通信受限
rank1   真算 93.4%   等待  0.1%(2 个)   stall 6.5%   →  计算受限
```

**这和 F3 是两条独立推导，结论一致。** F3 是从 host CLOCK_MONOTONIC 对齐推出
rank1 晚启动 1153.79 us、rank0 在空等；归责走查从设备侧时间戳独立得到
rank0 通信受限 / rank1 计算受限。而且 F1 的 36.72% 与归责的 `waitShare` 精确相等，
4 个 `*_wait` 全部落在归责路径上。

### 两条路径不能互换

静态 CPM 的 14 个节点里，9 个也在观测路径上，**另外 5 个观测路径从不经过**。

- 动只在 CPM 上的节点 → 降**依赖下界**
- 动只在观测路径上的节点 → 去掉 **stall**

面板上用红色左边框区分，并把 5 个 CPM-only 的 tag 直接列出来。
提优化建议时必须说清在动哪一条。

### 容差

参考工具用 2 个时钟 tick。decode_csa 有 `clock_freq_hz = 50 MHz` → **0.04 us**。
decode_fwd_layers **没记时钟频率**，退回到时间戳精度的 2 个量子（0.02 us），
界面上标明用的是哪一种，不假装有时钟。

### 还没有的

- 参考工具按 `(task, core-block)` 切片建图，这里按 task 建图、用泳道块算同核前驱。
  块级的 core-wait 比任务级更细。
- 没有 `CPM_static.json` / `CPM_observed.json` 那种把 off-path 任务改名的 Perfetto 导出。
- 归责用的 span 含 setup（泳道上块从 setup 起画），参考工具用的是裸 kernel tick。
  对 core-wait 而言含 setup 更对 —— 核在 setup 期间确实被占着。

---

## scope 和 kernel 不是一回事

这两个词在通用 trace 工具里会混成一个，在 PTO 里不是。

```
源码        with pl.spmd(NUM_QK_CORES, name_hint="qk_pv") as qk_tid:
              └─ 一个 InCoreScopeStmt            ← 这是 scope
                 │
OutlineIncoreScopes (09_after_…)
                 └─ Function(InCore) 名叫 qk_pv
                    │
ExpandMixedKernel (23_after_…)
                    ├─ qk_pv_aic  FunctionType::AIC   ← 这是 kernel
                    └─ qk_pv_aiv  FunctionType::AIV   ← 这也是 kernel
                       两个被一个 Group 函数依次调用
```

**一个 scope 编译出 1 个或 2 个 kernel。** 纯 Cube 或纯 Vec 的 scope，
`ExpandMixedKernel` 只把 `FunctionType::InCore` 改成 `AIC` / `AIV`，
不拆不改名；**同时含 Cube 和 Vec 算子的混合 scope 才被拆成两半**，各带 `_aic` / `_aiv` 后缀。

这个 dump 直接印证：

| | decode_csa | decode_fwd_layers |
|---|---|---|
| scope（源码 pl.spmd 区域） | 62 | 38 |
| kernel（`name_map.json` 里的名字） | **64** | **39** |
| 差额 = 混合 scope 数 | 2（`qk_pv`、`indexer_score_leaf_wave`） | 1（`fa_fused`） |

### 之前这里是错的

两半**共用一次 Group launch，所以在 trace 里是同一个 `taskId`**，
只有 `event-hint` 里的 `FuncId` 能把它们分开：

```
taskId 8589934645  tag r2t53  FuncId 48+49  72 blocks
taskId 12884901902 tag r3t14  FuncId 42+43  72 blocks
（84 个 task 里只有这 2 个带双 FuncId）
```

原来的 `build-data.cjs` 按 `taskId` 分组、`funcId` 只取第一个事件的，
于是 Vec 侧的 31626 us 被记到了 Cube 侧 `qk_pv_aic` 的名下。现在按 FuncId 重新分组，
每个 task 带一个 `kernels[]`，不变量 `Σkernel.coreTime == scope.coreTime` 在两个 case 上精确成立。

## 引擎配对：AIC / AIV

L2 右栏的「引擎配对」分区回答的是「Cube 和 Vec 各花了多少、混合核的两半谁拖谁」。

```
AIC (Cube)   37740 us · 30.1% · 32.22% 占用
AIV (Vec)    87833 us · 69.9% · 37.50% 占用
Cube : Vec   1 : 2.33

qk_pv                         46662 us
  qk_pv_aiv   V   31626   48/48   最长 881
  qk_pv_aic   C   15036   24/24   最长 830
```

**两侧最长块 830.3 / 881.2 us，相差 1.061 倍，而整段 span 只有 886.4 us。**
如果两半真的并行，span 会接近单侧最长块；现在 span ≈ 两侧最长块本身，
说明 Cube 段和 Vec 段是在**块内串行**跑的。这是 F9 的直接证据 ——
原来 F9 是靠「span ≈ 单块时长」侧面推的，现在是把两半拆开量出来的。

scope 排行也多了一列引擎标记：`C` / `V` / `C+V`，悬停给两侧的 core-time、块数和最长块。

## spmd 展开

`pl.spmd(N)` 把一个 scope 铺到 N 个核上。trace 只记块和 core id，
不记这次 launch 的形状，所以「铺了多宽、跑了几波、铺得匀不匀」要按 scope 重新汇总。

| 列 | 含义 |
|---|---|
| 核 | 这个 scope 最宽一次展开占了几个核 |
| 块 | 块数 |
| 波 | 块数 / 核数。1 波 = 一次填满；>1 波 = 同一批核要跑好几轮，每轮之间有一次完成回收 |
| 离散 | 最长块 / 中位块。>2 = 同一次展开里各块负载不均，最慢的那块决定 scope 什么时候结束 |

两个 case 的形状完全不同，这一维一眼能看出来：

```
decode_csa          48 个多核 scope / 14 个单核，最宽 72 核
                    22 个多波 scope，最多 21.33 波
                    qr_hadamard_matmul  24 核 256 块 10.7 波 离散 7.82
                    kv_score_proj       24 核 512 块 21.3 波 离散 2.97

decode_fwd_layers    4 个多核 scope / 34 个单核，最宽 72 核
                    没有多波、不均或变宽的展开
                    426 个 task 里 422 个是单核单波
```

没有值得看的展开时，这一段不拿 `1/1/1` 的行凑数，直接说「本 case 的形状问题在别处」。

## scope → 源码：dump 里没有，但能重建

泳道上的算子名不是编译器发明的。每个外联 scope 都以源码里的
`pl.spmd(..., name_hint="X")` 命名：

```python
# decode_sparse_attn_csa.py:208
with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", deps=[qk_plan_tid, cache_ready_dep],
             allow_early_resolve=True) as qk_tid:
```

从入口 `decode_csa.py` 追传递导入(13 个模块)、索引其中所有 `name_hint`，
就能把 trace 里的 scope 打回源码。**映射的粒度是 scope 不是 kernel** ——
混合 scope 拆出的 `_aic` / `_aiv` 共用一个 `name_hint`，指向同一处源码：

| | |
|---|---|
| 覆盖 | **62 / 62** scope |
| 唯一定位到 文件:行 | **54** |
| 多候选(同名 hint 出现在多处) | **8** |
| 未匹配 | 0 |

编译器加的两级后缀要先折回去：

- `_aic` / `_aiv` —— `ExpandMixedKernel` 拆 mixed kernel。
  `qk_pv_aic` → `qk_pv`。现在 scope 本身就叫 `qk_pv`，这一层是精确命中，不用去后缀
- `_0` —— **同一处源码被实例化两次**，不是两处源码。
  `decode_csa.py` 只 import 了 `compressor_ratio4`，却在 line 353 与 892 各调一次，
  两次 tile 常量不同，实测块数 512 / 256 正好对上 —— 于是有
  `kv_score_proj` 与 `kv_score_proj_0`

另外 `_spmd` 是**前端追踪时追加**的：源码写 `name_hint="csa_merge_pack_publish"`，
`00_frontend.py` 里才变成 `..._spmd`。别拿前端 IR 当源码读。

### 界面上怎么呈现

- L2 scope 排行每行第二列给 `文件:行`，多候选标 `+N` 并染成 warning 色
- L1 kernel inspector 多一行 `源码`；多候选或去后缀匹配的，下面补一张卡说明
- ISA 页签新增 `kernel → 源码` 段，写明**来源不在 dump 内**、依据是什么、
  以及**不能做的事：只给出 scope 写在哪里，不把实测块时长归到某一行**
- decode_fwd_layers 的 Qwen3 源码树不在仓库里，同一段显示为缺失并说明可重建

### 泳道图例去掉了

原来按算子着色时画 8 个色块，是 `slice(0, 8)` 的显示上限，不是统计量 ——
62 个 scope 列 8 个既不完整，在 decode_fwd_layers 上还会重名(426 个任务只有 38 个
callable，前 8 个任务里 5 个都是 `up_proj`，颜色还完全一样)。

62 路分类本来就没有可读的图例。现在改成一行读数：

```
62 scope 各一色 · 颜色只用于区分相邻块，名字看悬停或右侧排行
```

按引擎着色仍保留 AIC / AIV / MIX 三项 —— 那是真能查的图例。

---

## 诚实的空缺

- **dump 内没有 kernel → 源码映射**。decode_csa 的这一条已由模型源码按 name_hint 重建（62/62 覆盖、54 唯一、
  8 个多候选，见上一节）；decode_fwd_layers 的 Qwen3 源码树不在仓库里，仍然缺。重建出来的是「scope 写在哪里」，
  **不是**「哪一行耗了多少时间」——perf hint 自带的行号与它是两条独立证据，不要互相当作确认。
- **没有 PTOAS / VPTO 级产物**。ISA 视图只给这份 dump 能支撑的结论（工具链指纹、布局与内存空间分配、L0 tile 清单、
  512B 约束），并列出需要补齐哪些产物（TileLib 模板选择记录、VPTO 指令排布报告、cycle cost model 预测、PMU counter）
  才能把结论推进到指令层。
- **没有 PMU**。trace 里没有硬件 counter，门禁标为 `off`，并注明 PMU 打开会改变调度，不能与本基线直接比较。
- **只有 2 次调用**。所以工具不显示 mean/median，只显示每次调用的值，并在门禁里把这点标成 warn。
- **只有一次采集**。每个百分比都来自单次 run。同一负载两次采集的 stall 占比能差几个百分点，所以不要拿「各采一次」的两个配置做对比 —— 参考 skill 的原话是 one capture, one sample。另外 makespan 含首轮 warm-up，不是稳态。
- **归责是任务级，不是块级**。参考工具按 (task, core-block) 切片建 happens-before 图；这里按 task 建图，只有同核前驱用到了泳道块。块级的 core-wait 会比任务级更细。

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

## 右栏的密度：默认开三个，其余折叠

L2 页曾经把 7 个分区全部展开、配 12 段说明文字，右栏要滚 6.3 屏，
而其他每个页签只要 1–2 屏。分区本身不是问题，**一次全给**才是。

现在按「这一页是来回答哪三个问题」排，只开这三个：

| 开 | 回答 |
|---|---|
| 关键路径归责 | 这次为什么花了这么久 |
| scope 排行 | 时间花在哪个 scope |
| 空转窗口 | 哪段时间核在空转 |

统计口径、引擎配对、spmd 展开折起来，各占一行，点一下展开，
折叠状态记在 `S.folded` 里，跨页签和重绘都保持。**没有删任何内容。**

说明文字从 12 段降到 4 段。去掉的不是信息，是位置：

- 列的定义搬进表头的 `title=`（slack 怎么算、波和离散度什么意思）
- 推导过程搬进脚注上的 `两条路怎么选 ?` / `怎么算的 ?`，悬停可见
- 归责闭合检查从整张卡压成结论卡里的一行 `✓ compute + stall = makespan`

留下的 4 段卡都是**会让人读错数字的那种**：通信受限的判据、
最长空转窗口是谁挡住的、混合核两半的比值、Worker/Scheduler 重复计数。

表格行数也收了：scope 排行 14→10、空转窗口 8→5、路径节点 10→6，
表头上写清是「前 N / 共 M」，点进去可以下钻。

结果是 L2 从 6.3 屏降到 2.6 屏，和其余页签（1.0–2.5 屏）齐平。

---

## 交互速查

| 操作 | 结果 |
|---|---|
| Inspector 分区标题的 `+` / `−` | 展开 / 折叠该分区（统计口径、引擎配对、spmd 展开默认折叠） |
| 悬停表头 / 带点下划线的字段 | 该列或该行的定义与推导，正文里不再重复 |
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
