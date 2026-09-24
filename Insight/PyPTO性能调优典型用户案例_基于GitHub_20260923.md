# PyPTO 性能调优典型用户案例（基于 GitHub Issue 与经验文档）

> 整理日期：2026-09-23
> 资料范围：`hw-native-sys` 组织下 pypto、pypto-lib、simpler、PTOAS、pto-isa、pypto-serving、pypto-skills 七个仓库的 Issue / PR，以及社区公开的调优经验文档
> 标注约定：**【实录】** 数据和做法直接来自公开来源；**【还原】** 依据多个来源拼接或推断出的过程，来源本身没有完整写出；**【Mock】** 为补全流程而虚构的环节，只用于说明流程，不代表真实数据

---

## 一、结论摘要

1. **真实的调优几乎都是“先量后改、边改边推翻”的循环。** 在公开案例中，至少有 4 个结论先被提出、后来被推翻或修正（其中 3 个是作者自己推翻，1 个是由其他人在评论中指出的）：带宽瓶颈其实是调度停顿；2.3 倍差距其实是统计范围不一致；某个 scope 被认为不在关键路径，但在另一个模型形态下恰好在关键路径上；R1 的收益大半来自那次运行偶然偏快。这个领域最缺的是**快速、可靠地证伪假设**，而不是优化手段本身。
2. **瓶颈分布在五个层级，用户需要逐层下钻。** 依次是：服务链路（bind/H2D、LM head）→ 模型层（scope 排名、关键路径）→ 调度（派发开销、伪并行）→ 核内（MTE2 / Vector / Cube 流水）→ 编译器和 ISA（同步插入、MemoryReuse、SyncPeriod）。同一个现象可能出自不同层级，例如“核很空”既可能是调度串行，也可能是同步过于保守。
3. **最常见的四类根因**：
   - 派发和调度开销：细碎 task、Tail OH 过高；
   - 伪串行：共享 GM 句柄上的 WAW 或循环携带依赖，导致 task 数多但核空闲；
   - 访存事务过碎：逐行散读、ND2NZ 短 burst；
   - 编译器或 ISA 过于保守：自动插入的同步、MemoryReuse 引入的 WAR、SyncPeriod 过粗。
4. **验证环节最容易出错。** 社区自发形成了一套测量纪律：同 session A/B、多跑取中位数、选一个与改动无关的大 scope 作为噪声锚点、看 busy/core-time 而不看单次 wall、统计范围必须对齐（attention 对 attention）、PMU 会让 wall 膨胀约 15%。这些纪律目前只写在 Issue 和个人日志里，工具层面没有沉淀。
5. **8 月中旬以后的新趋势**【实录】：DeepSeek-V4 / DSpark 相关的 `Perf:` PR 大量出现（9 月上中旬 pypto-lib 几乎每天都有）；每日算子性能 CI 已经接入（pypto-lib#1274、#1284、#1307）；A5（Ascend950）平台开始专项跟踪性能（pypto-lib#1233、#1340）；实验结论开始主动报告“无稳定收益”（pto-isa#329、simpler#2286、simpler#2301）。

---

## 二、资料来源与方法

| 来源 | 范围 | 抓取时间 | 用途 |
| --- | --- | --- | --- |
| `github_issues/*/..._issues.json`（本地归档） | 7 个仓库共 1,465 个 Issue/PR | 2026-08-17 | 按关键词粗筛，得到 351 条性能相关候选 |
| `github_issues/pypto-lib/detailed/` | pypto-lib 的 Issue、PR 及评论 | 2026-09-04 | 读取跟踪 Issue 的后续评论和 PR 合入状态 |
| GitHub REST API（实时） | 2026-08-17 之后的 Issue/PR、PTOAS#643 评论、pto-isa#329 | 2026-09-23 | 补充归档之后的新内容 |
| [swimlane-tuning-log.zh.md](https://github.com/wangqin1723-max/pypto-lib/blob/9be7112769fdfbe07a089018ee9b88159ef0156b/docs/swimlane-tuning-log.zh.md) | 20 组 DeepSeek-V4 decode 调优实验 | 固定版本 `9be7112` | 索引见 pypto-lib#828；是目前最完整的一手调优过程记录 |
| `repo/pto/docs/zh-cn/dev/03-runtime-dfx.md` | 运行时 DFX 开关说明 | 本地镜像 | 核对工具名和开关 |

**筛选方法**：先用关键词（perf / 性能 / latency / swimlane / 泳道 / bottleneck / tiling / pipeline / 融合等，标题命中加权）粗筛，再人工挑选同时满足以下三点的条目：

- 有明确的现象和基线数据；
- 有归因过程；
- 有修改前后的对比数据。

**局限**：

- 性能数据绑定特定的代码版本、shape 和 a2a3（910B）硬件，只能用来说明方法，不能当作性能承诺。
- 部分图片和 trace 附件没有下载，结论以正文文字为准。
- 本地归档只保存了 pypto-lib 的评论；其他仓库只对关键条目实时补读了评论。

---

## 三、典型用户与场景总览

| # | 案例 | 用户角色（推断） | 调优对象 | 瓶颈类型 | 结果 | 主要来源 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | DeepSeek-V4 compressor 分步调优 | 模型开发者 | 模型子模块（12 个 kernel） | 派发开销 + 热点 kernel | 485.48 → 379.88 µs（**−21.8%**）；改用 spmd 后再降 9% | pypto-lib#314 |
| 2 | 泳道图诊断伪并行 | 模型开发者 | decode indexer compressor | WAW 链导致串行化 | 1815 → ~1025 µs（**−43%**） | 调优日志 §6、§8 |
| 3 | `b_trans` 一招两面 | 模型开发者 / 编译器开发者 | matmul 权重加载 | 访存事务粒度 | DSv4 场景 −8.3%；Qwen3 浅 K 场景反而 **+16~23%** | 调优日志 §19、pypto#2309 |
| 4 | 生成的 GEMM 与手写实现对标 | 算子开发者 | 单算子 GEMM | 自动同步过于保守 | 0.37× → 83% → 与手写持平 | PTOAS#643 |
| 5 | FlashAttention 版本回归定位 | ISA / 算子开发者 | 单算子 FA | 跨核同步节拍过粗 | 大 S 下恢复 10~12%，与 torch_npu 持平 | pto-isa#172 |
| 6 | “2.3 倍差距”口径校准 | 性能分析者 | Qwen3-14B attention 与 CCE 对比 | 测量范围不一致 | 2.3× → 实际 0.95×（基本持平） | pypto-lib#622、#465，pypto#2040 |
| 7 | 8 卡服务端到端拆解 | 服务化开发者 | DeepSeek-V4 serving | 每步重复 bind/H2D | 定位到 bind 占 75.74% | pypto-serving#95 |
| ★ | **完整全流程还原**：HCA softmax_pool | 模型开发者 | 单 scope → 整图 | MTE2 散读 + 冗余转置 | scope core-time **−71%**，整图 **−8.7%** | 调优日志 §20，pypto-lib#641 |

---

## 四、完整全流程还原：DeepSeek-V4 HCA 解码注意力中的 `softmax_pool` 调优

> 选择理由：这是公开资料中四个环节都有一手数据的案例，覆盖识别瓶颈、分析、修改和验证，而且包含两个典型的测量陷阱。第 0 步和第 7 步来源没有记录，用【Mock】/【还原】补全。

### 4.0 触发：目标从哪里来【Mock】

- **场景设定**：团队在推进 DeepSeek-V4 decode 的 TPOT 优化。服务侧拆解（参考案例 7 的方法）显示，消除 bind/H2D 之后 Device wall 成为主要瓶颈；层级泳道显示 HCA 注意力（`decode_attention_hca`）是占比较高的子图之一。
- **用户动作**：把 `decode_attention_hca` 设为本轮目标，先确认单跑能通过精度验证，并记录代码版本和环境。
- **说明**：这一步为串起全流程而虚构；来源只记录了从子图开始的调优。

### 4.1 识别瓶颈：整图 scope 排名【实录】

- **输入**：`decode_attention_hca` 整图，a2a3 真机，打开 `--enable-l2-swimlane`。
- **观察**：整图 wall 为 1303.6 µs；`softmax_pool` 的 core-time 为 9495 µs，是**第 2 忙的 scope**（avg 18.54 µs，n=512），超过 `qk_pv` 的 40%。
- **修正早期判断**：此前在 ratio4 compressor 中得出“softmax_pool 不在关键路径、优化后 wall 不变”（§18）。作者指出这个结论**在 ratio128 / HCA 形态下不成立**，所以这里的优化能直接体现在 wall 上。
- **工具坑**：这几次泳道的 Latency/Total 列没有采到数据。作者用 `l2_swimlane_records.json` 重建 wall：`max(finish) − min(dispatch)`，tick 按 50 MHz 换算。

### 4.2 问题分析：核内下钻【实录】

用 incore op-sim 查看 `softmax_pool.clean.json`，单 task 的流水构成如下：

| 观察项 | 数据 | 含义 |
| --- | --- | --- |
| veccore span | 20.58 µs | 单 task 时长 |
| MTE2 占比 | 69%，由 **256 笔 `[1,64]` 散读**组成，每笔约 211 ns | 访存事务碎，瓶颈在加载 |
| VECTOR 占比 | 63%，但大部分是 `MOV_UB_TO_UB` 逐行 staging 和两次 `VNCHWCONV` 转置 | Vector 时间主要花在搬运上，不是计算 |
| 真正的 softmax 计算（VEXP/VDIV/VCADD） | 合计 < 1 µs | 真正的计算只占很小一部分 |

**提出假设**：

1. 分页 state 池按行逐条读取，导致事务过碎；
2. 由于 `row_max/row_sum` 只能沿最后一维归约，只好先转置，由此多出一段转置。

**可行性论证**：窗口起点 `state_pos0` 一定是 `COMPRESS_RATIO=128` 的倍数，所以也一定是 `BLOCK_SIZE=8` 的倍数。因此 128 个 state 刚好覆盖 16 个完整物理块，没有残头残尾，可以按块读取。

**分析时的坑**：kernel 带有数据依赖门控 `position_ids % 128 >= 126`。自动生成的 golden 把 `position_ids` 清零，导致门控不进入、循环跑 0 轮，trace 退化，看起来“很快”。修正方法是手动把 `v1.bin` 写成 127 后重跑 `msprof op simulator`。

### 4.3 修改：两轮叠加【实录】

**R1：逐行散读改为按物理块批量读取**，softmax 的数学计算一字不改：

```python
# 改前：128 轮，每轮读 2 个 [1,64]，每行查一次 block table，逐行 staging
for s in pl.pipeline(STATE_LEN, stage=2):
    slot_score = compress_state_flat[blk:blk+1, OUT_DIM+kv_col0 : +HEAD_TILE]
    softmax_score_state[s:s+1, :] = slot_score
# 改后：16 轮，每轮读 2 个 [8,64]，每块查一次 block table，整块直接落入 UB
for blk_i in pl.pipeline(NUM_STATE_BLOCKS, stage=2):
    slot_score = compress_state_rows[row0:row0+8, OUT_DIM+h0 : OUT_DIM+h0+HEAD_TILE]
    softmax_score_state[s0:s0+8, :] = slot_score
```

**R2：用列归约替代转置，并把 `HEAD_TILE` 从 64 加宽到 128**：

```python
score_max  = pl.col_max(softmax_score_state)                       # 沿 state 轴直接归约
score_exp  = pl.col_expand_expdif(softmax_score_state, score_max)  # 融合计算 exp(x - max)
score_sum  = pl.col_sum(score_exp)
score_prob = pl.col_expand_mul(score_exp, pl.recip(score_sum))     # 用 ×recip 代替 ÷
pooled_chunk = pl.col_sum(pl.mul(softmax_kv_state, score_prob))
```

**修改时碰到的约束**：

- API 坑：`col_expand_sub` / `col_expand_div` 在 Python API 中存在，但**没有注册 codegen**（`No codegen registered`），只能绕开：减法用 `col_expand_expdif`，除法用 `col_expand_mul(x, recip(sum))`。
- 容量墙：`POOL_HEAD_TILE=256` 时 3 个 `[128,256]` FP32 tile 超过 192 KB UB，所以 128 是上限。
- 代价：kernel 从 MTE2-bound 变为 VECTOR-bound，单 task 的 VECTOR 时间从 4.85 µs 增加到 13.46 µs；但每个 batch 的净 core-time 从 61.8 µs 降到 43.3 µs（−30%）。

### 4.4 对比验证【实录】

**核内（op-sim 单 task）**：

| 指标 | 改前 | R1 块读 | 变化 |
| --- | --- | --- | --- |
| MTE2 cycles | 450,249 | 43,474 | −90.3% |
| MTE2 load 笔数 | 256 | 32 | 减少到 1/8 |
| VECTOR cycles | 473,236 | 91,948 | −80.6% |
| veccore span | 20.58 µs | 7.73 µs | −62.4% |

**整图（真机 a2a3，3 次运行，控制噪声）**：

| 版本 | 整图 wall | softmax_pool core-time | `qk_pv`（无关 scope，作噪声锚点） |
| --- | --- | --- | --- |
| 原始 | 1303.6 µs | 9495 µs | 22301 µs |
| R1 | 1276.2 µs | 3203 µs | **19754 µs**（这次运行偏快） |
| R2 | **1190.1 µs** | **2792 µs（−71%）** | 21918 µs（接近原始） |

**精度**：standalone 与整图输出全部通过，两轮 `max_error_ratio=0.0`，结果逐位一致。

**关键的验证判断**：R1 的 wall 只降了 27 µs，而那次运行中与改动无关的 `qk_pv` 快了约 2500 µs，说明 R1 的 wall 收益大半是那次 session 偏快，并非改动带来的。只有原始版本与 R2 的 `qk_pv` 基本一致，所以 **−113.5 µs（−8.7%）** 才是可信的归因。

### 4.5 收口与沉淀【实录 + 还原】

- 【实录】PR [pypto-lib#641](https://github.com/hw-native-sys/pypto-lib/pull/641) 于 2026-06-30 合入，描述中写明改前改后的 build 目录，便于追溯。
- 【实录】作者把经验提炼为可复用的法则：“分页池的 pooling 如果 MTE2-bound 且窗口按块对齐，就把逐行散读换成按块跨步读取”；以及“wall 的 A/B 必须锚定一个无关的大 scope 来判断噪声”。
- 【还原】pypto-lib 在 9 月接入每日算子性能 CI（#1274、#1284）。本案例是否纳入监控，来源没有说明；按正常流程，下一步应当把该 scope 的 core-time 纳入回归监控。

### 4.6 流程小结

| 阶段 | 用户在做什么 | 依赖的工具和产物 | 在本案例中卡在哪里 |
| --- | --- | --- | --- |
| 识别 | 整图按 scope core-time 排名，找出最忙的 scope | L2 泳道、`l2_swimlane_records.json` | 泳道列缺数据，只能手动重建 wall；历史结论“不在关键路径”误导判断 |
| 分析 | 看核内流水占比，分清是在搬运还是在计算 | incore op-sim、`msprof op simulator` | 数据门控让 trace 退化，看起来“很快” |
| 修改 | 改加载粒度和归约方向，扫 tile 上限 | DSL 源码、UB 容量 | API 存在但没有 codegen；UB 容量上限 |
| 验证 | 看核内 cycles、整图多次运行、精度逐位对比 | 泳道、golden | 单次运行的 wall 会骗人，需要锚定无关 scope |

---

## 五、七个典型案例

### 案例 1：模型子模块分步调优（DeepSeek-V4 `compressor_ratio128`）

**来源**：[pypto-lib#314](https://github.com/hw-native-sys/pypto-lib/issues/314)（2026-05-18，已关闭）

- **识别瓶颈**【实录】：基线（B=64，S=1，a2a3）Total Test Time 为 394.84 µs，共 94 个 task；Tail OH 合计 727.8 µs，占 latency 的 28.3%。
  - `softmax_pool` 占总执行时间的 72%，是第一热点；
  - `kv_cache_write` 为 64 次一行一派发，latency/exec 达 6.4 倍，是最大的调度开销来源。
- **问题分析**【实录】：拆成 4 条原则，并逐条写出风险：
  1. `pl.range` → `pl.pipeline`，风险是 Mat 超过 512 KB；
  2. 合并相邻的 `pl.at`，风险是 AIC/AIV 同步开销；
  3. 在 UB/Mat 限制内加大 tile；
  4. 调整 `pl.parallel` 的 chunk，扫描最优值。
- **修改与验证**【实录】：每步都按同一测量协议执行：运行 → 读 Total Test Time 和每个函数的 Exec/Latency → 通过则提交，失败则记录阻塞原因并回退。

| 步骤 | 改动 | 前 → 后 |
| --- | --- | --- |
| 1 | `kv_cache_write` 设置 chunk=16，派发数 64 → 4，同时简化 RoPE | 483.64 → 395.58 µs（−18.2%） |
| 3 | `kv_score_proj` 使用 pipeline(stage=2)，RMSNorm 使用 stage=4 | 427.82 → 418.12 µs（−2.3%） |
| 4 | 按 batch 并行（BATCH_CHUNK=32） | 412.10 → 403.10 µs（−2.2%） |
| 5 | `softmax_pool` 使用 pipeline(stage=2)，并融合 rmsnorm | 403.10 → 379.88 µs（−5.8%） |
| 合计 | task 数 94 → ~34，Avg Tail OH 16.0 → ~3.1 µs | **485.48 → 379.88 µs（−21.8%）** |

- **后续**【实录】：4 处 `pl.parallel + pl.at` 改为 `pl.spmd`，409.70 → 372.76 µs（−9.0%），收益主要来自 Head OH 从 2.29 µs 降到 0.57 µs。这一步依赖 pypto#1414 修复 `pl.spmd` 在 if/else 分支里的 SSA 问题。
- **可复用经验**：本案例中**最大的一次收益来自减少派发次数，而不是计算优化**；作者也坦白了几点不足：部分步骤没有单独测量，硬件资源紧张时只能从 commit message 回溯数据，两台设备之间存在噪声。

### 案例 2：用泳道图诊断伪并行（DeepSeek-V4 decode indexer compressor）

**来源**：调优日志 §6、§8（索引见 [pypto-lib#828](https://github.com/hw-native-sys/pypto-lib/issues/828)）

- **识别瓶颈**【实录】：泳道图中段全是细碎的核。按时间窗口统计占用：MIDDLE 段 1500 µs（占 72%）内，AIC 只有 1.9%、AIV 只有 5.8% 在忙。每个 scope 有几百个 task，却只落在 1~6 个核上。
- **问题分析**【实录】：
  - 排除“padding 空算”的假设。根因是 per-batch `pl.parallel` 循环体内对共享 GM 张量反复原地 `pl.assemble` 重新赋值，形成跨迭代的 WAW 链，64 个 batch 被串行执行，同一循环体内的计算 scope 也被连带串行。
  - 用三次真机实验定位：拆成 3-pass，1866 µs，计算并不是瓶颈；拆成 4-pass，scatter 单独成一个 pass 后只剩 1 个核，时间涨到 2593 µs；证明 scatter 必须和 softmax 留在同一个 pass。
- **修改**【实录】：把 `pl.at` 从循环内移到外层，包住 `for o0`，每个 batch 从 4 个碎 scope 合为 1 个，缩短依赖链：

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="state_scatter_paged"):
    for o0 in pl.range(0, OUT_DIM, OUT_CHUNK):   # 原来 for 在 pl.at 外
        ...
```

- **验证**【实录】：1815 → ~1025 µs（**−43%**，两次运行分别为 1039.7 / 1012.0 µs）；scatter 的 task 数 512 → 128，占用核数 5 → 42；uniform 与 hetero start_pos 两种输入精度全部通过。
- **反例（§6）**【实录】：作者曾误判 `kv_score_proj` 是 HBM 带宽瓶颈。把 N 从 64 加到 128 后，x 的重读减半、单 task 也更快，但 Total 反而从 582 µs 退化到 621 µs，因此推翻带宽假设。窗口内 Cube 利用率只有约 21%，说明瓶颈是依赖和调度停顿。
- **作者总结的法则**：调某个 scope 之前，先量它所在窗口的 Cube/AIV 利用率。低于约 30% 就不是计算瓶颈；这时重新切 tile 或增加 task 基本是零和，真正的杠杆是缩短依赖链或做融合。**task 多不等于并行。**

### 案例 3：`b_trans`，同一招在两个场景结果相反

**来源**：调优日志 §19、[pypto-lib#628](https://github.com/hw-native-sys/pypto-lib/pull/628)（已合入）、[pypto#2309](https://github.com/hw-native-sys/pypto/issues/2309)（2026-08-07，未关闭）

**正面：DeepSeek-V4 `kv_score_proj`**【实录】

- 识别：incore 显示 MTE2 五条子队列的并集 23.58 µs，**0 gap（100% 占满）**；Cube 只忙 23%。所有 load 都是 `MOV_OUT_TO_L1_MULTI_ND2NZ`。
- 分析链，逐条排除：
  - 加深 pipeline：stage=3 时 Mat 需要 576 KB，超过 512 KB 上限；
  - 裸带宽墙：每条通道只有约 21 GB/s，远低于 HBM 上限，应当是 transaction-bound；
  - NZ 预打包：声明 `pl.NZ` 后，codegen 静默忽略，生成的 `.pto` 逐字节相同。
- 修改：权重改存为 `[OUT_DIM, D]`，调用 `matmul(..., b_trans=True)`，加载方式从 ND2NZ 变为 DN2ZN，短 burst 变为长 burst。
- 验证：standalone busy −15.1%，wall −8.3%；CSA 整图取 3 次运行的中位数，wall −12.9%，并注明单次运行曾出现 +20% 的假象。
- 连带成本：权重签名要一路改到 `decode_layer` 和 `decode_fwd`；漏改一处时，运行时报的是 507018，而不是 shape 错误，很难排查。还踩过一次坑：CI 报错实际来自旧 base 上的另一个文件，与本次改动无关。

**反面：Qwen3-14B 浅 K tile**【实录】

- 生成代码只有 `alloc_tile` 的 layout 字段不同，但运行时开销随 inner-K 深度急剧变化：TK=256 时 +2.2%（噪声内），TK=64/TN=512 时 +15.8%，TK=64/TN=1024 时 +23.2%。
- 而 TK=64 正是 Qwen3 `out_proj` 和 `gate/up/down` 的配置，这部分占 decode 权重字节的 89%。
- 作者特意说明复现方法：必须扫描 tile shape，并用 40 层权重栈来避开缓存命中和派发开销，否则会漏掉这个现象。

**启示**：优化手段是否有效取决于 shape。**案例库里需要记录“在什么条件下有效”，只记录“某招有效”并不够。**

### 案例 4：生成的 GEMM 对标手写实现和 torch_npu

**来源**：[PTOAS#643](https://github.com/hw-native-sys/PTOAS/issues/643)（2026-05-08，已关闭，含 6 条评论）

- **识别瓶颈**【实录】：`ptoas --enable-insert-sync` 生成的 GEMM 在 4 种 shape 下都只有 torch_npu 的 **0.37~0.41 倍**；6144³ 时为 3.836 ms（120.9 TFLOPS），而 torch_npu 是 1.408 ms（329.5 TFLOPS）。手写 kernel 约 1.506 ms。各 shape 的差距稳定，说明是系统性问题，不是个别 shape 的问题。
- **问题分析**【实录】：对比生成的 C++ 和手写 kernel，发现内层 K 循环里有大量 `wait_flag`、`pipe_barrier`，甚至出现 `PIPE_ALL`，把 TLOAD → TEXTRACT → TMATMUL 之间的重叠打断了。
- **修改**【实录，维护者给出】：
  1. 只用**一个 if/else** 区分 ping/pong，不要拆成多个 if/else，因为编译器会把跨分支依赖的 set/wait 外提；改完加上新版 ptoas，达到 torch_npu 的 **83%**。
  2. 要与手写持平，改为**全展开（unroll）**使用 ping-pong buffer，消除控制流导致的同步外提，维护者附了展开版 `.pto`。
- **协作过程**【实录】：维护者首先要的是能复现的 `.pto`；提交者说明 `.pto` 是 DSL 现场生成的构建产物，并给出不需要硬件的生成命令。
- **启示**：DSL 的写法（分支结构）会直接决定编译器同步插入的质量，而用户很难从源码预判这一点。

### 案例 5：FlashAttention 版本回归，定位到一行代码

**来源**：[pto-isa#172](https://github.com/hw-native-sys/pto-isa/issues/172)（2026-06-17，已关闭）

- **识别瓶颈**【实录】：`flash_atten` 在 S ≥ 32768 时相对 torch_npu 从约 1.00× 掉到 0.89~0.90×。
- **问题分析**【实录】：做干净的 A/B 隔离：同一张卡、同一个 ptoas、同一份 MLIR 和 C++，只切换头文件根目录，两棵头文件树**仅 `TPush.hpp:37` 一行不同**。定位到 commit `014920a8` 把 `SyncPeriod` 从 `SlotNum/2` 改成了 `SlotNum`，该 commit 自己写着 `Not-tested: Full NPU hardware validation pending`。
- **机理**【实录】：`SyncPeriod=8` 时，Cube 先填满 8 个 slot 才检查，之后硬等 Vec 一次排空，同步变得粗糙而突发，打断了稳态重叠。S 越大，稳态循环占比越高，回归越明显。
- **验证**【实录】：case1~case8 全量测试，采用 `--timing sync`，因为它比 event 计时更稳。大 S 下恢复 +9.6%~+11.1%，达到约 185 TFLOP/s，与 torch_npu 持平；`err_kernel` 在 A/B 两版逐字节一致。
- **建议的处置**【实录】：不要全局回退，因为那个 commit 可能是为了其他配置的正确性。给出三个选项：只在 C2V 单生产者/单消费者路径上恢复；把 `SyncPeriod` 暴露为模板参数；保留新默认值，同时补上正确性回归测试。
- **启示**：回归定位的核心是**把变量缩到一行**，并在改动时权衡正确性。

### 案例 6：“2.3 倍差距”其实是统计范围不一致（Qwen3-14B attention 对比 CCE）

**来源**：[pypto-lib#622](https://github.com/hw-native-sys/pypto-lib/issues/622)、[pypto-lib#465](https://github.com/hw-native-sys/pypto-lib/issues/465)、[pypto#2040](https://github.com/hw-native-sys/pypto/issues/2040)

- **识别**【实录】：TraCR trace 显示 CCE 约 548 µs，pypto 约 1263 µs，差距约 2.3 倍。这个数字被转到 pypto 仓库，作为 codegen 根因分析的依据（#2040，怀疑 A2/A3 上 C2V/V2C 仍经过 GM 往返）；同期 pypto-lib#765 把 decode attention 换成了 CANN FAI。
- **口径纠正**【实录】：
  - 评论指出 1263 µs 是**整个 decode layer**（约 35 个 kernel）的 on-core busy，而 CCE 只统计了 paged attention。只取 `fa_fused + online_softmax` 时约为 350 µs。
  - 统一 runtime、各自用最佳 tiling 重新测量：pypto 约 326 µs，CCE 最佳约 343 µs，**比值 0.95×，基本持平**。CCE 默认 tiling 被 `is_long_seq` 阈值误导到慢路径，约 462 µs。
  - 继续用 PMU 分析：MTE2 只有约 65~70% 在忙；HBM 实际读取字节与理论值吻合，没有过量读取，所以 0.78 TB/s 不是带宽上限，而是管道空闲。
- **补充观察**【实录，#607】：standalone attention 的气泡在整层里被 MLP/投影掩盖，所以**优化 attention 本身要看 standalone，衡量端到端要看整层**。
- **对比表的写法**【实录，#465】：Qwen3-14B 单层按 stage 与 ASC 小算子逐项对比（整层 1169.4 µs 对比 970.02 µs，比值 83%），明确写出方法和统计口径：3 次均值、CoV、dep_gen 溢出后如何用编排提交序列还原 task 名，以及交叉校验。
- **启示**：跨实现对比前，必须先对齐**统计范围、runtime、tiling 和计时方法**。一个未经校准的差距数字，可能会被转到其他团队，影响他们的分析判断。

### 案例 7：服务端到端拆解（DeepSeek-V4 Flash W8A8，8 卡）

**来源**：[pypto-serving#95](https://github.com/hw-native-sys/pypto-serving/issues/95)（2026-07-16，未关闭）

- **识别瓶颈**【实录】：先定义 7 个层级的指标：Serving wall、Packed outer wall、Runtime Host wall、Bind/H2D、Device wall、D2H、LM head/sampling，并注明 Device 的三个观察窗口互相重叠、不能相加。
  - Steady decode 平均 7488.8 ms/token，其中 packed dispatch 占 95.93%；
  - 在 runtime 内部，**Bind/H2D 平均 5231.2 ms，占 75.74%**，Device wall 只有 1365.8 ms；
  - Prefill 的共享 buffer 首次分配耗时 368 s，占 82.83%，属于冷启动成本。
- **问题分析**【实录】：每个 decode step 都重新 bind 和 H2D 固定权重与长生命周期 Cache。估算优化空间：消除这部分后，单 rank 理论下限约为 1675.6 ms。作者明确注明这是计算值，不是实测值。
- **修改方案**【实录】：按优先级列出：P0 权重和 Cache 常驻 Device；P1 复用通信域、LM head 与 sampling 迁到 Device；P2 预分配、输入张量池化。同时规定验证条件：同设备、同 prompt、同样 20 token；分别报告各层耗时；优化后的数据单独记录，不覆盖基线。
- **启示**：在模型级，**首要瓶颈往往不在 kernel**。先按层级拆清楚，再决定是否需要优化 kernel。

---

## 六、跨案例的共性：调优旅程

### 6.1 四个阶段中的用户行为

| 阶段 | 典型动作 | 常用工具和产物 | 常见错误（来自案例） |
| --- | --- | --- | --- |
| **识别瓶颈** | 分层拆解耗时；按 scope/kernel 的 core-time 排名；按时间窗口统计核占用 | STRACE、L2 泳道（`--enable-l2-swimlane`）、`l2_swimlane_records.json`、TraCR + Perfetto | 统计范围不一致（案例 6）；dep_gen 溢出导致 task 没有名字（#465）；泳道列缺数据（§20） |
| **问题分析** | 区分计算、访存和停顿三类瓶颈；做 split test（tile 砍半，看单 task 是否也减半）；读 IR dump 和生成的 C++ | incore op-sim、`msprof op simulator`、PMU（`--enable-pmu`）、`passes_dump/*.py`、生成的 `.cpp` | 把停顿误判为带宽问题（§6）；数据门控导致 trace 退化（§20）；PMU 会让 wall 膨胀约 15%（#622） |
| **修改** | 调整派发粒度（chunk/spmd/合并 scope）；改变依赖结构；加 pipeline；调整 tile/layout；改编译选项或 DSL 写法 | `pl.pipeline`、`pl.spmd`、`pl.parallel(chunk=)`、`b_trans`、`col_*` 归约 | UB/Mat 容量墙；API 存在但没有 codegen；跨文件签名需要一路改；同一个手段在不同 shape 下效果相反 |
| **对比验证** | 同 session A/B；多跑取中位数；锚定无关 scope；逐位或按容差对比精度；隔离变量到一行 | golden、`ratio_allclose`、`--timing sync`、ABCCBA 交错运行 | 单次 wall 不可信（§19、§20）；本地与 CI 的 pypto 版本不一致（§19） |

### 6.2 社区自发形成的“测量纪律”【实录，汇总】

1. **先定口径，再比数字**：统计范围、runtime、tiling、计时方式四项要对齐（案例 6）。
2. **看 busy/core-time，wall 只作辅助**：wall 受 session 影响很大（§19 中不同 session 可相差 2.7 倍）。
3. **同一 session 做 A/B，至少 3 次取中位数**；更严格的做法是 ABCCBA 交错运行，每个变体 600 个样本，不剔除慢样本（pto-isa#329）。
4. **锚定一个无关的大 scope 判断噪声**（§20）。
5. **变量隔离到最小**：只切头文件根目录（案例 5），或只改一处计算（pto-isa#329）。
6. **报告负面结果**：“无稳定收益”也要写清楚（pto-isa#329、simpler#2286、simpler#2301）。
7. **推算值与实测值分开标注，优化后数据不覆盖基线**（案例 7）。

### 6.3 痛点与产品机会【推断，供规划参考】

| 痛点（有案例证据） | 可能的产品能力 |
| --- | --- |
| 从整图到 scope 再到核内，每一层都要换工具，并手动在 JSON 中拼接数据 | 统一的下钻视图：scope 排名 → 窗口占用率 → 核内流水，一键跳转 |
| 判断瓶颈属于计算、访存还是停顿，全凭经验 | 自动给出窗口利用率和 split test 建议，把 §6/§8 的法则做成诊断规则 |
| 测量纪律靠个人自觉，结论经常被推翻 | 内置实验记录：同 session A/B、自动选噪声锚点、多跑统计、标注统计口径 |
| 优化手段是否有效取决于 shape，经验难以迁移 | 案例库记录“适用条件、反例、shape 范围”，而不只是“某招有效” |
| DSL 写法对编译器同步和 MemoryReuse 的影响不可见 | 源码 ↔ 同步指令 ↔ 流水气泡的关联视图（参见 PTOAS#643、pypto#1475） |
| 服务端首要瓶颈常在 kernel 之外 | 服务链路分层耗时拆解模板（参照 pypto-serving#95 的 7 层指标） |

---

## 七、2026-08-17 之后的新动态（实时补充）

- **A5（Ascend950）性能专项**：pypto-lib#1233（MXFP8/MXFP4 端到端）、#1340（V4.1-Flash 端到端与算子性能）；PTOAS 中 VPTO 调度器相关问题集中出现（#1327、#1448、#1506）。
- **性能看护进入 CI**：每日算子性能 CI 已接入，包括 MTP、DSpark 和 prefill 用例（pypto-lib#1218、#1219、#1274、#1284、#1307）。
- **运行时和 JIT 自身的开销**：DP8 TaskArgs 打包 13.4 ms（pypto#2629）、Tensor wire 转换回归约 9 ms/rank（pypto#2532，已关闭）；多个 `perf(jit)` PR 在缩短工具链探测时间。
- **负面结果和严格实验设计增多**：pto-isa#329（TCOLSUM 批量化）用 ABCCBA 交错、600 个样本，结论是“不能确认稳定收益”；PTOAS#1569（冗余同步耗尽 event ID）。
- **pypto 仓库补充了性能文档**：pypto#2496、#2519（分别于 8/24、8/26 合入）补充了关键路径、依赖冗余分析和派发判定相关的性能文档（本地 `repo/pto` 镜像尚未同步，建议后续拉取后对照）。

---

## 附录：主要来源索引

| 来源 | 类型 | 日期 | 状态 |
| --- | --- | --- | --- |
| [pypto-lib#314](https://github.com/hw-native-sys/pypto-lib/issues/314) DSv4 compressor_ratio128 优化跟踪 | Issue | 2026-05-18 | 已关闭 |
| [pypto-lib#828](https://github.com/hw-native-sys/pypto-lib/issues/828) 性能优化案例集索引 | Issue | 2026-07-24 | 未关闭 |
| [swimlane-tuning-log.zh.md @9be7112](https://github.com/wangqin1723-max/pypto-lib/blob/9be7112769fdfbe07a089018ee9b88159ef0156b/docs/swimlane-tuning-log.zh.md) | 经验文档 | 2026-05~07 | 固定版本 |
| [pypto-lib#628](https://github.com/hw-native-sys/pypto-lib/pull/628) b_trans | PR | 2026-06-29 | 已合入 |
| [pypto-lib#641](https://github.com/hw-native-sys/pypto-lib/pull/641) softmax_pool 块读与列归约 | PR | 2026-06-30 | 已合入 |
| [pypto#2309](https://github.com/hw-native-sys/pypto/issues/2309) b_trans 浅 K 场景开销 | Issue | 2026-08-07 | 未关闭 |
| [PTOAS#643](https://github.com/hw-native-sys/PTOAS/issues/643) insert-sync 下 GEMM 变慢 | Issue | 2026-05-08 | 已关闭 |
| [pto-isa#172](https://github.com/hw-native-sys/pto-isa/issues/172) flash_atten SyncPeriod 回归 | Issue | 2026-06-17 | 已关闭 |
| [pypto-lib#622](https://github.com/hw-native-sys/pypto-lib/issues/622) pypto 与 CCE 的 PA 差距量化 | Issue | 2026-06-26 | 未关闭 |
| [pypto-lib#607](https://github.com/hw-native-sys/pypto-lib/issues/607) Qwen3-14B decode attention profiling | Issue | 2026-06-24 | 未关闭 |
| [pypto-lib#465](https://github.com/hw-native-sys/pypto-lib/issues/465) Qwen3-14B 单层与 ASC 小算子对比 | Issue | 2026-06-05 | 未关闭 |
| [pypto#2040](https://github.com/hw-native-sys/pypto/issues/2040) fa_fused 与 CANN FAI 差距 | Issue | 2026-07-14 | 未关闭 |
| [pypto#1475](https://github.com/hw-native-sys/pypto/issues/1475) MemoryReuse 导致 matmul 串行 | Issue | 2026-05-22 | 未关闭 |
| [pypto-serving#95](https://github.com/hw-native-sys/pypto-serving/issues/95) DSv4 8 卡 serving 瓶颈 | Issue | 2026-07-16 | 未关闭 |
| [pto-isa#329](https://github.com/hw-native-sys/pto-isa/issues/329) TCOLSUM 批量化 | Issue | 2026-09-21 | 已关闭 |

> 状态以抓取时为准：本地归档 2026-08-17 / 09-04，实时补读 2026-09-23。
