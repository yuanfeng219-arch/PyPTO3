# `_jit_decode_fwd_layers_20260625_184941` 实际检测报告

## 1. 报告范围

数据目录：`D:\project\PyPTO3\Data\_jit_decode_fwd_layers_20260625_184941`

本报告基于目录中现有的静态产物和运行记录整理，未修改原始数据。分析覆盖：

- IR pass dump 清单；
- `deps.json` 依赖图结构；
- L2 swimlane 的 AICore 执行时间；
- 内存分配报告；
- 编译器性能提示；
- debug runner 的输入形状与验证能力。

由于当前工作区没有可调用的 `simpler_setup.tools.deps_viewer` 和 `simpler_setup.tools.critical_path` 实现，本报告没有冒充官方工具输出关键路径或 reduction 数值；相关结论标为“静态分析”或“待官方工具确认”。

## 2. 结论摘要

1. 这是一个 `a2a3` 默认目标的 JIT decode forward 数据集，输入 hidden state 形状为 `[16, 5120]`，包含 attention、KV cache、MLP 和 residual/RMS 等阶段。
2. IR dump 完整度较好：发现 42 个按序编号的 pass snapshot，覆盖前端、SSA、循环变换、内存复用、依赖推导和 runtime scope materialization。
3. 依赖图规模较大：426 个 task、48 个 tensor、1,369 条 edge，图深度为 13；边以 `explicit` 为主（1,364 条），另有 4 条 `creator` 和 1 条 `tensormap`。
4. AICore 记录覆盖 72 个 core，时间窗口为 47,849 ticks；按 50 MHz 时钟折算约 956.98 µs。累计 kernel 执行时间为 1,321,420 ticks，明显高于 wall-span，说明存在大量并行执行。
5. 按 kernel family 累计执行时间，`rms_recip`、`q_seed`、`q_proj`、`x_gamma0` 是主要时间贡献者；但这不是关键路径结论。
6. `perf_hints.log` 有 16 条提示、合计 57 次 occurrence，核心问题是 Vec 上 tile 的 innermost dimension 小于 a2a3 建议的 512 B，最小只有 4 B。
7. 内存报告中 `Right` 空间多次达到 100% 使用率；`Vec` 最高约 69.6%，`Mat` 最高约 53.1%。100% 是容量使用风险信号，但目前不能单独证明发生了溢出或性能瓶颈。
8. debug runner 的 `_user_compare` 为空且没有 `golden.py` 证据，因此本目录不能证明数值正确性已完成校验。

## 3. 数据清单

| 产物 | 实际情况 |
|---|---|
| `passes_dump/` | 42 个 `.py` snapshot |
| `dfx_outputs/deps.json` | 426 tasks、48 tensors、1,369 edges |
| `dfx_outputs/name_map_*.json` | 39 个 callable name mapping |
| `dfx_outputs/l2_swimlane_records.json` | 579 条 AICore 记录、579 条 AICPU 记录 |
| `dfx_outputs/merged_swimlane_*.json` | 50,556 个 trace events |
| `ptoas/` | 38 个 `.pto` 和对应 `.cpp` |
| `kernels/` | AIC/AIV kernel 源文件及目标文件 |
| `orchestration/` | `decode_fwd_layers.cpp` 与 `.so` |
| `report/perf_hints.log` | 16 条提示，57 次 occurrence |
| `report/memory_after_AllocateMemoryAddr.txt` | 38 个 compute function 的内存报告 |

## 4. 运行输入与目标

来源：`debug/run.py`、`kernel_config.py`

### 4.1 运行配置

| 项目 | 实际值 |
|---|---|
| 默认 platform | `a2a3` |
| device id 默认值 | `0` |
| runtime | `tensormap_and_ringbuffer` |
| AICPU thread 数 | 4 |
| orchestration function | `aicpu_orchestration_entry` |
| kernel 数 | 39 个 function id（0–38） |

### 4.2 关键输入形状

| 输入 | 形状 | dtype |
|---|---:|---|
| `hidden_states` | `[16, 5120]` | bfloat16 |
| `wq` | `[5120, 5120]` | bfloat16 |
| `wk` / `wv` | `[5120, 1024]` | bfloat16 |
| `k_cache` / `v_cache` | `[524288, 128]` | bfloat16 |
| `wo` | `[5120, 5120]` | bfloat16 |
| `w_gate` / `w_up` | `[5120, 17408]` | bfloat16 |
| `w_down` | `[17408, 5120]` | bfloat16 |
| output `out` | `[16, 5120]` | bfloat16 |

### 4.3 数值验证状态

`debug/run.py` 中 `_user_compare(...)` 直接 `return`，没有实际断言；当前目录也未发现同级 `golden.py`。因此：

> 当前数据可以支持编译、执行和性能结构分析，但不能据此宣称 decode forward 数值正确。

## 5. IR lowering 分析

`passes_dump/` 中共有 42 个 snapshot，编号从 `00_frontend.py` 到 `41_after_MaterializeRuntimeScopes.py`。中间阶段包括：

- `InlineFunctions`、`ConvertToSSA`、`Simplify`；
- `UnrollLoops`、`SplitChunkedLoops`、`InterchangeChunkLoops`；
- `ConvertTensorToTileOps`、`MemoryReuse`、`AllocateMemoryAddr`；
- `LowerPipelineLoops`、`SkewCrossCorePipeline`、`SplitVectorKernel`；
- `AutoDeriveTaskDependencies`、`ExpandManualPhaseFence`；
- `MaterializeCommDomainScopes`、`MaterializeRuntimeScopes`。

这表明数据足以进行 lowering 过程追踪。若要生成交互式 HTML trace，应使用与该 dump 同一 worktree 的 IR trace CLI，并验证输出为自包含 HTML；当前没有在本次只读分析中生成新 HTML。

## 6. 依赖图分析

输入：`dfx_outputs/deps.json`

### 6.1 图规模

| 指标 | 数值 |
|---|---:|
| tasks | 426 |
| tensors | 48 |
| edges | 1,369 |
| 计算出的最长边数深度 | 13 |
| cycle | 静态 DFS 未发现 |

### 6.2 Edge source

| source | 数量 | 占比 |
|---|---:|---:|
| `explicit` | 1,364 | 99.63% |
| `creator` | 4 | 0.29% |
| `tensormap` | 1 | 0.07% |

### 6.3 静态观察

出度最高的 kernel family 包括：

- `gate_seed`：85 条；
- `up_seed`：85 条；
- `down_seed`：85 条；
- `q_seed`：50 条；
- `out_seed`：50 条。

入度最高的 kernel family 包括：

- `dcr_xgamma`：87 条；
- `post_rms_reduce`：50 条；
- `rope_qkv`：19 条；
- 各个 `qk_norm*`：约 16 条。

这显示 orchestration 中存在明显的 seed fan-out 和若干汇聚点。它们可能增加调度 bookkeeping 或形成依赖链压力，但仅凭度数不能断定这些边可删除。

对非 `creator` 边做的保守 reachability 检查，只识别出 1 条明确被其他路径覆盖的 `explicit` 边：

`8589934598 -> 8589934681`

这不是官方 `deps_viewer --edge-mode reduced` 的最终结果。由于官方工具未加载，建议使用原仓库的 reduction 工具重新确认，尤其要同时比较 `reduced` 和 `reduced_dataflow`。

## 7. AICore 执行时间分析

来源：`dfx_outputs/l2_swimlane_records.json`

### 7.1 时间窗口

| 指标 | 数值 |
|---|---:|
| clock frequency | 50 MHz |
| AICore rows | 579 |
| 覆盖 core 数 | 72 |
| 起始 tick | 4,409,875,909,784 |
| 结束 tick | 4,409,875,957,633 |
| makespan/span | 47,849 ticks ≈ 956.98 µs |
| kernel duration 累计 | 1,321,420 ticks |

累计 duration 是各并行 task duration 的总和，不能与 makespan 直接比较；两者的差异正是并行执行的结果。

### 7.2 Kernel family 累计时间 Top 10

| 排名 | kernel | 调度次数 | 累计 ticks | 最大单次 ticks |
|---:|---|---:|---:|---:|
| 1 | `rms_recip` | 72 | 270,572 | 8,436 |
| 2 | `q_seed` | 62 | 156,705 | 8,440 |
| 3 | `q_proj` | 37 | 135,924 | 8,443 |
| 4 | `x_gamma0` | 72 | 134,183 | 7,737 |
| 5 | `k_seed` | 34 | 66,726 | 7,864 |
| 6 | `v_seed` | 28 | 58,178 | 7,844 |
| 7 | `v_proj` | 27 | 55,681 | 2,439 |
| 8 | `fa_work_build` | 27 | 53,725 | 2,363 |
| 9 | `qk_norm` | 27 | 51,736 | 2,339 |
| 10 | `qk_norm_0` | 26 | 47,480 | 2,159 |

初步建议优先查看 `rms_recip`、`q_seed`、`q_proj` 和 `x_gamma0` 的指令、访存和依赖关系。但这只是累计执行时间排序，不能代替 observed path 或 static CPM 排序。

## 8. 性能提示分析

来源：`report/perf_hints.log`

### 8.1 总体情况

- 提示类型：`PH001 TileInnermostDimGranularity`；
- 提示位置：16 个源码位置；
- 合计 occurrence：57 次；
- 建议：a2a3 后端 Vec 目标的 innermost dimension 尽量达到 512 B。

### 8.2 主要问题

日志中多次出现以下粒度：

- `fp32[1]`：4 B；
- `bfloat16[64]`：128 B；
- `bfloat16[128]`：256 B；
- `fp32[64]`：256 B。

这些都低于建议的 512 B，最小粒度只有建议值的 1/128。出现位置集中在 `decode_layer.py` 的 426、573、600、608、615、656–689、797–810、958 和 1000 行附近。

### 8.3 建议

1. 先确认这些小 tile 是否来自标量 tail、边界处理或真实主路径。
2. 对主路径上的 load/store，评估扩大 innermost tile 或合并相邻访问的可行性。
3. 修改前后使用相同 shape、相同 target 和相同测量方式重新采集。
4. 不要仅凭 PH001 数量推断加速幅度；需要结合 trace 和真实设备 wall-clock。

## 9. 内存分配分析

来源：`report/memory_after_AllocateMemoryAddr.txt`

### 9.1 观察到的峰值

| memory space | 示例峰值使用率 | 观察 |
|---|---:|---|
| `Right` | 100.0% | 多个函数达到空间上限 |
| `Vec` | 69.6% | 仍有余量，但可能受 tile 布局影响 |
| `Mat` | 53.1% / 50.8% | 未达到上限 |
| `Acc` | 50.0% | 未达到上限 |
| `Left` | 约 1.6% | 使用较低 |

`Right` 达到 100% 需要重点关注，因为任何额外临时 buffer 或对齐浪费都可能触发容量压力。但该报告本身没有证明分配失败，也没有给出端到端性能损失。

### 9.2 建议检查

- 对达到 100% 的函数展开完整 buffer live range；
- 检查是否存在仅由尾部小 tile 引入的额外 buffer；
- 对照 `MemoryReuse` 和 `AllocateMemoryAddr` 两个 pass，确认复用是否被依赖或 scope 边界阻断；
- 若要改内存规划，修改后必须重新生成 `.pto`、kernel 和内存报告。

## 10. 当前不能确认的事项

以下事项不能从现有目录直接得出最终结论：

- 官方 dependency reduction 的准确可删除边数量；
- static CPM 与 observed path 的准确时长及 stall 分类；
- 是否存在数值误差；
- a2a3 真机上的实际 latency、带宽和吞吐；
- PH001 对端到端性能的实际影响；
- `Right` 100% 使用率是否导致 spill、失败或仅是合法的精确占满。

## 11. 推荐下一步

### 优先级 P0：补齐正确性证据

在 `debug/run.py` 的 `_user_compare` 中加入实际 reference 对比，或提供 `golden.py`，并记录最大绝对误差、相对误差和通过阈值。

### 优先级 P1：运行官方图分析

在包含 `simpler_setup.tools.deps_viewer` 的消费者仓库环境中运行：

```bash
python -m simpler_setup.tools.deps_viewer \
  D:/project/PyPTO3/Data/_jit_decode_fwd_layers_20260625_184941/dfx_outputs/deps.json \
  --edge-mode reduced

python -m simpler_setup.tools.deps_viewer \
  D:/project/PyPTO3/Data/_jit_decode_fwd_layers_20260625_184941/dfx_outputs/deps.json \
  --edge-mode reduced_dataflow \
  --func-names D:/project/PyPTO3/Data/_jit_decode_fwd_layers_20260625_184941/dfx_outputs/name_map__jit_decode_fwd_layers_20260625_184941.json
```

### 优先级 P1：运行官方关键路径分析

```bash
python -m simpler_setup.tools.critical_path \
  D:/project/PyPTO3/Data/_jit_decode_fwd_layers_20260625_184941 \
  --stdout
```

运行前应确认工具实际发现的是 `l2_swimlane_records.json`、`deps.json` 和 name map，并检查报告中的 tiling check、kernel name resolution、warm-up 和 rank coverage。

### 优先级 P2：围绕 PH001 做小范围实验

选择 occurrence 较多且位于主路径的 load/store，单独改变 tile shape，重新生成 kernel、执行同一 workload，并对比：

- makespan；
- CUBE/VECTOR 或 AIC/AIV 时间；
- `Right`/`Vec` 内存峰值；
- 数值误差；
- 真实设备 wall-clock。

## 12. 证据路径

- [debug runner](../Data/_jit_decode_fwd_layers_20260625_184941/debug/run.py)
- [kernel configuration](../Data/_jit_decode_fwd_layers_20260625_184941/kernel_config.py)
- [dependency graph](../Data/_jit_decode_fwd_layers_20260625_184941/dfx_outputs/deps.json)
- [swimlane records](../Data/_jit_decode_fwd_layers_20260625_184941/dfx_outputs/l2_swimlane_records.json)
- [IR pass dump](../Data/_jit_decode_fwd_layers_20260625_184941/passes_dump)
- [performance hints](../Data/_jit_decode_fwd_layers_20260625_184941/report/perf_hints.log)
- [memory report](../Data/_jit_decode_fwd_layers_20260625_184941/report/memory_after_AllocateMemoryAddr.txt)
