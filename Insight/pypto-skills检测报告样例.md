# PyPTO Skills 检测报告样例

> **报告性质：模拟样例**  
> 本报告用于展示 `pypto-user` 与 `pypto-developer` skills 可能生成的报告形式。以下路径、版本、数值、时间和结论均为示例数据，不代表一次真实运行。

## 1. 检测概况

| 项目 | 示例结果 |
|---|---|
| 检测对象 | PyPTO Transformer 单卡推理案例 |
| 工作区 | `/workspace/pypto-lib` |
| 运行目标 | `a5` 模拟器；单卡设备验证未执行 |
| 案例 | `examples/transformer/run_decoder.py --layers 2 --batch 1` |
| Skills 版本 | `pypto-user 0.2.0` |
| 检测时间 | `2026-08-27 14:30 +08:00` |
| 总体结论 | 环境和基础链路通过；主要瓶颈疑似依赖链过长和数据等待 |

## 2. 执行摘要

1. Python、PyTorch、PyPTO、PTOAS 和 Tile-ISA 版本与当前 checkout 的 pin 一致。
2. 模拟器 smoke test 通过，并完成编译、输入生成、golden reference、运行和数值校验。
3. IR lowering trace 成功生成，包含 42 个 pass snapshot，HTML 报告为自包含文件。
4. In-core profiling 显示总计约 12,840 cycles，其中 CUBE 占比较高，目标 kernel 确实执行了矩阵计算。
5. 依赖图中发现 17 条结构性冗余边；dataflow 模式额外证明 6 条 creator 边可安全移除。
6. 关键路径占 makespan 的 76%，应优先检查依赖链和上游 producer，而不是先优化等待中的 consumer。

## 3. 环境与运行检查

对应 skill：[`setup-and-run`](../repo/pypto-skills/plugins/pypto-user/skills/setup-and-run/SKILL.md)

### 3.1 环境 gate

| 检查项 | 结果 | 示例证据 |
|---|---|---|
| framework / torch import | PASS | `import pypto, torch` 成功 |
| harness import | PASS | `from golden import run, run_jit` 成功 |
| PTOAS 可执行 | PASS | `ptoas --version` = `0.2.0` |
| PTOAS pin | PASS | framework pin = `0.2.0` |
| Tile-ISA | PASS | checkout HEAD = `8f31c2a`，与 pin 一致 |
| 设备环境 | NOT RUN | 本次只执行模拟器，不占用共享设备 |

### 3.2 Smoke test 与模型阶梯

| 阶段 | 结果 | 说明 |
|---|---|---|
| Simulator smoke test | PASS | compile、input、golden、runtime、validation 均出现，进程退出码为 0 |
| Operator rung | PASS | `matmul_fp16` 数值误差在阈值内 |
| One-layer rung | PASS | 1 层 decoder forward 校验通过 |
| Two-layer forward | PASS | 2 层案例校验通过 |
| Real-device run | NOT RUN | 需要用户分配设备并确认 device id |

## 4. IR lowering trace

对应 skill：[`generate-ir-trace`](../repo/pypto-skills/plugins/pypto-user/skills/generate-ir-trace/SKILL.md)

| 检查项 | 示例结果 |
|---|---|
| 输入 dump | `/workspace/pypto-lib/build/ir-trace-runs/run.abc123/passes_dump` |
| dump provenance | 当前 worktree 生成，非历史目录 |
| snapshot 数量 | 42 |
| HTML 报告 | `/workspace/pypto-lib/reports/decoder-lowering.html` |
| HTML 完整性 | PASS：doctype、CSS、trace-data、JavaScript 均存在 |
| 外部资源 | 0 个 |
| 覆盖范围 | lowering、memory planning、instruction selection |

结论：可以使用该 HTML 报告逐个 pass 检查算子从高层 IR 到目标指令的变化。当前没有发现因报告损坏或外部资源缺失导致的解释风险。

## 5. In-core profiling

对应 skill：[`incore-profiling`](../repo/pypto-skills/plugins/pypto-user/skills/incore-profiling/SKILL.md)

| 指标 | 示例值 |
|---|---:|
| 目标 | `a5` / camodel |
| 函数 | `decoder_layer_0_attention` |
| 总 cycles | 12,840 |
| CUBE cycles | 8,420（65.6%） |
| VECTOR cycles | 3,120（24.3%） |
| Scalar / sync cycles | 1,300（10.1%） |
| manifest | PASS |
| trace 清理 | PASS |
| 产物目录 | `build_output/incore_decoder_layer_0_attention_sample/` |

初步判断：该 kernel 的矩阵计算确实执行，不能把近空 trace 误判为性能很好。CUBE 占主导，后续应结合 instruction CSV 和真实设备测量，确认矩阵计算、访存或同步是否限制吞吐。

限制：in-core simulator 是单核 synthetic case，不能直接代表多核设备上的端到端性能。

## 6. 依赖冗余分析

对应 skill：[`dependency-redundancy`](../repo/pypto-skills/plugins/pypto-user/skills/dependency-redundancy/SKILL.md)

输入：`outputs/decoder_rank0/deps.json`

| 项目 | 示例结果 |
|---|---:|
| task 数量 | 48 |
| edge 数量 | 132 |
| 图深度 | 6 |
| explicit edges | 20 |
| tensormap edges | 64 |
| creator edges | 48 |
| cycle warning | 未发现 |
| `reduced` 移除 | 17 条 |
| `reduced_dataflow` 移除 | 23 条 |

### 6.1 解释

- 17 条结构性冗余边已经被更长路径隐含，可以考虑从 orchestration 中移除。
- dataflow 模式额外证明 6 条 creator 边满足字节级 INOUT 流转条件，可以移除。
- 仍保留的 creator 边包含 reuse generation、缺失 stride 信息或无法证明完整字节流转的情况。
- 移除依赖边只表示减少调度 bookkeeping，不等于已经证明端到端性能提升。

示例冗余边：`task_07 -> task_19`、`task_11 -> task_24`、`task_18 -> task_31`。

## 7. Critical path 分析

对应 skill：[`critical-path-analysis`](../repo/pypto-skills/plugins/pypto-user/skills/critical-path-analysis/SKILL.md)

输入：`outputs/decoder_run/`

| 指标 | Rank 0 示例结果 |
|---|---:|
| makespan | 1,000,000 ticks |
| static CPM | 760,000 ticks（76.0%） |
| observed path compute | 510,000 ticks（51.0%） |
| data-wait | 270,000 ticks（27.0%） |
| core-wait | 170,000 ticks（17.0%） |
| front-gap | 50,000 ticks（5.0%） |
| tiling check | exact |
| kernel names | 已解析 |
| capture round | 第 1 轮，包含 warm-up |

### 7.1 性能归类

初步归类：**依赖受限（dependency-bound）**，同时存在明显的数据等待。

依据：静态 CPM 已达到 makespan 的 76%，说明依赖图本身构成较高的延迟下限；仅增加 core 数量预计不能解决主要问题。观察路径上的 `data-wait` 主要由 `kv_cache_update` 和 `attention_matmul` 上游 producer 引起，应优先检查这些 producer 的完成时间和依赖 fan-in。

建议顺序：

1. 检查 17 条结构性冗余边是否可以减少依赖 fan-in。
2. 对 `data-wait` 对应的上游 producer 做 kernel-family 级分析。
3. 在修改后重新采集同一 workload，并与独立 wall-clock 对齐。
4. 只有确认依赖链缩短后仍存在 `core-wait`，才考虑 core assignment 或并发度调整。

## 8. 开发流程安全检查

对应 skills：[`git-commit`](../repo/pypto-skills/plugins/pypto-developer/skills/git-commit/SKILL.md)、[`github-pr`](../repo/pypto-skills/plugins/pypto-developer/skills/github-pr/SKILL.md)、[`fix-pr`](../repo/pypto-skills/plugins/pypto-developer/skills/fix-pr/SKILL.md)

本示例假设后续需要提交依赖图优化代码：

- 只允许暂存明确归属本任务的文件。
- 提交前必须执行仓库策略指定的验证命令。
- 创建或更新 PR 前必须确认 base repository、head repository、分支和权限。
- PR review 修复需要先列出完整反馈，再由用户确认处理范围。
- 远程推送和分支清理需要使用精确 OID/lease 校验。

## 9. 最终结论与限制

### 结论

示例运行已经证明基础环境和模型执行链路可用。当前最值得优先验证的是依赖图优化，而不是直接改 kernel 算术实现：静态关键路径占比较高，且存在可被工具证明的冗余依赖。

### 限制

- 所有数字均为模拟值。
- profiling 只代表单核模拟器，不代表真实设备。
- 关键路径只覆盖一次 capture，且包含 warm-up。
- 依赖冗余结论只适用于当前 `deps.json` 对应的 workload topology。
- 是否产生实际加速，需要在相同输入、形状、设备和测量方法下重新运行对比。

## 10. 建议交付物

- [ ] `decoder-lowering.html`：自包含 IR trace
- [ ] `incore_*/`：原始 simulator 数据、清理后的 trace 和 summary
- [ ] `deps_viewer_reduced.txt`
- [ ] `deps_viewer_reduced_dataflow.txt`
- [ ] `critical_path_report.txt`
- [ ] 一份包含版本、命令、设备、输入形状和限制条件的复现记录
