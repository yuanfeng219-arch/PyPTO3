# 性能评估结果

- **Operator**: softmax
- **Device**: npu:1 (source=cli)
- **Warmup**: 3
- **Repeats**: 3
- **Seed**: 42
- **Timing method**: msprof.op_summary.Task_Duration
- **Timing source**: PipeUtilization
- **Target Op Name**: `_Z25softmax_tile_group_kernelPDhS_iiii`
- **Case selector**: `function_adapter:PERFORMANCE_CASES.test_function`
- **Profiling mode**: compare
- **Collection ID**: compare_20260831_230218_210028
- **Performance cases**: `case_manifest`; source=`/data/x00952168/softmax-opencode-benchmark-41-20260831-194403/custom/softmax/PERFORMANCE_CASES.json`; sha256=`cc8b75f0eb84fb70ab2dbb641cdcdefbd7d68f74032c94f49394381e7d7a90ce`
- **Golden target source status**: `joined`
- **Default target**: `golden_per_iteration_e2e_us / pypto_target_kernel_us >= 1.0` for every P0 case
- **Default target status**: `met` (valid_for_target_met=true)
- **Comparison scope**: `golden_e2e_to_pypto_target_kernel`
- **Golden source**: `torch_npu.kernel_details.all_golden_npu_kernel_sum` (iterations=1, normalized per iteration)
- **Ratio semantics**: this ratio is valid for the Stage 5 default Golden target; it is not baseline-to-final optimization speedup.

## Golden 默认目标对比

| Case | Shape | DType | PyPTO目标kernel(us) | Golden每迭代E2E(us) | 默认目标比值(E2E/kernel) |
| ---- | ----- | ----- | ------------- | -------- | -------------- |
| p0_batch1_n128 | [1,128] | fp16 | 3.43 | 11.87 | 3.461 |
| p0_batch4_n2048 | [4,2048] | fp16 | 4.01 | 18.55 | 4.620 |
| p0_batch32_n4096 | [32,4096] | fp16 | 4.95 | 26.56 | 5.370 |

### 重复采样证据

- `p0_batch1_n128`: samples_us=[3.428, 3.273, 3.44]; selected_op_names=['_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii']; evidence=`/data/x00952168/softmax-opencode-benchmark-41-20260831-194403/custom/softmax/docs/perf/round_002/case_p0_batch1_n128_b057c21c`
- `p0_batch4_n2048`: samples_us=[4.069, 3.607, 4.015]; selected_op_names=['_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii']; evidence=`/data/x00952168/softmax-opencode-benchmark-41-20260831-194403/custom/softmax/docs/perf/round_002/case_p0_batch4_n2048_230a7547`
- `p0_batch32_n4096`: samples_us=[4.946, 5.026, 4.913]; selected_op_names=['_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii']; evidence=`/data/x00952168/softmax-opencode-benchmark-41-20260831-194403/custom/softmax/docs/perf/round_002/case_p0_batch32_n4096_ed66c84d`

## Pipe / Bound 路由（诊断，不是最终 Roofline 结论）

| Case | Repeat 路由 | Scalar 候选 | Roofline 结论 | 搬运/计算流水证据 |
| ---- | ----------- | ----------- | ------------- | ----------------- |
| p0_batch1_n128 | ['no_dominant_pipe', 'no_dominant_pipe', 'no_dominant_pipe'] | false | ['insufficient_evidence'] | ['unverified'] |
| p0_batch4_n2048 | ['no_dominant_pipe', 'no_dominant_pipe', 'no_dominant_pipe'] | false | ['insufficient_evidence'] | ['unverified'] |
| p0_batch32_n4096 | ['no_dominant_pipe', 'no_dominant_pipe', 'no_dominant_pipe'] | false | ['insufficient_evidence'] | ['unverified'] |

> `op_summary` 的 pipe ratio 只用于选择下一项实验；它不包含事件先后关系，不能单独证明 compute/movement bound 或搬运与计算已经重叠。最终结论在下方「Roofline / 流水 / Scalar 终态证据」及 `performance.json` 的 `terminal_evidence` 字段中给出，引用工作量/流量模型及 DAG 依赖证据。

## Roofline / 流水 / Scalar 终态证据

> 以下终态由 `performance.json` 的 `per_case[].terminal_evidence` 结构化字段提供，与 `PERFORMANCE_REPORT.md` §6 一致。

| Case | roofline_terminal | pipeline_evidence | scalar_dominant | completion_eligible |
| ---- | ----------------- | ----------------- | --------------- | ------------------- |
| p0_batch1_n128 | compute_bound | not_applicable_with_dag | false (14.7%) | true |
| p0_batch4_n2048 | compute_bound | not_applicable_with_dag | false (13.7%) | true |
| p0_batch32_n4096 | compute_bound | not_applicable_with_dag | false (11.7%) | true |

### Roofline 数值推导（per case）

平台峰值（`950PR_958x.ini`, DAV_3510, Ascend950PR_9589）：
- Peak VF throughput per core = vec_freq × 1 instr/cycle = 1.650e9 instr/s
- Peak HBM BW (vector-only) = vector_core_cnt × ddr_rate × vec_freq = 64 × 16 × 1.65e9 = 1.69e12 B/s

| Case | VF instr/core | GM bytes/core | Compute floor (us) | Memory floor (us) | C/M ratio | Achieved VF util | Achieved BW util | Terminal |
| ---- | ------------- | ------------- | ------------------- | ------------------ | --------- | ---------------- | ---------------- | -------- |
| p0_batch1_n128 | 28 | 512 | 0.0170 | 0.0003 | 56.0 | 18.9% | 0.1% | compute_bound |
| p0_batch4_n2048 | 929 | 24576 | 0.5630 | 0.0291 | 19.4 | 101.4% | 4.8% | compute_bound |
| p0_batch32_n4096 | 1841 | 49152 | 1.1158 | 0.3200 | 3.5 | 81.0% | 41.8% | compute_bound |

> 数据均 L2 驻留（全部输入 << 128 MB L2，hit rate 99.8%），HBM 带宽非瓶颈。compute floor 显著高于 memory floor（3.5x–56x），VF 利用率均高于 BW 利用率 → binding ceiling 为 compute。

### Pipeline 证据

- **status**: `not_applicable_with_dag`（全部 P0 case）
- **DAG**: 每核 1 tile，单线性链 `{MTE2 load tile_0 → VF compute tile_0 → MTE3 store tile_0}`，无第二 work item 可流水，无合法重叠边。
- **timeline 捕获**: 已尝试 `msprof_perf_summary.py --timeline`，失败于 preflight：`_profile_env(tile_fwk=False)` 设置 `ASCEND_RT_VISIBLE_DEVICES=1` 与 `TILE_FWK_DEVICE_ID=1` 冲突（device remap 使 TILE_FWK_DEVICE_ID=1 越界，NPU error 107001）。DAG 证明独立于 timeline 捕获。

### Scalar 证据

- 全部 P0 case scalar ratio < 30%（14.7%, 13.7%, 11.7%），`scalar_route_triggered=false`，scalar 非关键路径。

## PyPTO measurement summary

| Metric | Value |
| ---- | -- |
| Cases | 3 |
| Valid PyPTO measurements | 3 |
| Mean PyPTO target-kernel (us) | 4.130 |
| Median PyPTO target-kernel (us) | 4.015 |
| Total PyPTO target-kernel (us) | 12.389 |
| Golden target cases | 3 |
| Default target threshold | 1.0 |
| Default target status | met |
| Mean Golden-E2E/PyPTO-kernel target ratio | 4.484 |
| Target-met cases (ratio >=1) | 3 |
| Target-missed cases (ratio <1) | 0 |

### 按数据类型汇总

| DType | 用例数 | 平均默认目标比值 | 达标(>=1) | 未达标(<1) |
| ----- | ------ | ------------------- | ------------- | -------- |
| fp16 | 3 | 4.484 | 3 | 0 |

## 简短分析

- Golden 每迭代 E2E / PyPTO target-kernel 平均目标比值为 4.484；默认目标要求每个 P0 case 均不低于 1.0。该比值不是 baseline→final 优化加速比。
- 详细瓶颈分析见 msprof 归档目录（op_summary_*.csv + summary.txt）。

## 深度瓶颈分析

- 深度采集轮次：`/data/x00952168/softmax-opencode-benchmark-41-20260831-194403/custom/softmax/docs/perf/round_002`
- `p0_batch1_n128`：`/data/x00952168/softmax-opencode-benchmark-41-20260831-194403/custom/softmax/docs/perf/round_002/case_p0_batch1_n128_b057c21c/summary.txt` 及七组 `op_summary_*.csv`
- `p0_batch4_n2048`：`/data/x00952168/softmax-opencode-benchmark-41-20260831-194403/custom/softmax/docs/perf/round_002/case_p0_batch4_n2048_230a7547/summary.txt` 及七组 `op_summary_*.csv`
- `p0_batch32_n4096`：`/data/x00952168/softmax-opencode-benchmark-41-20260831-194403/custom/softmax/docs/perf/round_002/case_p0_batch32_n4096_ed66c84d/summary.txt` 及七组 `op_summary_*.csv`


