# 性能评估结果

- **Operator**: softmax
- **Device**: npu:1 (source=cli)
- **Warmup**: 3
- **Repeats**: 3
- **Seed**: 42
- **Timing method**: msprof.quick.Task_Duration
- **Timing source**: task_time
- **Target Op Name**: `_Z25softmax_tile_group_kernelPDhS_iiii`
- **Case selector**: `function_adapter:PERFORMANCE_CASES.test_function`
- **Profiling mode**: quick
- **Collection ID**: quick_20260831_224844_173104
- **Performance cases**: `case_manifest`; source=`/data/x00952168/softmax-opencode-benchmark-41-20260831-194403/custom/softmax/PERFORMANCE_CASES.json`; sha256=`cc8b75f0eb84fb70ab2dbb641cdcdefbd7d68f74032c94f49394381e7d7a90ce`
- **Golden target source status**: `joined`
- **Default target**: `golden_per_iteration_e2e_us / pypto_target_kernel_us >= 1.0` for every P0 case
- **Default target status**: `unavailable` (valid_for_target_met=false)
- **Comparison scope**: `golden_e2e_to_pypto_target_kernel`
- **Golden source**: `torch_npu.kernel_details.all_golden_npu_kernel_sum` (iterations=1, normalized per iteration)
- **Ratio semantics**: this ratio is valid for the Stage 5 default Golden target; it is not baseline-to-final optimization speedup.

## Golden 默认目标对比

| Case | Shape | DType | PyPTO目标kernel(us) | Golden每迭代E2E(us) | 默认目标比值(E2E/kernel) |
| ---- | ----- | ----- | ------------- | -------- | -------------- |
| p0_batch1_n128 | [1,128] | fp16 | 3.28 | 11.87 | 3.617 |
| p0_batch4_n2048 | [4,2048] | fp16 | 4.07 | 18.55 | 4.560 |
| p0_batch32_n4096 | [32,4096] | fp16 | 4.87 | 26.56 | 5.460 |

### 重复采样证据

- `p0_batch1_n128`: samples_us=[3.049, 3.304, 3.28]; selected_op_names=['_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii']; evidence=`quick-summary-only`
- `p0_batch4_n2048`: samples_us=[4.068, 4.103, 3.75]; selected_op_names=['_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii']; evidence=`quick-summary-only`
- `p0_batch32_n4096`: samples_us=[5.188, 4.865, 4.798]; selected_op_names=['_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii', '_Z25softmax_tile_group_kernelPDhS_iiii']; evidence=`quick-summary-only`

## PyPTO measurement summary

| Metric | Value |
| ---- | -- |
| Cases | 3 |
| Valid PyPTO measurements | 3 |
| Mean PyPTO target-kernel (us) | 4.071 |
| Median PyPTO target-kernel (us) | 4.068 |
| Total PyPTO target-kernel (us) | 12.213 |
| Golden target cases | 3 |
| Default target threshold | 1.0 |
| Default target status | unavailable |
| Mean Golden-E2E/PyPTO-kernel target ratio | 4.546 |
| Target-met cases (ratio >=1) | 3 |
| Target-missed cases (ratio <1) | 0 |

### 按数据类型汇总

| DType | 用例数 | 平均默认目标比值 | 达标(>=1) | 未达标(<1) |
| ----- | ------ | ------------------- | ------------- | -------- |
| fp16 | 3 | 4.546 | 3 | 0 |

## 简短分析

- Golden 每迭代 E2E / PyPTO target-kernel 平均目标比值为 4.546；默认目标要求每个 P0 case 均不低于 1.0。该比值不是 baseline→final 优化加速比。
- 详细瓶颈分析见 msprof 归档目录（op_summary_*.csv + summary.txt）。

## 深度瓶颈分析

- quick 模式仅用于筛选，没有七组 aic-metrics 深度归档；最终证据必须用同配置 `--compare` 重采。


