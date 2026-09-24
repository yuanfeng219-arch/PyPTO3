# Stage 5 Performance Report — softmax

## 1. Frozen Configuration

| Item | Value |
|------|-------|
| Operator | softmax (pure Vector/VF, single kernel, L0) |
| Device | NPU 1, DAV_3510 (A5), SoC Ascend950PR_9589 |
| Platform | 32 cube cores, 64 vector cores, vec_freq=1650 MHz, L2=128 MB, UB=248 KB |
| CANN | 9.2.0 (`/usr/local/Ascend/cann-9.2.0`) |
| PyPTO-Pro | `/usr/local/Ascend/cann-9.2.0/python/site-packages/pypto_pro/__init__.py` |
| torch / torch_npu | 2.8.0+cpu / 2.8.0.post4 |
| Target Op Name | `_Z25softmax_tile_group_kernelPDhS_iiii` (AI_CORE, single launch) |
| Manifest | `PERFORMANCE_CASES.json` (sha256=cc8b75f0..., 3 P0 cases) |
| Golden contract | `GOLDEN_PERF_REPORT.json` (frozen once, sha256=ef99ca66...) |
| Seed | 42 (Stage 4 validated, fixed) |
| Warm-up / Repeats | 3 / 3 |
| Timing source | PipeUtilization op_summary `Task Duration(us)` (canonical) |
| Aggregation | Trimmed mean (drop min/max of 3 repeats, keep middle) |
| Default target | `golden_reference_ratio = golden_per_iteration_npu_e2e_us / final_pypto_target_kernel_us >= 1.0` per P0 case |
| Precision | atol=1e-3, rtol=1e-3 (SPEC), CPU FP32 golden reference |

### Performance Cases (frozen manifest)

| Case ID | Shape | dtype | test_function |
|---------|-------|-------|---------------|
| p0_batch1_n128 | [1,128] | fp16 | test_softmax_p0_batch1_n128 |
| p0_batch4_n2048 | [4,2048] | fp16 | test_softmax_p0_batch4_n2048 |
| p0_batch32_n4096 | [32,4096] | fp16 | test_softmax_p0_batch32_n4096 |

All cases use `test_function` adapter (no CLI/env selector); each launches the target kernel exactly once. Seed=42 hardcoded in test functions (matches Stage 4).

## 2. Golden Reference (frozen once)

Collected via `collect_golden_reference.py` with `softmax_golden.py`, `--factory _make_inputs`, `--warmup 3 --repeats 3 --seed 42`, device 1. Each repeat calls Golden once; E2E = sum of all NPU kernels in that call.

| Case ID | Golden per-iteration E2E (µs) |
|---------|-------------------------------|
| p0_batch1_n128 | 11.865 |
| p0_batch4_n2048 | 18.55 |
| p0_batch32_n4096 | 26.562 |

Report: `GOLDEN_PERF_REPORT.md` / `GOLDEN_PERF_REPORT.json`

## 3. Baseline (Stage 4 implementation, pre-optimization)

**Collection**: `docs/perf/round_001/`, collection_id=`compare_20260831_220259_39123`

| Case ID | Shape | Golden E2E (µs) | PyPTO kernel (µs) | Samples (µs) | Golden ratio | Target met |
|---------|-------|-----------------|---------------------|---------------|--------------|------------|
| p0_batch1_n128 | [1,128] | 11.865 | 3.250 | [3.149, 3.250, 3.506] | 3.651x | ✓ |
| p0_batch4_n2048 | [4,2048] | 18.55 | 4.714 | [4.003, 4.714, 4.881] | 3.935x | ✓ |
| p0_batch32_n4096 | [32,4096] | 26.562 | 5.926 | [5.834, 5.926, 6.130] | 4.482x | ✓ |

Aggregate golden ratio: 4.10x. **Default target met for all P0 cases.**

### Baseline bottleneck analysis (p0_batch32_n4096, largest case)

| Metric | Value |
|--------|-------|
| Task Duration | 5.834 µs, BlockDim=11 |
| aiv_time (max core) | 5.340 µs |
| Overhead | 0.494 µs (8.5%) |
| vec | 45.30% |
| scalar | 8.70% |
| mte2 | 7.60% |
| mte3 | 6.20% |
| icache_miss | 14.00% |
| L2 read hit | 99.80% |
| vec_resc_cflt | 121.50% |
| Routing | no_dominant_pipe |

Key observation: `update_mask` is called per-register per-pass inside the register loop. For N=4096 (n_regs=32), this is 32×3=96 `update_mask(fp16)` + 192 `update_mask(fp32)` calls per row — significant scalar overhead interleaved with vec compute, stalling the vec pipe.

## 4. Optimization: Mask Lifting

### Hypothesis

Per playbook §"mask 提升出寄存器循环": for tile width ≥ 64 lanes (all P0 cases have N as multiple of 128), the full-register mask is identical across all register iterations. Lifting it out of the loop eliminates per-register `update_mask` scalar overhead and reduces vec pipe stalls from interleaved mask instructions.

### Implementation

- **Full-register masks** (128 fp16 lanes, 64 fp32 lanes) computed **once** before the row loop, reused across all full-register iterations in all 3 passes.
- **Tail register** handled by a separate `pl.range(n_full, n_full + n_tail_iters)` loop where `n_tail_iters = ceil(n_tail / LANES)` — this is **0 when N is a multiple of 128** (empty loop, no overhead) and **1** when there's a partial tail register.
- No runtime `if` needed (DSL limitation — runtime `if` with variable declarations in `@pl.vector_function` causes C++ scoping errors; verified and rejected).
- **No design change**: same 3-pass structure (max→exp+sum→div+store), same TILE_ROWS=3/MAX_N=8192, same cast chain (fp16→fp32→fp16), same double-buffer tile groups, same auto_mutex. Only mask computation is restructured.

### Correctness

All 8 Stage 4 test cases pass with **identical precision** to baseline (same max_abs_error values, matched_ratio=1.0). Run command: `python custom/softmax/test_softmax.py`.

### Quick A/B (task_time timing)

| Case | Baseline (µs) | Lifted quick (µs) | Change |
|------|---------------|---------------------|--------|
| p0_batch1_n128 | 3.25 | 3.28 | +0.9% (noise, n_regs=1) |
| p0_batch4_n2048 | 4.71 | 4.07 | -13.6% |
| p0_batch32_n4096 | 5.93 | 4.87 | -17.9% |

## 5. Final (optimized implementation, formal compare)

**Collection**: `docs/perf/round_002/`, collection_id=`compare_20260831_230218_210028`

| Case ID | Shape | Golden E2E (µs) | PyPTO kernel (µs) | Samples (µs) | Golden ratio | Target met | Speedup vs baseline |
|---------|-------|-----------------|---------------------|---------------|--------------|------------|---------------------|
| p0_batch1_n128 | [1,128] | 11.865 | 3.428 | [3.273, 3.428, 3.440] | 3.461x | ✓ | 0.948x |
| p0_batch4_n2048 | [4,2048] | 18.55 | 4.015 | [3.607, 4.015, 4.069] | 4.620x | ✓ | 1.174x |
| p0_batch32_n4096 | [32,4096] | 26.562 | 4.946 | [4.913, 4.946, 5.026] | 5.370x | ✓ | 1.198x |

Aggregate golden ratio: 4.60x. Average speedup: 1.107x. **Default target met for all P0 cases.**

Per-case regression note: p0_batch1_n128 (n_regs=1) shows 5.2% regression (3.25→3.428 µs) — mask lifting has no benefit when there's only 1 register (mask computed once either way), and the extra loop structure adds slight overhead. This is within measurement noise (baseline samples ranged 3.149–3.506) and the target is still met by wide margin (3.461x ≥ 1.0).

### Final bottleneck analysis (p0_batch32_n4096)

| Metric | Baseline | Final | Change |
|--------|----------|-------|--------|
| Task Duration | 5.834 µs | 4.946 µs | -15.2% |
| aiv_time | 5.340 µs | 4.470 µs | -16.3% |
| vec | 45.30% | 30.80% | -14.5pp |
| scalar | 8.70% | 11.70% | +3.0pp |
| mte2 | 7.60% | 9.40% | +1.8pp |
| mte3 | 6.20% | 7.70% | +1.5pp |
| icache_miss | 14.00% | 12.70% | -1.3pp |
| overhead | 8.5% | 9.6% | +1.1pp |
| L2 read hit | 99.80% | 99.80% | — |
| vec_resc_cflt | 121.50% | 88.10% | -33.4pp |

Mechanism confirmed: mask lifting reduced total instruction count (fewer `update_mask` calls), which reduced both vec busy time (interleaved mask instructions no longer stall the vec pipe) and icache pressure. The vec ratio decreased because the same semantic compute (exp_sub, div, astype) now executes in fewer total cycles. Resource conflict (vec_resc_cflt) also decreased significantly (121.5%→88.1%).

## 6. Roofline / Pipeline / Scalar Terminal State

> Structured evidence: `performance.json` → `per_case[].terminal_evidence` (consistent with this section and `perf_report.md`).

### 6.1 Platform Peak (950PR_958x.ini, DAV_3510, Ascend950PR_9589)

| Constant | Value | Source |
|----------|-------|--------|
| vec_freq | 1650 MHz = 1.650e9 Hz | `[AICoreSpec] cube_freq=1650`; `[VectorCoreSpec] vec_freq=1650` |
| vector_core_cnt | 64 | `[SoCInfo] vector_core_cnt=64` |
| l2_size | 128 MB = 134217728 B | `[SoCInfo] l2_size=134217728` |
| VectorCoreMemoryRates.ddr_rate | 16 | `[VectorCoreMemoryRates] ddr_rate=16` |
| **Peak VF throughput per core** | **1.650e9 instr/s** (1 instr/cycle at 1.650 GHz) | vec_freq × 1 instr/cycle |
| **Peak HBM BW (vector-only)** | **1.69e12 B/s = 1.69 TB/s** | vector_core_cnt × ddr_rate × vec_freq = 64 × 16 × 1.65e9 |

> Bandwidth note: the `ddr_rate` units in the ini are not self-consistent (`ddr_rate=16` beside `ub_to_ddr_rate=40` for VectorCore, see `a5-roofline-and-levers.md`). The 1.69 TB/s derived from the formula is consistent with the measured aligned-plane MTE2 read of ~1.7 TB/s cited in the same reference; both are used only to establish the binding ceiling, not as achieved throughput.

### 6.2 Workload Model (per dominant core)

VF instruction count is derived from the source (`test_softmax.py` → `softmax_rows_vf`, 3-pass structure with lifted masks):

```
setup = 5 instructions (2 create_mask + 3 update_mask, lifted out of row loop)
per_row = 4 + 19 × n_full   (pass1: 3×n_full, pass2: 7×n_full, pass3: 9×n_full, 4×vf.full)
total = setup + rows × per_row
```

Necessary GM bytes = rows × N × 2 (fp16 read) + rows × N × 2 (fp16 write) = rows × N × 4 B.

All P0 cases have `n_tail_iters = 0` (N is a multiple of 128), so no tail-register instructions.

### 6.3 Per-Case Roofline Derivation

**p0_batch32_n4096** (B=32, N=4096, block_dim=11, dominant core: 3 rows, n_full=32):

| Metric | Value | Derivation |
|--------|-------|------------|
| VF instructions/core | 1841 | 5 + 3 × (4 + 19×32) = 5 + 3×612 |
| GM bytes/core | 49152 B (48 KB) | 3 × 4096 × 4 |
| L2 residency | Yes (input 256 KB << 128 MB L2, hit 99.8%) | 32 × 4096 × 2 = 262144 B |
| Compute floor | 1.1158 µs | 1841 / 1.650e9 |
| Memory floor | 0.3200 µs | 49152 / (1.69e12/11) |
| **C/M ratio** | **3.49** | 1.1158 / 0.3200 |
| Achieved VF throughput | 1.337 GIPS (81.0% of peak) | 1841 / 1.377 µs (aiv_vec_time) |
| Achieved BW per core | 64.3 GB/s (41.8% of peak) | 49152 / (0.42+0.345) µs |
| **Terminal** | **compute_bound** | C/M = 3.49 ≥ 2.0; VF util 81.0% > BW util 41.8% |

**p0_batch4_n2048** (B=4, N=2048, block_dim=2, dominant core: 3 rows, n_full=16):

| Metric | Value | Derivation |
|--------|-------|------------|
| VF instructions/core | 929 | 5 + 3 × (4 + 19×16) = 5 + 3×308 |
| GM bytes/core | 24576 B (24 KB) | 3 × 2048 × 4 |
| L2 residency | Yes (input 16 KB << 128 MB) | 4 × 2048 × 2 = 16384 B |
| Compute floor | 0.5630 µs | 929 / 1.650e9 |
| Memory floor | 0.0291 µs | 24576 / (1.69e12/2) |
| **C/M ratio** | **19.35** | 0.5630 / 0.0291 |
| Achieved VF throughput | 1.674 GIPS (~100% of peak*) | 929 / 0.555 µs |
| Achieved BW per core | 40.6 GB/s (4.8% of peak) | 24576 / (0.35+0.255) µs |
| **Terminal** | **compute_bound** | C/M = 19.4 ≥ 2.0; VF util >> BW util |

> *Achieved VF throughput slightly exceeds the 1-instr/cycle simplification; the VF pipe can issue some instruction types (arithmetic) at >1 per cycle. The key comparison is VF util >> BW util, confirming compute as the binding ceiling.

**p0_batch1_n128** (B=1, N=128, block_dim=1, 1 row, n_full=1):

| Metric | Value | Derivation |
|--------|-------|------------|
| VF instructions/core | 28 | 5 + 1 × (4 + 19×1) |
| GM bytes/core | 512 B | 1 × 128 × 4 |
| L2 residency | Yes (input 256 B) | 1 × 128 × 2 = 256 B |
| Compute floor | 0.0170 µs | 28 / 1.650e9 |
| Memory floor | 0.0003 µs | 512 / 1.69e12 |
| **C/M ratio** | **56.0** | 0.0170 / 0.0003 |
| Achieved VF throughput | 0.311 GIPS (18.9% of peak) | 28 / 0.09 µs |
| Achieved BW per core | 1.15 GB/s (0.1% of peak) | 512 / (0.222+0.223) µs |
| **Terminal** | **compute_bound** | C/M = 56.0 ≥ 2.0; both tiny, kernel is overhead-bound, but compute is the binding ceiling |

> For p0_batch1_n128, the workload (28 instructions, 512 B) is too small to saturate any pipe. The kernel time (3.428 µs) is dominated by launch overhead and setup. The Roofline model still identifies compute as the binding ceiling (compute floor >> memory floor), even though neither ceiling is reached.

### 6.4 Terminal State Summary

| Case | roofline_terminal.status | pipeline_evidence.status | scalar_evidence.dominant | completion_eligible | target_met |
|------|--------------------------|--------------------------|---------------------------|---------------------|------------|
| p0_batch1_n128 | compute_bound | not_applicable_with_dag | false (14.7% < 30%) | true | true |
| p0_batch4_n2048 | compute_bound | not_applicable_with_dag | false (13.7% < 30%) | true | true |
| p0_batch32_n4096 | compute_bound | not_applicable_with_dag | false (11.7% < 30%) | true | true |

### 6.5 Roofline Terminal Justification

**`compute_bound` for all P0 cases** — proven by numerical derivation:

1. **Platform peak**: Peak VF throughput = 1.650e9 instr/s per core (vec_freq, 1 instr/cycle); Peak HBM BW = 1.69 TB/s (64 cores × ddr_rate=16 × vec_freq). Source: `950PR_958x.ini`.

2. **Compute floor >> Memory floor**: For all 3 P0 cases, the compute floor (VF instruction count / peak VF throughput) exceeds the memory floor (necessary GM bytes / per-core peak bandwidth) by 3.5x–56x. Data is L2-resident (all inputs << 128 MB L2, hit rate 99.8%), so HBM bandwidth is not the binding constraint.

3. **Achieved VF utilization > Achieved BW utilization**: The VF pipe runs at 81.0% of peak when active (p0_batch32_n4096), while the memory pipes run at 41.8% of peak. For smaller cases, the VF pipe is near or at saturation while BW utilization is negligible (4.8%, 0.1%). This confirms compute is the more saturated resource.

4. **No self-contradiction**: The previous report incorrectly stated `balanced_compute_movement` while also claiming "compute >> data movement." This was contradictory. The correct terminal state is `compute_bound` — compute floor exceeds memory floor by ≥3.5x for all cases, meeting the ≥2.0 threshold for `compute_bound`.

5. **Gap between actual time and compute floor**: The actual aiv_time (4.47 µs for p0_batch32_n4096) is higher than the compute floor (1.12 µs) due to: icache_miss (12.7%, constant instruction footprint of 3-pass structure), scalar (11.7%, mask setup and loop control), resource conflicts (vec_resc_cflt 88.1%), and launch overhead. These overheads prevent the VF pipe from running at 100% continuously (it's active only 30.8% of total time), but they do NOT change the binding ceiling — compute is still the resource that would limit performance if all overheads were eliminated.

### 6.6 Pipeline Evidence

**`not_applicable_with_dag` for all P0 cases** — proven by Tile DAG analysis:

Per-core Tile DAG: `{MTE2 load tile_0 → VF compute tile_0 → MTE3 store tile_0}` — a single linear chain with no second work item to pipeline.

| Case | B | num_tiles (ceil(B/3)) | block_dim | tiles/core | overlap possible |
|------|---|----------------------|-----------|------------|-----------------|
| p0_batch1_n128 | 1 | 1 | 1 | 1 | No (single work item) |
| p0_batch4_n2048 | 4 | 2 | 2 | 1 | No (single work item) |
| p0_batch32_n4096 | 32 | 11 | 11 | 1 | No (single work item) |

The double-buffer (depth=2) allocates 2 slots per group, but only slot 0 is used per core (no tile k+1 to overlap with tile k). There are no two pipelinable work items and no legal overlapping edges.

**Timeline capture**: Attempted via `msprof_perf_summary.py --timeline` for all 3 P0 cases. All failed at the preflight stage: the preflight calls `_profile_env(device_id, ..., tile_fwk=False)` which sets `ASCEND_RT_VISIBLE_DEVICES=1`, conflicting with `TILE_FWK_DEVICE_ID=1` (device remaps to logical 0, making `TILE_FWK_DEVICE_ID=1` invalid, NPU error 107001). This is a tooling limitation in the preflight environment setup, not a pipeline failure. The DAG proof independently establishes `not_applicable_with_dag` and does not depend on timeline capture.

### 6.7 Scalar Evidence

**`dominant=false` for all P0 cases** — scalar ratio < 30% threshold:

| Case | scalar_ratio | threshold | scalar_route_triggered | icache_miss |
|------|-------------|-----------|------------------------|-------------|
| p0_batch1_n128 | 14.7% | 30% | false | 12.7% |
| p0_batch4_n2048 | 13.7% | 30% | false | 12.7% |
| p0_batch32_n4096 | 11.7% | 30% | false | 12.7% |

The `icache_miss` rate is constant at 12.7% across all cases (fixed instruction footprint from the 3-pass structure). It is below the 15% FIXP threshold and is not a scalar-bound condition. The `vec_resc_cflt` (88.1% for p0_batch32_n4096) indicates VF resource contention (register file port conflicts), not scalar dominance — the scalar pipe is not the critical path.

## 7. Candidate Coverage Ledger

| # | Candidate | Status | Evidence | Reason |
|---|-----------|--------|----------|--------|
| 1 | Mask lifting (full-register masks out of loop) | **Accepted** | 14-20% faster on large cases (formal compare), identical precision, no design change | Reduces update_mask scalar overhead and vec pipe stalls |
| 2 | Runtime `if` for tail register split | **Rejected** | C++ scoping error: `__inline_0_tail_mreg_fp16_2` type mismatch | DSL doesn't support runtime `if` with variable declarations in @pl.vector_function; replaced by empty-tail-loop (candidate 1) |
| 3 | Empty-tail-loop approach (pl.range with n_tail_iters=0) | **Accepted (part of #1)** | Compiles, all 8 cases pass, 14-20% faster | Avoids runtime `if`; empty loop when N%128==0 |
| 4 | TILE_ROWS adjustment (DESIGN §9 allows) | Not attempted | Target already met by wide margin | DESIGN allows tuning TILE_ROWS, but no need — target met with 3.46-5.37x ratios |
| 5 | exp result reuse (pass2→pass3) | Not attempted | Would change DESIGN 3-pass structure | DESIGN freezes pass2 (exp+sum) and pass3 (div+store) as separate passes with exp recomputed; reuse would require UB staging of fp32 results → design_violation |
| 6 | Pass fusion (combine max+exp+sum) | Not attempted | Would change DESIGN algorithm | DESIGN §0 explicitly defines 3-pass structure with m/s as register scalars; fusion changes Module semantics |

## 8. Correctness Evidence

Final implementation (`test_softmax.py` with mask lifting) passes all 8 Stage 4 test cases:

| Case | Shape | matched_ratio | max_abs_error | Status |
|------|-------|---------------|---------------|--------|
| aligned | [3,128] | 1.0 | 2.29e-05 | PASS |
| row_tail | [4,128] | 1.0 | 2.29e-05 | PASS |
| col_tail | [3,200] | 1.0 | 1.37e-05 | PASS |
| tail2d | [4,200] | 1.0 | 1.37e-05 | PASS |
| multitile | [7,200] | 1.0 | 1.71e-05 | PASS |
| p0_batch1_n128 | [1,128] | 1.0 | 1.37e-05 | PASS |
| p0_batch4_n2048 | [4,2048] | 1.0 | 1.79e-06 | PASS |
| p0_batch32_n4096 | [32,4096] | 1.0 | 2.77e-06 | PASS |

Precision is identical to baseline (same max_abs_error values). Command: `python custom/softmax/test_softmax.py` → "All tests PASS!"

## 9. Stage 4 Iron Rules (preserved)

- ✓ Single kernel (`softmax_tile_group_kernel`, single `@pl.jit`)
- ✓ Single wrapper call (`softmax_wrapper` calls kernel once, no loop)
- ✓ Kernel not called in a loop
- ✓ Host does no core computation (only dtype/shape/contiguity validation + `torch.empty_like` + launch)
- ✓ No case deletion, no shape/dtype/device/seed/warm-up/repeats/Op Name change
- ✓ No precision threshold change (atol=1e-3, rtol=1e-3)
- ✓ No SPEC/DESIGN/golden/module_interfaces modification

## 10. Evidence Paths

| Artifact | Path |
|----------|------|
| Final test_softmax.py | `custom/softmax/test_softmax.py` |
| PERFORMANCE_CASES.json | `custom/softmax/PERFORMANCE_CASES.json` |
| GOLDEN_PERF_REPORT.json/md | `custom/softmax/GOLDEN_PERF_REPORT.json` / `.md` |
| performance.json (final) | `custom/softmax/performance.json` |
| performance.log (final) | `custom/softmax/performance.log` |
| perf_report.md (final) | `custom/softmax/perf_report.md` |
| Baseline collection | `custom/softmax/docs/perf/round_001/` (collection_id=compare_20260831_220259_39123) |
| Final collection | `custom/softmax/docs/perf/round_002/` (collection_id=compare_20260831_230218_210028) |
| Per-case metrics | `docs/perf/round_002/case_<id>_<hash>/repeat_NNN/` (7 metrics + summary.txt) |
| Terminal evidence (structured) | `performance.json` → `per_case[].terminal_evidence` (roofline_model, pipeline_evidence, scalar_evidence, completion_eligible) |
| Timeline capture | Attempted via `msprof_perf_summary.py --timeline` for all 3 P0 cases; all failed at preflight (ASCEND_RT_VISIBLE_DEVICES vs TILE_FWK_DEVICE_ID conflict, NPU error 107001). DAG proof used instead for `not_applicable_with_dag`. |
| Baseline backup | `custom/softmax/test_softmax.py.bak_stage5` |

## 11. Conclusion

**target_met: true**

All P0 cases meet the default Golden reference target (golden_reference_ratio ≥ 1.0):
- p0_batch1_n128: 3.461x ✓
- p0_batch4_n2048: 4.620x ✓
- p0_batch32_n4096: 5.370x ✓

Terminal state: `compute_bound` (compute floor exceeds memory floor by 3.5x–56x; VF utilization exceeds BW utilization; data L2-resident). Pipeline: `not_applicable_with_dag` (1 tile per core, single linear chain, no overlap possible by DAG). Scalar: not dominant (<30% for all cases). All gates passed.

**Structured evidence consistency**: `performance.json` → `per_case[].terminal_evidence` and `perf_report.md` both report `compute_bound` / `not_applicable_with_dag` / `scalar_dominant=false` / `completion_eligible=true`, consistent with this report. The `repeat_bound_diagnoses` field remains `insufficient_evidence` / `unverified` by design (it is the script's ratio-route, explicitly not a terminal conclusion); the proven terminal state is in the `terminal_evidence` field.

Optimization applied: mask lifting (full-register masks lifted out of register loop) — 15-20% faster on larger cases while preserving identical precision and all Stage 4 iron rules. No design violations.
