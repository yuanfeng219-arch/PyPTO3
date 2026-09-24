# 算子需求规范

## 机器合同

> 下列 JSON 是公式、公开接口、验收配置的唯一机器事实源。正文只解释语义与证据，不复制字段值。

```json machine-contract
{
  "schema_version": 1,
  "op_name": "softmax",
  "formula": "y[b,i] = exp(x[b,i] - m[b]) / s[b]; m[b] = max_j(x[b,j]); s[b] = sum_j(exp(x[b,j] - m[b]))",
  "supported_dtypes": ["float16"],
  "inputs": [
    {
      "name": "x",
      "shape": ["B", "N"],
      "dtype": "float16",
      "value_range": [-65504, 65504]
    }
  ],
  "outputs": [
    {
      "name": "y",
      "shape": ["B", "N"],
      "dtype": "float16",
      "value_range": [0, 1]
    }
  ],
  "default_params": {},
  "tolerance": {
    "atol": 0.001,
    "rtol": 0.001
  },
  "dynamic_axes_ranges": {
    "B": [1, 1024],
    "N": [1, 8192]
  },
  "shape_constraints": [
    "N >= 1 (reduction axis must be non-empty so each row has a well-defined max and sum)",
    "B >= 1 (batch dimension must be non-empty)"
  ],
  "p0_cases": [
    {
      "name": "p0_batch1_n128",
      "params": {},
      "input_shapes": {"x": [1, 128]},
      "output_shapes": {"y": [1, 128]}
    },
    {
      "name": "p0_batch4_n2048",
      "params": {},
      "input_shapes": {"x": [4, 2048]},
      "output_shapes": {"y": [4, 2048]}
    },
    {
      "name": "p0_batch32_n4096",
      "params": {},
      "input_shapes": {"x": [32, 4096]},
      "output_shapes": {"y": [32, 4096]}
    }
  ],
  "perf_target": null
}
```

## 语义说明

### 1. 功能与分类

对二维输入 `x`（shape `[B, N]`，dtype `float16`）的**最后一个维度（axis=-1，即列维 N）**做数值稳定 softmax。每个行 `b` 独立归一化：先减行内最大值做数值稳定，再取指数、按行求和、逐元素相除，得到同 shape 的概率分布。属于 **row-wise reduction + elementwise** 类算子，无辅助输入、无可选参数、无随机性、无稀疏性、无量化。输出 `y` 与输入同 shape 同 dtype，每行元素之和为 1（在 fp16 精度范围内）。

### 2. 公式符号与依据

- `x[b, i]`：输入，`b ∈ [0, B)` 为行索引，`i ∈ [0, N)` 为列索引。
- `m[b] = max_j(x[b, j])`：第 `b` 行沿列维的最大值（标量，逐行）。
- `e[b, i] = exp(x[b, i] - m[b])`：减最大值后的指数（数值稳定，最大指数为 `exp(0)=1`，不会上溢）。
- `s[b] = sum_j(e[b, j])`：第 `b` 行的指数和（标量，逐行，`s[b] >= 1` 恒成立）。
- `y[b, i] = e[b, i] / s[b]`：归一化输出，`y[b, i] ∈ [0, 1]`，`sum_i y[b, i] = 1`。

依据：用户明确给出的公式（`✓ 高`）。max-subtract 是标准数值稳定做法，保证 `exp` 的输入 `<= 0`，避免 fp16 上溢（fp16 最大有限值 65504，`exp` 上溢阈约 11.09）。

### 3. 算法描述

逐行（每个 `b` 独立）执行以下三步，行间无依赖：

1. **行内 max**：`m[b] = max_{j=0..N-1} x[b, j]`。
2. **行内 sum of exp**：`s[b] = sum_{j=0..N-1} exp(x[b, j] - m[b])`。
3. **逐元素除**：`y[b, i] = exp(x[b, i] - m[b]) / s[b]`，对所有 `i ∈ [0, N)`。

注：步骤 2 和 3 都需要 `exp(x - m)`，实现上可复用同一中间量；是否分两趟（先 sum 再 div）或单趟融合由 Stage 3 设计，本 SPEC 不规定。

### 4. 数据流说明

单输入 `x[B, N]` → 单输出 `y[B, N]`，无辅助输入、无模型参数、无 bias。`m[b]` 与 `s[b]` 是 kernel 内部逐行 reduction 中间标量，不是公开输入/输出。`B` 与 `N` 均动态；行与行之间完全独立，可跨核并行；列维 `N` 是 reduction 轴。tile/分核/同步由 Stage 3 设计。

### 5. 接口语义

公开签名：`softmax(x: Tensor[B, N], float16) -> y: Tensor[B, N], float16`。

- 输入 `x`：2D，shape `[B, N]`，dtype `float16`，value_range `[-65504, 65504]`（覆盖全部有限 fp16 值，任何有限 fp16 logit 均合法）。
- 输出 `y`：2D，shape `[B, N]`（与 `x` 相同），dtype `float16`，value_range `[0, 1]`（softmax 输出为概率，每元素 ∈ [0,1]，每行和为 1）。
- 无可选参数（无 `axis`/`dim` 参数：本 SPEC 固定沿最后一维；无 `dtype` 参数：输入输出 dtype 一致均为 float16）。
- 输出是新分配的 tensor，不要求 in-place。

### 6. 功能与可选参数依据

- **axis/dim**：不作为公开参数。用户需求明确"对最后一个维度"，固定 `axis=-1`。若后续需支持任意轴，应另起 class/SPEC。优先级 P0（本版本固定 axis=-1）。
- **数值稳定（max-subtract）**：必须，P0。来源用户公式（`✓ 高`）。
- **累加精度提升**：实现细节（建议 fp32 累加 max/sum 以保证 fp16 数值精度），属 Stage 3 设计，本 SPEC 仅在 kernel 契约补充中标注语义要求。
- **mask / 量化 / 随机性 / 稀疏性**：不涉及。
- 无 optional 参数需做四项语义分析。

### 7. 精度语义

- 验收容差：`atol=1e-3, rtol=1e-3`（依据：官方样例 `pro_ops/vf_api/test_softmax_tile_group_vf.py` 的 `torch.testing.assert_close(..., rtol=1e-3, atol=1e-3)`，`✓ 高`）。
- 比较方式：NPU 输出与 CPU golden 均转为 float32 后逐元素比较（与官方样例一致）。
- 输出 dtype 为 float16，但 max/sum 累加建议在 float32 进行以控制误差（实现由 Stage 3 决定）。
- max-subtract 保证 `exp` 输入 `<= 0`，无上溢风险；极大负差值使 `exp` 下溢为 0（fp16 下溢阈约 -16.6），数学上对应概率为 0，语义正确。

### 8. 动态 Shape 与约束依据

- `B`（batch 维）：动态，范围 `[1, 1024]`，作为输入第 0 维独立出现。
- `N`（reduction 维）：动态，范围 `[1, 8192]`，作为输入第 1 维独立出现。
- 输出 shape 严格等于输入 shape `[B, N]`。
- P0 case 的具体 `(B, N)` 均落在上述范围内：`(1,128)`、`(4,2048)`、`(32,4096)`。
- `N >= 1` 保证每行 max/sum 良定义（非空 reduction）；`B >= 1` 保证至少一行。

### 9. 边界条件处理

- **N=1**：每行单元素，`m=x, exp(0)=1, s=1, y=1`。输出恒为 1。
- **极大正值**：如 `x=65504`，max-subtract 后 `exp(0)=1`，不上溢。
- **极小负值 / 大差值**：`exp(x-m)` 下溢为 0，对应概率 0，语义正确，不产生 NaN。
- **整行同值**：`m=x`，`exp(0)=1`，`s=N`，`y=1/N`。
- **NaN/Inf 输入**：fp16 可表示 Inf/NaN。若输入含 NaN，输出对应行为 NaN（标准浮点传播）；若输入含 +Inf，`max=Inf`，`Inf-Inf=NaN` → 输出 NaN。本 SPEC 的 value_range 为有限区间 `[-65504, 65504]`，**合法输入不含 Inf/NaN**；含 Inf/NaN 的行为不作为验收范围（golden 不构造此类输入）。
- **空维度**：`B>=1, N>=1`，不允许空维度。

### 10. P0 与性能目标依据

P0 case（用户明确指定，`✓ 高`）：

| name | B | N | 说明 |
|------|---|---|------|
| `p0_batch1_n128` | 1 | 128 | 单行小 N，验证最小配置与单核/单 tile 路径 |
| `p0_batch4_n2048` | 4 | 2048 | 中等 batch、中大 N，验证多 register reduction |
| `p0_batch32_n4096` | 32 | 4096 | 大 batch、大 N，验证多核扩展与多 register |

性能目标：**用户未给出可复算数值目标**（`✓ 高`，用户明确未指定）。`perf_target` 为 `null`。Stage 5 将使用默认 Golden 参考目标：每个 P0 case 的 `golden_reference_ratio = golden_per_iteration_npu_e2e_us / final_pypto_target_kernel_us >= 1.0`。此为系统默认值，非用户要求。

### 11. 参考与来源

- 用户陈述的算子名、公式、输入输出 dtype/shape、P0 case：`✓ 高`（用户直接给出）。
- 数值稳定 max-subtract 做法：`✓ 高`（用户公式含 `max_j`）。
- 验收容差 `atol=1e-3, rtol=1e-3`：`✓ 高`（官方样例 `pro_ops/vf_api/test_softmax_tile_group_vf.py` line 173）。
- 输入 value_range `[-65504, 65504]`：`✓ 高`（float16 有限值全区间，IEEE 754 半精度规范）。
- 输出 value_range `[0, 1]`：`✓ 高`（softmax 概率语义）。

### 12. 自动决策

- **目标设备**：运行时未显式指定 target，按 PyPTO-Pro 默认假设 **A5**（SoC 950）。依据：官方 softmax 样例 `@pytest.mark.soc("950")`，且可用设备 TILE_FWK_DEVICE_ID=1 为 A5 类卡。影响：KB 路由触发 `constraints/arch-a5.md`。此项为默认假设，非用户确认。
- **累加精度**：SPEC 不强制 fp32 累加，仅在 kernel 契约补充中标注"为达成 fp16 容差建议高精度累加"，具体由 Stage 3 设计。
- **无人值守模式**：用户提供了完整需求（算子名、公式、dtype、shape、P0 case），按无人值守模式直接冻结 SPEC，非阻塞字段（目标设备默认 A5）采用可追溯默认并在此披露。

## kernel 契约补充

> 本节记录后续阶段必须知道、但通用需求模板不表达的边界。topology/tile/同步字段标记"由 Stage 3 设计"，Stage 1 不猜测。

| 字段 | Stage 1 裁定 |
|------|--------------|
| 辅助张量语义 | softmax 无公开辅助输入、无模型参数。`m[b]`（行 max）与 `s[b]`（行 sum-of-exp）是 kernel 内部逐行 reduction 产生的标量中间量，不是公开输入/输出；实现位置与是否物化到 UB 交 Stage 3。 |
| cast 边界链 | 输入 `float16` → 行 max/sum 累加建议在 `float32` 中进行（数值稳定，保证 fp16 容差）→ `exp(x-m)` 可在 fp32 或 fp16 计算（由 Stage 3 据精度/性能裁定）→ 输出写回 `float16`。语义 dtype 链：`fp16(in) → fp32(acc) → fp16(out)`，其中累加段为强建议（非强制），其余 cast 交 Stage 3。 |
| 累加/写回语义 | 行级 reduction，每个 `b` 独立。`m[b]` 为行内 max（覆盖式取大），`s[b]` 为行内 sum-of-exp（累加）。输出 `y` 为**覆盖写**到新分配的输出 tensor（非 in-place，非跨块累加，非原子累加）。行间无数据依赖，可跨核并行。 |
| 目标设备 | 默认 **A5**（SoC 950）。运行时/build 未显式指定 target，按 Pro 默认假设；SPEC/MEMORY 已标注此为默认假设。KB 路由据此触发 `constraints/arch-a5.md`。 |
| topology / tile / 同步 | **由 Stage 3 设计**。Stage 1 仅确认计算拓扑为"逐行 reduction（max + sum）+ elementwise（exp/div）"，行间独立可并行；不规定 tile 形状、UB 布局、分核策略、同步方式或尾块处理。 |

---
*生成时间: 2026-08-31*
*确认状态: 已确认（无人值守模式：用户提供完整需求，非阻塞字段默认 A5 已披露）*
