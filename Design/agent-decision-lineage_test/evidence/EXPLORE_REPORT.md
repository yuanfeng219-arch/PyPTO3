---
schema_version: 2
op_name: softmax
feasibility: 可行
---

# PyPTO-Pro 资料探索报告

> **生成时间**: 2026-08-31
> **全量资料索引**: `custom/softmax/PRO_MATERIAL_INDEX.md`

---

## 1. 概述

### 1.1 输入摘要

- 算子：`softmax`，对 2D `float16` 输入 `[B, N]` 沿最后一维（axis=-1）做数值稳定 softmax，输出同 shape 同 dtype `float16`。
- SPEC：`custom/softmax/SPEC.md`（validate_spec.py PASS）。公式：`y[b,i]=exp(x[b,i]-m[b])/s[b]`，`m[b]=max_j(x[b,j])`，`s[b]=sum_j(exp(x[b,j]-m[b]))`。
- P0 case：`[1,128]`、`[4,2048]`、`[32,4096]`。容差 `atol=1e-3, rtol=1e-3`（依据官方 softmax 样例 line 173）。
- 目标设备：A5（SoC 950），默认假设。`TILE_FWK_DEVICE_ID=1`。

### 1.2 算子分类

- **计算引擎类型**：Vector（VF 手写）。公式仅含逐元素运算与行内归约（max / sum-of-exp），无矩阵乘（Cube），无跨 engine 融合。
- **判断依据**：公式为 `max → exp_sub → sum → div` 的行内 reduction + elementwise 链，完全可用 VF 指令表达；官方 softmax 样例（`pro_ops/vf_api/test_softmax_tile_group_vf.py`）即用 `@pl.vector_function` + VF API 实现（fp32 版本）。

### 1.3 可能会涉及的 API 类别

> Vector 规则见 `references/performance-constraints.md`（强制 2：Vector 数值计算用 VF 手写）。本阶段记录默认 VF 映射，KB 模板例外交 Stage 3 裁定。

| 类别 | 是否涉及 | 关键 API | 指定算子是否覆盖 |
|------|---------|----------|------------------|
| 数据搬运 | 是 | `pl.load` / `pl.store` / `vf.load_align` / `vf.store_align` | 覆盖（softmax 样例 + add 样例 + load_align/store_align 文档） |
| VF 实现 | 是 | `vf.reduce_max` / `vf.reduce_sum` / `vf.exp_sub` / `vf.exp` / `vf.div` / `vf.max` / `vf.add` / `vf.full` / `vf.astype` | 覆盖（softmax 样例 + 各 API 文档） |
| 矩阵计算（Cube） | 否 | — | 不适用（无 matmul） |
| 系统访问 | 是 | `pl.get_block_idx` / `pl.get_block_num` | 覆盖（softmax 样例 multicore striding） |
| 控制流 | 是 | `pl.section_vector` / `pl.range` | 覆盖（softmax 样例） |
| 工具 | 是 | 动态维度 `pl.DYNAMIC` / `pl.set_validshape` / `pl.make_tile_group` / `pl.TileType` / `pl.MemorySpace.Vec` | 覆盖（softmax 样例 + tail_block 教程） |

---

## 2. 公式分解

按 SPEC 算法逐行（每个 `b` 独立）分解为原子步骤。列维 `N` 是 reduction 轴，可能跨多个 VF 寄存器（fp16 时每寄存器 128 元素，fp32 时 64 元素）。

| 步骤 | 操作类型 | 数学表达 | vf.* 调用链（vec）/ pl.* 调用链（cube） | 说明 |
|------|----------|----------|--------------------------------------|------|
| 1 | 行内 reduction（max） | `m[b] = max_{j} x[b,j]` | `vf.reduce_max(reg, mreg)` 跨 n_regs 寄存器，`vf.max` 合并入 lane0 累加器，`vf.full(row_max, preg)` 广播 | 需初始化为该 dtype 最小值（fp16/ fp32 -inf 等价）；尾寄存器用 `vf.update_mask` 限定有效 lane |
| 2 | elementwise（exp_sub） | `e[b,i] = exp(x[b,i] - m[b])` | `vf.exp_sub(reg, row_max_b, mreg)` | 数值稳定：`x-m <= 0`，`exp` 不上溢。fp16 输入可经 `dtype=pl.DT_FP32` 在 fp32 计算 exp（见 §3.3） |
| 3 | 行内 reduction（sum） | `s[b] = sum_{j} e[b,j]` | `vf.reduce_sum(e_reg, mreg)` 跨 n_regs，`vf.add` 合并入 lane0，`vf.full(row_sum, preg)` 广播 | **关键精度约束**：DT_FP16 源在 fp16 累加（见 §3.3），需 Stage 3 裁定是否升 fp32 累加 |
| 4 | elementwise（div）+ 写回 | `y[b,i] = e[b,i] / s[b]` | `vf.div(e_reg, row_sum_b, mreg)` → `vf.store_align(out_tile, out, mreg)` | 输出写回 float16；若 e/s 在 fp32，需 `vf.astype` fp32→fp16 后 store |

---

## 3. API 文档探索

> **来源**: 探索方向 1 — 基于 `PRO_MATERIAL_INDEX.md` §A

### 3.1 API 映射结果

| 步骤 | 数学表达 | vf.* 调用链（vec）/ pl.* 调用链（cube） | 状态 | 约束满足 |
|------|----------|--------------------------------------|------|----------|
| 1 | `max_j x[b,j]` | `vf.reduce_max` → `vf.max`（合并） → `vf.full`（广播 lane0） | 直接可用 | ✓（DT_FP16/DT_FP32 均支持） |
| 2 | `exp(x-m)` | `vf.exp_sub(reg, max_b, mreg)` | 直接可用 | ⚠（fp16→fp32 高精度路径有 layout/半数元素约束，见 §3.3） |
| 3 | `sum_j e[b,j]` | `vf.reduce_sum` → `vf.add`（合并） → `vf.full`（广播） | 需组合 | ⚠（fp16 源在 fp16 精度累加，见 §3.3） |
| 4 | `e / s` | `vf.div` → `vf.store_align` | 直接可用 | ✓ |
| 搬运 | GM↔UB | `pl.load` / `pl.store`（tile 级） + `vf.load_align` / `vf.store_align`（寄存器级） | 直接可用 | ✓（32B 对齐） |
| 尾块 | 部分 lane / 部分 row | `vf.update_mask`（尾寄存器） + `pl.set_validshape`（尾 row-tile） | 直接可用 | ✓ |
| 多核 | 行 tile 跨核 | `pl.get_block_idx` / `pl.get_block_num` + `pl.range(core_id, n_tiles, num_cores)` | 直接可用 | ✓ |

### 3.2 替代方案

| 步骤 | 默认 VF 映射 | API 依据 | 目标版本 | 适用条件 | 交接状态 |
|------|---------------|----------|----------|----------|----------|
| 1 max | `vf.reduce_max` | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/reduce_max.md`（支持 DT_FP16/DT_FP32，结果在 dst[0]） | 950/A5 | reg_tensor，mask 空→dtype 最小值 | 待 Stage 3 核对已选 KB 模板 |
| 2 exp | `vf.exp_sub`（默认）/ `vf.exp`（无 max 时，本算子不用） | `.../composite_computation/exp_sub.md`（fp16→fp32 dtype 路径）/ `.../basic_arithmetic/exp.md` | 950/A5 | fp16 输入；max_subtract 保证 `<=0` 不上溢 | 待 Stage 3 裁定 fp16 直算 vs 升 fp32 |
| 3 sum | `vf.reduce_sum` | `.../reduction/reduce_sum.md`（fp16 源累加精度=fp16；fp32 源累加精度=fp32） | 950/A5 | reg_tensor，二叉树累加，确定性 | 待 Stage 3 裁定累加 dtype |
| 4 div | `vf.div` | `.../basic_arithmetic/div.md`（DT_FP16/DT_FP32） | 950/A5 | src0/src1/dst 同 dtype | 待 Stage 3 核对 |
| 广播 | `vf.full` | `.../data_movement/full.md`（Scalar 模式初始化 / Tensor 模式广播 lane0） | 950/A5 | scalar 模式可无 mask；tensor 模式需 mask | 待 Stage 3 核对 |
| 类型转换 | `vf.astype` | `.../type_conversion/astype.md`（DT_FP16↔DT_FP32 2x，layout ZERO/ONE） | 950/A5 | 升精度 / 降精度写回 | 待 Stage 3 裁定 cast 位置 |

### 3.3 API 约束

| API | 约束项 | 要求 | 结果 | 参数语义/寄存器级行为/同名差异 |
|-----|--------|------|------|----------------------|
| `vf.reduce_sum` | 累加精度 | DT_FP16 源在 **DT_FP16 精度**累加；DT_FP32 源在 DT_FP32 精度累加 | ⚠ | 对 fp16 softmax 的 sum 步骤，若直接对 fp16 exp 值 reduce_sum，累加在 fp16 进行，大 N（如 4096）时累加误差可能超 1e-3 容差。Stage 3 须裁定：将 exp 结果升 fp32 后再 reduce_sum，或整段在 fp32 计算。这是精度 vs UB/性能的权衡，非 API 不支持 |
| `vf.exp_sub` | fp16→fp32 高精度路径 | src0/src1=DT_FP16, dst=DT_FP32（`dtype=pl.DT_FP32`）；每调用只处理 128 个 fp16 中的 **64 个**（`layout=ZERO` 取偶数位 / `ONE` 取奇数位） | ⚠ | 即 exp 在 fp32 内部计算（`cast_f16_to_f32` 后 `exp`），精度高；但需 ZERO + ONE 两次调用覆盖 128 元素，exp 工作量翻倍。fp32→fp32 路径无此限制（一次 64 元素） |
| `vf.reduce_max` | mask 空时返回值 | mask 全空→该 dtype 最小值写入 dst[0] | ✓ | 初始化行 max 累加器应使用 `vf.full(NEG_INF, preg, dtype=...)`；fp16 最小有限值 -65504，但 softmax 行非空（N>=1），不会触发 |
| `vf.astype` | fp16↔fp32 位宽比 | 2x 扩展/缩窄；layout ZERO/ONE 决定半区；每调用处理半数元素 | ⚠ | fp16→fp32：128 fp16 → 64 fp32（一次）；fp32→fp16：64 fp32 → 128 fp16 槽位的一半。覆盖全量需 ZERO+ONE 两次 |
| `vf.load_align`/`vf.store_align` | 地址对齐 | Tile 地址 32 字节对齐 | ✓ | fp16 寄存器 128 元素（256B），fp32 寄存器 64 元素（256B）；寄存器宽度 VL 按 dtype 变化 |
| `vf.create_mask` | mask 粒度 | dtype 决定每元素掩码位数：fp16=128 元素×2bit，fp32=64 元素×4bit；总位宽 256bit | ✓ | 同一 mask_reg 不能跨 dtype 混用；fp16 计算与 fp32 计算需各自 create_mask |
| `vf.update_mask` | 尾寄存器掩码 | 标量值表示有效元素数，自动 `scalar = (scalar<VL)?0:(scalar-VL)` | ✓ | softmax 样例用 `valid = pl.min(LANES, n_cols - r*LANES)` + `vf.update_mask(valid, dtype=...)` 限定尾寄存器有效 lane |
| `pl.make_tile_group` | mutex_id 范围 | `[0, 31]`；每块 Tile ID 数一致，同 Tile 内 ID 不重复 | ✓ | N-buffer 用 `mutex_ids=[0,1,...]`；`auto_mutex=True` 自动插入同步 |
| `pl.set_validshape` | 调用顺序 | 必须在 `load` 之前调用以约束 GM 搬入 | ✓ | 尾 row-tile 用 `[valid_rows, cols]` 限定；softmax 样例对每个 tile `set_validshape` 后再 `load` |
| `@pl.jit(auto_mutex=True)` | 自动同步 | 依 tile group 的 mutex 元数据自动插入互斥 | ✓ | softmax 样例启用 auto_mutex，无需手写 event 同步 |

### 3.4 MemorySpace 约束

| 操作 | MemorySpace | 约束 |
|------|-------------|------|
| 输入/输出 Tile（UB） | `pl.MemorySpace.Vec` | softmax 全程在 UB（Vec）完成，无 L1/L0；与 softmax 样例一致 |
| GM Tensor | `pl.MemorySpace.DDR` | 输入 `x` / 输出 `y` 在 GM，经 `pl.load`/`pl.store` 与 UB tile 交换 |
| 中间 exp / sum | UB（reg_tensor） | `m[b]`/`s[b]` 为 reg_tensor 内 lane0 标量，经 `vf.full` 广播；不物化到独立 GM tensor |

---

## 4. 算子样例探索

> **来源**: 探索方向 2 — 基于 `PRO_MATERIAL_INDEX.md` §B（官方指定算子）
> **注意**: 官方指定算子为精选实现参考。pro_ops/ 下清单外文件不得参考。

### 4.1 全量样例参考（按 cube/vec 组成分类）

**纯 Cube 样例**（matmul 类，与 softmax 不相关，仅提取通用写法）：

| # | 示例路径 | 可复用点 |
|---|----------|----------|
| 1 | `pro_ops/matmul/test_matmul_8K_example.py` | matmul tile 管理 / K 累加链（与 softmax 无直接关系） |
| 2-6 | `pro_ops/matmul/test_matmul_perf_asw_*.py` | L1/L0 布局、set_mm_layout_transform（与 softmax 无直接关系） |

**纯 Vec 样例**（elementwise / vf_api 类，与 softmax 直接相关）：

| # | 示例路径 | 可复用点 |
|---|----------|----------|
| 12 | `pro_ops/vf_api/test_layernorm_tile_group_vf.py` | 行内 reduction（mean/var）VF 模式、update_mask 尾 lane、make_tile_group 双缓冲、auto_mutex、multicore striding |
| 13 | `pro_ops/vf_api/test_softmax_tile_group_vf.py` | **直接对应**：3-pass（max/sum/div）VF softmax、exp_sub、reduce_max/reduce_sum、update_mask 尾 lane、set_validshape 尾 row-tile、multicore、双缓冲 in/out tile group |
| 1 | `pro_ops/element_wise/test_add.py` | elementwise 双缓冲、section_vector、auto_mutex 通用写法 |

**VC 融合样例**（FA / lightning_indexer 类，与 softmax 不相关）：

| # | 示例路径 | 可复用点 |
|---|----------|----------|
| 7-11 | `pro_ops/fa/*.py`、`pro_ops/lightning_indexer/*.py` | section_cube→acc_to_vec→section_vector 衔接、cross_core 流水（softmax 无 Cube，不适用） |

### 4.2 可复用模式

**直接可复用**（来自 softmax 官方样例 `pro_ops/vf_api/test_softmax_tile_group_vf.py`）：

- **API 调用链**：`vf.create_mask(ALL)` → `vf.full(NEG_INF,...)` 初始化 max → 循环 `vf.load_align` / `vf.reduce_max` / `vf.max` 合并 / `vf.full` 广播 → `vf.exp_sub` / `vf.reduce_sum` / `vf.add` 合并 / `vf.full` 广播 → `vf.div` / `vf.store_align`（来源：`pro_ops/vf_api/test_softmax_tile_group_vf.py` line 76-110）
- **Tile 配置**：`TileType(shape=[TILE_ROWS, MAX_N], dtype=DT_FP32, target_memory=Vec, valid_shape=[-1,-1])`；`MAX_N` 为编译期 LANES 整数倍；`make_tile_group(addrs=[VA_IN0,VA_IN1], mutex_ids=[0,1])` 双缓冲 in/out（来源：同上 line 53-64, 120-124）
- **同步策略**：`@pl.jit(auto_mutex=True)`，全程 `section_vector()`，无手写 event 同步（来源：同上 line 113, 126）
- **循环结构**：外层 `for tile_id in pl.range(core_id, num_tiles, num_cores)` 跨核 striding；内层逐行 `for m in pl.range(0, n_rows)`；逐寄存器 `for r in pl.range(0, n_regs)` 三趟（max/sum/div）（来源：同上 line 79-110, 135-147）
- **尾块处理**：行尾 `valid_rows = pl.min(TILE_ROWS, rows - row_off)` + `pl.set_validshape(in_slot, [valid_rows, cols])`；寄存器尾 `valid = pl.min(LANES, n_cols - r*LANES)` + `mreg = vf.update_mask(valid, dtype=...)`（来源：同上 line 85-86, 137, 140）

**通用写法参考**（来自所有样例）：

- **API 用法**：`vf.load_align(tile, offset)` / `vf.store_align(tile, reg, preg, offset)` 的元素偏移语义；`vf.full(scalar, preg, dtype=)` 标量广播
- **分核策略**：`pl.get_block_idx()` striding by `pl.get_block_num()`，`num_cores = min(32, num_tiles)` 限核（来源：softmax 样例 line 164-165）
- **尾块处理**：`set_validshape` 必须在 `load` 之前；reduction 前尾 lane 用 `update_mask` 或 `fillpad(min)`（来源：tail_block 教程 line 80, 122）

### 4.3 差异分析

| 差异点 | 示例做法 | 本算子需求 | 调整建议 |
|--------|----------|------------|----------|
| **dtype** | softmax 样例为 **fp32**（输入/累加/输出全 fp32） | 本算子输入/输出 **fp16**，累加建议 fp32 | Stage 3 须设计 fp16↔fp32 cast 链：输入 fp16 load 后升 fp32 计算，或 exp_sub fp16→fp32；输出前降 fp16。详见 §3.3 精度约束 |
| **VF 寄存器宽度** | fp32 → LANES=64，MAX_N=512（8 寄存器/行） | fp16 直算时 LANES=128；若升 fp32 计算则 LANES=64 | MAX_N 须为所用 dtype LANES 的整数倍；fp16 直算路径 MAX_N 取 128 倍数，fp32 路径取 64 倍数 |
| **reduce_sum 精度** | fp32 累加（无精度问题） | fp16 源在 fp16 累加（可能超容差） | Stage 3 须将 sum 步骤升 fp32（exp 结果 cast fp32 后 reduce_sum，或整段 fp32） |
| **MAX_N 上限** | 样例 MAX_N=512，`assert cols <= MAX_N` | P0 最大 N=4096，远超 512 | Stage 3 须支持 N 跨多趟寄存器循环（样例已用 `n_regs = ceil(N/LANES)` 循环，仅需放大 MAX_N 或分多趟）；UB 容量 248KB 限制 MAX_N 上限 |

### 4.4 高参考价值样例推荐

| 样例路径 | 推荐理由 |
|----------|----------|
| `pro_ops/vf_api/test_softmax_tile_group_vf.py` | 与本算子数学完全一致（仅 dtype 差异），3-pass 结构、尾块处理、multicore、双缓冲全部直接可复用 |
| `pro_ops/vf_api/test_layernorm_tile_group_vf.py` | 同为行内 reduction（mean/var/sum）VF 算子，cross-reference 行内多寄存器归约 + update_mask 尾 lane 模式 |
| `pro_ops/element_wise/test_add.py` | elementwise 双缓冲 + section_vector + auto_mutex 通用骨架 |

---

## 5. 教程与设计指南探索

> **来源**: 探索方向 3 — 基于 `PRO_MATERIAL_INDEX.md` §C（40 个教程文档）

### 5.1 适用的设计模式

| 指南/教程文档（索引 §C） | 来源目录 | 设计模式 | 适用性 |
|---------------------------|----------|----------|--------|
| `tutorials/operator_development/tile_based_python_programming/tail_block_handling.md` | tutorials | `valid_shape=[-1,-1]` + `set_validshape` 尾块；reduction 前 `pad=min`/`fillpad` 或 `update_mask` 尾 lane | **高度适用**：softmax 行 max 归约尾块需 min 填充或 update_mask；教程明确列 "softmax 前的最大值归约 → pl.TilePad.min" |
| `tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md` | tutorials | `get_block_idx`/`get_block_num` 跨步切分；`block_dim=min(vector_core_num, total_tiles)`；UB 248KB 预算算例 | **高度适用**：softmax 行间独立，按行 tile 跨核 striding；UB 预算算例指导 MAX_N/TILE_ROWS 选择 |
| `tutorials/operator_development/tile_based_python_programming/Python_programming_overview.md` | tutorials | Tensor/Tile/TileGroup 基本模型 | 适用：理解 make_tile_group 双缓冲 |
| `tutorials/operator_development/tile_based_python_programming/Tile_vector_computation.md` | tutorials | Tile 级向量计算（pl.add/pl.relu 等 tile-op） | 部分适用：softmax 用 VF 寄存器级（vf.*）而非 tile-op，但 tile 级 load/store 通用 |
| `tutorials/operator_development/tile_based_python_programming/Reg_vector_computation.md` | tutorials | VF 寄存器级计算（vf.* / @pl.vector_function） | **高度适用**：softmax 核心即 VF 寄存器级 reduction + elementwise |
| `tutorials/operator_development/compilation_and_execution/JIT_compilation.md` | tutorials | `@pl.jit` 编译与启动 `kernel[None, block_dim]` | 适用：启动语法 |
| `tutorials/operator_development/tile_based_python_programming/tiling_key.md` | tutorials | TilingKey 编译期特化 | 不适用（softmax 无多模式分支） |
| `tutorials/operator_development/tile_based_python_programming/Cube_matrix_computation.md` | tutorials | Cube matmul | 不适用（softmax 无 Cube） |
| `tutorials/quick_start/SIMD/Add_operator.md` | tutorials | elementwise 入门 | 部分适用：双缓冲骨架 |
| `tutorials/debugging_and_optimization/functional_debugging.md` | tutorials | 功能调试 | 适用：Stage 4 调试参考 |
| `tutorials/debugging_and_optimization/performance_optimization.md` | tutorials | 性能优化 | 适用：Stage 5 参考 |
| `tutorials/programming_paradigm/abstract_hardware_architecture.md` | tutorials | A5 硬件架构 | 适用：理解 UB/VF 寄存器/核结构 |
| `tutorials/advanced_programming/*.md` | tutorials | superkernel / AOT / aclnn | 不适用（本流程为 JIT 单算子） |
| `tutorials/operator_development/tile_based_python_programming/TilingData.md` | tutorials | TilingData 传参 | 不适用（softmax 用动态 shape `pl.DYNAMIC`，无需 TilingData） |
| `tutorials/quick_start/SIMT/*.md`、`tutorials/programming_paradigm/simt_programming.md` | tutorials | SIMT 编程 | 不适用（softmax 用 SIMD/VF） |
| `tutorials/introduction.md`、`tutorials/quick_start/index.md`、`tutorials/operator_development/index.md` 等索引页 | tutorials | 总览 | 不适用（导航页） |

> 其余 §C 文档（如 `hardware_implementation.md`、`operator_graph_integration_development.md`、`ai_framework_operator_adaptation.md` 等）经评估与当前算子无直接设计模式关联，标记"不适用"以证遍历完整。

### 5.2 来自教程的关键约束与建议

| 来源 | 约束/建议 | 影响 |
|------|----------|------|
| `tail_block_handling.md` | `set_validshape` 必须在 `load` 之前调用；reduction 前尾块须 `pad=min`+`fillpad` 或 `update_mask` 限定 | Stage 3 须在 load 前 set_validshape；行 max 尾 lane 用 update_mask（样例做法）或 fillpad(min) |
| `tail_block_handling.md` | 测试须含两维均小于物理 Tile 的 shape，或两维均不整除 Tile 的大 shape | Stage 4 正确性 case 须覆盖尾块（SPEC 已含 [1,128] 等小 shape，建议补非整除 case） |
| `multi_core_partitioning_and_Tiling.md` | UB 容量 248KB（950PR/950DT）；并存 Tile 总字节 ≤ 248KB；`TILE_N` 取 Vector 对齐粒度整数倍（fp16 取 128、fp32 取 64） | Stage 3 MAX_N/TILE_ROWS/缓冲深度的 UB 预算约束 |
| `multi_core_partitioning_and_Tiling.md` | `block_dim = min(vector_core_num, total_tiles)`；跨步 `pl.range(core_id, total, num_cores)` 负载均衡 | Stage 3 多核切分策略 |
| `multi_core_partitioning_and_Tiling.md` | JIT 启动不截断超上限 block_dim；Host 须算合法值 | Stage 4 host 启动 `kernel[None, num_cores]` 须限核 |

---

## 6. Stage 3 设计事实输入

> **综合来源**: §3 API 约束 + §4 样例参考 + §5 教程指导。本节是 Stage 3 的消费入口，只汇总事实与待裁定项，不冻结 tile、Module、同步事件或 topology；这些由 Stage 3 决定。

### 6.1 Tile / 同步约束证据

| 设计问题 | 已证实约束 | 证据路径 | Stage 3 待裁定项 |
|----------|------------|----------|------------------|
| fp16 累加精度 | `vf.reduce_sum` 对 DT_FP16 源在 fp16 精度累加；fp32 源在 fp32 累加 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/reduce_sum.md`（约束说明表1） | 是否将 sum（及 max/exp）升 fp32 计算：方案A 整段 fp32（load 后 astype fp16→fp32，输出 astype fp32→fp16）；方案B 仅 exp/sum 升 fp32（exp_sub fp16→fp32，reduce_sum fp32，div fp32，输出 astype fp32→fp16）；方案C 全 fp16 直算（精度风险，需验证） |
| exp_sub fp16→fp32 半数元素 | `vf.exp_sub` fp16→fp32 路径每调用处理 64/128 元素（layout ZERO/ONE） | `.../composite_computation/exp_sub.md`（约束说明 + DT_FP16源→DT_FP32结果节） | 若选方案B，每寄存器需 ZERO+ONE 两次 exp_sub 覆盖 128 元素；exp 工作量翻倍 |
| astype fp16↔fp32 半数元素 | `vf.astype` 2x 转换每调用处理半数元素（layout ZERO/ONE） | `.../type_conversion/astype.md`（表3 浮点转浮点） | 若选方案A，load 后升 fp32 需 ZERO+ONE 两次 astype 覆盖 128 fp16→64+64 fp32；UB 翻倍 |
| VF 寄存器宽度 | fp16=128 元素/寄存器，fp32=64 元素/寄存器；总位宽 256bit | `.../mask_operations/create_mask.md`（dtype 表） | MAX_N 须为所用 dtype LANES 整数倍；n_regs=ceil(N/LANES) 循环覆盖大 N |
| UB 容量 | 248KB（950PR/950DT）；并存 Tile 总字节 ≤ 248KB | `tutorials/.../multi_core_partitioning_and_Tiling.md`（UB 切分节） | MAX_N × TILE_ROWS × dtype_bytes × 缓冲深度 ≤ 248KB；fp32 方案 UB 占用翻倍，可能需缩 TILE_ROWS/MAX_N 或减缓冲深度 |
| N=4096 超样例 MAX_N=512 | 样例 `assert cols <= MAX_N`，MAX_N=512 | `pro_ops/vf_api/test_softmax_tile_group_vf.py`（line 54, 153） | 须放大 MAX_N（受 UB 限）或分多趟寄存器循环（样例 n_regs 循环已支持，仅需 MAX_N >= 单趟上限并循环多趟） |
| 尾块处理 | `set_validshape` 须在 load 前；reduction 尾 lane 用 `update_mask` 或 `fillpad(min)` | `tutorials/.../tail_block_handling.md`；softmax 样例 line 85-86, 140 | 尾 row-tile 用 set_validshape；尾寄存器用 update_mask（样例做法） |
| 多核切分 | 行间独立，按行 tile 跨核 striding；`block_dim=min(vector_core_num, num_tiles)` | `tutorials/.../multi_core_partitioning_and_Tiling.md`；softmax 样例 line 135-147 | 行 tile 跨核分配；核数限核 |
| 同步 | `auto_mutex=True` + make_tile_group mutex_ids 自动同步 | softmax 样例 line 113；`.../resource_management/make_tile_group.md` | 全程 section_vector + auto_mutex，无手写 event |
| mask 粒度 | fp16/fp32 需各自 create_mask；不能跨 dtype 混用 | `.../mask_operations/create_mask.md` | 若方案A/B 混用 fp16/fp32，需分别 create_mask |

---

## 7. 环境常量快照

> 从当前仓库的 API 文档和官方指定算子中提取硬件/版本相关常量。每个值来自本次文档遍历或官方样例扫描。

| 常量 | 探测值 | 来源路径 | 备注 |
|------|--------|----------|------|
| UB 容量 | 248KB | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md`（UB 切分节） | Ascend 950PR/950DT（A5）；并存 Tile 总字节上限 |
| VF 寄存器宽度（fp32） | 64 元素/寄存器（LANES=64） | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/create_mask.md`（dtype 表：DT_FP32 → 64 元素×4bit） | softmax 样例 LANES=64（fp32） |
| VF 寄存器宽度（fp16） | 128 元素/寄存器 | `.../mask_operations/create_mask.md`（DT_FP16 → 128 元素×2bit） | fp16 直算路径 |
| mutex_id 取值范围 | [0, 31] | `.../resource_management/make_tile_group.md`（参数范围） | N-buffer mutex 分配 |
| cross_core event_id 范围 | 0~15 | `.../resource_management/make_tile_group.md`（fwd_ids/bwd_ids） | softmax 无 Cube，不使用 cross_core |
| 地址对齐 | 32 字节 | `.../vf_computation/data_movement/load_align.md`、`store_align.md` | Tile 地址 32B 对齐 |
| 目标 SoC | 950（A5） | `pro_ops/vf_api/test_softmax_tile_group_vf.py`（`@pytest.mark.soc("950")`）；所有 VF API 产品支持表均 "Ascend 950PR/950DT：支持" | A2/A3 不支持 VF API |
| block_dim 上限（仅 Vector） | `vector_core_num`（运行时查询 `get_platform_info().vector_core_num`） | `tutorials/.../multi_core_partitioning_and_Tiling.md` | Host 须限核；softmax 样例用 `min(32, num_tiles)` |
| Vector tile 对齐粒度 | fp16 取 128、fp32 取 64 | `tutorials/.../multi_core_partitioning_and_Tiling.md`（UB 切分节） | TILE_N 选型 |

---

## 8. 风险评估

### 8.1 阻断问题

| 问题 | 原因 | 建议 |
|------|------|------|
| 无阻断项 | 所有公式步骤均有可行 VF API 链；官方 softmax 样例直接对应（仅 dtype 差异，有 cast 路径） | 推进 Stage 3 |

### 8.2 注意事项

| 注意点 | 说明 |
|--------|------|
| **fp16 累加精度**（关键） | `vf.reduce_sum` 对 fp16 源在 fp16 精度累加；N=4096 时累加误差可能超 1e-3 容差。Stage 3 必须将 sum（建议连 exp/max）升 fp32 计算。这是本算子相对官方 fp32 样例的核心新增设计点 |
| **exp_sub / astype 半数元素** | fp16→fp32 路径每调用只处理 64/128 元素（layout ZERO/ONE），覆盖全量需两次调用，工作量翻倍。Stage 3 须在精度与性能/UB 间权衡 |
| **N=4096 超 MAX_N=512** | 官方样例 MAX_N=512 且 `assert cols<=MAX_N`；P0 最大 N=4096。Stage 3 须放大 MAX_N（受 248KB UB 限）或分多趟寄存器循环。样例的 `n_regs=ceil(N/LANES)` 循环结构已支持多寄存器，可扩展 |
| **UB 预算**（fp32 方案） | 若整段 fp32 计算，输入/输出/中间 tile 字节翻倍；248KB UB 下须缩 TILE_ROWS/MAX_N 或减缓冲深度。Stage 3 须做 UB 预算算例 |
| **matmul_8K_example 大小写不一致**（非阻断） | 官方清单 `pro_ops/matmul/test_matmul_8K_example.py`（大写 K）vs 缓存 `test_matmul_8k_example.py`（小写 k），文件实际存在，与 softmax 无关；记为缓存/清单大小写差异，不影响 softmax 开发。建议 orchestrator 后续修正清单大小写 |
| **api_list.md 缺失**（非阻断） | 缓存中未发现 `docs/pypto_api_list.md`；§A 已按实际子目录分组全量列出 312 个 API 文档作为替代索引来源 |
| **dtype 一致性** | fp16 与 fp32 计算需分别 `create_mask`；mask_reg 不跨 dtype 混用 |

---

## 9. 证据索引

> **全量资料索引**: `custom/softmax/PRO_MATERIAL_INDEX.md`

### 9.1 API 文档证据

| 信息 | 路径 |
|------|------|
| vf.reduce_max 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/reduce_max.md` |
| vf.reduce_sum 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/reduction/reduce_sum.md` |
| vf.exp_sub 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/composite_computation/exp_sub.md` |
| vf.exp 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/exp.md` |
| vf.div 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/div.md` |
| vf.max 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/max.md` |
| vf.add 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/basic_arithmetic/add.md` |
| vf.full 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/full.md` |
| vf.load_align 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/load_align.md` |
| vf.store_align 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/data_movement/store_align.md` |
| vf.create_mask 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/create_mask.md` |
| vf.update_mask 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/update_mask.md` |
| vf.astype 文档 | `docs/pypto_pro/api/SIMD-API/operation/vf_computation/type_conversion/astype.md` |
| pl.make_tile_group 文档 | `docs/pypto_pro/api/SIMD-API/operation/resource_management/make_tile_group.md` |
| pl.set_validshape 文档 | `docs/pypto_pro/api/SIMD-API/operation/memory_vector_computation/transpose_and_element_access/set_validshape.md` |
| pl.section_vector 文档 | `docs/pypto_pro/api/SIMD-API/operation/controlflow/section_vector_section_cube.md` |
| pl.TileType 文档 | `docs/pypto_pro/api/SIMD-API/basic_data_structures/TileType.md` |
| pl.MemorySpace 文档 | `docs/pypto_pro/api/SIMD-API/basic_data_structures/MemorySpace.md` |

### 9.2 算子样例证据

| 信息 | 路径 |
|------|------|
| softmax 官方样例（fp32，直接对应） | `pro_ops/vf_api/test_softmax_tile_group_vf.py` |
| layernorm 官方样例（行内 reduction cross-reference） | `pro_ops/vf_api/test_layernorm_tile_group_vf.py` |
| add 官方样例（elementwise 骨架） | `pro_ops/element_wise/test_add.py` |

### 9.3 指南与教程文档证据

| 信息 | 来源目录 | 路径 |
|------|----------|------|
| 尾块处理（含 softmax max 归约 min 填充建议） | tutorials | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/tail_block_handling.md` |
| 多核切分与 Tiling（UB 248KB、跨步切分） | tutorials | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md` |
| VF 寄存器级计算 | tutorials | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Reg_vector_computation.md` |
| Tile/TileGroup 概述 | tutorials | `docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/Python_programming_overview.md` |
| A5 硬件架构 | tutorials | `docs/pypto_pro/tutorials/programming_paradigm/abstract_hardware_architecture.md` |
| JIT 编译与启动 | tutorials | `docs/pypto_pro/tutorials/operator_development/compilation_and_execution/JIT_compilation.md` |

---

## 10. 结论

- **可行性**: **可行**。softmax 公式（max → exp_sub → sum → div）可由 VF API 完整表达，官方 softmax 样例（fp32）提供直接可复用的 3-pass 结构、尾块处理、multicore、双缓冲模式；fp16 输入输出有明确的 astype/exp_sub cast 路径。
- **主要问题**: 无阻断项。核心 Stage 3 设计点是 **fp16 累加精度**——`vf.reduce_sum` 对 fp16 源在 fp16 精度累加，N=4096 时可能超 1e-3 容差，须将 sum（建议连 exp/max）升 fp32 计算；以及 **exp_sub/astype fp16→fp32 半数元素**约束带来的工作量翻倍与 UB 预算权衡。其次是 **N=4096 超 MAX_N=512**，须放大 MAX_N 或多趟循环。这些均为 Stage 3 tile/dtype 设计决策，不影响 API 可行性结论。
- **替代路线**: 若 Stage 3 判定 fp16 直算精度不足，可走整段 fp32 计算（load 后 astype 升 fp32，输出降 fp16），代价是 UB 占用翻倍；或仅 exp/sum 升 fp32（exp_sub fp16→fp32 + reduce_sum fp32 + div fp32 + astype 输出）。三条路线均有 API 支持，Stage 3 据精度验证与 UB 预算裁定。
