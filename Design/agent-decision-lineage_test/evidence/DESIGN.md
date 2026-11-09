## 算子名称

- **op_name**: softmax
- **设计时间**: 2026-08-31
- **基于 SPEC**: `custom/softmax/SPEC.md`
- **基于 EXPLORE_REPORT**: `custom/softmax/EXPLORE_REPORT.md`

---

## Knowledge Bindings

> 完整、机器可读的 requirement 清单见 [`DESIGN_BINDINGS.json`](DESIGN_BINDINGS.json)。
> 本文只记录 R0–R8 的实际设计决策；每条活动 requirement 的 `planned_location` 必须准确指向
> 对应决策及最终 `test_softmax.py` file/symbol，不复制 Binding 分组或 requirement 表格。

---

## §0 Module 划分（R0 输出）

> 依据[Module划分](../references/module_partitioning.md)，先按数据依赖确定Module边界，再按Section边界继续拆分。R0记录每个Module的Section归属，具体Section代码结构在§4确定。

### 维度契约

- **kernel 接收与处理维度**：SPEC 输入 `x` 为 2D `[B, N]`（rank=2），dtype float16；输出 `y` 为 2D `[B, N]`，dtype float16。kernel 直接接收原始 2D `[B, N]`，不做 host 侧 reshape/permute。
- **kernel 内维度适配规则**：B 为行（batch）维，N 为列（reduction）维。kernel 内通过 `x.shape[0]`/`x.shape[1]` 读取运行时 B、N；行间独立、可跨核并行；列维 N 是归约轴，每行独立做 max→exp→sum→div。无需 stride/offset 重映射（输入即 2D RowMajor 连续）。

| SPEC 输入 shape | kernel 接收 | kernel 内索引/stride 映射 | 输出形态 |
|-----------------|-------------|---------------------------|----------|
| 2D `[B, N]` fp16 | 原始 `[B, N]` fp16 | 直接按 `[row_off, 0]` tile 偏移访问；行 m 的寄存器在 UB tile 内偏移 `m*MAX_N + r*LANES` | `[B, N]` fp16（与输入同 shape/dtype） |

### Wrapper 边界外操作

空

### 划分依据

依据 [module_partitioning.md > Softmax 示例](../references/module_partitioning.md) 的常规三遍划分：`m = max(x)` → `s = sum(exp(x-m))` → `y = exp(x-m)/s`。三遍都消费同一行块、且都属同一执行域（Vector/VF，无 Cube），无 Section 边界穿越；后两遍依赖前一遍的**最终**行级标量（行 max、行 sum），但本设计把整行放入单个 UB tile（MAX_N=8192 覆盖动态范围 N∈[1,8192]，见 §2/§3 UB 预算），因此三遍在**一次 `@pl.vector_function` 调用内**通过寄存器循环完成，`m`/`s` 为 VF 内寄存器标量、无需跨块持久化。数据依赖（后遍需前遍最终结果）在 VF 内由源码顺序保证，不引入 Section 边界或 cross-core 事件。故**单 Module**，无需按 Section 拆分。

### Module 列表

| Module | 目的 | 输入依赖 | 输出 | Section |
|-------|------|---------|------|---------|
| Module 1 | 数值稳定行 softmax 全流程：行 max → exp(x-max) → 行 sum → div → 写回 fp16 | `x`（primary，fp16 `[B,N]`） | `y`（fp16 `[B,N]`） | `section_vector` |

> 单 Module：本算子为纯 Vector（VF 手写），无 Cube 操作，三遍计算同属 Vector 执行域；整行装入单 UB tile 使 m/s 在一次 VF 调用内寄存器驻留，无跨块/跨 Section 数据传递，故不拆分。

### 归约轴容量结论

- **归约轴是否可能超单 tile**：**否**（本设计的 UB 规划下）。SPEC 动态范围 N∈[1,8192]；本设计 UB tile 取 `[TILE_ROWS=3, MAX_N=8192]` fp16，单槽 3×8192×2 = 49152 B = 48KB，双缓冲 in+out 共 4 槽 = 192KB ≤ 248KB（ub_size=253952，950PR_957x.ini，见 §3）。MAX_N=8192 ≥ 动态上界 8192，故**整行始终装入单个 UB tile**，归约在 tile 内跨寄存器完成（`n_regs = ceil(N/128)`），无需跨 tile 流式归约。
- **多Tile归约方案**：不涉及（单 tile 装得下整行）。`online-softmax-tail` 的流式 recurrence（跨块 m/l rescale）因此 `not_triggered`（见 DESIGN_BINDINGS.json 对应 requirement）。若未来需支持 N>8192，须回退本节重设为分块流式归约。

### Module 级数据流

```
GM x (fp16 [B,N])
  └─[pl.load, MTE2]→ in_slot (UB, fp16 [TILE_ROWS, MAX_N], double-buffer)
                       │  (整行块在 UB 内驻留，三遍寄存器循环在此 tile 上)
                       ▼
              @pl.vector_function softmax_rows_vf(in_slot, out_slot, ...)
                │  pass1: vf.load_align(fp16) → vf.reduce_max(fp16,exact) → vf.full 广播 m
                │  pass2: vf.load_align(fp16) → vf.exp_sub(fp16,fp16_m, dtype=FP32, ZERO/ONE) → vf.reduce_sum(fp32) → vf.full 广播 s
                │  pass3: vf.load_align(fp16) → vf.exp_sub(→fp32) → vf.div(fp32, s) → vf.astype(fp32→fp16, ZERO/ONE)+vf.add 重组 → vf.store_align(fp16)
                ▼
              out_slot (UB, fp16 [TILE_ROWS, MAX_N], double-buffer)
  └─[pl.store, MTE3]→ GM y (fp16 [B,N])
```
m/s 为 VF 内寄存器标量（lane0，经 `vf.full` 广播），不物化到独立 UB/GM tensor。无 Cube、无 cross-core、无 GM workspace。

---

## §1 API 映射（R1 输出）

> 按 `$CANNBOT_CONFIG_ROOT/references/performance-constraints.md` 中的“强制2：Vector默认使用VF”填写。每个Vector步骤冻结唯一实现：
> 已选 KB 模板（vec-row-reduce-broadcast）的 validated skeleton 是 **tile-op 形式**，未要求当前步骤使用 `pl.*`；本算子选用 VF 形式（依据官方 softmax 样例 `pro_ops/vf_api/test_softmax_tile_group_vf.py` 验证），故全部 Vector 步骤 `vector_selection.implementation = vf`。

### Module 1 API 调用序列

```
Module 1 — softmax_vf_row_reduce (section_vector, 单 @pl.vector_function):
  # 编译期常量（coder 必须逐字使用，见 §2.1）
  LANES = 128          # fp16 VF 寄存器宽度（元素数）
  MAX_N = 8192         # UB tile 列物理尺寸（LANES 整数倍，覆盖动态上界 N≤8192）
  TILE_ROWS = 3        # UB tile 行物理尺寸
  NEG_INF = -1e30      # reduce_max 有限负哨兵（float("inf") 在 PyPTO-Pro 不编译，见 §1数值边界）
  SLOT_BYTES = TILE_ROWS * MAX_N * 2   # fp16 单槽 49152 B

  # kernel 主体（@pl.jit(auto_mutex=True)）
  in_group  = pl.make_tile_group(type=fp16_tt, addrs=[VA_IN0, VA_IN1],  mutex_ids=[0,1])
  out_group = pl.make_tile_group(type=fp16_tt, addrs=[VA_OUT0,VA_OUT1], mutex_ids=[2,3])
  with pl.section_vector():
      rows, cols = x.shape[0], x.shape[1]
      num_cores, core_id = pl.get_block_num(), pl.get_block_idx()
      num_tiles = (rows + TILE_ROWS - 1) // TILE_ROWS
      for tile_id in pl.range(core_id, num_tiles, num_cores):
          row_off = tile_id * TILE_ROWS
          valid_rows = pl.min(TILE_ROWS, rows - row_off)
          in_slot  = in_group.next();  pl.set_validshape(in_slot,  [valid_rows, cols]); pl.load(in_slot,  x, [row_off,0])
          out_slot = out_group.next(); pl.set_validshape(out_slot, [valid_rows, cols])
          softmax_rows_vf(in_slot, out_slot, valid_rows, cols)        # 三遍寄存器循环
          pl.store(y, out_slot, [row_off, 0])

  # @pl.vector_function softmax_rows_vf 内部（每行 m 独立；m/s 为 VF 内寄存器标量）
  preg_fp16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)   # fp16 族 mask（load/exp_sub src/store）
  preg_fp32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)   # fp32 族 mask（reduce_sum/div on fp32）
  n_regs = (n_cols + LANES - 1) // LANES
  for m in pl.range(0, n_rows):
      base = m * MAX_N
      # ---- pass1: 行 max（fp16 精确归约，max of fp16 is exact）----
      row_max = vf.full(NEG_INF, preg_fp16, dtype=pl.DT_FP16)
      for r in pl.range(0, n_regs):
          valid = pl.min(LANES, n_cols - r*LANES)
          mreg_fp16 = vf.update_mask(valid, dtype=pl.DT_FP16)           # 列尾寄存器 mask-before-max
          reg = vf.load_align(in_tile, base + r*LANES)
          part = vf.reduce_max(reg, mreg_fp16)                          # fp16 max → lane0
          row_max = vf.max(row_max, part, preg_fp16)                   # 跨寄存器合并 lane0
      row_max_b = vf.full(row_max, preg_fp16)                          # 广播 fp16 max → 全 lane

      # ---- pass2: 行 sum-of-exp（exp_sub fp16→fp32 + reduce_sum fp32 累加）----
      row_sum = vf.full(0.0, preg_fp32, dtype=pl.DT_FP32)
      for r in pl.range(0, n_regs):
          valid = pl.min(LANES, n_cols - r*LANES)
          mreg_fp16 = vf.update_mask(valid, dtype=pl.DT_FP16)
          reg = vf.load_align(in_tile, base + r*LANES)
          # exp_sub fp16→fp32：每调用处理 64/128 元素（layout ZERO=偶数 lane / ONE=奇数 lane）
          e_even = vf.exp_sub(reg, row_max_b, mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)  # 64 fp32
          e_odd  = vf.exp_sub(reg, row_max_b, mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)   # 64 fp32
          k = valid
          mreg_fp32_z = vf.update_mask((k + 1)//2, dtype=pl.DT_FP32)    # ZERO 结果有效数 = ceil(k/2)
          mreg_fp32_o = vf.update_mask(k//2,       dtype=pl.DT_FP32)    # ONE  结果有效数 = floor(k/2)
          row_sum = vf.add(row_sum, vf.reduce_sum(e_even, mreg_fp32_z), preg_fp32)
          row_sum = vf.add(row_sum, vf.reduce_sum(e_odd,  mreg_fp32_o), preg_fp32)
      row_sum_b = vf.full(row_sum, preg_fp32)                           # 广播 fp32 sum → 全 lane

      # ---- pass3: div + 写回（exp_sub→fp32, div fp32, astype fp32→fp16 重组, store_align fp16）----
      for r in pl.range(0, n_regs):
          valid = pl.min(LANES, n_cols - r*LANES)
          mreg_fp16 = vf.update_mask(valid, dtype=pl.DT_FP16)
          reg = vf.load_align(in_tile, base + r*LANES)
          e_even = vf.exp_sub(reg, row_max_b, mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
          e_odd  = vf.exp_sub(reg, row_max_b, mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
          d_even = vf.div(e_even, row_sum_b, mreg_fp32_z)               # fp32 div, 64 元素
          d_odd  = vf.div(e_odd,  row_sum_b, mreg_fp32_o)
          # astype fp32→fp16：layout ZERO 填偶数 lane（奇数位置0），ONE 填奇数 lane（偶数位置0）
          out16_even = vf.astype(d_even, mreg_fp16, dtype=pl.DT_FP16, layout=pl.CastLayout.ZERO)  # 128 fp16, 偶数 slot
          out16_odd  = vf.astype(d_odd,  mreg_fp16, dtype=pl.DT_FP16, layout=pl.CastLayout.ONE)   # 128 fp16, 奇数 slot
          out16 = vf.add(out16_even, out16_odd, mreg_fp16)              # 不相交 slot 相加重组 128 fp16
          vf.store_align(out_tile + (base + r*LANES), out16, mreg_fp16)
```

**API 来源**（逐项核对 API 参考页 / 官方指定算子）：
- `vf.create_mask` / `vf.update_mask`：`docs/pypto_pro/api/SIMD-API/operation/vf_computation/mask_operations/`（fp16=128 元素×2bit，fp32=64×4bit；两族 mask 不可混用）。
- `vf.full`：`.../vf_computation/data_movement/full.md`（标量广播 lane0→全 lane；`dtype=` 指定）。
- `vf.load_align` / `vf.store_align`：`.../vf_computation/data_movement/load_align.md`、`store_align.md`（32B 地址对齐；元素偏移语义）。
- `vf.reduce_max`（fp16 精确）/ `vf.max`：`.../vf_computation/reduction/reduce_max.md`、`.../basic_arithmetic/max.md`（结果写 lane0）。
- `vf.exp_sub`（fp16→fp32，layout ZERO/ONE 处理 64/128）：`.../composite_computation/exp_sub.md`（约束表 src0=src1=DT_FP16→dst=DT_FP32；每调用处理 64 元素）。
- `vf.reduce_sum`（fp32 累加）/ `vf.add`：`.../reduction/reduce_sum.md`（DT_FP32 源在 DT_FP32 精度累加）、`.../basic_arithmetic/add.md`。
- `vf.div`：`.../basic_arithmetic/div.md`（src0/src1/dst 同 dtype，fp32）。
- `vf.astype`（fp32→fp16，layout ZERO/ONE 偶/奇 lane）：`.../type_conversion/astype.md`（表3 FP32→FP16 支持 ZERO/ONE；无 `vf.cast`）。
- `pl.make_tile_group` + `@pl.jit(auto_mutex=True)`：`.../resource_management/make_tile_group.md`（mutex_ids [0,31]）。
- `pl.set_validshape` / `pl.load` / `pl.store`：`.../memory_vector_computation/transpose_and_element_access/set_validshape.md`、`.../memory_data_movement/`。
- 整体 VF 3-pass 结构 + 双缓冲 in/out + multicore striding：**官方指定算子** `pro_ops/vf_api/test_softmax_tile_group_vf.py`（直接对应，仅 dtype fp32→fp16 差异，由 exp_sub/astype cast 路径覆盖）。

### Vector 选择（每步冻结）

| step | implementation | api_sequence | decision_reason | kb_template_evidence | target_version | applicable_conditions |
|------|----------------|--------------|-----------------|---------------------|----------------|------------------------|
| pass1 行 max | vf | vf.full→vf.load_align→vf.reduce_max→vf.max→vf.full | default_vf | n/a | 950/A5 | fp16 输入；mask 空→dtype 最小值；列尾 update_mask |
| pass2 exp+sum | vf | vf.load_align→vf.exp_sub(ZERO/ONE,dtype=FP32)→vf.reduce_sum→vf.add→vf.full | default_vf | n/a | 950/A5 | fp16 src→fp32 dst；每调用 64/128 元素需 ZERO+ONE |
| pass3 div+store | vf | vf.load_align→vf.exp_sub(ZERO/ONE)→vf.div→vf.astype(ZERO/ONE)→vf.add→vf.store_align | default_vf | n/a | 950/A5 | fp32 div；astype fp32→fp16 重组 128 fp16 |
| 数据搬运 | vf(VF内)+pl(kernel内) | pl.load/pl.store(tile级) + vf.load_align/vf.store_align(寄存器级) | default_vf | n/a | 950/A5 | GM↔UB 32B 对齐 |

> `decision_reason=default_vf`：已选 KB 模板 vec-row-reduce-broadcast 的 validated skeleton 是 tile-op 形式，**未要求**当前步骤使用 `pl.*`（其 Validation status 明确 VF 形式是 separate implementation choice）；本算子按 performance-constraints 强制2 默认选 VF，并经官方 softmax VF 样例独立验证。

### 超越函数数值安全边界

| API | 输入理论范围 | 目标 dtype 上限 | 是否溢出 | 防护措施 |
|-----|-------------|----------------|----------|----------|
| `vf.exp_sub(reg, row_max_b, ...)` | `x - m ≤ 0`（m = max_j x[b,j]，故每元素 x-m ≤ 0） | fp32 max ≈ 3.4e38；`exp(0)=1` | 否（max-subtract 保证 exp 输入 ≤0，最大输出 exp(0)=1，不上溢；极负差值 exp→0 下溢，对应概率 0，语义正确） | SPEC 公式含 max-subtract（数值稳定），无需额外防护；exp 在 fp32 内部计算（exp_sub cast_f16_to_f32 后 exp），精度满足 |
| `vf.div(e, row_sum_b, ...)` | 分母 `s = sum_j exp(x-m) ≥ exp(0) = 1 > 0`（每行含 max 元素，exp(0)=1） | — | 否（s≥1 恒成立，无除零；SPEC N≥1 保证每行非空） | max-subtract 数学保证 s≥1；无需 epsilon |
| `vf.reduce_sum(e, ...)` | 累加 N 个 ≤1 的 fp32 值，上界 ≤ N ≤ 8192 | fp32 远大于 8192 | 否 | fp32 累加（非 fp16），N=8192 下不溢出 |

**dtype/cast 链**（与 SPEC kernel-contract 一致）：`fp16(in GM/UB) → [exp_sub 内部 cast_f16_to_f32] → fp32(exp/sum/div, UB 寄存器) → [astype] → fp16(out UB/GM)`。`m`（行 max）以 fp16 存储（max of fp16 精确，作为 exp_sub src1 内部转 fp32 参与减法）；`s`（行 sum）以 fp32 累加（reduce_sum fp32）。widen 在 exp 之前（exp_sub 同时完成 cast+exp），narrow 仅在 store 前（astype），符合 precision.md「widen→compute→reduce, narrow only at store」。

---

## §2 Tile 规划（R2 输出）

### 2.1 关键常量定义

```python
# ── 算子关键常量（coder 必须逐字使用） ──
LANES = 128          # fp16 VF 寄存器宽度（元素数/寄存器）；fp32 时为 64
MAX_N = 8192         # UB tile 列物理尺寸，LANES 整数倍，覆盖动态上界 N∈[1,8192]
TILE_ROWS = 3        # UB tile 行物理尺寸；双缓冲 in+out 恰好 ≤ 248KB UB
NEG_INF = -1e30      # reduce_max 有限负哨兵（float("inf") 不编译，见 §1数值边界）
SLOT_BYTES = TILE_ROWS * MAX_N * 2   # = 3*8192*2 = 49152 B (48KB)，fp16 单槽
# UB 地址（32B 对齐；49152 = 1536*32）
VA_IN0  = 0x00000
VA_IN1  = VA_IN0  + SLOT_BYTES   # 0x0C000 (49152)
VA_OUT0 = VA_IN1  + SLOT_BYTES   # 0x18000 (98304)
VA_OUT1 = VA_OUT0 + SLOT_BYTES   # 0x24000 (147456)
# END  = VA_OUT1 + SLOT_BYTES   # 0x30000 (196608) = 192KB
```

> MAX_N=8192 取动态上界，使整行装入单 tile（§0 归约轴容量结论）；TILE_ROWS=3 受双缓冲 UB 预算约束（见 §3）。若 N<MAX_N，运行时 `set_validshape([valid_rows, cols])` 与 VF 内 `n_regs=ceil(N/128)` 只处理有效列，物理 tile 多余列不参与计算。

### Tile 属性表

| 用途 | 变量名 | shape | dtype | 内存空间 | layout | 大小(字节/槽) | 备注 |
|------|--------|-------|-------|---------|--------|----------|------|
| 输入暂存 | `in_slot` (in_group) | `[3, 8192]` | FP16 | UB(Vec) | `—` | 49152 | 双缓冲，mutex_ids=[0,1]；接收 GM fp16 load；三遍 VF 在此 tile 上读寄存器 |
| 输出暂存 | `out_slot` (out_group) | `[3, 8192]` | FP16 | UB(Vec) | `—` | 49152 | 双缓冲，mutex_ids=[2,3]；VF pass3 store_align 写入；pl.store 写回 GM |

> 无 fp32 工作 tile（VF 中间量寄存器驻留：exp_sub 直接产 fp32 寄存器、div 在 fp32 寄存器、astype 后转 fp16 寄存器）。`m`/`s` 为 VF 内寄存器标量（lane0 + vf.full 广播），不物化为 UB tile。无 scratch tile（VF 内无 store→load UB 往返）。无 L1/L0A/L0B/L0C（无 Cube）。

### TileGroup槽位访问表

| TileGroup | depth | 每个Tile的mutex ID | 访问方式 | 槽位表达式 | 同步及有界/游标说明 |
|---|---:|---|---|---|---|
| `in_group` | 2 | `[[0],[1]]`（Tile0:[0], Tile1:[1]） | `next()` | `in_group.next()` 每行块调用一次 | 双缓冲轮转：MTE2 load tile k+1 与 V compute tile k 重叠；`auto_mutex=True` 依 mutex_ids 自动管理 MTE2→V 依赖；`.next()` 自动回绕，无需手动下标 |
| `out_group` | 2 | `[[2],[3]]`（Tile0:[2], Tile1:[3]） | `next()` | `out_group.next()` 每行块调用一次 | 双缓冲轮转：V compute tile k 写出 与 MTE3 store tile k-1 重叠；`auto_mutex=True` 管理 V→MTE3 依赖 |

> `next()` 每行块调用一次（in 与 out 各一），游标由框架回绕。不使用 `group[i]` 显式下标（无跨 Section 共享缓冲、无 slot_idx 表达式需求）。mutex_ids 全局唯一：[0,1] in、[2,3] out，4 个 ID ∈ [0,31]。

### tile_dims stride 注意事项

不涉及。本设计使用 `pl.load(in_slot, x, [row_off, 0])` / `pl.store(y, out_slot, [row_off, 0])`（单元素偏移 `[row_off, 0]`，无 `tile_dims=[d0,d1]` 多维 stride 覆盖）。输入 2D RowMajor 连续，行 m 在 GM 中按 `row_off*cols` 偏移，load 按 tile shape `[3,8192]` 物理跨度搬运（cols≤8192 由 valid_shape 约束）。

---

## §3 片上空间布局（R3 输出）

> 纯 vector 算子，只有 UB(Vec) 一节。tile 按 `target_memory=Vec` 落 UB，地址从 `0x00000` 起算。

### 片上地址映射表（UB(Vec)）

| 内存空间 | 用途 | 变量名 | shape | dtype | layout | 起始字节 | 每槽字节数 | 槽位数 | 生命周期 | 结束字节（不含） | 备注 |
|---------|------|--------|-------|-------|--------|---------|-----------:|------:|----------|------------------|------|
| UB(Vec) | 输入暂存 slot0 | `in_group[0]` | `[3,8192]` | FP16 | `—` | `0x00000` | 49152 | 1 | 轮转 | `0x0C000` | mutex_id 0 |
| UB(Vec) | 输入暂存 slot1 | `in_group[1]` | `[3,8192]` | FP16 | `—` | `0x0C000` | 49152 | 1 | 轮转 | `0x18000` | mutex_id 1 |
| UB(Vec) | 输出暂存 slot0 | `out_group[0]` | `[3,8192]` | FP16 | `—` | `0x18000` | 49152 | 1 | 轮转 | `0x24000` | mutex_id 2 |
| UB(Vec) | 输出暂存 slot1 | `out_group[1]` | `[3,8192]` | FP16 | `—` | `0x24000` | 49152 | 1 | 轮转 | `0x30000` | mutex_id 3 |

> 地址使用半开区间。in_group 两槽连续 `[0x0, 0x18000)`，out_group 两槽连续 `[0x18000, 0x30000)`。in 与 out 区间不相交（双缓冲下同时 live，必须不重叠）。所有起始字节为 32 的倍数（49152 = 1536×32），满足 load_align/store_align 32B 对齐。

**各空间地址高水位**:
- UB(Vec): `0x30000` (196608 B) / 253952 (248KB, `ub_size` 950PR_957x.ini) = **77.4%**

### 分配方式选择

- **方案**: `make_tile_group` + `auto_mutex`（in_group/out_group 各 depth=2，mutex_ids=[0,1]/[2,3]；`@pl.jit(auto_mutex=True)` 自动管理 MTE2(load)→V(VF)→MTE3(store) 核内跨 Pipe 依赖）。无单次 scratch tile（VF 中间量寄存器驻留），故不使用 `make_tile`。
- **依据算子**: `pro_ops/vf_api/test_softmax_tile_group_vf.py`（in_group/out_group 双缓冲 + auto_mutex，line 113/123-124/139-147）。
- **理由**: 单 Vector Section，in/out tile 跨行块轮转（MTE2 装入下一行块 与 V 计算当前行块重叠）；`auto_mutex` 依 mutex_ids 自动插入 MTE2→V、V→MTE3 同步，无需手写 `sync_src`/`sync_dst`。三遍 max/sum/div 在一次 VF 调用内、寄存器驻留 m/s，阶段衔接为 V 管内源码顺序，无跨 Pipe 依赖需手写同步。

### double buffer 地址规划

| 项目 | 结论 | 受影响 tile/地址 |
|------|------|-----------------|
| buffer 数 | 2（in_group）+ 2（out_group）= 4 槽 | in_group[0,1]@[0x0,0x18000)，out_group[0,1]@[0x18000,0x30000) |
| PONG 地址 | 预留（每 group 第 2 槽即 PONG） | in PONG @0x0C000，out PONG @0x24000 |
| 槽位访问 | `next()` 自动轮转 | 游标由框架回绕；每行块 in/out 各一次 next() |

---

## §4 循环与 Section 结构（R4 输出）

> 依据[循环与Section结构设计](../references/loop_design.md)，把§0的Module放入具体Section代码结构。

### 参考样例

- **主要参考样例**: `pro_ops/vf_api/test_softmax_tile_group_vf.py`（PRO_MATERIAL_INDEX §B #13）
- **可复用结构点**: 单 `section_vector()`；外层 `for tile_id in pl.range(core_id, num_tiles, num_cores)` 跨核 striding 行块；行内 VF 三遍寄存器循环（pass1 max / pass2 exp+sum / pass3 div+store）；`set_validshape` 在 load 前约束行尾块；`update_mask` 约束列尾寄存器；`make_tile_group` + `auto_mutex` 双缓冲 in/out。
- **补充参考**: `pro_ops/vf_api/test_layernorm_tile_group_vf.py`（行内 reduction VF + update_mask 尾 lane cross-reference）。

### 本算子结构说明

- **Module / Section对应关系**: Module 1 → 单 `pl.section_vector()`（无 Cube Section）。三遍 max/sum/div 同属该 Vector Section，在一次 `@pl.vector_function` 调用内完成。
- **结果单元**: 一个行块 `[TILE_ROWS, MAX_N]`（最多 3 行 × N 列）的完整 softmax 输出——即每行块独立完成 max→sum→div 三遍并写回。
- **跨Tile状态生命周期**: `m`（行 max，fp16 寄存器标量）与 `s`（行 sum，fp32 寄存器标量）在 VF 内**每行**初始化（pass1 前 `vf.full(NEG_INF)` / `vf.full(0.0)`），在 pass1/pass2 跨寄存器 `r` 累积，pass3 消费。生命周期 = 单行单次 VF 调用内，**不跨行块、不跨 tile、不跨核**。无跨 tile 持久化状态。
- **Section结构**: 1 个 `section_vector()`；循环在 Section 内（Section 之前声明 tile group）。
- **循环嵌套**:
  - 外层（kernel，跨核 striding 行块）：`for tile_id in pl.range(core_id, num_tiles, num_cores)` → `row_off = tile_id * TILE_ROWS`。
  - 行内（VF 内）：`for m in pl.range(0, valid_rows)` → 每行独立三遍。
  - 寄存器遍历（VF 内，每遍）：`for r in pl.range(0, n_regs)` → 覆盖该行 N 列（每寄存器 128 fp16 元素）。
- **动态循环上界**: `num_tiles = (B + TILE_ROWS - 1) // TILE_ROWS`（ceiling）；`n_regs = (N + LANES - 1) // LANES`（ceiling，LANES=128）；`valid_rows = pl.min(TILE_ROWS, B - row_off)`；`valid = pl.min(LANES, N - r*LANES)`（列尾寄存器有效 lane 数）。B、N 由 `x.shape[0]`/`x.shape[1]` 运行时读取（`pl.DYNAMIC`）。
- **分核信息**: `core_id = pl.get_block_idx()`、`num_cores = pl.get_block_num()`（仅 Vector Section，等价 block_dim）；在 Section 内读取。行块跨核 striding，各核任务量最多相差 1（负载均衡）。host 侧 `block_dim = min(get_platform_info().vector_core_num, num_tiles)`（见 §5）。

> 同步点见 §6（不涉及 cross-core）；尾块处理见 §7。

### CV 手动预加载流水设计（`is_fusion=true`时必填）

不涉及（`is_fusion=false`，无 Cube/Vector 跨域流水）。

---

## §5 分核策略（R5 输出）

> 📌 权威依据：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/multi_core_partitioning_and_Tiling.md`。

### 分核方式

- **方案**: strided loop —— 一维跨步 `for tile_id in pl.range(core_id, num_tiles, num_cores)`，其中 `num_tiles = ceil(B / TILE_ROWS)`（行块总数）。选择理由：行间完全独立（softmax 每行归约互不依赖），按行块跨核 striding 实现负载均衡（各核行块数最多相差 1），且 GM 访问按行块连续（每核处理若干整行块）。`B` 维为 batch，无跨行状态，无需保留行/列两层循环。
- **host 侧 block_dim**: 仅 Vector Kernel → `vector_core_num`；`num_tiles = ceil(B / TILE_ROWS)`；`block_dim = min(get_platform_info().vector_core_num, num_tiles)`。`B ≥ 1`（SPEC 约束）保证 `num_tiles ≥ 1`，无零任务启动问题。
- **交付 launch 合同**: 恰好 1 次。唯一 kernel `softmax_tile_group_kernel` 由 `softmax_wrapper` 在 host 循环外调用一次：`softmax_tile_group_kernel[None, block_dim](x, y)`。
- **TensorList ABI（不涉及）**: N/A。`is_list=false`，单 tensor 输入，无 TensorList 形参。

---

## §6 核间同步（R6 输出）

> 依据[跨核同步](../references/cross_core_synchronization.md)。

### cross_core 涉及判定

- **是否涉及 cross_core**: **不涉及**
- **依据**: 单 Vector Section，无 Cube→Vector 或 Vector→Cube 数据交接；行块跨核 striding 写**不相交**的输出行区域（核 i 写行块 {i, i+num_cores, ...}），核间无共享输出、无共享状态、无跨 Block/subblock 数据依赖。in/out 双缓冲轮转的跨 Pipe 依赖（MTE2→V→MTE3）由 `auto_mutex=True` + `mutex_ids` 自动管理（核内），无需 `set_cross_core`/`wait_cross_core`。无 `INTER_BLOCK`/`INTER_SUBBLOCK`/`UNICAST_BLOCK` 需求。
- **流水实现**: 不涉及（无 CV 融合手动预加载）。

> 不涉及 cross_core，以下各表省略。VF 内三遍 max/sum/div 为 V 管内源码顺序依赖（无 store→load UB 往返，中间量寄存器驻留），无需 `vf.mem_bar`（无 V 管内 scratch store→load hazard）。

---

## §7 尾块处理（R7 输出）

> 📌 权威依据：`$PYPTO_DEVKIT_DIR/docs/pypto_pro/tutorials/operator_development/tile_based_python_programming/tail_block_handling.md`。

### 尾块处理方案

两类尾块，分别在两处独立处理（对应 online-softmax-tail 的「列尾 mask before max / 行尾 mask after subtract」位置规则，落在本设计的行块/寄存器粒度）：

**行尾块（B 维，tile 级）**：`set_validshape` + VF 行循环上界，load 前设置。

```python
m_tile_num = (B + TILE_ROWS - 1) // TILE_ROWS        # ceiling，覆盖行尾块
for tile_id in pl.range(core_id, m_tile_num, num_cores):
    row_off = tile_id * TILE_ROWS
    valid_rows = pl.min(TILE_ROWS, B - row_off)       # 满 tile=3，行尾=余数
    in_slot  = in_group.next()
    out_slot = out_group.next()
    pl.set_validshape(in_slot,  [valid_rows, N])      # load 前约束行尾 + 列
    pl.set_validshape(out_slot, [valid_rows, N])
    pl.load(in_slot, x, [row_off, 0])
    softmax_rows_vf(in_slot, out_slot, valid_rows, N)  # VF 内 for m in range(valid_rows)，行尾外行不处理
    pl.store(y, out_slot, [row_off, 0])
```

**列尾寄存器（N 维，寄存器级，reduction-axis tail）**：`update_mask` 在 `reduce_max`/`exp_sub`/`reduce_sum`/`store_align` **之前**（mask-before-max），并按 ZERO/ONE 半元素语义计算 fp32 尾 mask。

```python
n_regs = (N + LANES - 1) // LANES                     # ceiling，N=8192→64 regs
for r in pl.range(0, n_regs):
    k = pl.min(LANES, N - r*LANES)                    # 满 reg=128，列尾=余数
    mreg_fp16 = vf.update_mask(k, dtype=pl.DT_FP16)   # fp16 尾寄存器 mask（128 lane）
    reg = vf.load_align(in_tile, base + r*LANES)
    # exp_sub fp16→fp32 每调用处理 64/128 元素（layout ZERO=偶数 lane / ONE=奇数 lane）
    e_even = vf.exp_sub(reg, row_max_b, mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    e_odd  = vf.exp_sub(reg, row_max_b, mreg_fp16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    # fp32 结果有效数：ZERO=ceil(k/2)（偶数索引 0,2,...,2*ceil(k/2)-1），ONE=floor(k/2)
    mreg_fp32_z = vf.update_mask((k + 1)//2, dtype=pl.DT_FP32)
    mreg_fp32_o = vf.update_mask(k//2,       dtype=pl.DT_FP32)
    part_sum = vf.reduce_sum(e_even, mreg_fp32_z)     # 仅有效 fp32 lane 入累加
    ...
```

- **compact**: `None`（Vec ND 尾块不需要紧凑模式；无 fractal/cube/Acc 路径）。`tail_block_handling.md` 参数速查：向量逐元素 ND 动态尾块 → `valid_shape=[-1,-1]` + 逐块 `set_validshape`，compact 不需要。
- **无效区域是否会被读取**:
  - 行尾：`set_validshape([valid_rows, N])` + VF `for m in range(valid_rows)` → 行尾外行不 load、不计算、不 store。**不被读取**，不需 `fillpad`。
  - 列尾寄存器：尾寄存器的 padding lane 由 `update_mask` 屏蔽，`reduce_max`/`reduce_sum`/`store_align` 仅作用于有效 lane（mask-before-max 保证 padding 不污染 max/sum/store）。**mask 屏蔽**，不需 `fillpad`（`fillpad` 是 tile-op 路径，VF 用 register mask 替代；见 `tail-validshape.md` R03 invariant）。
  - pass1 max 初始化为 `NEG_INF=-1e30`（有限负哨兵），即使空 mask 也不产生 NaN（reduce_max mask 空→dtype 最小值，但 N≥1 保证每行至少 1 有效 lane，不触发全空）。

---

## §8 目标测试 case（R7.5 输出 → 交付 develop）

> 基于 R2 tile 尺寸（TILE_ROWS=3, MAX_N=8192, LANES=128）与 R7 尾块方案。SPEC P0 case 已给出，在其基础上追加 5 个基础 case 覆盖整除/单轴尾块/双轴尾块/跨多 tile。

| case 名 | 具体 shape | 覆盖场景 | design 适配确认 |
|---------|-----------|---------|----------------|
| `test_softmax_aligned` | `[3, 128]` | 全整除（B%3=0 单行块满，N%128=0 单寄存器满） | ✅ num_tiles=1, valid_rows=3, n_regs=1, 无尾块 |
| `test_softmax_row_tail` | `[4, 128]` | 单轴（行）尾块（B%3=1→行尾1行；N%128=0 无列尾） | ✅ num_tiles=2 (3+1)，第2块 valid_rows=1 |
| `test_softmax_col_tail` | `[3, 200]` | 单轴（列/寄存器）尾块（N%128=72→列尾72 lane；B%3=0 无行尾） | ✅ n_regs=2 (128+72)，第2寄存器 k=72，mask_before-max |
| `test_softmax_tail2d` | `[4, 200]` | 双轴尾块（行尾1行 + 列尾72 lane） | ✅ 行尾 set_validshape + 列尾 update_mask 同时生效 |
| `test_softmax_multitile` | `[7, 200]` | 跨多 tile + 尾块（B=7→3行块：2满+1尾1行；N=200列尾） | ✅ num_tiles=3，验证跨行块 m/s 不串扰（每行块独立 VF 调用） |
| `test_softmax_p0_batch1_n128` | `[1, 128]` | P0（用户指定）：单行小 N，单核单 tile 单寄存器 | ✅ num_tiles=1, valid_rows=1, n_regs=1 |
| `test_softmax_p0_batch4_n2048` | `[4, 2048]` | P0（用户指定）：中 batch 中大 N，行尾+多寄存器 | ✅ num_tiles=2, n_regs=16 |
| `test_softmax_p0_batch32_n4096` | `[32, 4096]` | P0（用户指定）：大 batch 大 N，多核+多寄存器，fp32 累加精度压力 | ✅ num_tiles=11，block_dim=min(vector_core_num,11)，n_regs=32，N=4096 fp16 vs fp32 累加验证 |

> 共 8 个 case（5 基础 + 3 P0），≥4。所有 shape 的 N ≤ 8192 = MAX_N（整行装入单 tile）。每个 case 适配确认由 R2 tile 尺寸 + R4 循环上界 + R7 尾块方案推导；逐个核对 design 可适配（无回溯）。
> 比较：NPU 输出 fp16 与 CPU golden fp32 均升 fp32 后 `torch.testing.assert_close(rtol=1e-3, atol=1e-3)`（与官方 softmax 样例 line 173 一致）。

---

## §9 综合评估（R8 输出）

### 准确性检查

| 检查项 | 结果 | 证据 |
|--------|------|------|
| API 调用链完整覆盖数学公式 | ✅ | pass1 reduce_max(max) → pass2 exp_sub(exp(x-m))+reduce_sum(sum) → pass3 exp_sub+div(y=e/s)+store；公式 y=exp(x-m)/s 每一步均有对应 vf.* API |
| §4 已参照官方样例确定循环与Section结构（含参考样例路径与结构说明） | ✅ | `pro_ops/vf_api/test_softmax_tile_group_vf.py`：单 section_vector + 行块 striding + 行内三遍寄存器循环 + set_validshape/update_mask 尾块 + 双缓冲 in/out + auto_mutex，本设计结构与之直接对应 |
| 数据依赖正确（Module 顺序 + sync 点） | ✅ | 单 Module 单 Section；m/s 在 VF 内源码顺序保证（pass1→pass2→pass3），无跨 Module/Section 依赖；行块间独立，无 sync 需求 |
| CV融合是否给出... | ✅ N/A | `is_fusion=false`，无 CV 融合流水 |
| dtype 精度满足要求 | ✅ | sum 累加 fp32（reduce_sum fp32）；max 精确 fp16；exp 内部 fp32（exp_sub cast_f16_to_f32）；div fp32；输出 astype fp16。N=4096 容差 1e-3 由 fp32 累加保证 |
| 归约类 API 的 `[M,1]`/`[1,N]` 输出已设 `layout` | ✅ N/A | 本设计用 VF 寄存器级 reduce（reduce_max/reduce_sum→lane0 标量+vf.full 广播），不写 tile-op `[rows,1]` DN tile，无 layout 需求 |
| Acc tile 物理 M×N×dtype_bytes ≥ fractal | ✅ N/A | 无 Cube/Acc tile |

### 泛化性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 目标测试 case（≥4）已按 tile 切分确定具体 shape，且逐个验证 design 可适配 | ✅ | §8 共 8 case，逐个适配确认（无回溯） |
| 是否正确处理了尾块（M/N 尾块） | ✅ | 行尾 set_validshape + VF 行循环上界；列尾 update_mask（mask-before-max）+ ZERO/ONE fp32 尾 mask 计数（ceil/floor(k/2)） |
| 归约轴可能超单Tile时，已给出覆盖完整归约轴的方案 | ✅ N/A | 归约轴 N≤8192=MAX_N，整行装入单 tile，不超单 tile；n_regs=ceil(N/128) 覆盖完整 N |
| 循环边界正确 | ✅ | num_tiles/n_regs 用 ceiling division；valid_rows/valid 用 pl.min clamp 到物理尺寸 |
| 超越函数在 dtype 范围内无溢出 | ✅ | §1 数值边界：exp_sub 输入 ≤0（max-subtract），exp≤1 不上溢；下溢→0 语义正确；div 分母 s≥1 无除零 |
| 窄 dtype 中间值量级上界在 dtype 范围内 / 升位宽落在产生增长那一步之前 | ✅ | widen→compute→reduce：exp_sub(dtype=FP32) 在 exp 前完成 fp16→fp32 cast（产生增长的 exp 之前）；reduce_sum fp32；narrow 仅 store 前 astype。无 fp16 中间累加溢出 |
| 跨 tile 状态初始化/持久化正确 | ✅ | m/s 每行 VF 内初始化（vf.full），不跨 tile/行块持久化，无跨 tile 状态串扰风险 |
| cross_core 同步方案正确 | ✅ N/A | 不涉及 cross_core（§6），单 Vector Section + 不相交输出行 + auto_mutex 核内同步 |

### 一致性检查

| 检查项 | 结果 | 说明 |
|--------|------|------|
| R0-R7 输出无矛盾 | ✅ | §0 单 Module/整行单 tile ↔ §2 MAX_N=8192/TILE_ROWS=3 ↔ §3 UB 192KB ↔ §4 n_regs=ceil(N/128) ↔ §7 尾块方案 ↔ §8 case 适配，交叉一致 |
| 所有决策有证据支撑 | ✅ | API 参考页（§1）、官方指定算子（§1/§4）、SPEC（维度/容差/P0）、EXPLORE_REPORT §7（UB/mask 约束）、KB patterns/constraints（DESIGN_BINDINGS.json） |
| R3 片上地址范围与逐空间容量检查 | ✅ | UB 高水位 0x30000(196608)/253952 = 77.4% ≤ 100%；in/out 区间不相交；32B 对齐 |
| `tile_dims` 使用时已关注大 stride 对性能的影响 | ✅ N/A | 不使用 tile_dims（单元素偏移 load/store） |
| 条件性检查（§6 不涉及 cross_core 时确认无 Cube↔Vector 或跨 Block 依赖） | ✅ | §0 单 Vector Module；§6 确认无 Cube、行块输出不相交、核间无共享状态 |

### 评估结论

- **整体**: **通过**
- **限制条件**: 
  - 动态范围 N∈[1,8192] 由 MAX_N=8192 单 tile 覆盖；若未来需 N>8192，须回退 §0 重设为分块流式归约（online-softmax-tail recurrence 转 applies）。
  - TILE_ROWS=3 受双缓冲 UB 预算约束（192KB/248KB）；Stage 5 性能优化可调（如单缓冲大 TILE_ROWS 或调整 MAX_N/TILE_ROWS 平衡）。
  - fp16→fp32 的 ZERO/ONE 半元素语义使每寄存器 exp 工作量翻倍（widen-compute-narrow 3x 向量开销，与 vec-row-reduce-broadcast 模式测量一致），属 fp16+fp32累加的固有开销。
- **修改记录**: 无（R0–R8 首轮通过，无回溯）。

---

## §10 Tile 数据流全景图

```
GM x (fp16 [B, N])
  │
  ├─[pl.load, MTE2]─→ in_group.next()  (UB fp16 [3,8192], mutex_id 0/1, double-buffer @0x0/0xC000)
  │                     │  set_validshape([valid_rows, N]) 在 load 前
  │                     ▼
  │            ┌───────────────────────────────────────────────────────────┐
  │            │  @pl.vector_function softmax_rows_vf(in_slot, out_slot,…)   │
  │            │  (auto_mutex 管 MTE2→V→MTE3；m/s 寄存器驻留，不落 UB)        │
  │            │                                                               │
  │            │  for m in range(valid_rows):  ── 每行独立 ──                  │
  │            │    base = m * MAX_N                                           │
  │            │                                                               │
  │            │    pass1 行 max (fp16 exact):                                │
  │            │      row_max = vf.full(NEG_INF, fp16)                        │
  │            │      for r in range(n_regs):                                 │
  │            │        mreg_fp16 = update_mask(min(128,N-r*128))  ◀── 列尾 mask-before-max
  │            │        reg = vf.load_align(in_slot, base+r*128)  ── (fp16 128 elem) ─┐
  │            │        row_max = vf.max(row_max, vf.reduce_max(reg, mreg_fp16))     │ |
  │            │      row_max_b = vf.full(row_max, fp16)  ── 广播 ─                  │ |
  │            │                                                                     │ |
  │            │    pass2 行 sum-of-exp (fp32 累加):                     reuse in_slot│
  │            │      row_sum = vf.full(0.0, fp32)                                    │ |
  │            │      for r in range(n_regs):                                         │ |
  │            │        reg = vf.load_align(in_slot, base+r*128)  ◀──────────────────┘ |
  │            │        e_even = vf.exp_sub(reg, row_max_b, mreg_fp16, dtype=FP32, layout=ZERO) ── fp16→fp32 (64 偶 lane)
  │            │        e_odd  = vf.exp_sub(reg, row_max_b, mreg_fp16, dtype=FP32, layout=ONE)  ── fp16→fp32 (64 奇 lane)
  │            │        row_sum = vf.add(row_sum, vf.reduce_sum(e_even, mreg_fp32_z), fp32)  ◀── widen→compute→reduce
  │            │        row_sum = vf.add(row_sum, vf.reduce_sum(e_odd,  mreg_fp32_o), fp32)
  │            │      row_sum_b = vf.full(row_sum, fp32)  ── 广播 ─                       |
  │            │                                                                        │
  │            │    pass3 div + 写回 (fp32 div → astype fp16 → store):                  │
  │            │      for r in range(n_regs):                                           │
  │            │        reg = vf.load_align(in_slot, base+r*128)  ◀── reuse in_slot ────┘
  │            │        e_even/e_odd = vf.exp_sub(reg, row_max_b, ..., dtype=FP32, ZERO/ONE)
  │            │        d_even = vf.div(e_even, row_sum_b, mreg_fp32_z)   ── fp32 div
  │            │        d_odd  = vf.div(e_odd,  row_sum_b, mreg_fp32_o)
  │            │        out16_even = vf.astype(d_even, mreg_fp16, dtype=FP16, layout=ZERO) ── 128 fp16 偶 slot
  │            │        out16_odd  = vf.astype(d_odd,  mreg_fp16, dtype=FP16, layout=ONE)  ── 128 fp16 奇 slot
  │            │        out16 = vf.add(out16_even, out16_odd, mreg_fp16)  ── 重组 128 fp16 (不相交 slot)
  │            │        vf.store_align(out_slot + (base+r*128), out16, mreg_fp16)  ── 写 out_slot
  │            └───────────────────────────────────────────────────────────────────────────┘
  │                     │
  │                     ▼
  │            out_group.next()  (UB fp16 [3,8192], mutex_id 2/3, double-buffer @0x18000/0x24000)
  │                     │  set_validshape([valid_rows, N]) 在 store 前
  │                     ▼
  └─[pl.store, MTE3]─→ GM y (fp16 [B, N])
```

### 图例

| 标记 | 含义 |
|------|------|
| `[pl.load, MTE2]` (tile 级 load_tile 角色) | MTE2 搬运：GM → UB（fp16）；本设计按官方 softmax 样例用 `pl.load(in_slot, x, [row_off,0])`（单元素偏移 tile 级 load，等效于 load_tile 的 GM→UB tile 搬运角色；多 tile_dims 变体 `pl.load_tile` 此处不需要） |
| `[pl.store, MTE3]` (tile 级 store_tile 角色) | MTE3 搬运：UB → GM（fp16）；`pl.store(y, out_slot, [row_off,0])`（等效 store_tile 角色） |
| `vf.load_align` | V 流水：UB → 寄存器（fp16 128 elem/寄存器） |
| `vf.store_align` | V 流水：寄存器 → UB（fp16 128 elem/寄存器） |
| `→` | V 流水操作，标注 vf.* API 名 |
| `vf.exp_sub(dtype=FP32, ZERO/ONE)` | fp16→fp32 widen（exp 内部 cast_f16_to_f32），偶/奇 lane 各 64 元素 |
| `vf.astype(layout=ZERO/ONE)` | fp32→fp16 narrow，偶/奇 slot；vf.add 重组 |
| `reuse in_slot` | 同一 in_slot 在 pass1/2/3 三遍被读（整行在 UB 内驻留，不重读 GM） |
| `---` | 无跨 Module 持久化（单 Module，m/s 寄存器驻留不落 UB） |
| `◀── 列尾 mask-before-max` | update_mask 在 reduce_max/exp_sub/reduce_sum/store 之前 |

**验证**：图中每一块 tile（in_group/out_group，2 组各 2 槽）与每一个操作（load/store/exp_sub/reduce_max/reduce_sum/div/astype/add/full/update_mask）均能在 §3 地址表与 §1 API 序列中找到对应条目；m/s 为寄存器标量（无 UB tile 条目，符合 §0 不物化）；无跨 Module 虚线（单 Module）。无矛盾。
