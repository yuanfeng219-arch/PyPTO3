# MEMORY — softmax (Stage 1 规划)

## 任务摘要

- 算子：`softmax`，对 2D 输入 `[B, N]`（float16）沿最后一维做数值稳定 softmax，输出同 shape 同 dtype float16。
- 公式：`y[b,i] = exp(x[b,i]-m[b])/s[b]`，`m[b]=max_j(x[b,j])`，`s[b]=sum_j(exp(x[b,j]-m[b]))`。
- 用户需求完整（算子名/公式/dtype/shape/P0 case 均给出），按无人值守模式冻结 SPEC。
- P0 case：`[1,128]`、`[4,2048]`、`[32,4096]`。容差 `atol=1e-3, rtol=1e-3`（官方 softmax 样例）。
- 性能目标：用户未给数值目标 → Stage 5 用默认 `golden_reference_ratio >= 1.0`。

## 确认状态与产物指针（Stage 1 全部完成）

- SPEC.md：`custom/softmax/SPEC.md`（validate_spec.py PASS，2026-08-31；含 kernel 契约补充节）。
- EXPLORE_REPORT.md：`custom/softmax/EXPLORE_REPORT.md`（10 章完整，feasibility=可行，无 unsupported 阻断）。
- PRO_MATERIAL_INDEX.md：`custom/softmax/PRO_MATERIAL_INDEX.md`（§A=312 / §B=13 / §C=40，与缓存一致）。
- KB_SELECTION.json：`custom/softmax/KB_SELECTION.json`（flat 布局，class_id="."；收尾自检 OK）。
- MEMORY.md：本文件。

## kernel 契约补充裁定

- 辅助张量：无公开辅助输入；`m[b]`/`s[b]` 为 kernel 内逐行 reduction 标量中间量，实现位置交 Stage 3。
- cast 链：`fp16(in) → fp32(acc，强建议) → fp16(out)`；其余 cast 交 Stage 3。
- 累加/写回：行级覆盖式 max、累加式 sum；输出覆盖写到新 tensor（非 in-place）；行间独立可并行。
- 目标设备：默认 A5（SoC 950），非用户确认的默认假设；KB 触发 `constraints/arch-a5.md`。
- topology/tile/同步：由 Stage 3 设计；Stage 1 仅确认"逐行 reduction + elementwise"计算拓扑。

## 公式步骤 → API 候选映射（简表，详见 EXPLORE_REPORT §2/§3）

1. 行 max：`vf.reduce_max` → `vf.max` 合并 → `vf.full` 广播（DT_FP16/FP32 均支持；mask 空返 dtype 最小值）
2. exp(x-m)：`vf.exp_sub`（fp16→fp32 高精度路径 `dtype=DT_FP32`，每调用 64/128 元素需 ZERO+ONE；或 fp32 直算）
3. 行 sum：`vf.reduce_sum` → `vf.add` 合并 → `vf.full` 广播（**关键：fp16 源在 fp16 精度累加**，须升 fp32）
4. div + 写回：`vf.div` → `vf.store_align`（输出 fp16，若 e/s 在 fp32 须 `vf.astype` fp32→fp16）

## Stage 3 关键待裁定项（来自 EXPLORE_REPORT §6）

- **fp16 累加精度**（核心）：`reduce_sum` fp16 源在 fp16 累加，N=4096 可能超 1e-3 容差。三方案：A 整段 fp32（UB 翻倍）；B 仅 exp/sum 升 fp32（exp_sub fp16→fp32 + reduce_sum fp32，exp 工作量翻倍）；C 全 fp16 直算（精度风险）。三条均有 API 支持。
- **exp_sub/astype 半数元素**：fp16→fp32 每调用处理 64/128，覆盖全量需 ZERO+ONE 两次。
- **N=4096 超 MAX_N=512**：须放大 MAX_N（UB 限）或多趟寄存器/多 tile 循环；样例 n_regs 循环结构可扩展。
- **UB 预算**：248KB；fp32 方案字节翻倍，须缩 TILE_ROWS/MAX_N 或减缓冲深度。
- 尾块：set_validshape（load 前）+ update_mask（尾寄存器）；多核：get_block_idx striding。

## KB 路由结果（KB_SELECTION.json）

- topologies：`row-reduction`（softmax 显式命中）+ `multi-phase-fusion`（max/sum/div 依赖阶段，状态 m/s 跨阶段携带）。
- properties：dtypes=[float16], ranks=[2], unaligned_shapes=true, tail_blocks=true, long_axis=true, mixed_precision=true, dynamic_dims=true, is_list=false。
- optional_patterns（3）：vec-row-reduce-broadcast（行归约+广播，softmax 验证骨架）、online-softmax-tail（大 N 分块流式尾块/精度）、buffer-reuse-lifetime（3-pass 缓冲复用，softmax 验证骨架）。
- required_constraints（9）：vec、sync-stitch、tiling、memory-layout、tail-validshape、precision、vec-mask-width、arch-a5、wrapper-boundary。
- no_matching_pattern=false。收尾自检 OK。

## 已冻结事实 / 阻塞 / 尝试历史

- 无阻断项；feasibility=可行。
- 非阻断注意：`pro_ops/matmul/test_matmul_8K_example.py` 清单大小写 vs 缓存 `8k`（文件存在，非 softmax 相关）；`docs/pypto_api_list.md` 缓存缺失（§A 已按子目录分组全量替代）。
- [2026-08-31] 冻结 SPEC.md，validate_spec.py PASS。
- [2026-08-31] 完成 EXPLORE_REPORT + PRO_MATERIAL_INDEX（扫描 §A=312/§B=13/§C=40）。
- [2026-08-31] 冻结 KB_SELECTION.json，收尾自检 OK。
