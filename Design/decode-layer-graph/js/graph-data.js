// Compute graph extracted from Design/Toolkit Studio/decode_layer.py (_decode_layer + hosts).
// Each node = one task group inside the per-layer inline body. `col` = pipeline depth
// (dependency stage), used by the layered SVG layout. `deps` = upstream node ids.
// tags: scope (manual = inside pl.manual_scope, auto-dep suppressed), unit (AIC cube /
// AIV vector / mixed), count = number of device tasks the group expands to.

const TENSOR = {
  hidden_states: { t: "hidden_states", shape: "[16, 5120]", dt: "FP32", role: "跨层 carry：本层残差流输入（上一层 dcr 写）" },
  normed_in:     { t: "normed_in", shape: "[16, 5120]", dt: "BF16", role: "本层 x·γ（上一层 dcr_xgamma / layer0 的 x_gamma0 产出），仅 QKV 消费" },
  inv_rms:       { t: "inv_rms_states", shape: "[16, 1]", dt: "FP32", role: "1/rms 行标量，延迟到 rope 里乘" },
  q_proj:        { t: "q_proj", shape: "[16, 5120]", dt: "FP32", role: "Q 投影（split-K 原子加）" },
  k_proj:        { t: "k_proj", shape: "[16, 1024]", dt: "FP32", role: "K 投影" },
  v_proj:        { t: "v_proj", shape: "[16, 1024]", dt: "FP32", role: "V 投影" },
  q_norm:        { t: "q_proj_norm", shape: "[16, 5120]", dt: "FP32", role: "QK-norm 后的 Q（仅 γ，recip 延迟）" },
  k_norm:        { t: "k_proj_norm", shape: "[16, 1024]", dt: "FP32", role: "QK-norm 后的 K" },
  q_inv:         { t: "q_inv_states", shape: "[8·16·5, 1]", dt: "FP32", role: "每 (头,行,组) 的 1/rms，延迟到 rope" },
  k_inv:         { t: "k_inv_states", shape: "[8·16, 1]", dt: "FP32", role: "每 (头,行) 的 1/rms" },
  all_q:         { t: "all_q_padded", shape: "[16·8·16, 128]", dt: "BF16", role: "RoPE 后的 Q，pad 到 16 行/头供 cube" },
  k_cache:       { t: "k_cache", shape: "paged", dt: "BF16", role: "分页 KV：RoPE 后的 K 写入物理页" },
  v_cache:       { t: "v_cache", shape: "paged", dt: "BF16", role: "分页 KV：V 写入物理页" },
  work_table:    { t: "fa_work_table / fa_total", shape: "[256,1]/[1,1]", dt: "INT32", role: "稠密真实块工作表 + 块计数" },
  oi_tmp:        { t: "all_oi_tmp", shape: "[16·8·32·16,128]", dt: "FP32", role: "每块 SV 部分和" },
  cur_mi:        { t: "all_cur_mi", shape: "[…,1]", dt: "FP32", role: "每块行最大（softmax 稳定项）" },
  cur_li:        { t: "all_cur_li", shape: "[…,1]", dt: "FP32", role: "每块行和（softmax 归一）" },
  attn_out:      { t: "attn_out", shape: "[16, 5120]", dt: "BF16", role: "在线 softmax 归并后的注意力输出" },
  attn_proj:     { t: "attn_proj_fp32", shape: "[16, 5120]", dt: "FP32", role: "out_proj 结果（split-K/N 原子加）" },
  post_partial:  { t: "post_norm_partial", shape: "[16, 5120]", dt: "FP32", role: "原始残差 h1 = attn_proj + hidden（down 后加回）" },
  mlp_norm_in:   { t: "mlp_norm_in", shape: "[16, 5120]", dt: "BF16", role: "h1 · post_γ，gate/up 的输入" },
  inv_rms_tile:  { t: "inv_rms_tile", shape: "[16, 1]", dt: "FP32", role: "post-RMS 的 1/rms，延迟到 silu" },
  gate_acc:      { t: "gate_acc_all", shape: "[16, 17408]", dt: "FP32", role: "gate 投影累加" },
  up_acc:        { t: "up_acc_all", shape: "[16, 17408]", dt: "FP32", role: "up 投影累加" },
  mlp_tile:      { t: "mlp_tile", shape: "[16, 17408]", dt: "BF16", role: "SiLU(gate)·up 激活" },
  down_acc:      { t: "down_acc_all", shape: "[16, 5120]", dt: "FP32", role: "down 投影累加" },
  out:           { t: "out", shape: "[16, 5120]", dt: "FP32", role: "本层输出残差流（= 下一层 hidden_states）" },
  normed_out:    { t: "normed_out", shape: "[16, 5120]", dt: "BF16", role: "下一层的 x·γ（与 out 同任务产出）" },
};

const NODES = [
  // ── stage 0: inputs ──
  { id: "in_hidden", label: "hidden_states", col: 0, kind: "io", unit: "carry", scope: "auto",
    reads: [], writes: ["hidden_states"], deps: [], carryIn: true,
    desc: "上一层 dcr_xgamma 写入的 FP32 残差流；layer0 由 copy_hidden 从 BF16 embed 转来。" },
  { id: "in_normed", label: "normed_in", col: 0, kind: "io", unit: "carry", scope: "auto",
    reads: [], writes: ["normed_in"], deps: [], carryIn: true,
    desc: "本层 x·γ（BF16）。由上一层的 dcr_xgamma 融合产出，layer0 为 x_gamma0。仅 QKV 消费。" },

  // ── stage 1: RMSNorm reduce + QKV seeds + work table ──
  { id: "rms_recip", label: "rms_recip", col: 1, kind: "vec", unit: "AIV", scope: "manual", count: 1,
    reads: ["hidden_states"], writes: ["inv_rms"], deps: ["in_hidden"],
    formula: "inv_rms[b] = 1/√(mean_k x² + ε)",
    desc: "Scope 1a：对 hidden_states 做平方和归约得到每行 1/rms。与 γ 缩放解耦——1/rms 延迟折进 rope，使 x_gamma / QKV 不等归约。stage=4 流水。" },
  { id: "q_seed", label: "q_seed", col: 1, kind: "seed", unit: "AIV", scope: "manual", count: 1,
    reads: [], writes: ["q_proj"], deps: ["in_hidden"],
    desc: "把 q_proj 按 QKV_N_TILE 分块清零，供 split-K 原子加。无显式 dep：靠 q_proj 的 WAR 排在上一层 qk_norm 之后。" },
  { id: "k_seed", label: "k_seed", col: 1, kind: "seed", unit: "AIV", scope: "manual", count: 1,
    reads: [], writes: ["k_proj"], deps: ["in_hidden"], desc: "K 累加器清零。" },
  { id: "v_seed", label: "v_seed", col: 1, kind: "seed", unit: "AIV", scope: "manual", count: 1,
    reads: [], writes: ["v_proj"], deps: ["in_hidden"], desc: "V 累加器清零。" },
  { id: "fa_work_build", label: "fa_work_build", col: 1, kind: "vec", unit: "AIV", scope: "manual", count: 1,
    reads: [], writes: ["work_table"], deps: [],
    formula: "fa_work_table[w] = b·MCB + p",
    desc: "Scope 2 预处理：前缀和游标把 ragged batch 里的真实 seq-block 压成无空洞工作表，并写 fa_total。输入 seq_lens 是外部张量，无 dep。" },

  // ── stage 2: QKV projections ──
  { id: "q_proj", label: "q_proj", col: 2, kind: "cube", unit: "AIC", scope: "manual", count: "10×5",
    reads: ["normed_in"], writes: ["q_proj"], deps: ["q_seed", "in_normed"],
    formula: "q = (x·γ) @ Wq   （1/rms 延迟）",
    desc: "split-K(5) × split-N(10) 的 50 个原子加任务，1:1 对齐 normed slab（QKV_K_SLICE=DOWN_TN）。TM=16 TN=256 TK=256，内层 pipeline。" },
  { id: "k_proj", label: "k_proj", col: 2, kind: "cube", unit: "AIC", scope: "manual", count: "2×5",
    reads: ["normed_in"], writes: ["k_proj"], deps: ["k_seed", "in_normed"],
    formula: "k = (x·γ) @ Wk", desc: "K 投影：2 个 outer-N × 5 split-K 原子加。" },
  { id: "v_proj", label: "v_proj", col: 2, kind: "cube", unit: "AIC", scope: "manual", count: "2×5",
    reads: ["normed_in"], writes: ["v_proj"], deps: ["v_seed", "in_normed"],
    formula: "v = (x·γ) @ Wv", desc: "V 投影：2 个 outer-N × 5 split-K 原子加。" },

  // ── stage 3: QK-norm ──
  { id: "qk_norm", label: "qk_norm", col: 3, kind: "vec", unit: "AIV", scope: "manual", count: 8,
    reads: ["q_proj", "k_proj", "inv_rms"], writes: ["q_norm", "k_norm", "q_inv", "k_inv"], deps: ["q_proj", "k_proj", "rms_recip"],
    formula: "γ 缩放 + 1/√(mean_d x²+ε) 归约（融合，一次读）",
    desc: "Scope 2：每 KV 头一个任务，融合 γ 乘（q/k_proj_norm）与 1/rms 归约（q/k_inv）。recip 延迟折进 rope。控制实验：显式乘 inv_rms 再在两步内抵消，保持与优化路数值一致。" },

  // ── stage 4: RoPE + KV write ──
  { id: "rope_qkv", label: "rope_qkv", col: 4, kind: "vec", unit: "AIV", scope: "manual", count: "32 core",
    reads: ["q_norm", "k_norm", "q_inv", "k_inv", "v_proj", "inv_rms"], writes: ["all_q", "k_cache", "v_cache"],
    deps: ["qk_norm", "v_proj", "rms_recip"],
    formula: "旋转 (lo,hi) 半维；折入延迟的 1/rms、qk_inv",
    desc: "单个 SPMD grid（32 core，NUM_KV_HEADS·BATCH 项）。K/V 经 slot_mapping 写入分页物理页，Q pad 到 Q_HEAD_PAD 行写 all_q_padded。合并了原来的 2 个分组 grid。" },

  // ── stage 5: MLP accumulator seeds (hoisted between rope and attn) ──
  { id: "down_seed", label: "down_seed", col: 5, kind: "seed", unit: "AIV", scope: "manual", count: 1,
    reads: [], writes: ["down_acc"], deps: ["in_hidden"],
    desc: "MLP down 累加器清零。提前到 rope 与 attn 之间，让向量清零与 fa_fused 的 cube/vec 重叠。" },
  { id: "gate_seed", label: "gate_seed", col: 5, kind: "seed", unit: "AIV", scope: "manual", count: 1,
    reads: [], writes: ["gate_acc"], deps: ["in_hidden"], desc: "gate 累加器清零（提前）。" },
  { id: "up_seed", label: "up_seed", col: 5, kind: "seed", unit: "AIV", scope: "manual", count: 1,
    reads: [], writes: ["up_acc"], deps: ["in_hidden"], desc: "up 累加器清零（提前）。" },

  // ── stage 6: fused attention ──
  { id: "fa_fused", label: "fa_fused", col: 6, kind: "mixed", unit: "AIC+AIV", scope: "manual", count: "24 core",
    reads: ["all_q", "k_cache", "v_cache", "work_table"], writes: ["oi_tmp", "cur_mi", "cur_li"],
    deps: ["rope_qkv", "fa_work_build"],
    formula: "scores=QKᵀ·scale → row_max/exp/row_sum → SV",
    desc: "一个混合 cube+vec root，块级稠密静��派发：24 个常驻 core 对工作表 grid-stride，core = w % 24。每步处理一个真实 seq-block(×8 头)，等代价负载均衡（≈1.25×）。gp 循环用 pipeline(stage=2) 让 i+1 的 QK 与 i 的 softmax 重叠。" },

  // ── stage 7: online softmax ──
  { id: "online_softmax", label: "online_softmax", col: 7, kind: "vec", unit: "AIV", scope: "auto", count: "48 core",
    reads: ["oi_tmp", "cur_mi", "cur_li"], writes: ["attn_out"], deps: ["fa_fused"],
    formula: "跨块归并：mi'=max, α/β=exp(mi−mi'), o=Σβ·oᵢ / Σ l",
    desc: "顶层 SPMD（48 core），把每个 (b,kvh) lane 的所有块部分和在线归并成 attn_out。捕获 attn_done_tid 供 manual_scope 的 out_proj 作显式 dep（manual 内 auto-dep 被抑制）。" },

  // ── stage 8: out_proj + seed ──
  { id: "out_seed", label: "out_seed", col: 8, kind: "seed", unit: "AIV", scope: "manual", count: 1,
    reads: [], writes: ["attn_proj"], deps: ["in_hidden"], desc: "attn_proj_fp32 按 OUT_TN 分块清零，供 out_proj 原子加。" },
  { id: "out_proj", label: "out_proj", col: 8, kind: "cube", unit: "AIC", scope: "manual", count: "10×5",
    reads: ["attn_out"], writes: ["attn_proj"], deps: ["out_seed", "online_softmax"],
    formula: "attn_proj = attn_out @ Wo",
    desc: "split-K(5) × split-N(10) = 50 个原子加任务。显式 dep 到 out_seed 与 online_softmax 的 attn_done_tid。" },

  // ── stage 9: residual + post-RMS ──
  { id: "residual_rms_cast", label: "residual_rms_cast", col: 9, kind: "vec", unit: "AIV", scope: "manual", count: 5,
    reads: ["attn_proj", "hidden_states"], writes: ["post_partial", "mlp_norm_in"], deps: ["out_proj"],
    formula: "h1 = attn_proj + hidden；mlp_in = h1 · post_γ",
    desc: "分 5 个 K-slice：写原始残差 h1（post_norm_partial，down 后加回，不做 γ）与 gate/up 输入 mlp_norm_in（h1·post_γ 再转 BF16）。post_γ 是 per-K 无法像 inv_rms 那样延迟。" },
  { id: "post_rms_reduce", label: "post_rms_reduce", col: 9, kind: "vec", unit: "AIV", scope: "manual", count: 1,
    reads: ["attn_proj", "hidden_states"], writes: ["inv_rms_tile"], deps: ["out_proj"],
    formula: "post_inv_rms[b] = 1/√(mean(h1²)+ε)",
    desc: "读全部 attn_proj+hidden 求 post-RMS 的 1/rms，延迟折进 silu。deps=[out_tids]（全部 out_proj）。" },

  // ── stage 10: gate/up ──
  { id: "gate_proj", label: "gate_proj", col: 10, kind: "cube", unit: "AIC", scope: "manual", count: "17×5",
    reads: ["mlp_norm_in"], writes: ["gate_acc"], deps: ["residual_rms_cast", "gate_seed"],
    formula: "gate = mlp_in @ Wgate", desc: "split-K(5) × 17 outer-N 原子加，与 up 交错发射。" },
  { id: "up_proj", label: "up_proj", col: 10, kind: "cube", unit: "AIC", scope: "manual", count: "17×5",
    reads: ["mlp_norm_in"], writes: ["up_acc"], deps: ["residual_rms_cast", "up_seed"],
    formula: "up = mlp_in @ Wup", desc: "split-K(5) × 17 outer-N 原子加。" },

  // ── stage 11: silu ──
  { id: "silu", label: "silu", col: 11, kind: "vec", unit: "AIV", scope: "manual", count: 17,
    reads: ["gate_acc", "up_acc", "inv_rms_tile"], writes: ["mlp_tile"], deps: ["gate_proj", "up_proj", "post_rms_reduce"],
    formula: "mlp = (g·σ(g)) · u，g/u 先乘 post_inv_rms",
    desc: "17 个任务（每 MLP_TN 一个），内层 pipeline 4 个子块。此处才把延迟的 post_inv_rms 乘上。" },

  // ── stage 12: down ──
  { id: "down_proj", label: "down_proj", col: 12, kind: "cube", unit: "AIC", scope: "manual", count: "5×17",
    reads: ["mlp_tile"], writes: ["down_acc"], deps: ["silu", "down_seed"],
    formula: "down = mlp @ Wdown",
    desc: "split-K(17) × split-N(5) = 85 个原子加。TaskId 提升到编排作用域，供 manual_scope 之外的 dcr_xgamma 用 deps= 门控。" },

  // ── stage 13: fused output (outside manual_scope) ──
  { id: "dcr_xgamma", label: "dcr_xgamma", col: 13, kind: "cube", unit: "AIV", scope: "auto", count: 5,
    reads: ["down_acc", "post_partial"], writes: ["out", "normed_out"], deps: ["down_proj"], carryOut: true,
    formula: "out = down_acc + post_partial；normed_out = out · γ_next",
    desc: "在 manual_scope 之外的 auto-dep 区，单个 SPMD(DOWN_ON=5) 派发。同一寄存器内的 FP32 chunk 同时产出 out（残差，供下一层 hidden + rms_recip）与 normed_out（下一层 x·γ，供下一层 QKV），省掉 out_partial 与额外 copy。分块写让 5 个 slab 并行且不互相 WAW。" },

  { id: "out_next", label: "out → 下一层", col: 14, kind: "io", unit: "carry", scope: "auto",
    reads: [], writes: [], deps: ["dcr_xgamma"], carryOut: true,
    desc: "out 成为下一层的 hidden_states，normed_out 成为下一层的 normed_in。prev_out_tids / prev_normed_tids 全部填成 dcr_tid。" },
];

// cross-iteration carry edges (dashed) — dcr feeds next layer's stage-1 consumers.
const CARRY_EDGES = [
  { from: "dcr_xgamma", to: "rms_recip", label: "prev_out_tids → hidden" },
  { from: "dcr_xgamma", to: "q_proj", label: "prev_normed_tids → x·γ" },
];
