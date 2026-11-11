/* CSA Operator Lens · L1 整层 / L2 Kernel 的静态数据。
 * 模块说明的措辞改写自 Design/PyPTO-Compute-Explorer/js/csa-model.js；
 * 行号、Shape 与常量全部按 Data/DeepseekV4/deepseek_v4_flash_dspark/ 当前源码重新核对（2026-11-10）。
 * 默认配置：TP=2，DECODE_BATCH=64 → 每 Rank B=32 个请求，S=1+7=8，T=256；AllGather 后 2T=512。 */
(function () {
  'use strict';

  const CSA = 'decode_csa.py';
  const SPARSE = 'decode_sparse_attn_csa.py';

  const FACTS = {
    tp: 2, B: 32, S: 8, T: 256, KV_T: 512, D: 4096, H: 64, HEAD_DIM: 512, ROPE: 64,
    WIN: 128, IDX_TOPK: 512, SPARSE_BLOCKS: 5, ATTN_K_TILE: 128, NUM_QK_CORES: 24,
    WB_TILE: 8, PUBLISH_WORKERS: 48, PUBLISH_T_TILE: 8, H_TILE: 16, O_GROUPS: 8, LOCAL_O_GROUPS: 4, BLOCK_SIZE: 32
  };

  // ── L1：五个语义阶段 × 四条列（Local T 行 / Gathered 2T 行 / 依赖 / Cache 状态） ──
  const COLUMNS = [
    { id: 'local', x: 186, label: 'LOCAL · T 行', note: '本 Rank 的 query token' },
    { id: 'full', x: 368, label: 'GATHERED · 2T 行', note: '整组 token / Group 归属' },
    { id: 'dep', x: 550, label: 'TASK 依赖', note: 'TaskID 汇合' },
    { id: 'cache', x: 732, label: 'CACHE · 状态', note: '跨 step 持久' }
  ];
  const STAGES = [
    { id: 's1', label: '输入准备', en: 'Input Preparation', y0: 44, y1: 104, lines: [[218, 289]] },
    { id: 's2', label: 'KV 与索引准备', en: 'KV & Index Preparation', y0: 112, y1: 318, lines: [[291, 382]] },
    { id: 's3', label: '稀疏注意力', en: 'Sparse Attention', y0: 326, y1: 392, lines: [[384, 393]] },
    { id: 's4', label: '输出重分布', en: 'Output Redistribution', y0: 400, y1: 466, lines: [[395, 484]] },
    { id: 's5', label: '输出投影', en: 'Output Projection', y0: 474, y1: 540, lines: [[486, 498]] }
  ];

  // kind：op 计算 · comm 跨 Rank 通信 · state Cache / 状态 · sync Task 依赖
  // scope：local / full / group / cache —— 数据归属，读这一层图最重要的一件事
  const MODULES = [
    { id: 'pre', stage: 's1', col: 'local', y: 74, kind: 'op', scope: 'local', label: 'HC Pre · RMSNorm',
      sub: 'x_hc → x_normed_t',
      what: 'HC Pre 混合 4 路 hidden，再做 Attention RMSNorm。',
      why: '为 Q 与 KV 两条 Projection 提供同一份归一化输入。',
      data: 'x_hc [T,4,4096] FP32 → x_mixed [T,4096] BF16 → x_normed_t [T,4096] BF16。',
      how: '两个独立 pl.scope：hc_pre（227–228）与 rms_norm（274–275）。',
      src: [[CSA, 221, 228], [CSA, 273, 275]] },
    { id: 'gather', stage: 's1', col: 'full', y: 74, kind: 'comm', scope: 'full', label: 'Token AllGather',
      sub: '[T,D] × 2 → [2T,D]',
      what: '每个 Rank 收到按 source rank 排序的整组 token 行。',
      why: 'KV 与两路 Compressor 要维护整组请求的 Cache，必须看到全部 token。',
      data: 'x_normed_t [T,4096] × 2 Rank → x_normed_full [2T,4096]，默认 [512,4096]；行 [0,T) 来自 Rank 0，[T,2T) 来自 Rank 1。',
      how: 'decode_cp_token_allgather_step 原地写 x_normed_full：Put → payload wait → readback → 信号回收。',
      src: [[CSA, 277, 289]] },
    { id: 'rope', stage: 's1', col: 'dep', y: 74, kind: 'sync', scope: 'dep', label: 'RoPE 表 · rope_tid',
      sub: 'csa_rope_interleave',
      what: '预先生成 local 与 full 两套交错 cos / sin 表，并用 rope_tid 发布。',
      why: '后续 KV Projection、两路 Compressor 与 Indexer 都要读这些表；late_dep 把它们挂在 rope_tid 之后。',
      data: 'idx_cos_il / idx_sin_signed [T,64] FP32；cmp_cos_il_full / cmp_sin_signed_full [2T,64] FP32。',
      how: 'pl.at(CORE_GROUP) 单任务；298 行 late_dep = task_dummy(deps=[rope_tid]) 传给 4 个消费者。',
      src: [[CSA, 230, 271], [CSA, 298, 298]],
      kernels: ['rope_interleave'] },

    { id: 'q', stage: 's2', col: 'local', y: 140, kind: 'op', scope: 'local', label: 'Q Projection',
      sub: 'q · qr · qr_scale',
      what: '本地 x_normed_t 生成 Q，以及给 Indexer 用的量化 qr。',
      why: 'query 只服务本 Rank 的 token，不需要扩成 2T 行。',
      data: 'q [T,64,512] BF16；qr [T,1024] INT8；qr_scale [T,1] FP32。',
      how: 'rope_prepare(local) + q_proj_rope。它在 AllGather 之后的 scope 里调用——local 输入不等于已经和 AllGather 重叠。',
      src: [[CSA, 304, 313]] },
    { id: 'kv', stage: 's2', col: 'full', y: 140, kind: 'op', scope: 'full', label: 'KV Projection',
      sub: 'gathered → kv_full',
      what: '用 gathered token stream 生成 kv_full。',
      why: '每个 Rank 都要为整组请求更新自己那份 Original KV Cache。',
      data: 'kv_full [2T,512] BF16，使用 full-stream RoPE。',
      how: 'rope_prepare(full) + kv_proj_rope(…, late_dep)。',
      src: [[CSA, 299, 302], [CSA, 315, 319]] },
    { id: 'write', stage: 's2', col: 'dep', y: 140, kind: 'op', scope: 'full', label: 'KV Writeback',
      sub: 'SPMD × 64 · 写 Original',
      what: '按 slot mapping 把 kv_full 写进 Original KV Cache。',
      why: 'slot mapping 把这一步的 token 对齐到请求的分页 Cache；slot < 0 的行跳过。',
      data: 'ori_slot_mapping [2T] INT64；kv_cache 按 [块数×32, 512] 展平后逐行写。',
      how: 'pl.spmd(kv_wb_blocks = 2T / 8 = 64)，每个 block 写 8 个 token，产生 ori_cache_write_tid。',
      src: [[CSA, 321, 333]],
      kernels: ['cache_writeback'] },
    { id: 'original', stage: 's2', col: 'cache', y: 140, kind: 'state', scope: 'cache', label: 'Original KV Cache',
      sub: 'kv_cache · BF16',
      what: '未压缩的近期 KV，按页存放。',
      why: 'Sparse Attention 的滑动窗口块（SB0）从这里读。',
      data: 'kv_cache [N_ori, 32, 1, 512] BF16；BLOCK_SIZE=32 行一页。',
      how: 'KV Writeback 写；qk_pv 用 window_swa_indices 成段读。',
      src: [[CSA, 186, 186], [CSA, 321, 333]] },

    { id: 'indexer', stage: 's2', col: 'local', y: 206, kind: 'op', scope: 'local', label: 'Indexer',
      sub: 'Top-k = 512',
      what: '本地 query 对索引 Cache 打分，选出 Top-k 压缩位置。',
      why: '主注意力之前先挑出值得看的压缩上下文。',
      data: 'idx_topk [T,512] INT32；idx_topk_scores [T,512] FP32。',
      how: '读 x_normed_t、qr、qr_scale；显式等待 idx_cache_write_tid。',
      src: [[CSA, 375, 382]] },
    { id: 'main', stage: 's2', col: 'full', y: 206, kind: 'op', scope: 'full', label: 'Main Compressor',
      sub: '4:1 压缩 · 写 Compressed',
      what: '对 gathered stream 做 4:1 压缩，更新压缩 KV Cache。',
      why: '为长上下文提供压缩 KV；与 Original Writeback 是两条独立的依赖分支。',
      data: 'compress_state FP32（按请求组织）；cmp_kv BF16。',
      how: 'compressor_ratio4(…, late_dep) 返回 cmp_cache_write_tid。',
      src: [[CSA, 335, 360]] },
    { id: 'ready', stage: 's2', col: 'dep', y: 206, kind: 'sync', scope: 'dep', label: 'cache_ready_dep',
      sub: 'ori + cmp 写入汇合',
      what: 'Original 与 Main 压缩两路 Cache 写入的汇合点。',
      why: '只等其中一路就读 KV 会读到旧数据。Indexer 另有自己的 idx_cache_write_tid——三种依赖不是一个全局 barrier。',
      data: 'task_dummy(deps=[ori_cache_write_tid, cmp_cache_write_tid])。',
      how: '作为参数传进 sparse_attn_csa，最终挂在 qk_pv 的 deps 上（decode_sparse_attn_csa.py 208 行）。',
      src: [[CSA, 361, 361], [SPARSE, 208, 208]] },
    { id: 'compressed', stage: 's2', col: 'cache', y: 206, kind: 'state', scope: 'cache', label: 'Compressed KV Cache',
      sub: 'cmp_kv · BF16',
      what: 'Main Compressor 维护的压缩 KV。',
      why: 'qk_pv 的 SB1–SB4 按 Indexer 选出的 Top-k 从这里逐行读。',
      data: 'cmp_kv [N_cmp, 32, 1, 512] BF16；cmp_block_table [B, CMP_MAX_BLOCKS]。',
      how: 'Main Compressor 写；qk_pv 查块表后逐行 gather。',
      src: [[CSA, 187, 188], [CSA, 352, 360]] },

    { id: 'idxcomp', stage: 's2', col: 'full', y: 272, kind: 'op', scope: 'full', label: 'Indexer Compressor',
      sub: '写 Indexer Cache',
      what: '为索引检索生成独立的量化 Cache。',
      why: '索引特征和主注意力 KV 不是同一份数据，不能画成一个 Cache。',
      data: 'idx_kv_cache INT8 + idx_kv_scale FP32；inner_compress_state FP32。',
      how: '消费 x_normed_full；返回 idx_cache_write_tid，显式传给 Indexer。',
      src: [[CSA, 363, 374]] },
    { id: 'indexcache', stage: 's2', col: 'cache', y: 272, kind: 'state', scope: 'cache', label: 'Indexer KV Cache',
      sub: 'INT8 + FP32 scale',
      what: 'Indexer 打分用的压缩量化索引。',
      why: '它只负责挑位置，本身不是注意力要读的 KV 值。',
      data: 'idx_kv_cache [N_idx, 32, 1, 128] INT8；idx_kv_scale [N_idx, 32, 1, 1] FP32。',
      how: 'Indexer Compressor 写，Indexer 读。',
      src: [[CSA, 189, 191], [CSA, 366, 374]] },

    { id: 'attention', stage: 's3', col: 'local', y: 359, kind: 'op', scope: 'local', label: 'Sparse Attention',
      sub: '4 个 Kernel · 可下钻', drill: true,
      what: '本地 Q 同时看滑动窗口（Original）和 Top-k 压缩位置（Compressed）。',
      why: 'query 归属仍是本地 token；输出随后按 O Group 重新分配给各 Rank。',
      data: '输入 q、kv_cache、cmp_kv、idx_topk；输出每个 (token, 头块, 稀疏块) 的部分结果 m / l / o。',
      how: 'sparse_attn_csa：kv_touch → qk_plan → qk_pv（SPMD × 24）→ rope_cs，等待 cache_ready_dep，返回 qk_tid 与 rope_tid。',
      src: [[CSA, 384, 393], [SPARSE, 105, 358]],
      kernels: ['kv_touch', 'qk_plan', 'qk_pv', 'rope_cs'] },

    { id: 'publish', stage: 's4', col: 'local', y: 433, kind: 'comm', scope: 'local', label: 'Merge · Publish',
      sub: 'SPMD × 48 · Put+Notify',
      what: '合并 5 个稀疏块的部分结果，归一化、做输出 RoPE，再按 O Group 发给目标 Rank。',
      why: '每个 Rank 只持有 4 个 O Group 的权重，所以本地 token 的各 Head 结果要送到对应 Rank。',
      data: 'attention_grouped [8×T_PAD, 4096] BF16 → 对端 attention_window。',
      how: 'pl.spmd(48, deps=[qk_tid, attn_rope_tid])；grid-stride 处理 (T/8)×(64/16)=128 个打包任务；Put 与 Notify 融合在生产者里。',
      src: [[CSA, 395, 477]],
      kernels: ['merge_publish'] },
    { id: 'a2a', stage: 's4', col: 'full', y: 433, kind: 'comm', scope: 'group', label: 'All-to-All 完成',
      sub: 'o_group_a2a',
      what: '完成 Publish 已经发起的 Group 重分布。',
      why: '这不是再发一次 payload：它等待对端信号、读回本 Rank 的 Group 窗口并回收信号。',
      data: 'attention_local_flat → [4, 2·T_PAD, 4096] BF16，即本 Rank 4 个 Group × 整组 token。',
      how: 'o_group_a2a(…, publish_tid, 48)：等待 → 本地 gather → 完成 / 回收。',
      src: [[CSA, 479, 486]] },

    { id: 'oproj', stage: 's5', col: 'full', y: 507, kind: 'comm', scope: 'group', label: 'O Proj · ReduceScatter',
      sub: '部分和 → owner FP32 累加',
      what: '用本 Rank 的 4 个 O Group 权重算出所有 token 的部分输出，再送回 token owner 累加。',
      why: '分组权重的贡献要跨 Rank 相加才是完整 hidden；ReduceScatter 既 reduce 又 scatter，不只是切 token。',
      data: 'wo_a [4,1024,4096] BF16；wo_b [4096, 4096] INT8；attn_out [T,4096] BF16。',
      how: 'o_proj_reduce_scatter 原地写 attn_out：Put / Notify → 等待 → owner 按来源 Rank 累加 → 回收。',
      src: [[CSA, 487, 495]] },
    { id: 'post', stage: 's5', col: 'local', y: 507, kind: 'op', scope: 'local', label: 'HC Post',
      sub: '→ x_out [T,4,4096]',
      what: '把本地注意力输出与 HC residual 混合，得到本层输出。',
      why: '回到本 Rank 的 local token 边界。',
      data: 'x_out [T,4,4096] FP32。',
      how: '独立 pl.scope，位于 O Proj / ReduceScatter 之后。',
      src: [[CSA, 497, 499]] }
  ];

  // [from, to, type]：data 数据流 · sync Task 依赖 · comm 跨 Rank
  const EDGES = [
    ['pre', 'gather', 'comm'], ['pre', 'q', 'data'], ['pre', 'indexer', 'data'],
    ['gather', 'kv', 'data'], ['gather', 'main', 'data'], ['gather', 'idxcomp', 'data'],
    ['rope', 'kv', 'sync'], ['rope', 'main', 'sync'],
    ['kv', 'write', 'data'], ['write', 'original', 'data'], ['write', 'ready', 'sync'],
    ['main', 'compressed', 'data'], ['main', 'ready', 'sync'],
    ['idxcomp', 'indexcache', 'data'], ['idxcomp', 'indexer', 'sync'], ['indexcache', 'indexer', 'data'],
    ['q', 'indexer', 'data'], ['q', 'attention', 'data'], ['indexer', 'attention', 'data'],
    ['original', 'attention', 'data'], ['compressed', 'attention', 'data'], ['ready', 'attention', 'sync'],
    ['attention', 'publish', 'sync'], ['publish', 'a2a', 'comm'], ['a2a', 'oproj', 'data'], ['oproj', 'post', 'comm']
  ];

  // ── L2：Kernel（一次派发）。dispatch：spmd 有 Lane 视图，at 是 CORE_GROUP 单任务 ──
  const KERNELS = {
    rope_interleave: { id: 'rope_interleave', name: 'csa_rope_interleave', module: 'rope', file: CSA, line: 234, range: [234, 271],
      dispatch: 'at', level: 'CORE_GROUP', produces: 'rope_tid', deps: [],
      what: 'pl.at(CORE_GROUP)：整段在一个 core group 上作为一个任务执行，不做 SPMD 拆分。',
      loops: 'pl.range(0, t_dim, 4) 与 pl.range(0, kv_dim, 4)：每次 4 行，生成 local / full 两套交错 cos / sin。' },
    cache_writeback: { id: 'cache_writeback', name: 'csa_cache_writeback', module: 'write', file: CSA, line: 323, range: [321, 333],
      dispatch: 'spmd', lanes: 64, laneExpr: 'kv_wb_blocks = kv_dim // 8 = 512 / 8', produces: 'ori_cache_write_tid', deps: [],
      what: '每个 block 负责连续 8 个 token：wb_t0 = block × 8，块内 pl.range(8) 逐行写。',
      lineMap: { lane: [324, 325], item: [326, 327], valid: [328, 333] } },
    merge_publish: { id: 'merge_publish', name: 'csa_merge_pack_publish', module: 'publish', file: CSA, line: 397, range: [395, 477],
      dispatch: 'spmd', lanes: 48, laneExpr: 'ATTENTION_PUBLISH_WORKERS = 48', produces: 'publish_tid', deps: ['qk_tid', 'attn_rope_tid'],
      what: 'grid-stride：worker w 依次处理打包任务 w、w+48、w+96；共 (T/8) × (64/16) = 128 个，每个是 8 token × 16 头。',
      lineMap: { lane: [402, 403], item: [404, 411], valid: [412, 467] } },
    kv_touch: { id: 'kv_touch', name: 'kv_touch', module: 'attention', file: SPARSE, line: 131, range: [127, 132],
      dispatch: 'at', level: 'CORE_GROUP', produces: null, deps: [],
      what: 'WAR 标记：一次 1 行的自拷贝，让 ori_kv 带上 add_inout，保住外层 KV 写回与 qk_pv 读取之间的依赖边。',
      loops: '无循环，allow_early_resolve=True。' },
    qk_plan: { id: 'qk_plan', name: 'csa_slots_build_valid_qk_plan', module: 'attention', file: SPARSE, line: 143, range: [134, 196],
      dispatch: 'at', level: 'CORE_GROUP', produces: 'qk_plan_tid', deps: [],
      what: '算出 sparse_bias、valid_block_mask，并把 T×5 个 (token, 稀疏块) 排成 qk_order：非空块在前，空块追加在尾部（不删除）。',
      loops: '按 8 行 token tile 生成掩码与 bias；Pass 1（183 行）写非空块，Pass 2（190 行）追加空块。' },
    qk_pv: { id: 'qk_pv', name: 'qk_pv', module: 'attention', file: SPARSE, line: 208, range: [198, 313],
      dispatch: 'spmd', lanes: 24, laneExpr: 'NUM_QK_CORES = 24（= a2a3 的 AIC 数）', produces: 'qk_tid', deps: ['qk_plan_tid', 'cache_ready_dep'], deep: true,
      what: '一次 pl.spmd(24) 派发；每条 Lane 从 qk_order[lane] 起，每次跨 24 领下一个 (token, 稀疏块)。',
      lineMap: { lane: [209, 214], item: [215, 221], valid: [222, 224], invalid: [305, 313], body: [225, 304] } },
    rope_cs: { id: 'rope_cs', name: 'rope_cs', module: 'attention', file: SPARSE, line: 322, range: [315, 346],
      dispatch: 'at', level: 'CORE_GROUP', produces: 'rope_tid', deps: [],
      what: '为 merge_norm 的输出 RoPE 生成 cos / sin 与 j^1 交换索引。',
      loops: 'pl.range(rope_cs_blocks)：每块 8 行。' }
  };
  // 各模块下的 Kernel 链（L2 上方的派发链）
  const CHAINS = {
    attention: { file: SPARSE, range: [105, 358], order: ['kv_touch', 'qk_plan', 'qk_pv', 'rope_cs'],
      edges: [['qk_plan', 'qk_pv', 'qk_plan_tid']], external: [['cache_ready_dep', 'qk_pv'], ['qk_pv', 'qk_tid →'], ['rope_cs', 'rope_tid →']] },
    write: { file: CSA, range: [151, 499], order: ['cache_writeback'], edges: [], external: [['cache_writeback', 'ori_cache_write_tid →']] },
    publish: { file: CSA, range: [151, 499], order: ['merge_publish'], edges: [], external: [['qk_tid', 'merge_publish'], ['merge_publish', 'publish_tid →']] },
    rope: { file: CSA, range: [151, 499], order: ['rope_interleave'], edges: [], external: [['rope_interleave', 'rope_tid →']] }
  };

  // ── 源码行内标签（L1 / L2）：PTO 框架族（作用域、派发、依赖）与 GM 族（跨 Rank 通信） ──
  const TAGS = {};
  TAGS[CSA] = {
    227: ['SCOPE · hc_pre', 'fw', 'pl.scope：把内联体外提成独立的作用域；作用域之间按数据依赖排队。'],
    234: ['AT · CORE_GROUP', 'fw', 'pl.at(level=CORE_GROUP)：整段作为一个任务在一个 core group 上执行，as rope_tid 捕获它的 TaskId。'],
    274: ['SCOPE · rms_norm', 'fw', 'pl.scope：RMSNorm 独立作用域。'],
    281: ['SCOPE · AllGather', 'fw', 'pl.scope：AllGather 独立作用域。'],
    285: ['COMM · AllGather', 'gm', 'Token AllGather：本 Rank T 行 → 整组 2T 行，原地写 x_normed_full。'],
    297: ['SCOPE · 主体', 'fw', 'pl.scope：Projection、Cache、Indexer、Attention、Publish、O Proj 都在这一个作用域里，靠 TaskId 显式排依赖。'],
    298: ['DEP · late_dep', 'fw', 'task_dummy(deps=[rope_tid])：一个只承载依赖的空任务，传给 4 个消费者。'],
    309: ['CALL · Local T', 'fw', 'q_proj_rope：只处理本 Rank 的 T 行。'],
    315: ['CALL · Gathered 2T', 'fw', 'kv_proj_rope：处理 AllGather 后的 2T 行。'],
    323: ['SPMD × 64', 'fw', 'pl.spmd(kv_wb_blocks)：2T / 8 = 64 个 block，每个写 8 个 token；as ori_cache_write_tid。'],
    353: ['CALL · Gathered 2T', 'fw', 'compressor_ratio4：在 2T 行上做 4:1 压缩，返回 cmp_cache_write_tid。'],
    361: ['DEP · cache_ready', 'fw', 'task_dummy(deps=[ori_cache_write_tid, cmp_cache_write_tid])：两路 Cache 写入汇合。'],
    366: ['CALL · Gathered 2T', 'fw', 'indexer_compressor：返回 idx_cache_write_tid。'],
    375: ['CALL · Local T', 'fw', 'indexer：本地 query 打分，等待 idx_cache_write_tid。'],
    388: ['CALL · 4 Kernel ↘', 'fw', 'sparse_attn_csa：kv_touch → qk_plan → qk_pv → rope_cs。点 Sparse Attention 可下钻到 L2。'],
    397: ['SPMD × 48', 'fw', 'pl.spmd(ATTENTION_PUBLISH_WORKERS=48, deps=[qk_tid, attn_rope_tid])。'],
    403: ['RANGE · 步长 48', 'fw', 'grid-stride：worker 从自己的编号起，每次跨 48。'],
    458: ['COMM · Put', 'gm', 'pld.tensor.put：把打包好的 Group 行直接写进目标 Rank 的 attention_window。'],
    471: ['COMM · Notify', 'gm', 'pld.system.notify(AtomicAdd)：告诉对端本 Rank 已发布完。'],
    479: ['COMM · All-to-All', 'gm', 'o_group_a2a：等待对端信号、读回本 Rank 的 Group 窗口、回收信号。'],
    489: ['COMM · ReduceScatter', 'gm', 'o_proj_reduce_scatter：部分和送回 token owner，owner 做 FP32 累加。'],
    497: ['SCOPE · hc_post', 'fw', 'pl.scope：HC Post 独立作用域。']
  };
  TAGS[SPARSE] = {
    131: ['AT · kv_touch', 'fw', 'pl.at(CORE_GROUP)：WAR 标记任务，没有输出 TaskId。'],
    143: ['AT · qk_plan', 'fw', 'pl.at(CORE_GROUP) as qk_plan_tid：生成掩码、bias 与 qk_order。'],
    183: ['PASS 1 · 非空块', 'fw', '先把 valid_block_mask > 0 的 (token, 稀疏块) 依次写进 qk_order。'],
    190: ['PASS 2 · 空块追加', 'fw', '再把空块追加到尾部；空块不删除，qk_pv 仍要为它写 −inf / 0。'],
    208: ['SPMD × 24 · qk_pv', 'fw', 'pl.spmd(24, deps=[qk_plan_tid, cache_ready_dep]) as qk_tid：一次派发、24 条逻辑 Lane，不是 24 次 Submit。'],
    209: ['Lane 号', 'fw', 'pl.tile.get_block_idx()：拿当前 Lane（block）号。'],
    214: ['RANGE · 步长 24', 'fw', '每条 Lane 从自己的编号起，每次跨 24 领下一个 Work Item。'],
    216: ['READ · qk_order', 'gm', '从 GM 读排序后的 Work Item 编号：item = token × 5 + 稀疏块。'],
    224: ['分支 · 非空块', 'fw', '运行期标量分支：非空块走 gather + 两次 matmul；305 行 else 是空块。'],
    281: ['PIPELINE · stage=2', 'fw', 'pl.pipeline：两个头块乒乓在飞。细节在 L3 / L4。'],
    305: ['分支 · 空块', 'fw', '空块不做 matmul，直接写 mi=−3e38、li=0、oi=0。'],
    322: ['AT · rope_cs', 'fw', 'pl.at(CORE_GROUP) as rope_tid：输出 RoPE 的 cos / sin 表。']
  };

  // ── qk_pv Work Item 规则：取自 csa-qk-work.js 的纯函数，输入为 Synthetic ──
  // 4 类上下文长度；两个 Rank 用不同的组合，只为示意分布差异（Synthetic）
  const POSITIONS = [[7, 1023, 2047, 4088], [63, 3071, 511, 1535]];   // 远小于 FLASH 的 max_position_embeddings（1,048,576），只为覆盖短 / 中上下文
  function syntheticValid(position, sb) {
    if (sb === 0) return { valid: true, kv: Math.min(FACTS.WIN, position + 1), reason: 'SB0：滑动窗口块，qk_plan 固定写 1' };
    const upper = Math.floor((position + 1) / 4), count = Math.min(FACTS.IDX_TOPK, upper);
    const lo = (sb - 1) * FACTS.ATTN_K_TILE, n = Math.max(0, Math.min(count - lo, FACTS.ATTN_K_TILE));
    return n > 0
      ? { valid: true, kv: n, reason: 'Top-k 在本块有 ' + n + ' 个合法压缩位置' }
      : { valid: false, kv: 0, reason: '上下文只有 ' + upper + ' 个压缩位置，落不到本块' };
  }
  function buildQkScenario(rank) {
    rank = rank ? 1 : 0;
    const items = [];
    for (let t = 0; t < FACTS.T; t++) {
      const req = Math.floor(t / FACTS.S), pos = POSITIONS[rank][req % 4] + (t % FACTS.S);
      for (let sb = 0; sb < FACTS.SPARSE_BLOCKS; sb++) {
        const v = syntheticValid(pos, sb);
        items.push({ id: t * FACTS.SPARSE_BLOCKS + sb, token: t, req: req, rank: rank, pos: pos, sb: sb, valid: v.valid, kv: v.kv, reason: v.reason });
      }
    }
    const queue = items.filter(function (i) { return i.valid; }).concat(items.filter(function (i) { return !i.valid; }));
    const N = FACTS.NUM_QK_CORES;
    queue.forEach(function (it, q) { it.q = q; it.lane = q % N; it.iter = Math.floor(q / N); });
    const lanes = [];
    for (let l = 0; l < N; l++) lanes.push({ index: l, items: queue.filter(function (it) { return it.lane === l; }) });
    return { items: items, queue: queue, lanes: lanes, validCount: items.filter(function (i) { return i.valid; }).length, iters: Math.ceil(queue.length / N) };
  }

  window.LensData = {
    FILES: { CSA: CSA, SPARSE: SPARSE }, FACTS: FACTS, COLUMNS: COLUMNS, STAGES: STAGES, MODULES: MODULES, EDGES: EDGES,
    KERNELS: KERNELS, CHAINS: CHAINS, TAGS: TAGS, buildQkScenario: buildQkScenario
  };
})();
