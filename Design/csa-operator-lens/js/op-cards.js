/* CSA Operator Lens · 右侧详情的“算子卡片”数据：功能摘要、数学表达式、输入输出示例、支持的场景及限制。
 * 依据（2026-11-10 核对）：
 *   数学 —— 各模块文件里的 golden_* torch 参考实现（hc_pre / rmsnorm / qkv_proj_rope / decode_compressor_ratio4 /
 *           decode_indexer / decode_sparse_attn_csa / decode_o_proj / hc_post）与 decode_csa.py 的 golden_decode_csa_tp1；
 *   形状 —— decode_csa.py 签名 + config.py 的 FLASH 预设（decode_csa.py 以 FLASH as M 导入）；示例取默认 TP=2、T=256；
 *   场景 —— decode_csa.py 命令行 --platform / --tp 的 choices；
 *   限制 —— 各文件模块级的 raise ValueError 与写死的常量。
 * 公式用简化 TeX 写，lens.js 渲染成排版后的公式。 */
(function (root) {
  'use strict';
  const CSA = 'decode_csa.py', SP = 'decode_sparse_attn_csa.py';

  // ── 张量：sym 是源码里的形状写法，dims 是默认配置（TP=2，T=256，2T=512）下的数 ──
  // scope：local 本 Rank T 行 · full AllGather 后 2T 行 · cache 跨 step 状态 · weight 权重 · meta 索引 / 标量 · tid TaskId
  const T = {
    x_hc: { sym: 'T, 4, D', dims: [256, 4, 4096], dt: 'FP32', sc: 'local' },
    hc_fn: { name: 'hc_attn_fn', sym: '24, 4D', dims: [24, 16384], dt: 'FP32', sc: 'weight' },
    x_mixed: { sym: 'T, D', dims: [256, 4096], dt: 'BF16', sc: 'local' },
    post_comb: { name: 'post · comb', sym: 'T, 4 | T, 16', dims: [256, 20], dt: 'FP32', sc: 'local' },
    x_normed_t: { sym: 'T, D', dims: [256, 4096], dt: 'BF16', sc: 'local' },
    x_normed_full: { sym: '2T, D', dims: [512, 4096], dt: 'BF16', sc: 'full', seg: [['Rank 0', .5], ['Rank 1', .5]] },
    rope_tab: { name: 'cos / sin 表', sym: 'T | 2T, 64', dims: [512, 64], dt: 'FP32', sc: 'meta' },
    wq: { name: 'wq_a · wq_b', sym: 'D, 1024 · 1024, 64·512', dims: [1024, 32768], dt: 'INT8', sc: 'weight' },
    q: { name: 'q', sym: 'T, 64, 512', dims: [256, 64, 512], dt: 'BF16', sc: 'local' },
    qr: { name: 'qr · qr_scale', sym: 'T, 1024', dims: [256, 1024], dt: 'INT8', sc: 'local' },
    wkv: { sym: 'D, 512', dims: [4096, 512], dt: 'BF16', sc: 'weight' },
    kv_full: { sym: '2T, 512', dims: [512, 512], dt: 'BF16', sc: 'full', seg: [['Rank 0', .5], ['Rank 1', .5]] },
    slot: { name: 'ori_slot_mapping', sym: '2T', dims: [512], dt: 'INT64', sc: 'meta' },
    kv_cache: { sym: 'N_ori, 32, 1, 512', dims: [512, 32, 512], dt: 'BF16', sc: 'cache' },
    cmp_w: { name: 'cmp_wkv · cmp_wgate', sym: '1024, D', dims: [1024, 4096], dt: 'BF16', sc: 'weight' },
    cmp_state: { name: 'compress_state', sym: 'N, 2, 2048', dims: [2, 2048], dt: 'FP32', sc: 'cache' },
    cmp_kv: { sym: 'N_cmp, 32, 1, 512', dims: [256, 32, 512], dt: 'BF16', sc: 'cache' },
    inner_w: { name: 'inner_wkv · inner_wgate', sym: '256, D', dims: [256, 4096], dt: 'BF16', sc: 'weight' },
    idx_cache: { name: 'idx_kv_cache', sym: 'N_idx, 32, 1, 128', dims: [256, 32, 128], dt: 'INT8', sc: 'cache' },
    idx_w: { name: 'idx_wq_b · weights_proj', sym: '1024, 64·128 · D, 64', dims: [1024, 8192], dt: 'INT8', sc: 'weight' },
    idx_topk: { sym: 'T, 512', dims: [256, 512], dt: 'INT32', sc: 'meta' },
    swa: { name: 'window_swa_indices', sym: 'T, 128', dims: [256, 128], dt: 'INT32', sc: 'meta' },
    tids: { name: 'ori / cmp 写入 TaskId', sym: '2', dims: [2], dt: 'TASK_ID', sc: 'tid' },
    ready: { name: 'cache_ready_dep', sym: '1', dims: [1], dt: 'TASK_ID', sc: 'tid' },
    part: { name: 'm · l · o 分块', sym: 'T·4·5·16, 512', dims: [81920, 512], dt: 'FP32', sc: 'local', seg: [['SB0', .2], ['SB1', .2], ['SB2', .2], ['SB3', .2], ['SB4', .2]] },
    sink: { name: 'attn_sink', sym: '64', dims: [64], dt: 'FP32', sc: 'weight' },
    grouped: { name: 'attention_grouped', sym: '8·T_PAD, 4096', dims: [2048, 4096], dt: 'BF16', sc: 'local', seg: [['G0–3 → R0', .5], ['G4–7 → R1', .5]] },
    window: { name: 'attention_window', sym: '4·2T_PAD, 4096', dims: [2048, 4096], dt: 'BF16', sc: 'full', seg: [['来自 R0', .5], ['来自 R1', .5]] },
    local_groups: { name: 'attention_local_groups', sym: '4, 2T_PAD, 4096', dims: [4, 512, 4096], dt: 'BF16', sc: 'full' },
    wo: { name: 'wo_a · wo_b', sym: '4, 1024, 4096 · D, 4096', dims: [4096, 4096], dt: 'INT8', sc: 'weight' },
    attn_out: { sym: 'T, D', dims: [256, 4096], dt: 'BF16', sc: 'local' },
    x_out: { sym: 'T, 4, D', dims: [256, 4, 4096], dt: 'FP32', sc: 'local' },
    // qk_pv 的一个 Work Item（Tile 级）
    q_tile: { name: 'Q 块', sym: '32, 512', dims: [32, 512], dt: 'BF16', sc: 'local' },
    kv_tile: { name: 'KV 块（L1）', sym: '128, 512', dims: [128, 512], dt: 'BF16', sc: 'cache' },
    bias: { name: 'sparse_bias', sym: '1, 128', dims: [1, 128], dt: 'FP32', sc: 'meta' },
    mlo: { name: 'mi · li · oi', sym: '32, 1 | 32, 512', dims: [32, 514], dt: 'FP32', sc: 'local' }
  };
  Object.keys(T).forEach(function (k) { if (!T[k].name) T[k].name = k; });

  // ── 场景（命令行与常量）：[行名, [[选项, 'def' 默认 | 'on' 支持 | 'off' 不支持, 说明]]] ──
  const LAYER_SCEN = [
    ['平台', [['a2a3', 'def', '默认'], ['a2a3sim', 'on'], ['a5', 'on'], ['a5sim', 'on']], '--platform 的 choices；本 Demo 的分配数据只来自 a2a3 编译产物'],
    ['TP', [['1', 'on'], ['2', 'def', '默认'], ['4', 'on'], ['8', 'off', '_TP_CHOICES 只允许 1 / 2 / 4']], 'decode_csa.py 20–35 行，导入时按 TP 固化形状'],
    ['阶段', [['Decode', 'def'], ['Prefill', 'off', '另见 prefill_csa.py']], ''],
    ['压缩比', [['4 · CSA', 'def'], ['128 · HCA', 'off', '另见 decode_hca.py']], 'COMPRESS_RATIO = 4（decode_csa.py 123 行）']
  ];
  const L = function (kind, text, check, src) { return { kind: kind, text: text, check: check || '', src: src || null }; };

  // ── 卡片 ──
  const CARDS = {
    pre: {
      summary: '把 4 路 HC 残差流按学到的权重混成 1 路，再做 Attention 前的 RMSNorm；顺带算出 HC Post 要用的 post / comb 系数。',
      flow: { in: ['x_hc', 'hc_fn'], out: ['x_normed_t', 'post_comb'] },
      math: [
        { k: '混合系数', f: 'm = \\frac{x W_{fn}}{\\sqrt{\\text{mean}(x^2) + \\varepsilon}}' },
        { k: 'pre / post', f: 'pre = \\sigma(s_0 m_{pre} + b) + \\varepsilon,\\quad post = 2\\sigma(s_1 m_{post} + b)' },
        { k: 'comb', f: 'comb = \\text{Sinkhorn}_{20}(\\text{softmax}(s_2 m_{comb} + b))' },
        { k: '输出', f: '\\hat{x} = \\text{RMSNorm}(\\sum_{i=1}^{4} pre_i \\odot x_i)\\odot w' }
      ],
      sym: [['x_i', 'x_hc 第 i 路'], ['W_fn', 'hc_attn_fn'], ['s, b', 'hc_attn_scale / base']],
      ex: '每 Rank 32 个请求 × 8 个 token = 256 行；4 路各 4096 维混成 1 路。',
      scen: [['HC 路数', [['4', 'def', 'hc_mult']], '']],
      limits: [L('fix', 'Sinkhorn 迭代固定 20 次', 'hc_sinkhorn_iters = 20', ['hc_pre.py', 26]), L('run', 'HC 残差流全程 FP32，x_mixed 以 rint 转 BF16', '', ['hc_pre.py', 336])]
    },
    rope: {
      summary: '预先把 RoPE 的 cos / sin 交错展开、sin 按奇偶取符号，生成 local（T 行）与 full（2T 行）两套表。',
      flow: { in: ['x_normed_t'], out: ['rope_tab'] },
      math: [{ k: '交错展开', f: '\\cos_{il}[j] = \\cos[\\lfloor j/2 \\rfloor],\\quad \\sin_{il}[j] = (-1)^{j+1}\\sin[\\lfloor j/2 \\rfloor]' }],
      sym: [['j', '0 … 63 的 RoPE 维']],
      ex: 'RoPE 维 64；local 表 256 行，full 表 512 行。',
      scen: [],
      limits: [L('fix', 'RoPE 维固定 64（qk_rope_head_dim）'), L('run', '以 pl.at(CORE_GROUP) 单任务执行，用 rope_tid 发布', '', [CSA, 234])]
    },
    gather: {
      summary: '把各 Rank 的 T 行归一化 token 拼成整组 2T 行，每个 Rank 都拿到同一份完整副本，顺序是 [Rank 0 | Rank 1]。',
      flow: { in: ['x_normed_t'], out: ['x_normed_full'] },
      math: [{ k: 'AllGather', f: 'X_{full} = \\text{concat}_{rows}(X^{(0)}, X^{(1)}) \\in \\mathbb{R}^{2T \\times D}' }],
      sym: [['X^{(r)}', 'Rank r 的 x_normed_t']],
      ex: '两块 [256, 4096] 按 Rank 顺序叠成 [512, 4096]。',
      scen: [],
      limits: [L('run', '先 Put 到两个 Rank，再等 payload 信号、读回、回收信号', 'push → wait → readback → retire', ['decode_cp_token_allgather.py', 94])]
    },
    q: {
      summary: '本 Rank 的 T 行生成 Q：低秩 LoRA 投影 + RMSNorm，中间结果量化成 INT8 交给 Indexer 复用，最后对 RoPE 维旋转。',
      flow: { in: ['x_normed_t', 'wq'], out: ['q', 'qr'] },
      math: [
        { k: '低秩', f: 'q_r = \\text{RMSNorm}(x W_{qa})\\odot\\gamma_{cq}' },
        { k: '量化', f: '(q_r^{i8}, s) = \\text{quant}_{row}(q_r),\\quad q = (q_r^{i8} W_{qb}^{i8})\\cdot s \\cdot s_w' },
        { k: '每头归一化 + RoPE', f: 'q \\leftarrow \\frac{q}{\\text{rms}(q)},\\quad q_{448:512} \\leftarrow \\text{RoPE}(q_{448:512})' }
      ],
      sym: [['W_qa / W_qb', 'wq_a / wq_b'], ['s_w', 'wq_b_scale']],
      ex: '256 × 4096 → 低秩 1024 → 64 头 × 512 维；NoPE 448 + RoPE 64。',
      scen: [['权重精度', [['wq_a BF16', 'def'], ['wq_b INT8', 'def']], '']],
      limits: [L('fix', '每头 512 维 = NoPE 448 + RoPE 64'), L('run', '只处理本 Rank 的 T 行，不扩成 2T')]
    },
    kv: {
      summary: '对整组 2T 行做 KV 投影：一个 512 维的共享 KV 头（MLA），RMSNorm 后对 RoPE 维旋转。',
      flow: { in: ['x_normed_full', 'wkv'], out: ['kv_full'] },
      math: [{ k: 'KV', f: 'kv = \\text{RMSNorm}(X_{full} W_{kv})\\odot\\gamma_{ckv},\\quad kv_{448:512} \\leftarrow \\text{RoPE}' }],
      sym: [['W_kv', 'wkv [D, 512]']],
      ex: '512 × 4096 → 512 × 512（每个 token 一行 KV，所有头共享）。',
      scen: [],
      limits: [L('run', '读 gathered 2T 行：两个 Rank 都维护整组请求的 KV Cache')]
    },
    write: {
      summary: '按 slot mapping 把 2T 行新 KV 写进分页的 Original KV Cache，slot 为负的行跳过。',
      flow: { in: ['kv_full', 'slot'], out: ['kv_cache'] },
      math: [{ k: '写回', f: '\\text{cache}[\\lfloor s_t/32 \\rfloor,\\; s_t \\bmod 32] \\leftarrow kv_t,\\quad s_t \\ge 0' }],
      sym: [['s_t', 'ori_slot_mapping[t]']],
      ex: '512 行 KV 写入 512 页 × 32 行的 Cache；64 个 block 各写 8 行。',
      scen: [],
      limits: [L('fix', '每页 32 行（BLOCK_SIZE）'), L('div', '2T 要能被 8 整除（每 block 8 个 token）', '512 ÷ 8 = 64 ✓', [CSA, 279])]
    },
    main: {
      summary: '每凑满 4 个 token 压成 1 行：对重叠窗口里的 8 行 KV 用 gate 做 softmax 加权池化，再 RMSNorm + RoPE 写进压缩 Cache。',
      flow: { in: ['x_normed_full', 'cmp_w'], out: ['cmp_kv', 'cmp_state'] },
      math: [
        { k: '投影', f: 'v = X W_{kv}^{c},\\quad g = X W_{gate}^{c} + ape' },
        { k: '池化（pos+1 是 4 的倍数时）', f: 'c = \\sum_{i=1}^{8} \\text{softmax}_i(g_i)\\odot v_i' },
        { k: '写 Cache', f: 'cmp = \\text{RoPE}(\\text{RMSNorm}(c)\\odot w)' }
      ],
      sym: [['8 行', '2 × 4：本组 4 行 + 上一组 4 行（overlap）'], ['ape', 'cmp_ape 位置偏置']],
      ex: '512 行 → 只有满足 (pos+1) mod 4 = 0 的 token 产出 1 行压缩 KV；状态环 8 行、每页 2 行。',
      scen: [['压缩比', [['4', 'def'], ['128', 'off', '在 decode_hca.py']], '']],
      limits: [L('fix', '状态环长度 8 = 2 × 4（OVERLAP）', 'STATE_LEN = COFF × 4', ['decode_compressor_ratio4.py', 53]), L('run', '不在 4 的边界上的 token 只更新状态，不写压缩 KV')]
    },
    idxcomp: {
      summary: '和 Main Compressor 同样的 4:1 池化，但产出的是给 Indexer 打分用的 128 维索引键：Hadamard 旋转后量化成 INT8。',
      flow: { in: ['x_normed_full', 'inner_w'], out: ['idx_cache'] },
      math: [{ k: '池化', f: 'c = \\sum_{i=1}^{8} \\text{softmax}_i(g_i)\\odot v_i' }, { k: '索引键', f: 'k = \\text{quant}_{i8}(H\\cdot\\text{RoPE}(\\text{RMSNorm}(c)))' }],
      sym: [['H', 'hadamard_idx [128, 128]']],
      ex: '每 4 个 token 产出 1 行 128 维 INT8 键 + 1 个 FP32 scale。',
      scen: [],
      limits: [L('fix', '索引维 128（index_head_dim），与主 KV 的 512 维不是同一份数据')]
    },
    indexer: {
      summary: '本地 query 与压缩索引键逐个打分（64 个索引头 ReLU 后加权求和），选出 Top-512 个压缩位置交给 Sparse Attention。',
      flow: { in: ['qr', 'idx_w', 'idx_cache'], out: ['idx_topk'] },
      math: [
        { k: '查询', f: 'q = H\\cdot\\text{RoPE}(\\text{dequant}(q_r^{i8} W_{qb}^{i8}))' },
        { k: '打分', f: 'score_t = s_t \\sum_{h=1}^{64} w_h\\, \\text{ReLU}(q_h\\cdot k_t)' },
        { k: '选择', f: 'idx = \\text{Top}_{512}(score),\\quad t < \\lfloor (pos+1)/4 \\rfloor' }
      ],
      sym: [['w_h', 'weights_proj 给出的头权重'], ['s_t', 'idx_kv_scale']],
      ex: '每个 token 从至多 ⌊(pos+1)/4⌋ 个压缩位置里选 512 个；不足时补 −1。',
      scen: [],
      limits: [L('fix', 'Top-k 固定 512（index_topk）'), L('run', '候选不超过 (pos+1)//4 个，未来位置不可见')]
    },
    ready: {
      summary: '只承载依赖：Original 与 Compressed 两路 Cache 都写完，Sparse Attention 才能读。',
      flow: { in: ['tids'], out: ['ready'] },
      math: [{ k: '汇合', f: 'ready = \\text{done}(\\tau_{ori})\\land\\text{done}(\\tau_{cmp})' }],
      sym: [['τ', 'ori_cache_write_tid / cmp_cache_write_tid']],
      ex: '',
      scen: [],
      limits: [L('run', 'Indexer 另等 idx_cache_write_tid，三路依赖不合成一个全局 barrier')]
    },
    attention: {
      summary: '每个本地 token 只看 640 个位置：最近 128 个原始 KV（滑窗）+ Indexer 选出的 512 个压缩 KV，分 5 块各算一次局部 softmax。',
      flow: { in: ['q', 'swa', 'idx_topk', 'kv_cache', 'cmp_kv', 'ready'], out: ['part'] },
      math: [
        { k: '第 j 块打分', f: 'S_j = \\frac{q K_j^{\\top}}{\\sqrt{512}} + bias_j,\\quad j = 0 \\ldots 4' },
        { k: '局部 softmax', f: 'm_j = \\max S_j,\\quad l_j = \\sum e^{S_j - m_j},\\quad o_j = e^{S_j - m_j} V_j' }
      ],
      sym: [['K_0', '滑窗 128 行（Original）'], ['K_{1..4}', 'Top-k 512 行（Compressed）'], ['bias', '无效位置为 −∞']],
      ex: '256 个 token × 5 块 = 1280 个 Work Item，由 qk_pv 的 24 条 Lane 领取。',
      scen: [],
      limits: [L('fix', '5 个稀疏块 = (128 + 512) ÷ 128', 'SPARSE_BLOCKS = 5', [SP, 79]), L('div', 'T 要能被 8 整除（bias 与 publish 的 token tile）', '256 ÷ 8 = 32 ✓', [SP, 92]), L('fix', 'qk_pv 派发 24 条 Lane，按 a2a3 的 AIC 数调过', '其他芯片需重新扫参', [SP, 67])]
    },
    publish: {
      summary: '把 5 个块的局部结果按 online softmax 合并、加 attention sink 归一化、逆旋转 RoPE 维，再按 O Group 打包直接写进目标 Rank。',
      flow: { in: ['part', 'sink'], out: ['grouped', 'window'] },
      math: [
        { k: '合并', f: 'm = \\max_j m_j,\\quad l = \\sum_j e^{m_j - m} l_j,\\quad o = \\sum_j e^{m_j - m} o_j' },
        { k: '归一化', f: '\\hat{o} = \\frac{o}{l + e^{sink - m}}' },
        { k: '去向', f: '\\text{rank}(h) = \\lfloor \\lfloor h/8 \\rfloor / 4 \\rfloor' }
      ],
      sym: [['h', '头号 0 … 63；每 8 个头一个 O Group']],
      ex: '头 0–31（Group 0–3）发往 Rank 0，头 32–63（Group 4–7）发往 Rank 1。',
      scen: [],
      limits: [L('div', 'O Group 数要能被 TP 整除', '8 ÷ 2 = 4 ✓', [SP, 96]), L('fix', '48 个 worker grid-stride 领 128 个打包任务')]
    },
    a2a: {
      summary: '收尾 Publish 发起的重分布：等对端信号，把本 Rank 负责的 4 个 Group 的整组 token 读回成连续布局。',
      flow: { in: ['window'], out: ['local_groups'] },
      math: [{ k: '重排', f: 'A_g = [\\,W^{(0)}_g ;\\, W^{(1)}_g\\,],\\quad g \\in \\text{本 Rank 的 4 个 Group}' }],
      sym: [['W^{(r)}_g', '来自 Rank r 的 Group g 行']],
      ex: '[2048, 4096] 的窗口 → 4 个 Group × 512 行 × 4096。',
      scen: [],
      limits: [L('run', '不是再发一次 payload：只等待、读回、回收信号')]
    },
    oproj: {
      summary: '每个 Rank 用自己 4 个 O Group 的权重算出整组 token 的部分输出，再把部分和送回 token 所属 Rank 做 FP32 累加。',
      flow: { in: ['local_groups', 'wo'], out: ['attn_out'] },
      math: [
        { k: '分组低秩', f: 'a_g = A_g W^{a}_g,\\quad a_g^{i8} = \\text{quant}_{row}(a_g)' },
        { k: '部分和', f: 'y^{(r)} = \\sum_{g \\in G_r} (a_g^{i8} W^{b\\,i8}_g)\\cdot s_g \\odot s_w' },
        { k: 'ReduceScatter', f: 'y_t = \\sum_{r=0}^{1} y_t^{(r)},\\quad t \\in \\text{owner}' }
      ],
      sym: [['G_r', 'Rank r 持有的 4 个 Group']],
      ex: '每 Rank 先算 [512, 4096] 的部分和，ReduceScatter 后各留自己的 [256, 4096]。',
      scen: [],
      limits: [L('div', 'decode token 数要能被 TP 整除', '512 ÷ 2 ✓', ['decode_o_proj.py', 109]), L('div', 'O Group 数要能被 TP 整除', '8 ÷ 2 ✓', ['decode_o_proj.py', 111])]
    },
    post: {
      summary: '把注意力输出按 post 系数分回 4 路，并按 comb 混合原来的 4 路残差，得到本层输出。',
      flow: { in: ['attn_out', 'x_hc', 'post_comb'], out: ['x_out'] },
      math: [{ k: '输出', f: 'y_o = post_o\\cdot x_{attn} + \\sum_{i=1}^{4} comb_{i,o}\\cdot r_i,\\quad o = 1 \\ldots 4' }],
      sym: [['r_i', 'x_hc 第 i 路残差']],
      ex: '[256, 4096] 注意力输出 + [256, 4, 4096] 残差 → [256, 4, 4096]。',
      scen: [],
      limits: [L('run', 'HC 残差流全程 FP32，不做 BF16 舍入', '', ['hc_post.py', 162])]
    },
    // 状态节点
    original: { summary: '未压缩的近期 KV，按 32 行一页存放；Sparse Attention 的滑窗块从这里按页成段读取。', flow: { in: ['kv_full'], out: ['kv_cache'] }, math: [], sym: [], ex: '', scen: [], limits: [L('fix', '每页 32 行')] },
    compressed: { summary: '每 4 个 token 一行的压缩 KV；Sparse Attention 的 SB1–SB4 按 Top-k 逐行读取。', flow: { in: ['kv_full'], out: ['cmp_kv'] }, math: [], sym: [], ex: '', scen: [], limits: [L('fix', '压缩比 4')] },
    indexcache: { summary: 'Indexer 打分用的 INT8 索引键与 FP32 scale，不是注意力要读的 KV 值。', flow: { in: ['x_normed_full'], out: ['idx_cache'] }, math: [], sym: [], ex: '', scen: [], limits: [L('fix', '128 维 INT8')] }
  };

  // Kernel 级卡片：qk_pv 讲一个 Work Item（Tile）的计算；其余 Kernel 用所属模块的卡片
  CARDS['kernel:qk_pv'] = {
    summary: '一个 Work Item = 一个 token 的一个稀疏块：把 128 行 KV 拼进 L1，64 个头分 2 个头块，各做 Q·Kᵀ → 局部 softmax → P·V，写回 m / l / o。',
    flow: { in: ['q_tile', 'kv_tile', 'bias'], out: ['mlo'] },
    math: [
      { k: '打分', f: 'S = \\frac{Q K^{\\top}}{\\sqrt{512}} + bias \\in \\mathbb{R}^{32 \\times 128}' },
      { k: '局部 softmax', f: 'm = \\max_{row} S,\\quad P = e^{S - m},\\quad l = \\sum_{row} P' },
      { k: '加权', f: 'o = P^{bf16} V \\in \\mathbb{R}^{32 \\times 512}' }
    ],
    sym: [['Q', '32 个头 × 512 维（一个头块）'], ['K = V', 'L1 里同一块 KV，两次 matmul 复用']],
    ex: '每个 Work Item 处理 2 个头块；空块不做 matmul，直接写 m = −3e38、l = 0、o = 0。',
    scen: [['派发', [['SPMD × 24', 'def', 'NUM_QK_CORES']], ''], ['流水', [['stage = 2', 'def'], ['stage = 1', 'on', 'AIC / AIV 不重叠'], ['stage > 2', 'off', '只有 2 个头块，按 2 处理']], 'pl.pipeline(H // 32, stage=2)']],
    limits: [L('fix', 'L1 用满 192 KB：v2c 槽 32 + KV 128 + Q 32', 'pass 36 真实分配', [SP, 225]), L('run', '没写 UP_DOWN 切分：每个 AIC 配的第二个 AIV 只握手', 'valid_shape = [0, 0]', [SP, 208])]
  };
  CARDS['kernel:cache_writeback'] = CARDS.write;
  CARDS['kernel:merge_publish'] = CARDS.publish;

  root.LensCards = {
    T: T, LAYER_SCEN: LAYER_SCEN,
    forModule: function (id) { return CARDS[id] || null; },
    forKernel: function (K) { return CARDS['kernel:' + K.id] || CARDS[K.module] || null; },
    isKernelCard: function (K) { return !!CARDS['kernel:' + K.id] && K.id === 'qk_pv'; }
  };
})(window);
