/* ──────────────────────────────────────────────────────────────────────────────
 * Hardware-Native 主线 · CSA（DeepSeek-V4 qk_pv）
 *
 * 与 fa_fused 主线同一套页面，换成 decode_sparse_attn_csa.py 的 qk_pv。
 * 区别在于数据来源：硅片上的地址与大小不再是估算，而是真实编译产物
 * Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/36_after_AllocateMemoryAddr.py
 *   qk_pv_aic（2761–2950 行）：Mat / Left / Right / Acc 分配与 v2c 管道槽
 *   qk_pv_aiv（2951–3081 行）：Vec 分配、c2v 管道槽、dual_aiv_dispatch 与 subblock 1 的 valid_shape=[0,0]
 * 主线：qk_pv 没写 UP_DOWN 切分 → 每个 AIC 配的第二个 AIV 只做管道握手 → 加切分或保持的取舍。
 * 打开“UP_DOWN”后显示的分配是按行数减半的推算，页面上一律标“推算”。
 * ────────────────────────────────────────────────────────────────────────────── */
(function () {
  'use strict';

  const KB = 1024;
  const NUM_QK_CORES = 24;
  const H = 64, QK_M_TILE = 32, H_TILE = 16, HEAD_DIM = 512, ATTN_K_TILE = 128;
  const UB_NOMINAL = 192 * KB, UB_USABLE = 184 * KB;
  const DUMP = 'Data/DeepseekV4/_jit_l3_decode_csa_20260903_010617/passes_dump/36_after_AllocateMemoryAddr.py';

  // ── 分配：stage=2、不切分时与 pass 36 的真实分配逐字节一致，其余组合按同一布局推算 ──
  // 推算规则：在飞的头块数 = min(stage, H / QK_M_TILE = 2)，管道槽数取在飞数（真实产物里是 2 槽）；
  // 加 UP_DOWN 后每个 AIV 只算 16 行，管道槽、临时区、mi / li、工作区按行数减半。
  const HEAD_BLOCKS = H / QK_M_TILE;
  const VEC_NOTES = {
    slots: 'reserve_buffer · slot_size 16384 × slot_num 2', bias: 'mem_vec_14 · 512 B', tmp: 'mem_vec_17 · 16 KB',
    mi: 'mem_vec_18 · 128 B', li: 'mem_vec_22 · 128 B', work: 'mem_vec_27 · 32 KB；scaled / scores / sub / exp / bf16 全在 49920'
  };
  function effStage(stage) { return Math.min(stage, HEAD_BLOCKS); }
  function isReal(split, stage) { return !split && stage === 2; }
  // AIV：qk_pv_aiv 的 Vec 地址表（自低地址向高地址）
  function vecOf(split, stage) {
    if (stage === undefined) stage = state.stage;
    const rows = split ? 16 : 32, slots = effStage(stage), real = isReal(split, stage);
    const parts = [
      ['slots', 'slot', 'c2v 管道 ' + slots + ' 槽', slots * rows * ATTN_K_TILE * 4],
      ['bias', 'bias', 'qk_bias_row [1,128]', 512],
      ['tmp', 'tmp', 'row_max / row_sum 临时', rows * ATTN_K_TILE * 4],
      ['mi', 'row', 'qk_mi [' + rows + ',1]', rows * 4],
      ['li', 'row', 'qk_li [' + rows + ',1]', rows * 4],
      // 工作区大小由空块分支的零块 [H_TILE, 512] FP32 决定，切分后按一半计
      ['work', 'work', 'softmax 原地工作区', rows / 2 * HEAD_DIM * 4]
    ];
    let addr = 0;
    return parts.map(function (p) {
      const b = { id: p[0], kind: p[1], name: p[2], lo: addr, hi: addr + p[3], note: real ? VEC_NOTES[p[0]] : '推算' };
      addr = b.hi;
      return b;
    });
  }
  // AIC：Mat（L1）地址表
  function l1Of(stage) {
    const slot = effStage(stage) * 16384;
    return [
      { id: 'l1-slot', lo: 0, hi: slot, kind: 'slot', name: 'v2c' },
      { id: 'l1-kv', lo: slot, hi: slot + 131072, kind: 'kv', name: 'qk_kv [128,512] · 128 KB' },
      { id: 'l1-q', lo: slot + 131072, hi: slot + 163840, kind: 'q', name: 'Q' }
    ];
  }
  // Lens 估算口径（每个中间量各占一块，并按 stage=2 复制）——用来对照真实结果
  const NAIVE = (function () {
    const tile = 32 * 128 * 4, bf = 32 * 128 * 2, rows = 2 * 32 * 4;
    const chain = 4 * tile + bf;
    return { once: 32768 + 512 + chain + tile + rows, staged: 32768 + 512 + 2 * (chain + tile + rows) };
  })();

  function hwOf(split, stage) { const v = vecOf(split, stage); return v[v.length - 1].hi; }

  // ── 格式化 ──
  function kb(n) { const v = n / KB; return (Math.abs(v - Math.round(v)) < 0.05 ? Math.round(v) : v.toFixed(1)) + ' KB'; }
  function bytes(n) { return n < KB ? n + ' B' : kb(n); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (ch) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]; }); }
  function uniq(list) { return list.filter(function (v, i) { return list.indexOf(v) === i; }); }

  // ════════════════════════════════════════════════════════════════════════════
  // 源码中的代码行（去掉函数体的 4 格缩进）
  // ════════════════════════════════════════════════════════════════════════════
  const SOURCE = [
    { no: 208, t: 'with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", §deps=[qk_plan_tid, cache_ready_dep], allow_early_resolve=True) as qk_tid:' },
    { no: 209, t: '    qk_core = pl.tile.get_block_idx()' },
    { no: 213, t: '    qk_lane_iters = (qk_items - qk_core + NUM_QK_CORES - 1) // NUM_QK_CORES' },
    { no: 214, t: '    for qk_it in pl.range(qk_lane_iters):' },
    { no: 215, t: '        qk_flat = qk_core + qk_it * NUM_QK_CORES' },
    { no: 216, t: '        qk_item = pl.cast(pl.read(qk_order, [qk_flat]), pl.INDEX)' },
    { no: 217, t: '        qk_t = qk_item // SPARSE_BLOCKS' },
    { no: 218, t: '        qk_sb = qk_item - qk_t * SPARSE_BLOCKS' },
    { no: 219, t: '        qk_b = qk_t // S' },
    { no: 220, t: '        qk_token_base = qk_t * (H // H_TILE) * SPARSE_BLOCKS * H_TILE' },
    { no: 221, t: '        qk_s0 = qk_sb * ATTN_K_TILE' },
    { no: 222, t: '        qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]' },
    { no: 223, t: '        qk_block_valid = pl.read(valid_block_mask, [qk_t, qk_sb])' },
    { no: 224, t: '        if qk_block_valid > 0:' },
    { no: 225, t: '            qk_kv = pl.create_l1([ATTN_K_TILE, HEAD_DIM], pl.BF16)' },
    { no: 228, t: '            qk_win_rows = pl.min(pl.max(WIN - qk_s0, 0), ATTN_K_TILE)' },
    { no: 229, t: '            if qk_win_rows > 0:' },
    { no: 236, t: '                qk_pos = pl.cast(pl.read(position_ids, [qk_t, 0]), pl.INDEX)' },
    { no: 237, t: '                qk_win_len = pl.min(qk_pos + 1, WIN)' },
    { no: 238, t: '                qk_win_start = qk_pos - qk_win_len + 1' },
    { no: 239, t: '                qk_run_rows = pl.min(pl.max(qk_win_len - qk_s0, 0), qk_win_rows)' },
    { no: 245, t: '                qk_head = (qk_win_start + qk_s0) % BLOCK_SIZE' },
    { no: 246, t: '                for qk_run in pl.unroll(SWA_RUNS):' },
    { no: 247, t: '                    qk_run_lo = pl.max(qk_run * BLOCK_SIZE - qk_head, 0)' },
    { no: 248, t: '                    qk_run_hi = pl.min((qk_run + 1) * BLOCK_SIZE - qk_head, qk_run_rows)' },
    { no: 249, t: '                    if qk_run_hi > qk_run_lo:' },
    { no: 250, t: '                        qk_run_raw = pl.read(window_swa_indices, [qk_t, qk_s0 + qk_run_lo])' },
    { no: 253, t: '                        qk_run_src = pl.cast(pl.max(qk_run_raw, 0), pl.INDEX)' },
    { no: 254, t: '                        qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_run_lo, 0], [qk_run_src, 0],' },
    { no: 255, t: '                                              [ATTN_K_TILE, HEAD_DIM],' },
    { no: 256, t: '                                              valid_shape=[qk_run_hi - qk_run_lo, HEAD_DIM])' },
    { no: 257, t: '                qk_tail_n = qk_win_rows - qk_run_rows' },
    { no: 258, t: '                if qk_tail_n > 0:' },
    { no: 261, t: '                    qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_run_rows, 0], [0, 0],' },
    { no: 262, t: '                                          [ATTN_K_TILE, HEAD_DIM], valid_shape=[qk_tail_n, HEAD_DIM])' },
    { no: 264, t: '            for qk_r in pl.range(qk_win_rows, ATTN_K_TILE):' },
    { no: 265, t: '                qk_cmp_k = qk_s0 + qk_r - WIN' },
    { no: 266, t: '                if qk_cmp_k < CMP_TOPK:' },
    { no: 267, t: '                    qk_ridx = pl.read(cmp_sparse_indices, [qk_t, qk_cmp_k])' },
    { no: 268, t: '                    if qk_ridx >= 0:' },
    { no: 269, t: '                        qk_slot = qk_ridx' },
    { no: 270, t: '                        qk_cblk = pl.cast(pl.read(cmp_block_table, [qk_b, qk_slot // BLOCK_SIZE]), pl.INDEX)' },
    { no: 271, t: '                        qk_csrc = qk_cblk * BLOCK_SIZE + qk_slot % BLOCK_SIZE' },
    { no: 272, t: '                        qk_kv = pl.gather_row(qk_kv, cmp_kv_flat, [qk_r, 0], [qk_csrc, 0], [1, HEAD_DIM])' },
    { no: 273, t: '                    else:' },
    { no: 274, t: '                        qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])' },
    { no: 275, t: '                else:' },
    { no: 276, t: '                    qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_r, 0], [0, 0], [1, HEAD_DIM])' },
    { no: 281, t: '            for qk_hb in pl.pipeline(H // QK_M_TILE, stage=§):' },
    { no: 282, t: '                qk_h0 = qk_hb * QK_M_TILE' },
    { no: 283, t: '                qk_head_row = qk_t * H + qk_h0' },
    { no: 284, t: '                qk_q_tile = q_flat[qk_head_row : qk_head_row + QK_M_TILE, 0 : HEAD_DIM]' },
    { no: 285, t: '                qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)' },
    { no: 286, t: '                qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)' },
    { no: 289, t: '                qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)' },
    { no: 290, t: '                qk_mi = pl.row_max(qk_scores)' },
    { no: 293, t: '                qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))' },
    { no: 294, t: '                qk_li = pl.row_sum(qk_exp)' },
    { no: 295, t: '                qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")' },
    { no: 296, t: '                qk_oi = pl.matmul(qk_exp_bf16, qk_kv, out_dtype=pl.FP32)' },
    { no: 297, t: '                for qk_sub in pl.unroll(QK_M_TILE // H_TILE):' },
    { no: 298, t: '                    qk_h_idx = qk_hb * (QK_M_TILE // H_TILE) + qk_sub' },
    { no: 299, t: '                    qk_r0 = qk_sub * H_TILE' },
    { no: 300, t: '                    qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE' },
    { no: 301, t: '                    qk_row = qk_blk_base + qk_sb * H_TILE' },
    { no: 302, t: '                    sparse_blk_mi[qk_row : qk_row + H_TILE, 0 : 1] = qk_mi[qk_r0 : qk_r0 + H_TILE, 0 : 1]' },
    { no: 303, t: '                    sparse_blk_li[qk_row : qk_row + H_TILE, 0 : 1] = qk_li[qk_r0 : qk_r0 + H_TILE, 0 : 1]' },
    { no: 304, t: '                    sparse_blk_oi[qk_row : qk_row + H_TILE, 0 : HEAD_DIM] = qk_oi[qk_r0 : qk_r0 + H_TILE, 0 : HEAD_DIM]' },
    { no: 305, t: '        else:' },
    { no: 306, t: '            qk_oi_zero = pl.full([H_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)' },
    { no: 307, t: '            for qk_h_idx in pl.range(H // H_TILE):' },
    { no: 308, t: '                qk_blk_base = qk_token_base + qk_h_idx * SPARSE_BLOCKS * H_TILE' },
    { no: 309, t: '                qk_row = qk_blk_base + qk_sb * H_TILE' },
    { no: 310, t: '                for qk_hr in pl.range(H_TILE):' },
    { no: 311, t: '                    pl.write(sparse_blk_mi, [qk_row + qk_hr, 0], -3.0e38)' },
    { no: 312, t: '                    pl.write(sparse_blk_li, [qk_row + qk_hr, 0], 0.0)' },
    { no: 313, t: '                sparse_blk_oi[qk_row : qk_row + H_TILE, 0 : HEAD_DIM] = qk_oi_zero' }
  ];
  function range(a, b) { const out = []; for (let i = a; i <= b; i++) out.push(i); return out; }
  const SHOWN = SOURCE.map(function (r) { return r.no; });
  function shownIn(a, b) { return range(a, b).filter(function (no) { return SHOWN.indexOf(no) >= 0; }); }

  // ════════════════════════════════════════════════════════════════════════════
  // 行内标签：① 层级（缩进导轨 + 作用域标签）② Hardware-Native（落点标签 + API 下划线）
  // ════════════════════════════════════════════════════════════════════════════
  // 结构深度按 4 格缩进算；续行（上一行括号没闭）跟着语句本身的深度，不另算一层，
  // 否则 254–256 这种对齐到括号的续行会画出十几条导轨。
  const INDENT = 4;
  const META = (function () {
    const out = []; let open = 0, stmt = 0;
    SOURCE.forEach(function (row) {
      const cont = open > 0;
      if (!cont) stmt = Math.floor(row.t.match(/^ */)[0].length / INDENT);
      out.push({ no: row.no, depth: stmt, cont: cont, opens: !cont && /:$/.test(row.t) });
      for (let i = 0; i < row.t.length; i++) {
        const ch = row.t[i];
        if (ch === '(' || ch === '[') open++;
        else if (ch === ')' || ch === ']') open--;
      }
      if (open < 0) open = 0;
    });
    return out;
  })();
  // 每个开头行覆盖的行区间：联动时整段 body 点亮同一条导轨
  const SCOPES = META.map(function (m, i) {
    if (!m.opens) return null;
    let j = i + 1;
    while (j < META.length && META[j].depth > m.depth) j++;
    return { no: m.no, depth: m.depth, from: i, to: j - 1 };
  }).filter(function (s) { return !!s; });
  function scopeOf(no) {
    const i = SHOWN.indexOf(no);
    if (i < 0) return null;
    if (META[i].opens) return SCOPES.filter(function (s) { return s.from === i; })[0] || null;
    let best = null;
    SCOPES.forEach(function (s) { if (i > s.from && i <= s.to && (!best || s.depth > best.depth)) best = s; });
    return best;
  }
  // ① 层级标签：这一层循环跑多少次、这是哪个分支。名字与逻辑图里的框一致
  const SCOPE_TAG = {
    208: function () { return 'SPMD × 24'; },
    214: function () { return 'GRID-STRIDE'; },
    224: function () { return '有效块'; },
    246: function () { return 'UNROLL · 按页'; },
    264: function () { return '逐行 · 压缩段'; },
    281: function () { return 'PIPELINE · ' + effStage(state.stage) + ' 块在飞'; },
    297: function () { return 'UNROLL × 2'; },
    305: function () { return '空块'; },
    307: function () { return '× 4 头块'; },
    310: function () { return '× 16 行'; }
  };
  // ② Hardware-Native 落点标签：这一行把数据落到哪里。
  //    mat = AIC 的 Mat / L0 / Cube，vec = AIV 的 Vec 与管道，gm = 读写 GM
  const HW_TAG = [
    [/pl\.create_l1/, 'L1 · 128 KB', 'mat'],
    [/pl\.gather_row/, 'GM → L1', 'mat'],
    [/pl\.matmul/, 'CUBE → L0C', 'mat'],
    [/q_flat\[/, 'GM → L1 · Q', 'mat'],
    [/target_type=pl\.BF16/, 'VEC → V2C', 'vec'],
    [/sparse_bias\[/, 'GM → VEC · bias', 'vec'],
    [/pl\.row_max\(/, 'VEC · mi', 'vec'],
    [/pl\.row_sum\(/, 'VEC · li', 'vec'],
    [/pl\.(mul|col_expand_add|row_expand_sub|exp|full)\(/, 'VEC · 原地', 'vec'],
    [/pl\.write\(|sparse_blk_\w+\[[^\]]*\] =/, '→ GM', 'gm']
  ];
  function hwTag(t) {
    for (let i = 0; i < HW_TAG.length; i++) if (HW_TAG[i][0].test(t)) return HW_TAG[i];
    return null;
  }
  // 把硬件写进源码的 API 与参数：虚线下划线，和普通张量运算区分开
  const HW_API = {
    spmd: 1, pipeline: 1, unroll: 1, create_l1: 1, gather_row: 1, matmul: 1,
    split: 1, SplitMode: 1, UP_DOWN: 1, stage: 1, optimizations: 1,
    valid_shape: 1, out_dtype: 1, b_trans: 1, target_type: 1
  };

  // ════════════════════════════════════════════════════════════════════════════
  // 联动表：逻辑节点 = 源码行 + 硅片区域 + Vec 缓冲
  // ════════════════════════════════════════════════════════════════════════════
  const NODES = [
    { id: 'f-spmd', frame: true, lines: [208, 209], regions: ['cores'] },
    // 208 行的切分写法：逻辑图里不画，把这一行连到 AIC、两个 AIV 和中间的管道
    { id: 'split', hidden: true, lines: [208], regions: ['aic', 'aiv0', 'aiv1', 'pipe'] },
    { id: 'f-iter', frame: true, lines: shownIn(213, 221).concat([223, 224]), regions: ['cores', 'gm-plan'] },
    { id: 'f-pipe', frame: true, lines: [281], regions: ['pipe', 'l1-slot'], bufs: ['slots'] },
    { id: 'p-gather', frame: true, lines: [], group: ['kvl1', 'win', 'cmp'] },
    { id: 'p-score', frame: true, lines: [], group: ['q', 'kvref', 'qk', 'scale'] },
    { id: 'p-softmax', frame: true, lines: [], group: ['max', 'exp', 'sum', 'cast'] },
    { id: 'p-weight', frame: true, lines: [], group: ['pv', 'out'] },
    { id: 'kvl1', lines: [225], regions: ['l1-kv'] },
    { id: 'win', lines: shownIn(228, 262), regions: ['gm-ori', 'l1-kv'] },
    { id: 'cmp', lines: shownIn(264, 276), regions: ['gm-cmp', 'l1-kv'] },
    { id: 'empty', lines: shownIn(305, 313), regions: ['gm-out'], bufs: ['work'] },
    { id: 'q', lines: [282, 283, 284], regions: ['gm-q', 'l1-q', 'l0a'] },
    { id: 'kvref', lines: [225], regions: ['l1-kv', 'l0b'] },
    { id: 'qk', lines: [285], regions: ['cube', 'l0a', 'l0b', 'l0c', 'pipe'], bufs: ['slots'] },
    { id: 'scale', lines: [286, 289, 222], regions: ['gm-plan'], bufs: ['work', 'bias'] },
    { id: 'max', lines: [290], bufs: ['tmp', 'mi'] },
    { id: 'exp', lines: [293], bufs: ['work'] },
    { id: 'sum', lines: [294], bufs: ['tmp', 'li'] },
    { id: 'cast', lines: [295], regions: ['pipe', 'l1-slot'], bufs: ['work'] },
    { id: 'pv', lines: [296], regions: ['cube', 'l1-kv', 'l0b', 'l0c'] },
    { id: 'out', lines: shownIn(297, 304), regions: ['gm-out', 'l0c'], bufs: ['mi', 'li'] }
  ];
  const AIV_NODES = ['scale', 'max', 'exp', 'sum', 'cast'];
  const REGION_NODES = { aic: ['split', 'q', 'kvl1', 'qk', 'pv'], aiv0: ['split'].concat(AIV_NODES), aiv1: ['split'] };
  function nodeById(id) { return NODES.filter(function (n) { return n.id === id; })[0]; }
  function nodesWhere(fn) { return NODES.filter(fn).map(function (n) { return n.id; }); }
  function lineMapped(no) { return NODES.some(function (n) { return n.lines.indexOf(no) >= 0; }); }

  // ── 状态 ──
  const SECTIONS = [
    { name: '写法', q: '和 fa_fused 差在哪三处' },
    { name: '落点', q: '编译器真正给出的地址' },
    { name: '裁决', q: '第二个 AIV 在做什么' },
    { name: '改法', q: '加切分，还是保持' },
    { name: '复查', q: '要拿什么证据' }
  ];
  const state = { step: 1, split: false, stage: 2, chosen: null, hover: null, preview: null };
  function shownSplit() { return state.preview ? state.preview === 'A' : state.split; }

  // ════════════════════════════════════════════════════════════════════════════
  // 说明文字
  // ════════════════════════════════════════════════════════════════════════════
  function pipeCaption() {
    const st = state.stage, eff = effStage(st);
    if (st === 2) {
      return '<code>pl.pipeline(2, stage=2)</code>：64 个头分成 2 个头块，每块 32 行，两块同时在飞。编译后 AIC 与 AIV 之间是 <b>2 槽 × 16 KB</b> 的 C2V / V2C 管道'
        + '（c2v 槽在 AIV 的 Vec 0–32 KB，v2c 槽在 AIC 的 L1 0–32 KB）<span class="tag-real">真实</span>';
    }
    if (eff === 1) {
      return '<code>stage=1</code>：同一时刻只有 1 个头块在飞。AIC 做完 QK 要等 AIV 把 softmax 送回来才能做 PV，下一块 QK 也要排在后面，时间线上 AIC 与 AIV 交替空等。'
        + '推算管道只需 1 槽，Vec 和 L1 各省 16 KB <span class="tag-proj">推算</span>';
    }
    return '<code>stage=' + st + '</code>：循环只有 ' + HEAD_BLOCKS + ' 个头块（64 / 32），同时在飞的最多 2 个，多出的 stage 没有块可填，效果与 stage=2 相同 <span class="tag-proj">推算</span>';
  }
  function nodeCaption(id, split) {
    const per = split ? '16' : '32';
    const map = {
      'f-spmd': '<code>pl.spmd(NUM_QK_CORES)</code>：24 条 lane，正好对应 a2a3 的 24 个 AIC。每个 lane 是一个 block，配 1 个 AIC 和 2 个 AIV。',
      split: split
        ? '<b>推算</b>：加上 <code>optimizations=[pl.split(pl.SplitMode.UP_DOWN)]</code> 后，32 行 softmax 分给两个 AIV，各算 16 行。'
        : '208 行没有写切分。编译产物给 qk_pv_aiv 标了 <code>dual_aiv_dispatch</code>，但 subblock 1 的每个 tile 都是 <code>valid_shape=[0, 0]</code>。',
      'f-iter': '按 <code>qk_order</code> 取 (token, 稀疏块)：qk_plan 已把非空块排在前面，24 条 lane 先各拿一个重活。',
      'f-pipe': pipeCaption(),
      'p-gather': '<b>① 拼 KV 块</b>：先在 L1 开 128 KB，再把滑窗行按页成段搬进来、压缩行按 top-k 逐行搬进来。全部发生在 AIC 一侧。',
      'p-score': '<b>② 打分</b>：Q 块与 L1 里的 KV 做 Q·Kᵀ。K=512 被编译器自动切成 4 段 × 128，L0B 两块 32 KB 轮换。',
      'p-softmax': '<b>③ 局部 softmax</b>：全部在 AIV 的 Vec 里完成，5 个 [32,128] 中间量原地写在同一块地址（49920 起）。' + (split ? '推算：两个 AIV 各算 16 行。' : '<b>现在全部 32 行都在 AIV0，AIV1 只握手。</b>'),
      'p-weight': '<b>④ 加权求和</b>：P 与同一块 KV 做 P·V，结果 [32,512] FP32 占满 L0C 的 64 KB，由 AIC 按 16 行两次写回 GM。',
      kvl1: '<code>pl.create_l1</code>：在源码里显式指定 L1。编译后落在 Mat 32–160 KB，大小 128 KB。',
      win: '滑窗部分：窗口位置连续，分页 KV 里同一页的行也连续，所以按页拆成几段整段搬（BLOCK_SIZE = 32 行一页）。',
      cmp: '压缩部分：indexer 选出的 top-k 位置是散的，只能一行一行查压缩页表再搬。',
      empty: '块里没有有效位置：不做 matmul，直接把 mi 写 −3e38、li 写 0、oi 写 0。零块 [16,512] FP32 借用 Vec 工作区的 32 KB。',
      q: 'Q 块 [32,512] BF16 从 GM 载入 L1（Mat 160–192 KB），再按 K 切片进 L0A（2 × 8 KB）。',
      kvref: 'QK 和 PV 两次 matmul 都直接用 L1 里同一块 KV，不重复搬运。',
      qk: 'Q·Kᵀ：[32,512] × [512,128]。结果 [32,128] FP32 16 KB 在 L0C，经 C2V 管道送到 AIV。',
      scale: '乘缩放系数再加上 NEG_INF bias：无效位置 exp 后约等于 0，不需要另外的掩码。',
      max: '行最大 <code>qk_mi [' + per + ',1]</code>，借用 16 KB 临时区做规约。',
      exp: '<code>exp(scores − max)</code>：原地写回同一块工作区。',
      sum: '行求和 <code>qk_li [' + per + ',1]</code>，同样借用临时区。',
      cast: '转 BF16 后经 V2C 管道送回 AIC，作为 PV 的左操作数。',
      pv: 'P·V：[32,128] × [128,512]。结果 [32,512] FP32 64 KB，与 qk_raw 共用 L0C 的同一地址。',
      out: 'mi / li 由 AIV 写回；oi 由 AIC 直接从 L0C 按 16 行切两次写回 GM。297 行的循环本来就按 16 行一半组织。'
    };
    return map[id] || '';
  }
  function regionCaption(id, split) {
    const hw = hwOf(split);
    const map = {
      cores: '<b>pl.spmd × 24</b>：下方展开 block 0。每个 block 1 个 AIC + 2 个 AIV，用同一份分配。',
      'gm-q': '<b>q</b> [T, 64, 512] BF16：每次取一个 token 的 32 个头。',
      'gm-ori': '<b>ori_kv</b>：原始 KV 分页，一页 32 行；滑窗部分从这里读。',
      'gm-cmp': '<b>cmp_kv</b>：压缩 KV 分页；按 top-k 位置逐行读。',
      'gm-plan': '<b>qk_order / sparse_bias / valid_block_mask</b>：qk_plan 事先算好的取块顺序、加法 bias 和块有效标记。',
      'gm-out': '<b>sparse_blk_mi / li / oi</b>：每个 (token, 头块, 稀疏块) 的部分结果，留给 merge_norm 合并。',
      aic: '<b>AIC</b>：L1 用了 192 KB（v2c 槽 32 + KV 128 + Q 32），L0B 用满 64 KB，L0C 用满 64 KB。',
      'l1-slot': '<b>v2c 管道槽</b>：L1 0–' + kb(effStage(state.stage) * 16384) + '，' + effStage(state.stage) + ' 槽 × 16 KB，接收 AIV 送回的 exp_bf16' + (state.stage !== 2 ? '（推算）' : '') + '。',
      'l1-kv': '<b>qk_kv</b>：L1 32–160 KB，[128,512] BF16。源码 225 行显式 <code>create_l1</code>。',
      'l1-q': '<b>Q 块</b>：L1 160–192 KB，[32,512] BF16。',
      l0a: '<b>L0A · Left</b>：2 × 8 KB，Q 的 K 切片 [32,128] 轮换。',
      l0b: '<b>L0B · Right</b>：2 × 32 KB = 64 KB，正好用满。QK 时放 [128,128] 的 K 切片，PV 时放 [32,512]。',
      cube: '<b>Cube</b>：两次 matmul，K=512 自动切成 4 段累加（AutoTileMatmulL0）。',
      l0c: '<b>L0C · Acc</b>：64 KB。qk_raw 16 KB 与 qk_oi 64 KB 都从地址 0 开始，先后复用。',
      pipe: effStage(state.stage) === 1
        ? '<b>C2V / V2C 管道</b>：推算为 1 槽。只有一个槽时 AIC 送出结果后要等 AIV 取走并处理完，二者无法重叠。'
        : '<b>C2V / V2C 管道</b>：2 槽 × ' + (split ? '8' : '16') + ' KB。AIC 与 AIV 轮流使用这两个槽交接结果，从而互相重叠' + (isReal(split, state.stage) ? '。' : '（推算）。'),
      aiv0: '<b>AIV0（subblock 0）</b>：Vec 高水位 ' + kb(hw) + (isReal(split, state.stage) ? '' : '（推算）') + '，远低于 184 KB。',
      aiv1: split ? '<b>AIV1（subblock 1）</b>：推算后与 AIV0 各算 16 行。' : '<b>AIV1（subblock 1）</b>：分配和 AIV0 一样，但所有 tile 都是 <code>valid_shape=[0, 0]</code>，只跟 AIC 握手。',
      ub0: '<b>Vec 地址条</b>：c2v 槽 → bias → 临时区 → mi / li → softmax 工作区，高水位 ' + kb(hw) + '。',
      ub1: split ? '<b>AIV1 Vec</b>：推算，与 AIV0 相同。' : '<b>AIV1 Vec</b>：地址照样分配了，但没有有效数据写进来。'
    };
    return map[id] || '';
  }
  function bufCaption(id, split) {
    const b = vecOf(split).filter(function (x) { return x.id === id; })[0];
    if (!b) return '';
    return '<b>' + b.name + '</b> · ' + bytes(b.hi - b.lo) + ' · 地址 ' + b.lo + '–' + b.hi + (split ? '（推算）' : '') + '<br>' + b.note;
  }
  function lineCaption(no, split) {
    if (no === 208) return nodeCaption('split', split);
    const owner = NODES.filter(function (n) { return !n.hidden && n.lines.indexOf(no) >= 0; })[0];
    return owner ? nodeCaption(owner.id, split) : '';
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 源码
  // ════════════════════════════════════════════════════════════════════════════
  const KW = { 'for': 1, 'in': 1, 'with': 1, 'as': 1, 'if': 1, 'else': 1, 'True': 1, 'False': 1, 'None': 1 };
  function hl(text) {
    const re = /("[^"]*"|'[^']*')|(\d+(?:\.\d+)?(?:e[+-]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|(.)/g;
    let out = '', m, prev = '';
    while ((m = re.exec(text))) {
      const tok = m[0];
      if (m[1]) out += '<span class="tk-str">' + esc(tok) + '</span>';
      else if (m[2]) out += '<span class="tk-num">' + tok + '</span>';
      else if (m[3]) {
        const next = text.slice(re.lastIndex);
        let cls = 'tk-id';
        if (KW[tok]) cls = 'tk-kw';
        else if (tok === 'pl') cls = 'tk-mod';
        else if (/^\(/.test(next)) cls = 'tk-fn';
        else if (/^=(?!=)/.test(next)) cls = 'tk-kwarg';
        else if (/^[A-Z][A-Z0-9_]+$/.test(tok)) cls = 'tk-const';
        else if (prev === '.') cls = 'tk-attr';
        if (HW_API[tok]) cls += ' tk-hw';
        out += '<span class="' + cls + '">' + tok + '</span>';
      } else if (m[4]) out += tok;
      else out += '<span class="tk-punc">' + esc(tok) + '</span>';
      if (!m[4]) prev = tok;
    }
    return out;
  }
  function splitControl() {
    if (state.split) {
      return '<span class="split-on">' + hl('optimizations=[pl.split(pl.SplitMode.UP_DOWN)], ')
        + '</span><button class="split-del" type="button" data-act="split" data-on="0" title="去掉切分，回到真实编译产物" aria-label="去掉 UP_DOWN 切分">×</button> ';
    }
    return '<button class="split-add" type="button" data-act="split" data-on="1" title="加上 UP_DOWN 切分（推算）">+ UP_DOWN</button> ';
  }
  function stageControl() {
    let html = '<span class="stage-ctl" role="group" aria-label="pl.pipeline stage">';
    for (let s = 1; s <= 4; s++) {
      const on = state.stage === s;
      html += '<button type="button" data-act="stage" data-v="' + s + '" class="' + (on ? 'is-on' : '') + (on && s > HEAD_BLOCKS ? ' is-over' : '')
        + '" aria-pressed="' + on + '">' + s + '</button>';
    }
    return html + '</span>' + (state.stage !== 2 ? '<span class="stage-was">原 stage=2</span>' : '');
  }
  function renderCode() {
    const html = SOURCE.map(function (row, i) {
      let text;
      if (row.no === 208 || row.no === 281) {
        const parts = row.t.split('§');
        text = hl(parts[0]) + (row.no === 208 ? splitControl() : stageControl()) + hl(parts[1]);
      } else text = hl(row.t);
      // 一行最多一个标签：开头行标层级，其余标落点
      const scope = SCOPE_TAG[row.no], hw = hwTag(row.t);
      if (scope) text += '<span class="cl-tag t-scope">' + scope() + '</span>';
      else if (hw) text += '<span class="cl-tag t-' + hw[2] + '">' + hw[1] + '</span>';
      const modified = (row.no === 208 && state.split) || (row.no === 281 && state.stage !== 2);
      const cls = ['cl', lineMapped(row.no) ? 'is-mapped' : '', modified ? 'is-modified' : ''].join(' ');
      return '<div class="' + cls + '" data-line="' + row.no + '"><span class="cl-mark"></span><span class="cl-no">' + row.no
        + '</span><span class="cl-text" style="--d:' + META[i].depth + '">' + text + '</span></div>';
    }).join('');
    document.getElementById('hpCode').innerHTML = html;
  }

  // ── SVG 小工具 ──
  function txt(x, y, s, cls, anchor) {
    return '<text x="' + x + '" y="' + y + '" class="' + (cls || 'sv-text') + '"' + (anchor ? ' text-anchor="' + anchor + '"' : '') + '>' + s + '</text>';
  }
  function markers(prefix) {
    return '<marker id="' + prefix + 'Arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#555"/></marker>'
      + '<marker id="' + prefix + 'ArrowHot" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L8 4 L0 8 z" fill="#e8e8e8"/></marker>';
  }
  function region(id, inner) { return '<g class="sv-region" data-region="' + id + '">' + inner + '</g>'; }

  // ════════════════════════════════════════════════════════════════════════════
  // 算子逻辑图
  // ════════════════════════════════════════════════════════════════════════════
  // 节点 [x, y, w, 名字]（viewBox 720 × 336，节点高 26）
  const LG = {
    kvl1: [46, 84, 110, '开 L1 缓冲'], win: [168, 84, 160, '滑窗：按页成段搬'], cmp: [340, 84, 178, '压缩：按 top-k 逐行搬'],
    empty: [540, 84, 144, '写 0 / −inf'],
    q: [58, 182, 96, 'Q 块'], kvref: [162, 182, 96, 'KV（L1）'], qk: [58, 226, 96, 'Q · Kᵀ'], scale: [162, 226, 96, '× scale + bias'],
    max: [288, 182, 72, '行最大'], exp: [368, 182, 72, 'exp'], sum: [448, 182, 72, '行求和'], cast: [368, 226, 72, '转 BF16'],
    pv: [550, 182, 114, 'P · KV'], out: [550, 226, 114, '写回 mi / li / oi']
  };
  const LG_H = 26;
  const LG_EDGES = [
    ['kvl1', 'win', 'M156 97 H 168'], ['win', 'cmp', 'M328 97 H 340'],
    ['q', 'qk', 'M106 208 V 226'], ['kvref', 'qk', 'M210 208 V 216 H 130 V 226'], ['qk', 'scale', 'M154 239 H 162'],
    ['scale', 'max', 'M258 239 H 274 V 195 H 288'], ['max', 'exp', 'M360 195 H 368'], ['exp', 'sum', 'M440 195 H 448'],
    ['exp', 'cast', 'M404 208 V 226'], ['cast', 'pv', 'M440 239 H 536 V 195 H 550'], ['pv', 'out', 'M607 208 V 226']
  ];
  const LG_PHASES = [
    ['p-gather', 36, 62, 492, 56, '① 拼 KV 块 → L1'],
    ['p-score', 48, 156, 220, 150, '② 打分'],
    ['p-softmax', 278, 156, 252, 150, '③ 局部 SOFTMAX'],
    ['p-weight', 540, 156, 134, 150, '④ 加权']
  ];
  function logicNode(id) {
    const p = LG[id];
    if (id === 'q' || id === 'kvref') {
      return '<g class="lg-hit" data-node="' + id + '">' + PathGridGlyphs.tensor(id, p[3], id === 'q' ? 'Q' : 'KV · L1', p[0], p[1], p[2], LG_H, id === 'q' ? '#72d7af' : '#bca0f3') + '</g>';
    }
    const kinds = { qk:'matmul', pv:'matmul', max:'reduce', sum:'reduce', exp:'activation', scale:'gate', cast:'reshape', out:'cache', kvl1:'cache', win:'cache', cmp:'cache', empty:'reshape' };
    const icon = window.PtoOpIcons.find(function (item) { return item.id === kinds[id]; });
    return '<g class="lg-hit" data-node="' + id + '">' + VisualSystem.node('op', id, p[3], p[0], p[1], {width:p[2], height:LG_H, icon:icon && icon.path, fontSize:9}) + '</g>';
  }
  // AIC / AIV 时间线（示意）：QK、softmax、PV 按等长画，只用来看 stage 和切分改变了什么重叠关系
  function schedule(split, stage) {
    const QK = 2, PV = 2, SM = split ? 1 : 2, eff = effStage(stage), out = [];
    function add(lane, s, d, label, node) { out.push({ lane: lane, s: s, e: s + d, label: label, node: node }); return s + d; }
    if (eff === 1) {
      let t = 0;
      for (let hb = 0; hb < HEAD_BLOCKS; hb++) {
        const qkE = add('aic', t, QK, 'QK' + hb, 'qk');
        const smE = add('aiv', qkE, SM, 'SM' + hb, 'p-softmax');
        t = add('aic', smE, PV, 'PV' + hb, 'pv');
      }
    } else {
      // 两个头块都在飞：AIC 连着发两个 QK，再按 softmax 回来的顺序做 PV
      const qk0 = add('aic', 0, QK, 'QK0', 'qk'), qk1 = add('aic', qk0, QK, 'QK1', 'qk');
      const sm0 = add('aiv', qk0, SM, 'SM0', 'p-softmax');
      const sm1 = add('aiv', Math.max(qk1, sm0), SM, 'SM1', 'p-softmax');
      const pv0 = add('aic', Math.max(qk1, sm0), PV, 'PV0', 'pv');
      add('aic', Math.max(pv0, sm1), PV, 'PV1', 'pv');
    }
    return out;
  }
  function timelineSvg(split, stage) {
    const blocks = schedule(split, stage);
    const total = Math.max.apply(null, blocks.map(function (b) { return b.e; }));
    const x0 = 112, unit = 46, y = { aic: 356, aiv: 374 };
    let s = txt(24, 347, 'AIC / AIV 时间线', 'sv-title') + txt(150, 347, '示意：每段按等长画，真实时长需 profiling', 'sv-label');
    s += txt(24, y.aic + 10, 'AIC', 'sv-label') + txt(24, y.aiv + 10, split ? 'AIV0 + AIV1' : 'AIV0', 'sv-label');
    s += '<rect x="' + x0 + '" y="' + y.aic + '" width="' + 12 * unit + '" height="13" class="lg-tl-track"/>'
      + '<rect x="' + x0 + '" y="' + y.aiv + '" width="' + 12 * unit + '" height="13" class="lg-tl-track"/>';
    blocks.forEach(function (b) {
      s += '<g class="lg-hit" data-node="' + b.node + '"><rect x="' + (x0 + b.s * unit + 1) + '" y="' + y[b.lane] + '" width="' + ((b.e - b.s) * unit - 2)
        + '" height="13" class="lg-tl u-' + b.lane + '"/>' + txt(x0 + b.s * unit + 5, y[b.lane] + 10, b.label, 'sv-strong') + '</g>';
    });
    s += '<line x1="' + (x0 + total * unit) + '" y1="' + (y.aic - 4) + '" x2="' + (x0 + total * unit) + '" y2="' + (y.aiv + 17) + '" class="sv-zoom"/>'
      + txt(x0 + total * unit + 6, y.aiv + 10, '共 ' + total + ' 格', 'sv-strong');
    return s;
  }

  function renderLogic() {
    const split = shownSplit(), eff = effStage(state.stage);
    let s = '<svg viewBox="0 0 720 396" role="img" aria-label="qk_pv 的循环嵌套与计算链"><defs>' + markers('lg') + '</defs>';
    s += '<g class="lg-hit" data-node="f-spmd"><rect x="12" y="8" width="696" height="322" class="lg-frame"/>'
      + txt(24, 25, 'PL.SPMD × 24', 'sv-title') + txt(124, 25, '每条 lane 一个 block', 'sv-label') + '</g>';
    s += '<g class="lg-hit" data-node="f-iter"><rect x="24" y="36" width="672" height="286" class="lg-frame"/>'
      + txt(36, 53, 'FOR qk_it · GRID-STRIDE', 'sv-title') + txt(210, 53, '按 qk_order 取 (token, 稀疏块)', 'sv-label') + '</g>';
    s += txt(540, 77, '块无效时', 'sv-label');
    const pipeNote = eff === 1 ? '两个头块依次处理，AIC 与 AIV 不重叠'
      : (state.stage > HEAD_BLOCKS ? '只有 2 个头块，stage=' + state.stage + ' 实际按 2 处理' : '64 头分 2 个头块，两块同时在飞');
    s += '<g class="lg-hit" data-node="f-pipe"><rect x="36" y="128" width="648" height="186" class="lg-frame"/>'
      + txt(48, 145, 'PL.PIPELINE × 2 · STAGE=' + state.stage, 'sv-title')
      + txt(250, 145, pipeNote, state.stage > HEAD_BLOCKS ? 'sv-label sv-warn' : 'sv-label');
    // 每格一个 stage 位：亮 = 有头块在飞；超出头块数的位画成虚线空格
    for (let i = 0; i < Math.max(state.stage, HEAD_BLOCKS); i++) {
      const cls = i < eff ? ' is-on' : (i >= HEAD_BLOCKS ? ' is-ghost' : '');
      s += '<rect x="' + (668 - (Math.max(state.stage, HEAD_BLOCKS) - 1 - i) * 14) + '" y="136" width="10" height="10" class="lg-flight' + cls + '"/>';
    }
    s += '</g>';
    LG_PHASES.forEach(function (p) {
      s += '<g class="lg-hit" data-node="' + p[0] + '"><rect x="' + p[1] + '" y="' + p[2] + '" width="' + p[3] + '" height="' + p[4] + '" class="lg-phase"/>'
        + txt(p[1] + 10, p[2] + 15, p[5], 'sv-title') + '</g>';
    });
    // softmax 段底部：32 行由谁来算
    s += split
      ? txt(288, 278, 'AIV0 算行 0–15 · AIV1 算行 16–31', 'sv-text') + txt(288, 292, '推算 · 需重新编译确认', 'sv-label sv-warn')
      : txt(288, 278, '32 行全部在 AIV0', 'sv-text') + txt(288, 292, 'AIV1 只握手，不计算', 'sv-label sv-warn');
    s += LG_EDGES.map(function (e) { return '<path d="' + e[2] + '" class="lg-edge" data-from="' + e[0] + '" data-to="' + e[1] + '"/>'; }).join('');
    Object.keys(LG).forEach(function (id) { s += logicNode(id); });
    s += timelineSvg(split, state.stage);
    s += '</svg>';
    document.getElementById('hpLogic').innerHTML = s;
    document.getElementById('hpLogicMeta').textContent = 'qk_pv · stage=' + state.stage + (split ? ' · 推算：加 UP_DOWN 切分' : ' · 循环嵌套与计算链');
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 硅片画布（viewBox 720 × 470）：GM → 24 个 block → 展开 block 0（AIC + 两个 AIV）
  // ════════════════════════════════════════════════════════════════════════════
  const CORE = { x: 384, y: 93, pitch: 13, size: 10 };
  const VBAR = { top: 186, bottom: 440, w: 36 };
  function vY(addr) { return VBAR.bottom - addr / UB_NOMINAL * (VBAR.bottom - VBAR.top); }

  // 标签自下而上排，间距不足时往上推
  function placeLabels(items, gap) {
    const sorted = items.slice().sort(function (a, b) { return b.y - a.y; });
    let last = Infinity;
    sorted.forEach(function (it) { it.y = Math.min(it.y, last - gap); last = it.y; });
    return sorted;
  }

  function aivSvg(id, x, split) {
    const idle = id === 'aiv1' && !split;
    const list = vecOf(split), hw = hwOf(split);
    const bar = id === 'aiv0' ? 'ub0' : 'ub1';
    let s = region(id, '<rect x="' + x + '" y="144" width="' + (id === 'aiv0' ? 174 : 168) + '" height="308" class="sv-box"/>'
      + txt(x + 12, 162, id === 'aiv0' ? 'AIV0 · SUBBLOCK 0' : 'AIV1 · SUBBLOCK 1', idle ? 'sv-title sv-warn' : 'sv-title'));
    const bx = x + 14;
    s += region(bar, '<rect x="' + bx + '" y="' + VBAR.top + '" width="' + VBAR.w + '" height="' + (VBAR.bottom - VBAR.top) + '" class="sv-mem"/>'
      + txt(bx, VBAR.top - 8, 'VEC', 'sv-label'));
    s += '<g class="sv-ub" data-ub="' + bar + '">';
    list.forEach(function (b) {
      const y1 = vY(b.hi), h = Math.max(1, vY(b.lo) - y1);
      s += '<rect x="' + (bx + 1) + '" y="' + y1 + '" width="' + (VBAR.w - 2) + '" height="' + h + '" class="sv-buf k-' + b.kind
        + (idle ? ' is-idle' : '') + (split ? ' is-proj' : '') + '" data-buf="' + b.id + '"/>';
    });
    s += '</g>';
    // 184 / 192
    s += '<line x1="' + (bx - 3) + '" y1="' + VBAR.top + '" x2="' + (bx + VBAR.w + 3) + '" y2="' + VBAR.top + '" class="sv-line-nominal"/>'
      + '<line x1="' + (bx - 3) + '" y1="' + vY(UB_USABLE) + '" x2="' + (bx + VBAR.w + 3) + '" y2="' + vY(UB_USABLE) + '" class="sv-line-usable"/>'
      + txt(bx + VBAR.w + 6, VBAR.top + 3, '192', 'sv-label') + txt(bx + VBAR.w + 6, vY(UB_USABLE) + 9, '184 可用', 'sv-label sv-bad');
    // 高水位
    s += '<line x1="' + (bx - 6) + '" y1="' + vY(hw) + '" x2="' + bx + '" y2="' + vY(hw) + '" stroke="#e8e8e8" stroke-width="1.4"/>';
    const lx = bx + VBAR.w + 8;
    if (idle) {
      s += txt(lx, 252, '空转', 'sv-title sv-warn') + txt(lx, 268, 'valid_shape', 'sv-label') + txt(lx, 280, '= [0, 0]', 'sv-label')
        + txt(lx, 300, '只 tpop / tfree', 'sv-label') + txt(lx, 312, '和 AIC 握手', 'sv-label')
        + txt(lx, 340, '已分配 ' + kb(hw), 'sv-text') + txt(lx, 352, '没有有效数据', 'sv-label');
      return s;
    }
    // 段标签（合并 mi / li，跳过太小的 bias 单独标在旁边）
    const labels = [
      { y: (vY(list[0].lo) + vY(list[0].hi)) / 2 + 3, t: (split ? 'c2v 槽 ' : 'c2v 槽 ') + kb(list[0].hi - list[0].lo) },
      { y: vY(list[1].lo) + 3, t: 'bias 0.5 KB' },
      { y: (vY(list[2].lo) + vY(list[2].hi)) / 2 + 3, t: '临时区 ' + kb(list[2].hi - list[2].lo) },
      { y: vY(list[3].lo) + 3, t: 'mi / li ' + bytes(list[4].hi - list[3].lo) },
      { y: (vY(list[5].lo) + vY(list[5].hi)) / 2 + 3, t: 'softmax 原地 ' + kb(list[5].hi - list[5].lo) },
      { y: vY(hw) + 3, t: '高水位 ' + kb(hw), strong: true }
    ];
    placeLabels(labels, 11).forEach(function (l) { s += txt(lx, l.y, l.t, l.strong ? 'sv-strong' : 'sv-label'); });
    if (!isReal(split, state.stage)) s += txt(lx, 210, '推算', 'sv-label sv-warn');
    return s;
  }

  function renderSvg() {
    const split = shownSplit();
    let s = '<svg viewBox="0 0 720 470" role="img" aria-label="qk_pv 在一个 block 上的真实落点"><defs>' + markers('hp')
      + '<pattern id="csGrid" width="6" height="6" patternUnits="userSpaceOnUse"><rect x="1" y="1" width="4" height="4" rx=".5" fill="none" stroke="#555" stroke-width=".55"/></pattern>'
      + '<pattern id="csIdle" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="#1a1a1a"/><line x1="0" y1="0" x2="0" y2="5" stroke="#5a5a5a" stroke-width="1.2"/></pattern>'
      + '</defs>';
    // GM：与参考页面共享存储网格图元，保留 CSA 对象和焦点映射。
    s += '<rect x="12" y="4" width="696" height="78" rx="12" fill="#282828"/>' + txt(24, 25, 'GM · HBM', 'sv-title');
    [
      ['gm-q', 24, 124, 'q', '[T, 64, 512] BF16'],
      ['gm-ori', 156, 136, 'ori_kv · 滑窗', '分页 · 32 行/页'],
      ['gm-cmp', 300, 136, 'cmp_kv · 压缩', 'top-k 散读'],
      ['gm-plan', 444, 128, 'qk_order / bias', 'qk_plan 产出'],
      ['gm-out', 580, 116, 'mi / li / oi', '部分结果']
    ].forEach(function (item) {
      s += region(item[0], '<g transform="translate(' + item[1] + ' 32)">' + PathGridGlyphs.memory(item[3], item[4], item[2], 42) + '</g>');
    });

    // 24 个 block 概览
    let cores = '';
    for (let i = 0; i < NUM_QK_CORES; i++) {
      const x = CORE.x + i * CORE.pitch;
      cores += '<g class="sv-core-unit' + (i === 0 ? ' is-expanded' : '') + '"><path class="sv-core-pins" d="M' + (x + 2) + ' ' + (CORE.y - 1) + 'v12 M' + (x + 8) + ' ' + (CORE.y - 1) + 'v12"/><rect x="' + x + '" y="' + CORE.y + '" width="' + CORE.size + '" height="' + CORE.size + '" rx="2" class="sv-core' + (i === 0 ? ' is-zoom' : '') + '"/></g>';
    }
    s += region('cores', '<rect x="' + (CORE.x - 100) + '" y="86" width="' + (NUM_QK_CORES * CORE.pitch + 112) + '" height="24" class="sv-hit"/>'
      + txt(CORE.x - 8, 102, 'PL.SPMD × 24', 'sv-title', 'end') + '<g class="sv-cores">' + cores + '</g>');
    s += '<line x1="' + (CORE.x + 5) + '" y1="' + (CORE.y + CORE.size) + '" x2="' + (CORE.x + 5) + '" y2="122" class="sv-zoom"/>';
    s += '<rect x="12" y="122" width="696" height="340" class="sv-frame"/>'
      + txt(CORE.x + 13, 137, '展开 BLOCK 0 · 1 个 AIC + 2 个 AIV', 'sv-title');

    // AIC
    s += region('aic', '<rect x="24" y="144" width="270" height="308" class="sv-box"/>' + txt(36, 162, 'AIC · CUBE', 'sv-title'));
    // L1 条按 192 KB 定宽，stage 变了 v2c 槽跟着变宽变窄
    const l1x = 36, l1w = 246, l1k = l1w / (192 * KB), l1 = l1Of(state.stage), l1used = l1[l1.length - 1].hi;
    let l1s = txt(36, 178, 'L1 · MAT  已用 ' + kb(l1used) + (state.stage !== 2 ? '（推算）' : ''), state.stage !== 2 ? 'sv-label sv-warn' : 'sv-label');
    l1.forEach(function (b) {
      const x = l1x + b.lo * l1k, w = (b.hi - b.lo) * l1k;
      l1s += region(b.id, '<rect x="' + x + '" y="184" width="' + w + '" height="24" class="sv-l1 k-' + b.kind + '"/>'
        + (w > 16 ? txt(x + 5, 200, b.name, 'sv-strong') : ''));
    });
    s += l1s;
    s += region('l0a', '<rect x="36" y="218" width="84" height="56" class="sv-mem"/><rect x="36" y="218" width="84" height="56" rx="5" fill="url(#csGrid)" opacity=".45" pointer-events="none"/>' + txt(44, 233, 'L0A · LEFT', 'sv-label')
      + '<rect x="44" y="240" width="32" height="7" class="sv-fill-lime"/><rect x="80" y="240" width="32" height="7" class="sv-fill-lime"/>'
      + txt(44, 266, '2 × 8 KB', 'sv-text'));
    s += region('l0b', '<rect x="128" y="218" width="154" height="56" class="sv-mem"/><rect x="128" y="218" width="154" height="56" rx="5" fill="url(#csGrid)" opacity=".45" pointer-events="none"/>' + txt(136, 233, 'L0B · RIGHT', 'sv-label')
      + '<rect x="136" y="240" width="68" height="7" class="sv-fill-lav"/><rect x="206" y="240" width="68" height="7" class="sv-fill-lav"/>'
      + txt(136, 266, '2 × 32 KB = 64 KB 用满', 'sv-text'));
    s += region('cube', '<rect x="36" y="284" width="246" height="62" class="sv-mem"/>' + txt(44, 299, 'CUBE', 'sv-label')
      + txt(44, 316, 'QK [32,512] × [512,128]', 'sv-strong') + txt(44, 331, 'PV [32,128] × [128,512]', 'sv-strong')
      + txt(274, 299, 'K=512 → 4 × 128', 'sv-label', 'end'));
    s += region('l0c', '<rect x="36" y="356" width="246" height="56" class="sv-mem"/><rect x="36" y="356" width="246" height="56" rx="5" fill="url(#csGrid)" opacity=".45" pointer-events="none"/>' + txt(44, 371, 'L0C · ACC', 'sv-label')
      + '<rect x="44" y="378" width="230" height="7" class="sv-fill-amber"/><rect x="44" y="378" width="57" height="7" fill="#c9a24a"/>'
      + txt(44, 402, 'qk_raw 16 KB 与 qk_oi 64 KB 同址复用', 'sv-text'));
    s += txt(36, 440, '数据：pass 36 · AllocateMemoryAddr', 'sv-label');
    // GM → L1
    s += region('gm-ori', '<path d="M224 74 V 182" class="sv-arrow"/>');
    s += region('gm-cmp', '<path d="M368 74 V 89 H 262 V 182" class="sv-arrow"/>');

    // 管道：槽数跟在飞的头块数走
    const eff = effStage(state.stage);
    function slotRow(y) {
      let r = '';
      for (let i = 0; i < eff; i++) r += '<rect x="' + (eff === 1 ? 313 : 303 + i * 20) + '" y="' + y + '" width="16" height="12" class="sv-slot"/>';
      return r;
    }
    s += region('pipe', '<rect x="296" y="268" width="50" height="100" class="sv-hit"/>'
      + txt(321, 280, 'C2V', 'sv-label', 'middle') + '<path d="M296 290 H 346" class="sv-arrow"/>' + slotRow(296)
      + txt(321, 332, 'V2C', 'sv-label', 'middle') + '<path d="M346 342 H 296" class="sv-arrow"/>' + slotRow(348)
      + txt(321, 322, eff + ' × ' + (split ? '8K' : '16K'), state.stage !== 2 || split ? 'sv-label sv-warn' : 'sv-label', 'middle'));

    // 两个 AIV
    s += aivSvg('aiv0', 350, split);
    s += aivSvg('aiv1', 528, split);
    if (!isReal(split, state.stage)) s += '<rect x="346" y="140" width="354" height="316" class="sv-proj-frame"/>';
    s += '</svg>';
    document.getElementById('hpSvg').innerHTML = s;
    document.getElementById('hpChipMeta').textContent = isReal(split, state.stage)
      ? '真实分配 · pass 36 AllocateMemoryAddr · Vec 地址条按比例'
      : '推算（stage=' + state.stage + (split ? ' · 加 UP_DOWN' : '') + '）· 以 stage=2 的真实分配为基准 · 虚线框内为推算';
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 焦点
  // ════════════════════════════════════════════════════════════════════════════
  function fromNodes(ids) {
    ids = ids.reduce(function (acc, id) { return acc.concat([id], nodeById(id).group || []); }, []);
    const f = { lines: [], regions: [], bufs: [], nodes: ids.slice() };
    ids.forEach(function (id) {
      const n = nodeById(id);
      f.lines = f.lines.concat(n.lines);
      f.regions = f.regions.concat(n.regions || []);
      f.bufs = f.bufs.concat(n.bufs || []);
    });
    return f;
  }
  function sectionFocus() {
    if (state.step === 1) return fromNodes(['split', 'kvl1', 'scale']);
    if (state.step === 2) { const f = fromNodes(['p-softmax', 'f-pipe']); f.regions = ['aic', 'pipe']; return f; }
    if (state.step === 3) return fromNodes(['split']);
    if (state.step === 4) { const f = fromNodes(['out']); f.lines = f.lines.concat([208]); return f; }
    return { lines: [], regions: [], bufs: [], nodes: [] };
  }
  function focusFor(h) {
    if (!h) return null;
    let f;
    if (h.type === 'node') f = fromNodes([h.id]);
    else if (h.type === 'line') { f = fromNodes(nodesWhere(function (n) { return n.lines.indexOf(h.no) >= 0; })); f.lines = [h.no]; }
    else if (h.type === 'buf') { f = fromNodes(nodesWhere(function (n) { return !n.frame && (n.bufs || []).indexOf(h.id) >= 0; })); f.bufs = [h.id]; f.regions = []; }
    else {
      f = fromNodes(REGION_NODES[h.id] || nodesWhere(function (n) { return (n.regions || []).indexOf(h.id) >= 0; }));
      f.regions = [h.id];
      // AIV 盒子保留 softmax 用到的缓冲；地址条点亮全部；管道只点亮 c2v 槽；其余区域不连带 Vec
      if (h.id === 'ub0' || h.id === 'ub1') f.bufs = vecOf(state.split).map(function (b) { return b.id; });
      else if (h.id === 'pipe') f.bufs = ['slots'];
      else if (h.id !== 'aiv0') f.bufs = [];
    }
    return f;
  }
  function fmtLines(lines) {
    const sorted = uniq(lines).sort(function (a, b) { return a - b; });
    const parts = [];
    sorted.forEach(function (no) {
      const last = parts[parts.length - 1];
      if (last && SHOWN.indexOf(no) === SHOWN.indexOf(last[1]) + 1) last[1] = no;
      else parts.push([no, no]);
    });
    return parts.map(function (p) { return p[0] === p[1] ? p[0] : p[0] + '–' + p[1]; }).join(' / ') || '—';
  }
  function revealLines() {
    const box = document.getElementById('hpCode');
    const hot = box.querySelectorAll('.cl.is-hot');
    if (!hot.length) return;
    const first = hot[0], last = hot[hot.length - 1];
    const top = first.offsetTop, bottom = last.offsetTop + last.offsetHeight;
    if (top >= box.scrollTop && bottom <= box.scrollTop + box.clientHeight) return;
    const target = bottom - top < box.clientHeight ? (top + bottom - box.clientHeight) / 2 : top - 24;
    box.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
  }
  function applyFocus() {
    const f = focusFor(state.hover) || sectionFocus();
    // 只有真正悬停时才把无关部分压暗；章节默认焦点不压暗，避免页面长期发灰
    document.body.classList.toggle('is-focusing', !!state.hover);
    // 只有 AIV0 的缓冲参与联动；AIV1 空转时不点亮
    // 命中行所在的那一层：开头行到段尾点亮同一条缩进导轨，告诉你现在在第几层里
    const sc = state.hover && f.lines.length ? scopeOf(f.lines[0]) : null;
    document.querySelectorAll('.cl').forEach(function (el) {
      const no = Number(el.dataset.line), i = SHOWN.indexOf(no);
      el.classList.toggle('is-hot', f.lines.indexOf(no) >= 0);
      el.classList.toggle('is-inscope', !!sc && i >= sc.from && i <= sc.to);
      el.classList.toggle('is-scope', !!sc && i === sc.from);
      el.querySelector('.cl-text').style.setProperty('--ad', sc ? sc.depth : 0);
    });
    document.querySelectorAll('.sv-region').forEach(function (el) { el.classList.toggle('is-hot', f.regions.indexOf(el.dataset.region) >= 0); });
    document.querySelectorAll('.sv-ub').forEach(function (el) { el.classList.toggle('has-hot', f.bufs.length > 0); });
    document.querySelectorAll('.sv-buf').forEach(function (el) {
      const idle = el.classList.contains('is-idle');
      el.classList.toggle('is-hot', !idle && f.bufs.indexOf(el.dataset.buf) >= 0);
    });
    document.querySelectorAll('.lg-hit').forEach(function (el) { el.classList.toggle('is-hot', f.nodes.indexOf(el.dataset.node) >= 0); });
    document.querySelectorAll('.lg-edge').forEach(function (el) {
      el.classList.toggle('is-hot', f.nodes.indexOf(el.dataset.from) >= 0 || f.nodes.indexOf(el.dataset.to) >= 0);
    });
    document.querySelectorAll('.alloc tr').forEach(function (el) {
      el.classList.toggle('is-hot', (el.dataset.region && f.regions.indexOf(el.dataset.region) >= 0) || (el.dataset.buf && f.bufs.indexOf(el.dataset.buf) >= 0));
    });
    if (state.hover && (state.hover.type !== 'line' || state.hover.fromStory)) revealLines();
    renderCaption();
  }
  function renderCaption() {
    const h = state.hover, split = shownSplit();
    let key = '', body = '';
    if (h && h.type === 'line') { key = 'line ' + h.no + ' → 逻辑 · 硅片'; body = lineCaption(h.no, split); }
    else if (h && h.type === 'node') { key = '逻辑 → line ' + fmtLines(fromNodes([h.id]).lines) + ' · 硅片'; body = nodeCaption(h.id, split); }
    else if (h && h.type === 'buf') { key = 'Vec 缓冲 → line ' + fmtLines(focusFor(h).lines); body = bufCaption(h.id, split); }
    else if (h && h.type === 'region') { key = '硅片 → 逻辑 · line ' + fmtLines(focusFor(h).lines); body = regionCaption(h.id, split); }
    if (!body) {
      key = '悬停源码行、逻辑节点或硅片区域，三处会一起亮起';
      body = split
        ? '<b>推算</b>：加上 UP_DOWN 后两个 AIV 各算 16 行，每个 Vec 高水位约 ' + kb(hwOf(true)) + '。AIC 侧仍是真实分配。'
        : '真实编译产物：AIV0 的 Vec 用了 ' + kb(hwOf(false)) + '；<b>AIV1 分到了同样的地址，但所有 tile 都是 valid_shape=[0, 0]</b>。';
    }
    document.getElementById('hpCaption').innerHTML = '<span class="cap-k">' + key + '</span>' + body;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 右栏：五个可折叠章节
  // ════════════════════════════════════════════════════════════════════════════
  function next(n, label) { return '<button class="btn btn-primary" type="button" data-act="open" data-step="' + n + '">' + label + '</button>'; }

  function body1() {
    return '<h2 class="st-h">qk_pv 和 fa_fused 几乎同构，差在三处写法</h2>'
      + '<p class="st-p">都是 <code>pl.spmd(24)</code> 按 grid-stride 取块，块内 <code>pl.pipeline(stage=2)</code> 做 QK → softmax → PV，再交给后面的合并任务。不同在于：</p>'
      + '<table class="map-table"><thead><tr><th>行</th><th>写法</th></tr></thead><tbody>'
      + '<tr data-line="225"><td>225</td><td><code>pl.create_l1</code> 显式在 L1 开 [128, 512] 的 KV 缓冲，滑窗行和压缩行都拼进这一块</td></tr>'
      + '<tr data-line="289"><td>289</td><td>用加法 bias（无效位置加 NEG_INF）代替 <code>set_validshape</code> + <code>fillpad</code></td></tr>'
      + '<tr data-line="208"><td>208</td><td><code>pl.spmd</code> <b>没有写</b> <code>optimizations=[pl.split(pl.SplitMode.UP_DOWN)]</code>，fa_fused 写了</td></tr>'
      + '</tbody></table>'
      + '<p class="note">第三处在源码里只是少了一个参数，读代码看不出后果。下一步看编译器给出的真实分配。</p>'
      + '<p class="note">281 行的 <code>stage</code> 可以直接点选 1–4：管道槽数、L1 和 Vec 的占用，以及逻辑图下方的时间线会跟着变。只有 stage=2 是真实产物，其余是推算。</p>'
      + '<div class="st-actions">' + next(2, '展开：编译器把它放在哪') + '</div>';
  }

  function allocRow(name, sp, off, size, attr) {
    return '<tr ' + attr + '><td>' + name + '</td><td class="sp">' + sp + '</td><td class="num">' + off + '</td><td class="num">' + size + '</td></tr>';
  }
  function body2() {
    const rows = '<tr class="grp"><td colspan="4">AIC · qk_pv_aic</td></tr>'
      + allocRow('v2c 管道槽', 'Mat', '0', '2 × 16 KB', 'data-region="l1-slot"')
      + allocRow('<code>qk_kv</code> [128,512]', 'Mat', '32 KB', '128 KB', 'data-region="l1-kv"')
      + allocRow('<code>qk_q_tile</code> [32,512]', 'Mat', '160 KB', '32 KB', 'data-region="l1-q"')
      + allocRow('Q 的 K 切片', 'Left', '0 / 8 KB', '2 × 8 KB', 'data-region="l0a"')
      + allocRow('K / P 切片', 'Right', '0 / 32 KB', '2 × 32 KB', 'data-region="l0b"')
      + allocRow('qk_raw → qk_oi', 'Acc', '0', '64 KB', 'data-region="l0c"')
      + '<tr class="grp"><td colspan="4">AIV0 · qk_pv_aiv</td></tr>'
      + vecOf(false, 2).map(function (b) {
        return allocRow(b.name, 'Vec', b.lo < KB ? b.lo : kb(b.lo), bytes(b.hi - b.lo), 'data-buf="' + b.id + '"');
      }).join('');
    const stageNote = state.stage === 2 ? ''
      : '<div class="callout is-plain">下表是 stage=2 的真实分配。当前 stage=' + state.stage + '，推算为 ' + effStage(state.stage) + ' 个管道槽：'
        + 'L1 已用 ' + kb(l1Of(state.stage)[2].hi) + '，AIV0 的 Vec 高水位 ' + kb(hwOf(state.split)) + '，见硅片画布的虚线框。</div>';
    return '<h2 class="st-h">这是编译器真正给出的地址，不是估算 <span class="tag-real">真实</span></h2>'
      + stageNote
      + '<p class="st-p">悬停表格行，右边对应的存储块会亮起。</p>'
      + '<table class="alloc"><thead><tr><th>内容</th><th>空间</th><th>起始</th><th>大小</th></tr></thead><tbody>' + rows + '</tbody></table>'
      + '<p class="st-sec">和估算对照</p>'
      + '<div class="callout is-plain">按 Lens 工作台的口径（每个中间量各占一块，再按 stage=2 复制一份），Vec 会估成 <b>' + kb(NAIVE.staged) + '</b>，判为超过 192 KB。'
      + '真实只有 <b>' + kb(hwOf(false, 2)) + '</b>：5 个 [32,128] 中间量都原地写在 49920 起的同一块 16 KB 里；Vec 里也没有第二份 softmax 链。'
      + '<code>pl.pipeline</code> 在 pass 28 之后已改写成普通循环，AIC / AIV 之间靠 2 槽 × 16 KB 的管道交接。槽数是否由 stage 决定，产物里看不出来，需对照编译器实现。</div>'
      + '<p class="note">这也提醒 fa_fused 主线：那一页的 Vec 数字同样是估算，结论要以编译产物为准。</p>'
      + '<p class="note">来源：<code>' + DUMP + '</code> 第 2761–2771 行（AIC）、2951–2961 行（AIV）。</p>'
      + '<div class="st-actions">' + next(3, '展开：第二个 AIV 在做什么') + '</div>';
  }

  function body3() {
    return '<h2 class="st-h">每个 AIC 配 2 个 AIV，第二个只在握手 <span class="tag-real">真实</span></h2>'
      + '<p class="st-p">a2a3 上 24 个 AIC 配 48 个 AIV。编译产物给 <code>qk_pv_aiv</code> 标了双 AIV 派发，但两条分支不一样：</p>'
      + '<div class="evidence"><i># 2952</i>  pl.func_attr({"dual_aiv_dispatch": True})\n'
      + '<i># 2962</i>  if subblock_idx == 0:   <i># AIV0：完整的 softmax，写回 mi / li</i>\n'
      + '<i># 3023</i>  else:                  <i># AIV1</i>\n'
      + '<i># 3038</i>    qk_raw = tpop_from_aic(split=0)   <b>valid_shape=[0, 0]</b>\n'
      + '<i># 3048</i>    qk_exp_bf16 = cast(...)            <b>valid_shape=[0, 0]</b>\n'
      + '<i># 3055</i>    sparse_blk_mi = sparse_blk_mi      <i># 不写回</i></div>'
      + '<p class="st-p">AIV1 照常从管道取数、释放槽位，让 AIC 不被卡住，但 tile 的有效形状都是 0：不算 softmax，也不写 mi / li。<b>24 个 AIV 的算力没有用上</b>。</p>'
      + '<p class="st-sec">在哪一环能看到</p>'
      + '<ol class="timeline">'
      + '<li class="t-pass"><span class="t-k">读源码</span><span class="t-v"><b>看不到</b>：只是少写了一个参数</span></li>'
      + '<li class="t-pass"><span class="t-k">编译</span><span class="t-v"><b>通过</b>：没有警告</span></li>'
      + '<li class="t-pass"><span class="t-k">上板</span><span class="t-v"><b>结果正确</b>：空转不影响数值</span></li>'
      + '<li class="t-stop"><span class="t-k">编译产物 · 本页</span><span class="t-v"><b>能看到</b>：subblock 1 全部 valid_shape=[0, 0]</span></li>'
      + '<li class="t-fail"><span class="t-k">Profiling</span><span class="t-v"><b>才能回答</b>：AIV 是不是这一段的瓶颈</span></li>'
      + '</ol>'
      + '<div class="callout is-plain">这不是正确性问题。值不值得改，取决于 AIV 有没有拖慢这一段：如果 AIC 两次 K=512 的 matmul 加上 L1 拼块本来就更慢，AIV 空着也不影响端到端。</div>'
      + '<p class="note">原因推断：fa_fused 写了 <code>pl.split(pl.SplitMode.UP_DOWN)</code>，qk_pv 没写，编译器就把 32 行整块交给 AIV0（<code>tpop_from_aic(split=0)</code>）。这一点需要加上切分重新编译来确认。</p>'
      + '<div class="st-actions">' + next(4, '展开：怎么改') + '</div>';
  }

  function body4() {
    const opt = function (key, title, sub, res, cost, scope) {
      const chosen = state.chosen === key;
      return '<div class="opt' + (state.preview === key ? ' is-preview' : '') + (chosen ? ' is-chosen' : '') + '" data-opt="' + key + '" tabindex="0">'
        + '<div class="opt-head"><span class="opt-id">' + key + '</span><span class="opt-title">' + title + '</span><span class="opt-sub">' + sub + '</span></div>'
        + '<div class="opt-res">' + res + '</div><p class="opt-cost">' + cost + '</p>'
        + '<div class="opt-foot"><span class="opt-scope">' + scope + '</span>'
        + '<button class="btn' + (chosen ? '' : ' btn-primary') + '" type="button" data-act="choose" data-opt="' + key + '">' + (chosen ? '已采用' : '采用并复查') + '</button></div></div>';
    };
    return '<h2 class="st-h">加上切分，还是保持现状</h2>'
      + '<p class="st-p">悬停卡片，逻辑图和两个 AIV 会切到这个方案的样子。</p>'
      + opt('A', '加 UP_DOWN 切分', '208 行加一个参数',
        '<span>AIV 用上 <b class="is-good">2 / 2</b></span><span>每个 Vec <b>≈ ' + kb(hwOf(true)) + '</b><span class="tag-proj">推算</span></span>',
        '两个 AIV 各算 16 行。297 行的写回循环本来就按 16 行一半组织，形状上是顺的。<em>C2V 搬运怎么切、写回是否仍正确，要重新编译后看产物</em>。',
        '改动 · 算子 1 行')
      + opt('B', '保持现状', '不改',
        '<span>AIV 用上 <b class="is-warn">1 / 2</b></span><span>Vec <b>' + kb(hwOf(false)) + '</b>'
          + (isReal(false, state.stage) ? '<span class="tag-real">真实</span>' : '<span class="tag-proj">推算</span>') + '</span>',
        '如果瓶颈在 AIC（两次 K=512 的 matmul，加上 L1 里逐行拼压缩 KV），AIV 空着不影响端到端。<em>要拿 profiling 证明 AIV 不在关键路径上</em>。',
        '改动 · 无')
      + '<p class="note">两个方案都需要一份证据：A 要编译产物，B 要 profiling。本页只能指出现象和原因。</p>';
  }

  function body5() {
    if (!state.chosen) {
      return '<h2 class="st-h">还没有选方案</h2><div class="callout is-plain">先在“改法”里选一个，再看要补哪些证据。</div>'
        + '<div class="st-actions">' + next(4, '展开改法') + '</div>';
    }
    const row = function (name, a, b, ev, evText) {
      return '<tr><td>' + name + '</td><td class="v">' + a + '</td><td class="v">' + b + '</td><td><span class="ev ' + ev + '">' + evText + '</span></td></tr>';
    };
    const A = state.chosen === 'A';
    const table = '<table class="cmp"><thead><tr><th>项目</th><th>现状</th><th>' + (A ? '方案 A' : '方案 B') + '</th><th>依据</th></tr></thead><tbody>'
      + row('参与计算的 AIV', '1 / 2', A ? '2 / 2' : '1 / 2', A ? 'e-build' : 'e-est', A ? '需重新编译' : '编译产物')
      + row('每个 AIV 的 softmax 行数', '32 / 0', A ? '16 / 16' : '32 / 0', A ? 'e-build' : 'e-est', A ? '推算' : '编译产物')
      + row('每个 AIV 的 Vec 高水位', kb(hwOf(false)), A ? '≈ ' + kb(hwOf(true)) : kb(hwOf(false)), A ? 'e-build' : 'e-est', A ? '推算' : '编译产物')
      + row('AIC 侧分配', 'L0B 64 / 64 KB', '不变', 'e-est', '编译产物')
      + row('端到端耗时', '—', '—', 'e-board', '需 profiling')
      + '</tbody></table>';
    const checks = A
      ? ['重新编译，查看 <code>25_after_SplitVectorKernel</code>：subblock 1 的 tile 不再是 valid_shape=[0, 0]',
        '查看 <code>36_after_AllocateMemoryAddr</code>：两个 AIV 的 Vec 高水位与推算是否一致',
        '上板与 golden_sparse_attn 逐元素对比，确认 mi / li 的写回行号没有错位',
        'Profiling 对比改前改后的 qk_pv 耗时，确认 AIV 原本在关键路径上']
      : ['Profiling 看 qk_pv 的 AIC / AIV 时间线：AIV0 的 softmax 是否被 AIC 的 matmul 和 L1 拼块完全盖住',
        '如果 AIV0 在关键路径上，回到方案 A'];
    return '<h2 class="st-h">' + (A ? '加了切分之后，要拿编译产物和 profiling 证明' : '保持现状，要拿 profiling 证明') + '</h2>'
      + table
      + '<p class="st-sec">接下来要看</p>'
      + '<ul class="checklist">' + checks.map(function (t) { return '<li><span>' + t + '</span></li>'; }).join('') + '</ul>'
      + '<div class="st-actions"><button class="btn" type="button" data-act="open" data-step="4">换个方案</button>'
      + '<button class="btn" type="button" data-act="reset">从头再走一遍</button></div>';
  }

  function sectionSummary(n) {
    if (n === 1) return state.split ? { t: 'stage=' + state.stage + ' · 已加 UP_DOWN', cls: 'is-warn' }
      : { t: 'stage=' + state.stage + ' · 未写 UP_DOWN', cls: state.stage !== 2 ? 'is-warn' : '' };
    if (n === 2) return { t: 'Vec ' + kb(hwOf(state.split)) + ' · L0B 64/64' + (isReal(state.split, state.stage) ? '' : ' · 推算'), cls: isReal(state.split, state.stage) ? '' : 'is-warn' };
    if (n === 3) return state.split ? { t: '推算：2 个 AIV 都在算', cls: 'is-good' } : { t: '24 / 48 个 AIV 空转', cls: 'is-warn' };
    if (n === 4) return state.chosen ? { t: '已采用 ' + state.chosen + (state.chosen === 'A' ? ' · 加切分' : ' · 保持'), cls: 'is-good' } : { t: '2 个方案' };
    return state.chosen ? { t: state.chosen === 'A' ? '待重新编译 / profiling' : '待 profiling', cls: 'is-warn' } : { t: '—' };
  }

  function renderStory() {
    const bodies = [body1, body2, body3, body4, body5];
    document.getElementById('hpStory').innerHTML = SECTIONS.map(function (sec, i) {
      const n = i + 1, open = state.step === n, sum = sectionSummary(n);
      return '<section class="sec' + (open ? ' is-open' : '') + '" data-sec="' + n + '">'
        + '<button class="sec-head" type="button" data-act="toggle" data-step="' + n + '" aria-expanded="' + open + '" aria-controls="cs-body-' + n + '">'
        + '<span class="sec-no">0' + n + '</span><span class="sec-name">' + sec.name + '</span><span class="sec-q">' + sec.q + '</span>'
        + '<span class="sec-sum ' + (sum.cls || '') + '">' + sum.t + '</span><span class="sec-chev" aria-hidden="true"></span></button>'
        + (open ? '<div class="sec-body" id="cs-body-' + n + '">' + bodies[i]() + '</div>' : '')
        + '</section>';
    }).join('');
  }

  function renderStatus() {
    const el = document.getElementById('hpStatus');
    const span = Math.max.apply(null, schedule(state.split, state.stage).map(function (b) { return b.e; }));
    const tail = ' · stage=' + state.stage + '（' + effStage(state.stage) + ' 槽）· 时间线示意 ' + span + ' 格';
    if (state.split) {
      el.textContent = '推算 · 加 UP_DOWN 后两个 AIV 各算 16 行 · 每个 Vec ≈ ' + kb(hwOf(true)) + tail;
    } else {
      el.textContent = '! AIV1 空转（valid_shape=[0, 0]）· AIV0 Vec ' + kb(hwOf(false)) + ' / 184 KB' + tail;
    }
    el.className = 'hp-status-main is-warn';
    document.getElementById('hpStatusDim').textContent = isReal(state.split, state.stage)
      ? '数据：真实编译产物 · 未上板 · 未 profiling'
      : '以 stage=2 的真实产物为基准推算 · 需重新编译确认';
  }

  function render() { renderCode(); renderLogic(); renderSvg(); renderStory(); renderStatus(); applyFocus(); }
  function openSection(n, scroll) {
    state.step = n; state.preview = null; state.hover = null;
    render();
    revealLines();
    if (scroll && n) {
      const el = document.querySelector('.sec[data-sec="' + n + '"]');
      if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }
  function setSplit(on) { state.split = on; state.preview = null; render(); }

  // ════════════════════════════════════════════════════════════════════════════
  // 事件
  // ════════════════════════════════════════════════════════════════════════════
  document.addEventListener('click', function (e) {
    const act = e.target.closest('[data-act]');
    if (!act) return;
    const a = act.dataset.act;
    if (a === 'split') setSplit(act.dataset.on === '1');
    else if (a === 'stage') { state.stage = Number(act.dataset.v); state.preview = null; render(); }
    else if (a === 'toggle') { const n = Number(act.dataset.step); openSection(state.step === n ? 0 : n, false); }
    else if (a === 'open') openSection(Number(act.dataset.step), true);
    else if (a === 'choose') { state.chosen = act.dataset.opt; state.step = 5; setSplit(act.dataset.opt === 'A'); }
    else if (a === 'reset') { state.step = 1; state.chosen = null; state.hover = null; state.stage = 2; setSplit(false); }
  });


  function hoverFrom(target) {
    const buf = target.closest('.sv-buf');
    if (buf && !buf.classList.contains('is-idle')) return { type: 'buf', id: buf.dataset.buf };
    const reg = target.closest('.sv-region');
    if (reg) return { type: 'region', id: reg.dataset.region };
    const node = target.closest('.lg-hit');
    if (node) return { type: 'node', id: node.dataset.node };
    const line = target.closest('.cl.is-mapped');
    if (line) return { type: 'line', no: Number(line.dataset.line) };
    const row = target.closest('.map-table tr[data-line]');
    if (row) return { type: 'line', no: Number(row.dataset.line), fromStory: true };
    const arow = target.closest('.alloc tr[data-region], .alloc tr[data-buf]');
    if (arow) return arow.dataset.buf ? { type: 'buf', id: arow.dataset.buf, fromStory: true } : { type: 'region', id: arow.dataset.region, fromStory: true };
    return null;
  }
  document.addEventListener('mouseover', function (e) {
    const h = hoverFrom(e.target);
    const opt = e.target.closest('.opt[data-opt]');
    const preview = opt ? opt.dataset.opt : null;
    const changed = JSON.stringify(h) !== JSON.stringify(state.hover);
    if (preview !== state.preview) {
      state.preview = preview; state.hover = h;
      renderLogic(); renderSvg();
      document.querySelectorAll('.opt').forEach(function (el) { el.classList.toggle('is-preview', el.dataset.opt === preview); });
      applyFocus();
    } else if (changed) { state.hover = h; applyFocus(); }
  });
  document.addEventListener('focusin', function (e) {
    const opt = e.target.closest('.opt[data-opt]');
    if (opt && opt.dataset.opt !== state.preview) { state.preview = opt.dataset.opt; renderLogic(); renderSvg(); applyFocus(); }
  });

  render();
})();
