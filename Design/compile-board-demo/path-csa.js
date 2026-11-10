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
  // open：右栏当前展开的问题 id（见 problemList）
  const state = { open: 'aiv1', split: false, stage: 2, chosen: null, hover: null, preview: null };
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
      return '<code>stage=1</code>：同一时刻只有 1 个头块在飞。AIC 做完 QK 要等 AIV 把 softmax 送回来才能做 PV，下一块 QK 也要排在后面，AIC 与 AIV 交替空等。'
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
  // AIC / AIV 排布（示意）：QK、softmax、PV 按等长画，只用来看 stage 和切分改变了什么重叠关系；问题分析里 stage 那一条用它画对比
  function schedule(split, stage) {
    const QK = 2, PV = 2, SM = split ? 1 : 2, eff = effStage(stage), out = [];
    function add(lane, s, d, label) { out.push({ lane: lane, s: s, e: s + d, label: label }); return s + d; }
    if (eff === 1) {
      let t = 0;
      for (let hb = 0; hb < HEAD_BLOCKS; hb++) {
        const qkE = add('aic', t, QK, 'QK' + hb);
        const smE = add('aiv', qkE, SM, 'SM' + hb);
        t = add('aic', smE, PV, 'PV' + hb);
      }
    } else {
      // 两个头块都在飞：AIC 连着发两个 QK，再按 softmax 回来的顺序做 PV
      const qk0 = add('aic', 0, QK, 'QK0'), qk1 = add('aic', qk0, QK, 'QK1');
      const sm0 = add('aiv', qk0, SM, 'SM0');
      const sm1 = add('aiv', Math.max(qk1, sm0), SM, 'SM1');
      const pv0 = add('aic', Math.max(qk1, sm0), PV, 'PV0');
      add('aic', Math.max(pv0, sm1), PV, 'PV1');
    }
    return out;
  }
  function renderLogic() {
    const split = shownSplit(), eff = effStage(state.stage);
    let s = '<svg viewBox="0 0 720 338" role="img" aria-label="qk_pv 的循环嵌套与计算链"><defs>' + markers('lg') + '</defs>';
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
    // 展开哪个问题，三处就默认点亮它涉及的部分；问题分析栏收着时不点亮，免得高亮没有解释
    const pane = document.getElementById('hpStoryPane');
    if (pane && pane.hidden) return { lines: [], regions: [], bufs: [], nodes: [] };
    if (state.open === 'aiv1') return fromNodes(['split']);
    if (state.open === 'vec-est') { const f = fromNodes(['p-softmax']); f.regions = ['aiv0']; return f; }
    if (state.open === 'slots' || state.open === 'stage') return fromNodes(['f-pipe']);
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
  // 右栏：问题分析。每个问题按 问题描述 → 根因分析 → 修改建议 展开；真实 / 推算 / 待确认分开标
  // ════════════════════════════════════════════════════════════════════════════
  function evTag(kind) {
    if (kind === 'real') return '<span class="tag-real">真实</span>';
    if (kind === 'proj') return '<span class="tag-proj">推算</span>';
    return '<span class="tag-open">待确认</span>';
  }

  // ── 可视化小件：右栏问题分析用，HTML + CSS，颜色沿用硅片画布（AIC lavender · AIV sky · 空转斜纹）；小标签用 path.css 的 .meter-k ──
  function vizCores(split) {
    let top = '', bottom = '';
    for (let i = 0; i < NUM_QK_CORES; i++) { top += '<i class="on"></i>'; bottom += '<i class="' + (split ? 'on is-proj' : 'idle') + '"></i>'; }
    return '<div class="viz viz-cores" role="img" aria-label="' + NUM_QK_CORES + ' 个 AIC 各配 2 个 AIV：AIV0 全部在算，AIV1 ' + (split ? '推算后也在算' : '全部空转') + '">'
      + '<span class="vc-k meter-k">AIV0</span><span class="vc-row">' + top + '</span>'
      + '<span class="vc-k meter-k">AIV1</span><span class="vc-row">' + bottom + '</span>'
      + '<span></span><span class="vc-axis meter-k"><b>AIC 0</b><b>每列 1 个 AIC · 共 ' + NUM_QK_CORES + ' 个</b><b>23</b></span></div>';
  }
  // 横向环节轨道：s = miss（没拦住）/ hit（能看到）/ open（待回答）/ todo
  function vizTrack(items, label) {
    return '<ol class="viz viz-track" aria-label="' + label + '">' + items.map(function (it) {
      return '<li class="vt-' + it.s + '"><span class="vt-dot" aria-hidden="true"></span><span class="vt-k meter-k">' + it.k + '</span><span class="vt-v">' + it.v + '</span></li>';
    }).join('') + '</ol>';
  }
  // 一行 32 行的切分条：AIV0 / AIV1 各算几行
  function splitBar(label, a0, a1, proj) {
    const seg = function (n, who) {
      return '<span class="vs-seg' + (n ? (proj ? ' on is-proj' : ' on') : ' idle') + '" style="flex:' + Math.max(n, 11) + '">' + who + (n ? ' · ' + n + ' 行' : ' 空') + '</span>';
    };
    return '<div class="vs-row"><span class="vs-k meter-k">' + label + '</span><span class="vs-bar">' + seg(a0, 'AIV0') + seg(a1, 'AIV1') + '</span></div>';
  }
  function vizChain(nodes) {
    return '<div class="viz viz-chain">' + nodes.map(function (n, i) {
      return (i ? '<span class="ch-arrow" aria-hidden="true"></span>' : '')
        + '<div class="ch-node' + (n.cls ? ' ' + n.cls : '') + '"><span class="ch-k meter-k">' + n.k + '</span><span class="ch-v">' + n.v + '</span></div>';
    }).join('') + '</div>';
  }
  // Vec 容量条：同一把尺子（0–220 KB），184 可用线与 192 标称线
  const CAP_MAX = 220 * KB;
  function capPct(b) { return (b / CAP_MAX * 100).toFixed(2) + '%'; }
  function vizCap(rows) {
    return '<div class="viz viz-cap" role="img" aria-label="' + rows.map(function (r) { return r.k + ' ' + kb(r.b); }).join('，') + '；可用 184 KB，标称 192 KB">'
      + '<div class="vk-scale"><span style="left:' + capPct(UB_USABLE) + '" class="is-usable">184 可用</span><span style="left:' + capPct(UB_NOMINAL) + '">192</span></div>'
      + rows.map(function (r) {
        const over = Math.max(0, r.b - UB_NOMINAL);
        return '<div class="vk-row"><span class="vk-k meter-k">' + r.k + '</span><span class="vk-track">'
          + '<span class="vk-fill ' + r.cls + '" style="width:' + capPct(r.b - over) + '"></span>'
          + (over ? '<span class="vk-over" style="left:' + capPct(UB_NOMINAL) + ';width:' + capPct(over) + '"></span>' : '')
          + '<i style="left:' + capPct(UB_USABLE) + '"></i><em style="left:' + capPct(UB_NOMINAL) + '"></em></span>'
          + '<span class="vk-v">' + kb(r.b) + '</span></div>';
      }).join('') + '</div>';
  }
  // Vec 组成：估算与真实按同一把尺子堆叠
  function vizStack() {
    const tile = 32 * 128 * 4, chain = 4 * tile + 32 * 128 * 2, rows = 2 * 32 * 4;
    const est = [['slot', 32768, 'c2v 管道'], ['bias', 512, 'bias'], ['chain', 2 * chain, 'softmax 链 × 2'], ['tmp', 2 * tile, '临时 × 2'], ['row', 2 * rows, 'mi / li × 2']];
    const vec = vecOf(false, 2);
    const size = function (id) { return vec.filter(function (b) { return b.id === id; }).map(function (b) { return b.hi - b.lo; })[0] || 0; };
    const kindOf = { slots: 'slot', bias: 'bias', tmp: 'tmp', mi: 'row', li: 'row', work: 'work' };
    const real = vec.map(function (b) { return [kindOf[b.id], b.hi - b.lo, b.name]; });
    const bar = function (label, parts) {
      return '<div class="vk-row"><span class="vk-k meter-k">' + label + '</span><span class="vk-track">' + parts.map(function (p) {
        return '<span class="vk-seg k-' + p[0] + '" style="width:' + capPct(p[1]) + '" title="' + p[2] + ' · ' + bytes(p[1]) + '"></span>';
      }).join('') + '<i style="left:' + capPct(UB_USABLE) + '"></i><em style="left:' + capPct(UB_NOMINAL) + '"></em></span>'
        + '<span class="vk-v">' + kb(parts.reduce(function (a, p) { return a + p[1]; }, 0)) + '</span></div>';
    };
    return '<div class="viz viz-cap"><div class="viz-cap" role="img" aria-label="组成对比：Lens 估算 ' + kb(NAIVE.staged) + '，编译产物 ' + kb(hwOf(false, 2)) + '">'
      + bar('Lens 估算', est) + bar('编译产物', real) + '</div>'
      + '<table class="vk-cmp"><thead><tr><th></th><th>估算</th><th>真实</th></tr></thead><tbody>'
      + '<tr class="is-diff"><td><span class="vk-sw k-chain"></span>softmax 中间量</td><td>' + kb(2 * chain) + ' 各占一块</td><td>' + kb(size('work')) + ' 原地</td></tr>'
      + '<tr><td><span class="vk-sw k-tmp"></span>临时</td><td>' + kb(2 * tile) + '</td><td>' + kb(size('tmp')) + '</td></tr>'
      + '<tr><td><span class="vk-sw k-slot"></span>c2v 管道</td><td>32 KB</td><td>' + kb(size('slots')) + '</td></tr>'
      + '</tbody></table></div>';
  }
  function vizPipe() {
    return '<div class="viz viz-pipe" role="img" aria-label="AIC 与 AIV 之间 2 个管道槽，每槽 16 KB；槽数来源待确认">'
      + '<span class="vp-core k-aic">AIC</span><span class="vp-slots"><span>槽 0<small>16 KB</small></span><span>槽 1<small>16 KB</small></span><em>slot_num = 2 · 从哪来？</em></span><span class="vp-core k-aiv">AIV</span></div>';
  }
  // 时间线示意：用 schedule() 的排布，两条泳道等比画（只在 stage 那条问题里出现）
  function vizGantt(rows) {
    const total = Math.max.apply(null, rows.map(function (r) { return Math.max.apply(null, r.blocks.map(function (b) { return b.e; })); }));
    return '<div class="viz viz-gantt" role="img" aria-label="' + rows.map(function (r) { return r.label + ' 共 ' + Math.max.apply(null, r.blocks.map(function (b) { return b.e; })) + ' 格'; }).join('，') + '">' + rows.map(function (r) {
      const end = Math.max.apply(null, r.blocks.map(function (b) { return b.e; }));
      const lane = function (name) {
        return '<span class="vg-lane"><span class="vg-k meter-k">' + name.toUpperCase() + '</span><span class="vg-track">' + r.blocks.filter(function (b) { return b.lane === name; }).map(function (b) {
          return '<span class="vg-b u-' + name + '" style="left:' + (b.s / total * 100) + '%;width:' + ((b.e - b.s) / total * 100) + '%">' + b.label + '</span>';
        }).join('') + '</span></span>';
      };
      return '<div class="vg-case"><span class="vg-title meter-k">' + r.label + '<b>' + end + ' 格</b></span>' + lane('aic') + lane('aiv') + '</div>';
    }).join('') + '<p class="vg-note">示意：QK、softmax、PV 按等长画，只看重叠关系</p></div>';
  }
  function vizFlight(stage) {
    let cells = '';
    for (let i = 0; i < Math.max(stage, HEAD_BLOCKS); i++) {
      const used = i < Math.min(stage, HEAD_BLOCKS);
      cells += '<span class="vf-cell' + (i < stage ? (used ? ' on' : ' empty') : ' wait') + '">' + (i < stage ? 'stage ' + (i + 1) : '排队')
        + '<small>' + (i < stage ? (used ? '头块 ' + i : '无块可填') : '头块 ' + i) + '</small></span>';
    }
    return '<div class="viz viz-flight" role="img" aria-label="64 个头分成 2 个头块；stage=' + stage + ' 时同时在飞 ' + Math.min(stage, HEAD_BLOCKS) + ' 个">'
      + '<span class="vf-k meter-k">64 头 ÷ 32 = 2 个头块</span><span class="vf-row">' + cells + '</span></div>';
  }
  function vizChecks(items) {
    return '<ul class="checklist viz-checks">' + items.map(function (it) {
      return '<li><span><span class="ev ' + it.ev + '">' + it.phase + '</span>' + it.t + '</span></li>';
    }).join('') + '</ul>';
  }

  function aivEvidence() {
    return '<div class="evidence"><i># 3023</i>  else:  <i># AIV1</i>\n'
      + '<i># 3038</i>    qk_raw = tpop_from_aic(split=0)  <b>valid_shape=[0, 0]</b>\n'
      + '<i># 3048</i>    qk_exp_bf16 = cast(...)  <b>valid_shape=[0, 0]</b></div>';
  }
  function aivDesc() {
    if (state.split) {
      return '<div class="diagnosis-primary"><span>参与计算的 AIV</span><strong class="is-good">48 / 48</strong></div>'
        + vizCores(true)
        + '<p class="note">208 行已加 UP_DOWN，两个 AIV 各算 16 行 ' + evTag('proj') + '　切分前 24 / 48 空转，要重新编译确认。</p>'
        + '<details class="story-detail"><summary>切分前的编译产物证据</summary>' + aivEvidence() + '</details>';
    }
    return '<div class="diagnosis-primary"><span>空转的 AIV</span><strong class="is-warn">24 / 48</strong></div>'
      + vizCores(false)
      + '<p class="note">AIV1 照常取数、释放槽位，但 tile 全是 <code>valid_shape=[0, 0]</code>：不算 softmax，也不写 mi / li。' + evTag('real') + '</p>'
      + aivEvidence()
      + vizTrack([
        { k: '读源码', v: '看不到', s: 'miss' }, { k: '编译', v: '无警告', s: 'miss' }, { k: '上板', v: '数值正确', s: 'miss' },
        { k: '编译产物', v: '本页可见', s: 'hit' }, { k: 'Profiling', v: '才能回答', s: 'open' }
      ], '在哪一环能看到这个问题');
  }
  function aivCause() {
    return vizChain([
      { k: '源码 · 208 行', v: '没写 <code>pl.split(pl.SplitMode.UP_DOWN)</code>', cls: 'is-src' },
      { k: '编译器', v: '<code>tpop_from_aic(split=0)</code>：32 行整块交给 AIV0' },
      { k: '结果', v: 'AIV1 只剩握手', cls: 'is-warn' }
    ])
      + '<div class="viz viz-split" role="img" aria-label="当前 AIV0 32 行、AIV1 空；加 UP_DOWN 后各 16 行（推算）">' + splitBar('当前', 32, 0, false) + splitBar('加 UP_DOWN', 16, 16, true) + '</div>'
      + '<p class="note">' + evTag('open') + ' 推断：fa_fused 写了这个参数；要加上切分重新编译确认。</p>';
  }
  function aivFix() {
    const opt = function (key, title, sub, bar, res, cost, scope) {
      const chosen = state.chosen === key;
      return '<div class="opt' + (state.preview === key ? ' is-preview' : '') + (chosen ? ' is-chosen' : '') + '" data-opt="' + key + '" tabindex="0">'
        + '<div class="opt-head"><span class="opt-id">' + key + '</span><span class="opt-title">' + title + '</span><span class="opt-sub">' + sub + '</span></div>'
        + '<div class="viz viz-split is-compact">' + bar + '</div>'
        + '<div class="opt-res">' + res + '</div>'
        + '<details class="story-detail"><summary>代价与前提</summary><p class="opt-cost">' + cost + '</p></details>'
        + '<div class="opt-foot"><span class="opt-scope">' + scope + '</span>'
        + '<button class="btn' + (chosen ? '' : ' btn-primary') + '" type="button" data-act="choose" data-opt="' + key + '">' + (chosen ? '已采用' : '采用并复查') + '</button></div></div>';
    };
    let html = '<p class="note">悬停卡片预览方案 · 这不是正确性问题，值不值得改要看 AIV 是否拖慢这一段</p>'
      + opt('A', '加 UP_DOWN 切分', '208 行加一个参数', splitBar('AIV', 16, 16, true),
        '<span>每个 Vec <b>≈ ' + kb(hwOf(true)) + '</b><span class="tag-proj">推算</span></span>',
        '两个 AIV 各算 16 行。297 行的写回循环本来就按 16 行一半组织，形状上是顺的。<em>C2V 搬运怎么切、写回是否仍正确，要重新编译后看产物</em>。',
        '改动 · 算子 1 行')
      + opt('B', '保持现状', '不改', splitBar('AIV', 32, 0, false),
        '<span>Vec <b>' + kb(hwOf(false)) + '</b>'
          + (isReal(false, state.stage) ? '<span class="tag-real">真实</span>' : '<span class="tag-proj">推算</span>') + '</span>',
        '如果瓶颈在 AIC（两次 K=512 的 matmul，加上 L1 里逐行拼压缩 KV），AIV 空着不影响端到端。<em>要拿 profiling 证明 AIV 不在关键路径上</em>。',
        '改动 · 无')
      + '<p class="note">A 要编译产物证明，B 要 profiling 证明；本页只能指出现象和原因。</p>';
    if (!state.chosen) return html;
    const A = state.chosen === 'A';
    const row = function (name, a, b, ev, evText) {
      return '<tr><td>' + name + '</td><td class="v">' + a + '</td><td class="v">' + b + '</td><td><span class="ev ' + ev + '">' + evText + '</span></td></tr>';
    };
    const checks = A
      ? [{ ev: 'e-build', phase: '编译产物', t: 'pass 25 <code>SplitVectorKernel</code>：subblock 1 不再是 valid_shape=[0, 0]' },
        { ev: 'e-build', phase: '编译产物', t: 'pass 36 <code>AllocateMemoryAddr</code>：两个 AIV 的 Vec 与推算一致' },
        { ev: 'e-board', phase: '上板', t: '对比 golden：mi / li 写回行号没有错位' },
        { ev: 'e-board', phase: 'Profiling', t: '改前改后 qk_pv 耗时' }]
      : [{ ev: 'e-board', phase: 'Profiling', t: 'AIV0 的 softmax 是否被 AIC 完全盖住' },
        { ev: 'e-est', phase: '回退', t: '若 AIV0 在关键路径上，回到方案 A' }];
    return html
      + '<p class="st-sec">方案 ' + state.chosen + ' 的复查</p>'
      + '<table class="cmp"><thead><tr><th>项目</th><th>现状</th><th>方案 ' + state.chosen + '</th><th>依据</th></tr></thead><tbody>'
      + row('参与计算的 AIV', '1 / 2', A ? '2 / 2' : '1 / 2', A ? 'e-build' : 'e-est', A ? '需重新编译' : '编译产物')
      + row('每个 AIV 的 Vec 高水位', kb(hwOf(false)), A ? '≈ ' + kb(hwOf(true)) : kb(hwOf(false)), A ? 'e-build' : 'e-est', A ? '推算' : '编译产物')
      + row('端到端耗时', '—', '—', 'e-board', '需 profiling')
      + '</tbody></table>'
      + vizChecks(checks)
      + '<div class="st-actions"><button class="btn" type="button" data-act="reset">撤销方案，恢复原配置</button></div>';
  }

  // ── Lens 估算偏大 ──
  function allocRow(name, sp, off, size, attr) {
    return '<tr ' + attr + '><td>' + name + '</td><td class="sp">' + sp + '</td><td class="num">' + off + '</td><td class="num">' + size + '</td></tr>';
  }
  function vecDesc() {
    return '<div class="diagnosis-primary"><span>估算 / 真实</span><strong>' + (NAIVE.staged / hwOf(false, 2)).toFixed(1) + '×</strong></div>'
      + vizCap([{ k: 'Lens 估算', b: NAIVE.staged, cls: 'is-est' }, { k: '编译产物', b: hwOf(false, 2), cls: 'is-real' }])
      + '<p class="note">按 Lens 口径会判为超过 192 KB；AIV0 实际只用 ' + kb(hwOf(false, 2)) + '。' + evTag('real') + '</p>';
  }
  function vecCause() {
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
    return vizStack()
      + '<p class="note">估算让每个中间量各占一块并按 stage=2 复制；产物里 5 个 [32,128] 中间量原地写在 49920 起的同一块。</p>'
      + (state.stage === 2 ? '' : '<p class="note">当前 stage=' + state.stage + '：L1 ' + kb(l1Of(state.stage)[2].hi) + '、Vec ' + kb(hwOf(state.split))
        + '，按 ' + effStage(state.stage) + ' 个管道槽推算。<span class="tag-proj">推算</span></p>')
      + '<details class="story-detail"><summary>完整分配表 · pass 36</summary>'
      + '<p>悬停表格行，硅片上对应的存储块会亮起。</p>'
      + '<table class="alloc"><thead><tr><th>内容</th><th>空间</th><th>起始</th><th>大小</th></tr></thead><tbody>' + rows + '</tbody></table>'
      + '<p>来源：<code>' + DUMP + '</code> 第 2761–2771 行（AIC）、2951–2961 行（AIV）。</p></details>';
  }
  function vecFix() {
    return vizChain([
      { k: '容量结论', v: '以编译产物为准：pass 36 <code>AllocateMemoryAddr</code>', cls: 'is-src' },
      { k: '同样适用', v: 'fa_fused 主线的 Vec 数字也是估算' }
    ]);
  }

  // ── 管道槽数来源待确认 ──
  function slotDesc() {
    return vizPipe()
      + '<p class="note"><code>' + VEC_NOTES.slots + '</code>：槽数是否由 stage 决定，产物里看不出。' + evTag('open') + '</p>';
  }
  function slotCause() {
    return vizTrack([
      { k: '源码', v: 'stage=2', s: 'miss' },
      { k: 'pass 28', v: '改成普通循环', s: 'hit' },
      { k: 'pass 36', v: '2 槽 × 16 KB', s: 'hit' },
      { k: '槽数来源', v: '没留下', s: 'open' }
    ], '槽数在编译各阶段的痕迹');
  }
  function slotFix() {
    return vizChain([
      { k: '确认', v: '对照编译器实现，找 slot_num 的来源', cls: 'is-src' },
      { k: '确认之前', v: 'stage ≠ 2 时的槽数、L1、Vec 一律标推算' }
    ]);
  }

  // ── stage 设置（只在改了 281 行时出现）──
  function stageDesc() {
    const now = schedule(state.split, state.stage), base = schedule(state.split, 2);
    return vizGantt([{ label: 'stage=' + state.stage + '（当前）', blocks: now }, { label: 'stage=2（原值）', blocks: base }])
      + '<p class="note">' + (state.stage > HEAD_BLOCKS ? '和 stage=2 完全一样：多出的 stage 不起作用。' : 'AIC 与 AIV 交替空等，总时长拉长。') + evTag('proj') + '</p>';
  }
  function stageCause() {
    return vizFlight(state.stage)
      + '<p class="note">' + (state.stage > HEAD_BLOCKS
        ? 'stage 决定同时在飞的头块数，上限是 H / QK_M_TILE = 2；多出的位置没有块可填。'
        : '只有 1 个头块在飞：AIC 做完 QK 要等 AIV 送回 softmax 才能做 PV，下一块排在后面。') + '</p>';
  }
  function stageFix() {
    return '<p class="note">改回原值 stage=2：2 个头块同时在飞，也是唯一有真实产物的配置。</p>'
      + '<div class="st-actions"><button class="btn btn-primary" type="button" data-act="stage" data-v="2">恢复 stage=2</button></div>';
  }

  function problemList() {
    const list = [{
      id: 'aiv1', sev: state.split ? 'good' : 'warn', line: 208,
      title: state.split ? '已加 UP_DOWN，待重新编译' : 'AIV1 空转',
      pill: state.split ? '推算 2 / 2 在算' : '24 / 48 空转',
      parts: [aivDesc, aivCause, aivFix]
    }];
    if (state.stage !== 2) {
      list.push({ id: 'stage', sev: 'warn', line: 281,
        title: state.stage > HEAD_BLOCKS ? '多出的 stage 无效' : 'AIC 与 AIV 交替空等',
        pill: 'stage=' + state.stage + ' 推算', parts: [stageDesc, stageCause, stageFix] });
    }
    list.push({ id: 'vec-est', sev: 'info', region: 'aiv0', title: 'Lens 估算 Vec 偏大', pill: '真实 ' + kb(hwOf(false, 2)), parts: [vecDesc, vecCause, vecFix] });
    list.push({ id: 'slots', sev: 'info', line: 281, title: '管道槽数来源', pill: '待确认', parts: [slotDesc, slotCause, slotFix] });
    return list.filter(function (p) { return p.sev === 'warn'; }).concat(list.filter(function (p) { return p.sev !== 'warn'; }));
  }

  function renderStory() {
    const box = document.getElementById('hpStory');
    const list = problemList();
    const warn = list.filter(function (p) { return p.sev === 'warn'; }).length;
    document.getElementById('hpStoryMeta').textContent = list.length + ' 个问题' + (warn ? ' · ' + warn + ' 警告' : '');
    // 改 stage / 切分会重画：记住已经展开的折叠项，重画后照原样打开
    const opened = [].map.call(box.querySelectorAll('details.story-detail[open] > summary'), function (s) {
      const opt = s.closest('[data-opt]');
      return (opt ? opt.dataset.opt + ':' : '') + s.textContent;
    });
    box.innerHTML = list.map(function (p, i) {
      const open = state.open === p.id;
      const target = p.line ? ' data-line="' + p.line + '"' : ' data-region="' + p.region + '"';
      const cls = p.sev === 'warn' ? 'is-warn' : p.sev === 'good' ? 'is-good' : '';
      return '<section class="sec' + (open ? ' is-open' : '') + '" data-pid="' + p.id + '">'
        + '<button class="sec-head" type="button" data-act="toggle" data-pid="' + p.id + '"' + target + ' aria-expanded="' + open + '" aria-controls="cs-body-' + p.id + '">'
        + '<span class="sec-no">0' + (i + 1) + '</span><span class="sec-name">' + p.title + '</span>'
        + '<span class="sec-sum ' + cls + '">' + p.pill + '</span><span class="sec-chev" aria-hidden="true"></span></button>'
        + (open ? '<div class="sec-body" id="cs-body-' + p.id + '">'
          + '<p class="st-sec">问题描述</p>' + p.parts[0]()
          + '<p class="st-sec">根因分析</p>' + p.parts[1]()
          + '<p class="st-sec">修改建议</p>' + p.parts[2]()
          + '</div>' : '')
        + '</section>';
    }).join('');
    box.querySelectorAll('details.story-detail > summary').forEach(function (s) {
      const opt = s.closest('[data-opt]');
      if (opened.indexOf((opt ? opt.dataset.opt + ':' : '') + s.textContent) >= 0) s.parentNode.open = true;
    });
  }

  function renderStatus() {
    const el = document.getElementById('hpStatus');
    const tail = ' · stage=' + state.stage + '（' + effStage(state.stage) + ' 槽）';
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

  // ════════════════════════════════════════════════════════════════════════════
  // 硅片落点标题栏：资源占用摘要。数字与画布同源；推算值用琥珀色，和画布上的“推算”一致
  // ════════════════════════════════════════════════════════════════════════════
  function renderChipStats() {
    const real = isReal(state.split, state.stage), l1used = l1Of(state.stage)[2].hi, hw = hwOf(state.split), eff = effStage(state.stage);
    const stat = function (k, v, cap, proj) {
      return '<span class="cs-stat' + (proj ? ' is-proj' : '') + '"' + (proj ? ' title="推算"' : '') + '><span class="meter-k">' + k + '</span><b>' + v + '</b>'
        + (cap ? '<small>/ ' + cap + '</small>' : '') + '</span>';
    };
    const el = document.getElementById('hpChipStats');
    el.innerHTML = stat('L1', kb(l1used), '192 KB', state.stage !== 2)
      + stat('L0B', '64 KB', '64 KB', false)
      + stat('Vec', kb(hw), '184 KB', !real)
      + stat('AIV', state.split ? '2' : '1', '2 每 AIC', state.split)
      + stat('C2V', eff + ' × ' + (state.split ? '8' : '16') + ' KB', '', !real)
      + evTag(real ? 'real' : 'proj');
    el.setAttribute('aria-label', '资源占用：L1 ' + kb(l1used) + ' / 192 KB，L0B 64 / 64 KB，AIV0 Vec ' + kb(hw) + ' / 184 KB 可用，每个 AIC 有 '
      + (state.split ? '2' : '1') + ' / 2 个 AIV 在算，C2V 管道 ' + eff + ' 槽 × ' + (state.split ? '8' : '16') + ' KB；' + (real ? '真实分配' : '含推算'));
  }

  function render() { renderCode(); renderLogic(); renderSvg(); renderStory(); renderStatus(); renderChipStats(); applyFocus(); }
  function openSection(id, scroll) {
    state.open = id; state.preview = null; state.hover = null;
    render();
    revealLines();
    if (scroll && id) {
      const el = document.querySelector('.sec[data-pid="' + id + '"]');
      if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }
  function setSplit(on) { state.split = on; state.preview = null; render(); }

  // ════════════════════════════════════════════════════════════════════════════
  // 事件
  // ════════════════════════════════════════════════════════════════════════════
  document.addEventListener('click', function (e) {
    // 顶栏开关问题分析栏（页面内联脚本先切换 hidden）：默认点亮跟着更新
    if (e.target.closest('#hpStoryToggle')) { applyFocus(); return; }
    const act = e.target.closest('[data-act]');
    if (!act) return;
    const a = act.dataset.act;
    if (a === 'split') setSplit(act.dataset.on === '1');
    else if (a === 'stage') { state.stage = Number(act.dataset.v); state.preview = null; render(); }
    else if (a === 'toggle') {
      const id = act.dataset.pid;
      openSection(state.open === id ? null : id, false);
      // 重画后把焦点还给同一个问题标题，键盘可以继续往下走
      const again = document.querySelector('.sec-head[data-pid="' + id + '"]');
      if (again && document.activeElement === document.body) again.focus();
    }
    else if (a === 'choose') { state.chosen = act.dataset.opt; state.open = 'aiv1'; setSplit(act.dataset.opt === 'A'); }
    else if (a === 'reset') { state.open = 'aiv1'; state.chosen = null; state.hover = null; state.stage = 2; setSplit(false); }
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
    const arow = target.closest('.alloc tr[data-region], .alloc tr[data-buf]');
    if (arow) return arow.dataset.buf ? { type: 'buf', id: arow.dataset.buf, fromStory: true } : { type: 'region', id: arow.dataset.region, fromStory: true };
    // 问题标题：悬停时点亮它落在的源码行或硅片区域
    const head = target.closest('.sec-head[data-pid]');
    if (head) return head.dataset.line ? { type: 'line', no: Number(head.dataset.line), fromStory: true } : { type: 'region', id: head.dataset.region, fromStory: true };
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
    const head = e.target.closest('.sec-head[data-pid]');
    if (head) { state.hover = hoverFrom(head); applyFocus(); }
  });
  // 键盘焦点离开问题标题时收回联动，否则高亮会一直停在上一条
  document.addEventListener('focusout', function (e) {
    if (!e.target.closest('.sec-head[data-pid]')) return;
    if (e.relatedTarget && e.relatedTarget.closest && e.relatedTarget.closest('.sec-head[data-pid]')) return;
    state.hover = null; applyFocus();
  });

  render();

  // 首屏引导：先从源码 225 行点亮三处，再从硅片上的 KV 缓冲反向点亮；鼠标一动就交还给用户
  (function intro() {
    const beats = [{ type: 'line', no: 225 }, { type: 'region', id: 'l1-kv' }];
    let i = 0, current = null;
    function step() {
      if (current && state.hover !== current) return; // 用户已经接手
      current = beats[i++] || null;
      state.hover = current;
      applyFocus();
      if (current) setTimeout(step, 2200);
      else document.getElementById('hpCode').scrollTo({ top: 0, behavior: 'smooth' }); // 回到 208 行开头
    }
    setTimeout(step, 500);
  })();
})();
