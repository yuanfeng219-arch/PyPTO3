/* ──────────────────────────────────────────────────────────────────────────────
 * Hardware-Native 主线
 *
 * 只讲一条路：把 fa_fused 的流水从 stage=2 调到 4 → 落到 AIV 的 Vec(UB) →
 * 越过实际可用的 184 KB、但没越过分配器按名义容量发的 192 KB → 编译、运行都不报错，
 * 输出静默 NaN（抽象自 pto-isa#170）→ 三种改法各把压力推到不同地方 → 复查。
 *
 * 三个视图共用一个焦点：源码行 ⇄ 算子逻辑图（循环与计算链）⇄ 硅片落点（区域与 Vec 缓冲）。
 *
 * 源码取自 Design/Toolkit Studio/decode_layer.py 728–798 行（缩进去掉 8 格）。
 * Vec 高水位口径与 js/model.js 的 derive() 一致：每级 5 个 [Q_HEAD_PAD, SEQ_TILE] FP32
 * + 1 个同形 BF16 + 2 个 [Q_HEAD_PAD, 1] FP32，外加常驻区 Q_HEAD_PAD × (HEAD_DIM × 4 + 8)。
 * 本页不加载 model.js，避免和工作台页面的改动互相牵连。
 * ────────────────────────────────────────────────────────────────────────────── */
(function () {
  'use strict';

  // ── 固定事实 ──
  const Q_HEAD_PAD = 16;          // 40 Q 头 / 8 KV 头 = 5 行，对齐 Cube M 16
  const Q_HEAD_BATCH = 5;
  const HEAD_DIM = 128;
  const GP_SIZE = 8;
  const NUM_CORES = 24;
  const MAX_SEQ = 4096;
  const L0B_CAP = 64 * 1024;      // Right(L0B)，沿用 Lens 示例口径
  const UB_NOMINAL = 192 * 1024;  // 分配器按名义容量发预算
  const UB_RESERVED = 8 * 1024;   // 顶部运行时保留（pto-isa#170，a2a3）
  const UB_USABLE = UB_NOMINAL - UB_RESERVED;
  const ISSUE_URL = 'https://github.com/hw-native-sys/pto-isa/issues/170';
  // 与 model.js 相同的 ragged decode batch，取 BATCH=16
  const SEQ_LENS = [3972, 1184, 2560, 640, 3328, 896, 2048, 1472,
    3712, 512, 2816, 1920, 1088, 3456, 768, 2304];

  // ── 主线上的三种配置 ──
  const BASE = { stage: 2, seq: 128, page: 128, reuse: false };
  const CHANGED = { stage: 4, seq: 128, page: 128, reuse: false };
  const OPTIONS = {
    A: {
      cfg: { stage: 3, seq: 128, page: 128, reuse: false },
      title: '降一级流水', sub: 'stage = 3', scope: '改动 · 算子 1 行',
      to: '时间', cost: function () {
        return 'AIC / AIV 少一级重叠。<em>能少多少收益，只能靠 profiling 回答</em>，本页不给数字。';
      }
    },
    B: {
      cfg: { stage: 4, seq: 64, page: 64, reuse: false },
      title: '缩小 KV 块', sub: 'SEQ_TILE = 64', scope: '改动 · 算子 + 运行时 KV 分页',
      to: '运行时与 GM', cost: function (c) {
        return '<em>runtime.page_size 必须同步改为 64</em>（逻辑块 = 物理页），这是整个 KV cache 的分页参数；'
          + 'fa_fused 真实块 ' + calc(CHANGED).blocks + ' → ' + c.blocks + '，work table 容量 ×2。';
      }
    },
    C: {
      cfg: { stage: 4, seq: 128, page: 128, reuse: true },
      title: '复用 softmax 中间量', sub: 'stage 保持 4', scope: '改动 · 依赖编译器内存复用',
      to: '证据', cost: function () {
        return '前提是 <code>set_validshape</code> / <code>fillpad</code> 不新开缓冲。'
          + '<em>在编译产物的内存映射确认之前，这只是假设</em>，不能算已修复。';
      }
    }
  };

  // ── 每级 softmax 链的缓冲（自低地址向高地址） ──
  const BUFS = [
    { id: 'scores_scaled', line: 775, name: 'scores_scaled', kind: 'fp32' },
    { id: 'scores_valid', line: 781, name: 'scores_valid', kind: 'fp32', alias: true },
    { id: 'scores', line: 782, name: 'scores', kind: 'fp32', alias: true },
    { id: 'rexp', line: 784, name: 'row_expand_sub 临时量', kind: 'fp32' },
    { id: 'exp', line: 784, name: 'exp_scores', kind: 'fp32' },
    { id: 'exp_bf16', line: 786, name: 'exp_scores_bf16', kind: 'bf16' },
    { id: 'cur_mi', line: 783, name: 'cur_mi', kind: 'row' },
    { id: 'cur_li', line: 785, name: 'cur_li', kind: 'row' }
  ];
  const ALL_BUFS = BUFS.map(function (b) { return b.id; });

  function bufSize(b, c) {
    if (c.reuse && b.alias) return 0;
    if (b.kind === 'fp32') return Q_HEAD_PAD * c.seq * 4;
    if (b.kind === 'bf16') return Q_HEAD_PAD * c.seq * 2;
    return Q_HEAD_PAD * 4;
  }
  function bufById(id) { return BUFS.filter(function (b) { return b.id === id; })[0]; }

  function calc(c) {
    const stage = Math.min(c.stage, GP_SIZE);
    const carry = Q_HEAD_PAD * (HEAD_DIM * 4 + 8);
    const per = BUFS.reduce(function (s, b) { return s + bufSize(b, c); }, 0);
    const hw = per * stage + carry;
    // 自低地址排布，记下每块的地址区间，用来找出落进保留区的缓冲
    const layout = [{ id: 'carry', stage: -1, lo: 0, hi: carry }];
    let addr = carry;
    for (let s = 0; s < stage; s++) {
      BUFS.forEach(function (b) {
        const size = bufSize(b, c);
        if (!size) return;
        layout.push({ id: b.id, stage: s, lo: addr, hi: addr + size });
        addr += size;
      });
    }
    const intruders = layout.filter(function (r) { return r.hi > UB_USABLE; });
    const blocks = SEQ_LENS.reduce(function (s, L) { return s + Math.ceil(L / c.seq); }, 0);
    let verdict = 'ok';
    if (hw > UB_NOMINAL) verdict = 'compile';
    else if (hw > UB_USABLE) verdict = 'silent';
    else if (hw > UB_USABLE * 0.92) verdict = 'tight';
    return {
      cfg: c, stage: stage, per: per, carry: carry, hw: hw, layout: layout, intruders: intruders,
      verdict: verdict, bad: verdict === 'silent' || verdict === 'compile',
      kTile: c.seq * HEAD_DIM * 2, kDB: c.seq * HEAD_DIM * 4,
      acc: Q_HEAD_PAD * c.seq * 4, blocks: blocks, tableCap: SEQ_LENS.length * (MAX_SEQ / c.seq),
      pageOk: c.seq === c.page
    };
  }

  // ── 格式化 ──
  function kb(n) {
    const v = n / 1024;
    return (Math.abs(v - Math.round(v)) < 0.05 ? Math.round(v) : v.toFixed(1)) + ' KB';
  }
  function bytes(n) { return n < 1024 ? n + ' B' : kb(n); }
  function pct(n) { return Math.round(n * 100) + '%'; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (ch) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch]; }); }
  function same(a, b) { return a.stage === b.stage && a.seq === b.seq && a.page === b.page && a.reuse === b.reuse; }
  function chosenKey(c) {
    return Object.keys(OPTIONS).filter(function (k) { return same(OPTIONS[k].cfg, c); })[0] || null;
  }
  function uniq(list) { return list.filter(function (v, i) { return list.indexOf(v) === i; }); }

  // ════════════════════════════════════════════════════════════════════════════
  // 源码节选
  // ════════════════════════════════════════════════════════════════════════════
  const SOURCE = [
    { no: 728, t: 'with pl.spmd(' },
    { no: 729, t: '    NUM_CORES,' },
    { no: 730, t: '    name_hint="fa_fused",' },
    { no: 731, t: '    optimizations=[pl.split(pl.SplitMode.UP_DOWN)],' },
    { no: 732, t: '    deps=[work_tid, rope_grp_tids[0]],' },
    { no: 733, t: ') as fa_tid:' },
    { no: 734, t: '    fa_core = pl.get_block_idx()' },
    { fold: 9 },
    { no: 744, t: '    for fa_w in pl.range(fa_core, fa_total_blocks, NUM_CORES):' },
    { fold: 13 },
    { no: 758, t: '        fa_pbid = pl.cast(pl.tensor.read(block_table, [fa_b * max_blocks_per_seq + sb]), pl.INDEX)' },
    { fold: 5 },
    { no: 764, t: '        for gp in pl.pipeline(GP_SIZE, stage=§):' },
    { fold: 3 },
    { no: 768, t: '            q_padded = all_q_padded[q_pad_row_g : q_pad_row_g + Q_HEAD_PAD, :]' },
    { fold: 4 },
    { no: 773, t: '            k_tile = k_cache[cache_row : cache_row + SEQ_TILE, :]' },
    { no: 774, t: '            raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)' },
    { no: 775, t: '            scores_scaled = pl.mul(raw_scores, ATTN_SCALE)' },
    { fold: 5 },
    { no: 781, t: '            scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_BATCH, valid_len)' },
    { no: 782, t: '            scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)' },
    { no: 783, t: '            cur_mi = pl.row_max(scores)' },
    { no: 784, t: '            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))' },
    { no: 785, t: '            cur_li = pl.row_sum(exp_scores)' },
    { no: 786, t: '            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)' },
    { fold: 2 },
    { no: 789, t: '            v_tile = v_cache[cache_row : cache_row + SEQ_TILE, :]' },
    { no: 790, t: '            oi_tmp = pl.matmul(exp_scores_bf16, v_tile, out_dtype=pl.FP32)' },
    { fold: 5 },
    { no: 796, t: '            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [g_base + sb * Q_HEAD_PAD, 0])' },
    { no: 797, t: '            all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [g_base + sb * Q_HEAD_PAD, 0])' },
    { no: 798, t: '            all_cur_li = pl.assemble(all_cur_li, cur_li, [g_base + sb * Q_HEAD_PAD, 0])' }
  ];

  // ════════════════════════════════════════════════════════════════════════════
  // 联动的单一真相源：算子逻辑节点 = 源码行 + 硅片区域 + Vec 缓冲
  // 循环框也是节点；源码行和硅片区域上的联动都从这张表反查。
  // ════════════════════════════════════════════════════════════════════════════
  const NODES = [
    { id: 'f-spmd', frame: true, lines: [728, 729, 730, 731, 732, 733, 734], regions: ['cores', 'aic', 'aiv'] },
    { id: 'f-stride', frame: true, lines: [744], regions: ['cores', 'gm-table'] },
    { id: 'f-pipe', frame: true, lines: [764], regions: ['ub'], bufs: ALL_BUFS },
    { id: 'tbl', unit: 'gm', lines: [758], regions: ['gm-table', 'gm-kcache', 'gm-vcache'],
      name: 'block_table → 物理页', sub: function (c) { return 'fa_pbid · 一页 ' + c.page + ' 行'; } },
    { id: 'q', unit: 'gm', lines: [768], regions: ['l0a'], name: 'q_padded', sub: function () { return '[16,128] BF16'; } },
    { id: 'k', unit: 'gm', lines: [773], regions: ['gm-kcache', 'l0b'], name: 'k_tile', sub: function (c) { return '[' + c.seq + ',128] BF16'; } },
    { id: 'qk', unit: 'aic', lines: [774], regions: ['cube', 'l0c', 'ch-c2v'], name: 'QK matmul', sub: function () { return 'Q·Kᵀ → FP32'; } },
    { id: 'scale', unit: 'aiv', lines: [775], bufs: ['scores_scaled'], name: '× scale', sub: function () { return 'scores_scaled'; } },
    { id: 'valid', unit: 'aiv', lines: [781], bufs: ['scores_valid'], name: 'set_validshape', sub: function () { return '只标 5 行有效'; } },
    { id: 'fill', unit: 'aiv', lines: [782], bufs: ['scores'], name: 'fillpad −inf', sub: function () { return '填充行不进 max'; } },
    { id: 'max', unit: 'aiv', lines: [783], bufs: ['cur_mi'], name: 'row_max', sub: function () { return 'cur_mi'; } },
    { id: 'exp', unit: 'aiv', lines: [784], bufs: ['rexp', 'exp'], name: 'exp(x − max)', sub: function () { return 'exp_scores'; } },
    { id: 'sum', unit: 'aiv', lines: [785], bufs: ['cur_li'], name: 'row_sum', sub: function () { return 'cur_li'; } },
    { id: 'cast', unit: 'aiv', lines: [786], bufs: ['exp_bf16'], regions: ['ch-v2c'], name: 'cast BF16', sub: function () { return '→ V2C 送回 Cube'; } },
    { id: 'v', unit: 'gm', lines: [789], regions: ['gm-vcache', 'l0b'], name: 'v_tile', sub: function (c) { return '[' + c.seq + ',128] BF16'; } },
    { id: 'sv', unit: 'aic', lines: [790], regions: ['cube', 'l0c', 'ch-v2c'], name: 'SV matmul', sub: function () { return 'P·V → oi_tmp'; } },
    { id: 'out', unit: 'gm', lines: [796, 797, 798], regions: ['gm-out'], name: '写回部分结果', sub: function () { return 'oi · mi · li → GM'; } }
  ];
  const SOFTMAX_NODES = ['scale', 'valid', 'fill', 'max', 'exp', 'sum', 'cast'];
  function nodeById(id) { return NODES.filter(function (n) { return n.id === id; })[0]; }
  function nodesWhere(fn) { return NODES.filter(fn).map(function (n) { return n.id; }); }
  // 源码行上能否悬停：有逻辑节点覆盖的行
  function lineMapped(no) { return NODES.some(function (n) { return n.lines.indexOf(no) >= 0; }); }

  function lineCaption(no, k) {
    const c = k.cfg, n = k.stage;
    const buf = BUFS.filter(function (b) { return b.line === no; });
    if (buf.length) {
      return buf.map(function (b) {
        const size = bufSize(b, c);
        const shape = b.kind === 'row' ? '[16, 1] FP32' : '[16, ' + c.seq + '] ' + (b.kind === 'bf16' ? 'BF16' : 'FP32');
        if (!size) return '<b>' + b.name + '</b> · 假设与 <code>scores_scaled</code> 共用地址，不另占 Vec';
        return '<b>' + b.name + '</b> <code>' + shape + '</code> · 每级 ' + bytes(size) + ' × ' + n + ' 级 = <b>'
          + bytes(size * n) + '</b> → AIV · Vec(UB)';
      }).join('<br>') + (no === 786 ? '<br>经 V2C 送回 Cube 做 SV matmul。' : '');
    }
    const map = {
      728: '<code>pl.spmd(NUM_CORES)</code>：' + NUM_CORES + ' 个 block 并行，<b>每个 block 用同一份片上地址分配</b>。',
      729: '<code>NUM_CORES = ' + NUM_CORES + '</code>：常驻 block 数。',
      730: '任务名 <code>fa_fused</code>：QK → softmax → SV 融合在一个根任务里。',
      731: '<code>UP_DOWN</code> 切分：同一个根任务里 AIC 做 matmul、AIV 做 softmax，经 C2V / V2C 交接。',
      732: '依赖 work table 构建与 RoPE 写 KV 完成之后再启动。',
      733: '返回任务句柄 <code>fa_tid</code>，供后面的 online_softmax 依赖。',
      734: '每个 block 取自己的编号，再从 work table 里按 grid-stride 取块。',
      744: 'grid-stride 从 work table 取号：batch 16 的真实块 <b>' + k.blocks + '</b> 个，摊到 ' + NUM_CORES + ' 个 block，每个 block 约 ' + Math.ceil(k.blocks / NUM_CORES) + ' 块。',
      758: 'block_table 把逻辑块翻成物理页。前提：<code>SEQ_TILE == page_size</code>（当前 ' + c.seq + ' / ' + c.page + '）。',
      764: '<code>stage=' + c.stage + '</code>：同时有 <b>' + n + ' 个头</b>在飞，Vec 里驻留 ' + n + ' 份 softmax 链，每份 ' + kb(k.per) + '，合计 <b>' + kb(k.per * n) + '</b>。',
      768: '<code>q_padded [16, 128] BF16</code> 4 KB → L0A(Left)。5 行真实 Q 填到 16 行，对齐 Cube M。',
      773: '<code>k_tile [' + c.seq + ', 128] BF16</code> ' + kb(k.kTile) + '：GM 物理页 → L0B(Right)；双缓冲估算 ' + kb(k.kDB) + ' / 64 KB。',
      774: 'QK matmul 在 Cube 上算，FP32 结果 <code>[16, ' + c.seq + ']</code> ' + kb(k.acc) + ' 进 L0C(Acc)，再经 C2V 交给 AIV。',
      789: '<code>v_tile</code> 与 k_tile 同一物理行范围 → L0B(Right)。',
      790: 'SV matmul：exp_scores_bf16 × v_tile，FP32 <code>[16, 128]</code> 8 KB 进 L0C(Acc)。',
      796: '按块写回 GM，留给后面的 online_softmax 归约。', 797: '按块写回 GM。', 798: '按块写回 GM。'
    };
    return map[no] || '';
  }
  function regionCaption(id, k) {
    const c = k.cfg;
    const map = {
      cores: '<b>pl.spmd × ' + NUM_CORES + '</b> · 每个 block 用同一份 Vec 分配，越界不是个别核的偶发问题。',
      aic: '<b>AIC · Cube 侧</b>：QK 与 SV 两次 matmul，操作数在 L0A / L0B，结果在 L0C。',
      aiv: '<b>AIV · Vector 侧</b>：softmax 链全部落在 Vec(UB)，stage 决定同时驻留几份。',
      'gm-kcache': '<b>k_cache</b>：分页存放，一页 ' + c.page + ' 行；fa_fused 每次读一个逻辑块。',
      'gm-vcache': '<b>v_cache</b>：与 k_cache 同一分页。',
      'gm-table': '<b>block_table / work table</b>：真实块 ' + k.blocks + ' 个，table 容量 ' + k.tableCap + '。',
      'gm-out': '<b>all_oi_tmp / all_cur_mi / all_cur_li</b>：每块的部分结果，留给 online_softmax。',
      l0a: '<b>L0A · Left</b>：q_padded 4 KB。容量本页未建模。',
      l0b: '<b>L0B · Right</b>：K / V 块双缓冲估算 ' + kb(k.kDB) + ' / 64 KB（' + pct(k.kDB / L0B_CAP) + '）。按完整块双缓冲估算，实际映射以编译产物为准。',
      cube: '<b>Cube</b>：QK <code>[16,128]×[128,' + c.seq + ']</code> 与 SV <code>[16,' + c.seq + ']×[' + c.seq + ',128]</code>。',
      l0c: '<b>L0C · Acc</b>：raw_scores ' + kb(k.acc) + ' 与 oi_tmp 8 KB。容量本页未建模。',
      'ch-c2v': '<b>C2V</b>：raw_scores 从 Cube 交给 Vector，搬整块 16 行。',
      'ch-v2c': '<b>V2C</b>：exp_scores_bf16 送回 Cube 做 SV。',
      ub: '<b>Vec(UB) 地址条</b>：自低地址向上，常驻区 ' + kb(k.carry) + ' + ' + k.stage + ' 级 × ' + kb(k.per) + ' = 高水位 <b>' + kb(k.hw) + '</b>。'
    };
    return map[id] || '';
  }
  function nodeCaption(id, k) {
    if (id === 'f-spmd') return lineCaption(728, k);
    if (id === 'f-stride') return lineCaption(744, k);
    if (id === 'f-pipe') {
      return lineCaption(764, k) + '<br>逻辑上 8 个头依次处理；流水把相邻 ' + k.stage + ' 个头叠在一起，所以中间量要同时存 ' + k.stage + ' 份。';
    }
    const n = nodeById(id);
    return n.lines.map(function (no) { return lineCaption(no, k); }).filter(Boolean).slice(0, 1).join('');
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 状态
  // ════════════════════════════════════════════════════════════════════════════
  const SECTIONS = [
    { name: '改动', q: '只改了一个调度参数' },
    { name: '落点', q: '它在硅片上复制了什么' },
    { name: '裁决', q: '硬件说了不，编译器没说' },
    { name: '改法', q: '压力推到哪里去' },
    { name: '复查', q: '本页能证明到哪一步' }
  ];
  // step = 当前展开的章节（0 = 全部收起）
  const state = { step: 1, cfg: Object.assign({}, BASE), preview: null, hover: null };

  function current() { return calc(state.cfg); }
  function shown() { return calc(state.preview ? OPTIONS[state.preview].cfg : state.cfg); }

  // ════════════════════════════════════════════════════════════════════════════
  // 源码
  // ════════════════════════════════════════════════════════════════════════════
  const KW = { 'for': 1, 'in': 1, 'with': 1, 'as': 1, 'if': 1, 'else': 1, 'True': 1, 'False': 1, 'None': 1, 'return': 1 };

  function hl(text, c) {
    const values = { NUM_CORES: NUM_CORES, GP_SIZE: GP_SIZE, Q_HEAD_PAD: Q_HEAD_PAD, Q_HEAD_BATCH: Q_HEAD_BATCH, SEQ_TILE: c.seq };
    const re = /("[^"]*"|'[^']*')|(\d+(?:\.\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|(.)/g;
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
        out += '<span class="' + cls + '">' + tok + '</span>';
        if (values[tok] !== undefined && cls === 'tk-const') {
          const changed = tok === 'SEQ_TILE' && c.seq !== BASE.seq;
          out += '<span class="cval' + (changed ? ' is-changed' : '') + '" title="当前配置下的值">' + values[tok] + '</span>';
        }
      } else if (m[4]) out += tok;
      else out += '<span class="tk-punc">' + esc(tok) + '</span>';
      if (!m[4]) prev = tok;
    }
    return out;
  }

  function stageControl(k) {
    let html = '<span class="stage-ctl" role="group" aria-label="pl.pipeline stage">';
    for (let s = 1; s <= 4; s++) {
      const on = state.cfg.stage === s;
      const bad = on && k.bad;
      html += '<button type="button" data-stage="' + s + '" class="' + (on ? 'is-on' : '') + (bad ? ' is-bad' : '')
        + '" aria-pressed="' + on + '">' + s + '</button>';
    }
    return html + '</span>';
  }

  function intruderLines(k) {
    const out = {};
    if (!k.bad) return out;
    out[764] = 1;
    k.intruders.forEach(function (r) { const b = bufById(r.id); if (b) out[b.line] = 1; });
    return out;
  }

  function renderCode() {
    const k = current(), c = state.cfg, alertLines = intruderLines(k);
    const html = SOURCE.map(function (row) {
      if (row.fold) return '<div class="cl-fold"><span></span><span></span><span>… ' + row.fold + ' 行</span></div>';
      let text;
      if (row.no === 764) {
        const parts = row.t.split('§');
        text = hl(parts[0], c) + stageControl(k) + hl(parts[1], c)
          + (c.stage !== BASE.stage ? '<span class="stage-was">原 stage=' + BASE.stage + '</span>' : '');
      } else text = hl(row.t, c);
      const modified = (row.no === 764 && c.stage !== BASE.stage) || ((row.no === 773 || row.no === 789) && c.seq !== BASE.seq)
        || ((row.no === 781 || row.no === 782) && c.reuse);
      const cls = ['cl', lineMapped(row.no) ? 'is-mapped' : '', modified ? 'is-modified' : '', alertLines[row.no] ? 'is-alert' : ''].join(' ');
      return '<div class="' + cls + '" data-line="' + row.no + '"><span class="cl-mark"></span><span class="cl-no">' + row.no
        + '</span><span class="cl-text">' + text + '</span></div>';
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

  // ════════════════════════════════════════════════════════════════════════════
  // 算子逻辑图：spmd × 24 ⊃ grid-stride 取块 ⊃ pipeline × 8 头 ⊃ 三段计算链
  // ════════════════════════════════════════════════════════════════════════════
  const LG = {
    // 节点位置（viewBox 720 × 300）
    tbl: [32, 58, 210], q: [44, 140, 96], k: [148, 140, 92], qk: [44, 186, 96], scale: [148, 186, 92],
    valid: [258, 140, 88], fill: [354, 140, 88], max: [450, 140, 88],
    exp: [258, 186, 88], sum: [354, 186, 88], cast: [450, 186, 88],
    v: [566, 140, 112], sv: [566, 186, 112], out: [566, 232, 112]
  };
  const LG_H = 34;
  const LG_EDGES = [
    'M92 174 V 186', 'M194 174 L 124 186', 'M140 203 H 148',
    'M240 203 H 248 V 157 H 258', 'M346 157 H 354', 'M442 157 H 450',
    'M494 174 V 180 H 302 V 186', 'M346 203 H 354', 'M302 220 V 226 H 494 V 220',
    'M538 203 H 566', 'M622 174 V 186', 'M622 220 V 232'
  ];

  function logicNode(id, k, badIds) {
    const n = nodeById(id), p = LG[id];
    const cls = 'lg-node u-' + n.unit + (badIds.indexOf(id) >= 0 ? ' is-bad' : '');
    return '<g class="lg-hit" data-node="' + id + '">'
      + '<rect x="' + p[0] + '" y="' + p[1] + '" width="' + p[2] + '" height="' + LG_H + '" rx="3" class="' + cls + '"/>'
      + '<rect x="' + p[0] + '" y="' + p[1] + '" width="3" height="' + LG_H + '" class="lg-stripe u-' + n.unit + '"/>'
      + txt(p[0] + 9, p[1] + 14, n.name, 'sv-strong') + txt(p[0] + 9, p[1] + 27, n.sub(k.cfg), 'sv-text')
      + '</g>';
  }

  function renderLogic() {
    const k = shown(), c = k.cfg;
    const badIds = k.bad ? uniq(k.intruders.map(function (r) {
      return nodesWhere(function (n) { return (n.bufs || []).indexOf(r.id) >= 0 && !n.frame; })[0];
    }).filter(Boolean)) : [];
    let s = '<svg viewBox="0 0 720 304" role="img" aria-label="fa_fused 的循环嵌套与计算链">'
      + '<defs>' + markers('lg') + '</defs>';
    // 三层循环框
    s += '<g class="lg-hit" data-node="f-spmd"><rect x="8" y="8" width="704" height="290" rx="4" class="lg-frame"/>'
      + txt(20, 25, 'PL.SPMD × ' + NUM_CORES + '  ·  每个 block 跑一份 fa_fused', 'sv-title') + '</g>';
    s += '<g class="lg-legend">'
      + '<rect x="560" y="17" width="10" height="10" class="lg-stripe u-aic"/>' + txt(574, 26, 'AIC', 'sv-label')
      + '<rect x="604" y="17" width="10" height="10" class="lg-stripe u-aiv"/>' + txt(618, 26, 'AIV', 'sv-label')
      + '<rect x="648" y="17" width="10" height="10" class="lg-stripe u-gm"/>' + txt(662, 26, 'GM 读写', 'sv-label') + '</g>';
    s += '<g class="lg-hit" data-node="f-stride"><rect x="20" y="34" width="680" height="258" rx="4" class="lg-frame"/>'
      + txt(32, 50, 'FOR fa_w IN range(core, total, ' + NUM_CORES + ')  ·  每个 block 轮到约 ' + Math.ceil(k.blocks / NUM_CORES) + ' 个 KV 块（共 ' + k.blocks + '）', 'sv-title') + '</g>';
    s += '<g class="lg-hit" data-node="f-pipe"><rect x="32" y="100" width="656" height="186" rx="4" class="lg-frame lg-frame-pipe' + (k.bad ? ' is-bad' : '') + '"/>'
      + txt(44, 116, 'FOR gp IN pl.pipeline(' + GP_SIZE + ', stage=' + c.stage + ')  ·  ' + GP_SIZE + ' 个 KV 头，同时在飞 ' + k.stage + ' 个', 'sv-title') + '</g>';
    s += logicNode('tbl', k, badIds);
    s += txt(252, 79, '逻辑块 → 物理页；SEQ_TILE 必须等于 page_size', 'sv-label');

    // 三段标题
    s += txt(44, 133, '① 打分', 'sv-label') + txt(258, 133, '② 局部 softmax', 'sv-label') + txt(566, 133, '③ 加权求和', 'sv-label');

    // softmax 段：在飞几个头，就叠几份
    for (let i = k.stage - 1; i >= 1; i--) {
      s += '<rect x="' + (250 + i * 4) + '" y="' + (124 + i * 4) + '" width="296" height="146" rx="4" class="lg-ghost' + (k.bad && i === k.stage - 1 ? ' is-bad' : '') + '"/>';
    }
    s += '<rect x="250" y="124" width="296" height="146" rx="4" class="lg-group' + (k.bad ? ' is-bad' : '') + '"/>';
    s += txt(258, 248, '× ' + k.stage + ' 份同时驻留 Vec · 每份 ' + kb(k.per), k.bad ? 'sv-text sv-bad' : 'sv-text');
    s += txt(258, 262, k.bad ? '第 ' + k.stage + ' 份的尾部落进 UB 保留区' : '份数 = 在飞的头数 = stage', k.bad ? 'sv-label sv-bad' : 'sv-label');

    s += LG_EDGES.map(function (d) { return '<path d="' + d + '" class="lg-edge"/>'; }).join('');
    ['q', 'k', 'qk', 'scale', 'valid', 'fill', 'max', 'exp', 'sum', 'cast', 'v', 'sv', 'out'].forEach(function (id) {
      s += logicNode(id, k, badIds);
    });
    s += txt(44, 262, 'Q 来自前序 RoPE，按头取 16 行', 'sv-label');
    s += '</svg>';
    document.getElementById('hpLogic').innerHTML = s;
    document.getElementById('hpLogicMeta').textContent = state.preview
      ? '预览 · 方案 ' + state.preview + ' · stage=' + c.stage + ' · SEQ_TILE=' + c.seq
      : 'fa_fused · 循环嵌套与计算链';
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 硅片画布（viewBox 720 × 470）
  // ════════════════════════════════════════════════════════════════════════════
  const BAR = { x: 418, w: 60, top: 160, bottom: 440 };
  const INSET = { x: 498, w: 80, top: 240, bottom: 400, lo: 176 * 1024, hi: 192 * 1024 };
  function ubY(addr) { return BAR.bottom - addr / UB_NOMINAL * (BAR.bottom - BAR.top); }
  function insetY(addr) { return INSET.bottom - (addr - INSET.lo) / (INSET.hi - INSET.lo) * (INSET.bottom - INSET.top); }

  function region(id, inner) { return '<g class="sv-region" data-region="' + id + '">' + inner + '</g>'; }

  function bufClass(r) {
    if (r.id === 'carry') return 'k-carry';
    return 'k-' + bufById(r.id).kind;
  }

  function renderSvg() {
    const k = shown(), c = k.cfg, ref = state.preview ? current() : null, bad = k.bad;
    let s = '<svg viewBox="0 0 720 470" role="img" aria-label="fa_fused 在一个 AI Core block 上的落点">'
      + '<defs>' + markers('hp')
      + '<pattern id="hpHatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="#2a1716"/><line x1="0" y1="0" x2="0" y2="5" stroke="#ff6b5b" stroke-width="1.4" opacity=".55"/></pattern>'
      + '<clipPath id="hpInsetClip"><rect x="' + INSET.x + '" y="' + INSET.top + '" width="' + INSET.w + '" height="' + (INSET.bottom - INSET.top) + '"/></clipPath>'
      + '</defs>';

    // GM
    s += '<rect x="12" y="8" width="696" height="74" class="sv-frame"/>' + txt(24, 25, 'GM · HBM', 'sv-title');
    s += region('gm-kcache', '<rect x="24" y="32" width="150" height="42" class="sv-mem"/>' + pages(30, 52, c.page)
      + txt(30, 46, 'k_cache · paged', 'sv-strong') + txt(30, 70, '页 = ' + c.page + ' 行', 'sv-label'));
    s += region('gm-vcache', '<rect x="182" y="32" width="150" height="42" class="sv-mem"/>' + pages(188, 52, c.page)
      + txt(188, 46, 'v_cache · paged', 'sv-strong') + txt(188, 70, '页 = ' + c.page + ' 行', 'sv-label'));
    s += region('gm-table', '<rect x="340" y="32" width="170" height="42" class="sv-mem"/>'
      + txt(346, 46, 'block / work table', 'sv-strong') + txt(346, 64, k.blocks + ' 个真实块', 'sv-text'));
    s += region('gm-out', '<rect x="518" y="32" width="178" height="42" class="sv-mem"/>'
      + txt(524, 46, 'all_oi_tmp / mi / li', 'sv-strong') + txt(524, 64, '每块的部分结果', 'sv-text'));

    // block 外框
    s += '<rect x="12" y="92" width="584" height="370" class="sv-frame"/>'
      + txt(588, 108, 'FA_FUSED · 1 个 BLOCK', 'sv-title', 'end');

    // AIC
    s += region('aic', '<rect x="24" y="116" width="262" height="336" class="sv-box"/>' + txt(36, 134, 'AIC · CUBE', 'sv-title'));
    s += '<rect x="36" y="142" width="238" height="28" class="sv-dimmem"/>' + txt(44, 160, 'L1 · Mat', 'sv-label') + txt(266, 160, '本例未建模', 'sv-label', 'end');
    s += region('l0a', '<rect x="36" y="180" width="112" height="62" class="sv-mem"/>' + txt(44, 195, 'L0A · LEFT', 'sv-label')
      + '<rect x="44" y="202" width="' + (96 * 4 / 64) + '" height="7" class="sv-fill-lime"/>'
      + txt(44, 224, 'q_padded', 'sv-strong') + txt(44, 236, '[16,128] · 4 KB', 'sv-text'));
    const l0bRatio = Math.min(1, k.kDB / L0B_CAP);
    s += region('l0b', '<rect x="156" y="180" width="118" height="62" class="sv-mem"/>' + txt(164, 195, 'L0B · RIGHT', 'sv-label')
      + '<rect x="164" y="202" width="102" height="7" fill="#262626"/>'
      + '<rect x="164" y="202" width="' + (102 * l0bRatio) + '" height="7" class="' + (k.kDB > L0B_CAP * 0.75 ? 'sv-fill-warn' : 'sv-fill-lav') + '"/>'
      + txt(164, 224, 'K / V 双缓冲', 'sv-strong') + txt(164, 236, kb(k.kDB) + ' / 64 KB 估算', 'sv-text'));
    s += region('cube', '<rect x="36" y="252" width="238" height="62" class="sv-mem"/>' + txt(44, 267, 'CUBE', 'sv-label')
      + txt(44, 286, 'QK  [16,128] × [128,' + c.seq + ']', 'sv-strong') + txt(44, 304, 'SV  [16,' + c.seq + '] × [' + c.seq + ',128]', 'sv-strong'));
    s += region('l0c', '<rect x="36" y="324" width="238" height="56" class="sv-mem"/>' + txt(44, 339, 'L0C · ACC', 'sv-label')
      + '<rect x="44" y="346" width="' + Math.max(6, k.acc / 1024 * 3) + '" height="7" class="sv-fill-amber"/>'
      + txt(44, 370, 'raw_scores ' + kb(k.acc) + ' · oi_tmp 8 KB', 'sv-text'));
    s += txt(36, 440, 'Left / Acc 容量本页未建模', 'sv-label');

    // GM → L1 → L0B 搬运
    s += region('gm-kcache', '<path d="M99 74 V 87 H 190 V 140" class="sv-arrow"/>');
    s += region('gm-vcache', '<path d="M257 74 V 140" class="sv-arrow"/>');
    s += region('l0b', '<path d="M215 170 V 178" class="sv-arrow"/>');

    // C2V / V2C
    s += region('ch-c2v', '<rect x="288" y="268" width="60" height="24" class="sv-hit"/><path d="M288 282 H 348" class="sv-arrow"/>' + txt(318, 276, 'C2V', 'sv-label', 'middle'));
    s += region('ch-v2c', '<rect x="288" y="298" width="60" height="24" class="sv-hit"/><path d="M348 312 H 288" class="sv-arrow"/>' + txt(318, 326, 'V2C', 'sv-label', 'middle'));

    // AIV
    s += region('aiv', '<rect x="350" y="116" width="236" height="336" class="sv-box"/>' + txt(362, 134, 'AIV · VECTOR', 'sv-title'));
    s += ubSvg(k, ref, bad);

    // 核阵列
    s += region('cores', '<rect x="604" y="92" width="104" height="370" class="sv-hitframe"/>'
      + txt(612, 108, 'PL.SPMD × ' + NUM_CORES, 'sv-title')
      + '<g class="sv-cores' + (bad ? ' is-bad' : '') + '">' + cores() + '</g>'
      + txt(612, 286, bad ? '同一份越界' : '同一份 Vec', bad ? 'sv-text sv-bad' : 'sv-text')
      + txt(612, 300, '分配 × ' + NUM_CORES, bad ? 'sv-text sv-bad' : 'sv-text')
      + txt(612, 326, 'grid-stride', 'sv-label')
      + txt(612, 340, k.blocks + ' 个真实块', 'sv-label')
      + txt(612, 354, '→ ' + NUM_CORES + ' 个 block', 'sv-label')
      + txt(612, 380, '每格一个 block', 'sv-label'));
    s += '</svg>';
    document.getElementById('hpSvg').innerHTML = s;
    document.getElementById('hpChipMeta').textContent = state.preview
      ? '预览 · 方案 ' + state.preview + ' · ' + OPTIONS[state.preview].title + '（虚线 = 当前配置的高水位）'
      : '示意 · 不按比例 · Vec 地址条按比例';
  }

  function pages(x, y, page) {
    const n = 128 / page * 4;      // 同一段长度里画出的页数随页大小变化
    const w = 128 / n;
    let out = '';
    for (let i = 0; i < n; i++) out += '<rect x="' + (x + i * w) + '" y="' + y + '" width="' + (w - 2) + '" height="7" class="sv-fill-lav" opacity=".75"/>';
    return out;
  }

  function cores() {
    let out = '';
    for (let i = 0; i < NUM_CORES; i++) {
      out += '<rect x="' + (614 + (i % 4) * 22) + '" y="' + (120 + Math.floor(i / 4) * 24) + '" width="17" height="17" rx="2" class="sv-core"/>';
    }
    return out;
  }

  function ubSvg(k, ref, bad) {
    let s = '<g class="sv-ub" id="hpUb">';
    // 地址条与缓冲
    s += region('ub', '<rect x="' + BAR.x + '" y="' + BAR.top + '" width="' + BAR.w + '" height="' + (BAR.bottom - BAR.top) + '" class="sv-mem"/>'
      + txt(BAR.x, BAR.top - 8, 'VEC · UB', 'sv-label'));
    k.layout.forEach(function (r) {
      const y1 = ubY(r.hi), h = Math.max(0.9, ubY(r.lo) - y1);
      s += '<rect x="' + (BAR.x + 1) + '" y="' + y1 + '" width="' + (BAR.w - 2) + '" height="' + h + '" class="sv-buf ' + bufClass(r)
        + '" data-buf="' + r.id + '" data-stage="' + r.stage + '"/>';
    });
    // 级标签
    for (let st = 0; st < k.stage; st++) {
      const lo = k.carry + st * k.per, hi = lo + k.per;
      const y = (ubY(lo) + ubY(hi)) / 2 + 3;
      s += txt(BAR.x + BAR.w / 2, y, 'stage ' + st, 'sv-strong', 'middle');
    }
    s += txt(BAR.x - 6, ubY(k.carry / 2) + 3, '常驻', 'sv-label', 'end');
    // 保留区与两条线
    s += '<rect x="' + BAR.x + '" y="' + BAR.top + '" width="' + BAR.w + '" height="' + (ubY(UB_USABLE) - BAR.top) + '" class="sv-reserved" opacity=".9"/>';
    if (k.hw > UB_USABLE) {
      s += '<rect x="' + (BAR.x + 1) + '" y="' + ubY(Math.min(k.hw, UB_NOMINAL)) + '" width="' + (BAR.w - 2) + '" height="'
        + Math.max(1.5, ubY(UB_USABLE) - ubY(Math.min(k.hw, UB_NOMINAL))) + '" class="sv-over"/>';
    }
    s += '<line x1="' + (BAR.x - 4) + '" y1="' + BAR.top + '" x2="' + (BAR.x + BAR.w + 14) + '" y2="' + BAR.top + '" class="sv-line-nominal"/>';
    s += '<line x1="' + (BAR.x - 4) + '" y1="' + ubY(UB_USABLE) + '" x2="' + (BAR.x + BAR.w + 14) + '" y2="' + ubY(UB_USABLE) + '" class="sv-line-usable"/>';
    s += txt(BAR.x + BAR.w + 18, BAR.top + 3, '192 · 分配器预算', 'sv-label');
    s += txt(BAR.x + BAR.w + 18, ubY(UB_USABLE) + 9, '184 · 实际可用', 'sv-text sv-bad');
    // 高水位
    const hwY = ubY(Math.min(k.hw, UB_NOMINAL));
    s += '<line x1="' + (BAR.x - 10) + '" y1="' + hwY + '" x2="' + BAR.x + '" y2="' + hwY + '" stroke="' + (bad ? '#ff6b5b' : '#e8e8e8') + '" stroke-width="1.5"/>';
    s += txt(BAR.x - 12, hwY - 2, '高水位', 'sv-label', 'end') + txt(BAR.x - 12, hwY + 10, kb(k.hw), bad ? 'sv-text sv-bad' : 'sv-strong', 'end');
    if (ref && ref.hw !== k.hw) {
      const ry = ubY(Math.min(ref.hw, UB_NOMINAL));
      s += '<line x1="' + (BAR.x - 2) + '" y1="' + ry + '" x2="' + (BAR.x + BAR.w + 2) + '" y2="' + ry + '" class="sv-line-ghost"/>';
    }
    // 顶部放大
    s += '<line x1="' + (BAR.x + BAR.w) + '" y1="' + ubY(INSET.hi) + '" x2="' + INSET.x + '" y2="' + INSET.top + '" stroke="#3a3a3a"/>'
      + '<line x1="' + (BAR.x + BAR.w) + '" y1="' + ubY(INSET.lo) + '" x2="' + INSET.x + '" y2="' + INSET.bottom + '" stroke="#3a3a3a"/>'
      + '<line x1="' + (BAR.x + BAR.w) + '" y1="' + ubY(INSET.lo) + '" x2="' + (BAR.x + BAR.w + 4) + '" y2="' + ubY(INSET.lo) + '" stroke="#555"/>';
    s += txt(INSET.x, INSET.top - 18, '顶部 16 KB', 'sv-label') + txt(INSET.x, INSET.top - 7, '放大 ×' + Math.round(((INSET.bottom - INSET.top) / (INSET.hi - INSET.lo)) / ((BAR.bottom - BAR.top) / UB_NOMINAL)), 'sv-label');
    s += '<rect x="' + INSET.x + '" y="' + INSET.top + '" width="' + INSET.w + '" height="' + (INSET.bottom - INSET.top) + '" class="sv-mem"/>';
    s += '<g clip-path="url(#hpInsetClip)">';
    k.layout.forEach(function (r) {
      if (r.hi <= INSET.lo) return;
      const y1 = insetY(Math.min(r.hi, INSET.hi)), y2 = insetY(Math.max(r.lo, INSET.lo));
      s += '<rect x="' + (INSET.x + 1) + '" y="' + y1 + '" width="' + (INSET.w - 2) + '" height="' + Math.max(1.5, y2 - y1) + '" class="sv-buf ' + bufClass(r)
        + '" data-buf="' + r.id + '" data-stage="' + r.stage + '"/>';
      if (y2 - y1 > 11) s += txt(INSET.x + 5, (y1 + y2) / 2 + 3, shortName(r), 'sv-text');
    });
    s += '<rect x="' + INSET.x + '" y="' + INSET.top + '" width="' + INSET.w + '" height="' + (insetY(UB_USABLE) - INSET.top) + '" class="sv-reserved" opacity=".75"/>';
    if (k.hw > UB_USABLE) {
      s += '<rect x="' + (INSET.x + 1) + '" y="' + insetY(Math.min(k.hw, UB_NOMINAL)) + '" width="' + (INSET.w - 2) + '" height="'
        + (insetY(UB_USABLE) - insetY(Math.min(k.hw, UB_NOMINAL))) + '" class="sv-over"/>';
    }
    s += '</g>';
    s += '<line x1="' + INSET.x + '" y1="' + insetY(UB_USABLE) + '" x2="' + (INSET.x + INSET.w) + '" y2="' + insetY(UB_USABLE) + '" class="sv-line-usable"/>';
    s += txt(INSET.x + INSET.w / 2, insetY(UB_USABLE) - 18, '保留 8 KB', 'sv-text sv-bad', 'middle');
    if (k.hw > INSET.lo) {
      const y = insetY(Math.min(k.hw, UB_NOMINAL));
      s += '<line x1="' + (INSET.x - 4) + '" y1="' + y + '" x2="' + INSET.x + '" y2="' + y + '" stroke="' + (bad ? '#ff6b5b' : '#e8e8e8') + '" stroke-width="1.5"/>';
    }
    s += txt(INSET.x, INSET.bottom + 13, '176 KB', 'sv-label');
    if (bad) s += txt(INSET.x, INSET.bottom + 28, '越界 ' + bytes(k.hw - UB_USABLE), 'sv-text sv-bad');
    else s += txt(INSET.x, INSET.bottom + 28, '余量 ' + bytes(UB_USABLE - k.hw), 'sv-text sv-good');
    return s + '</g>';
  }

  function shortName(r) {
    if (r.id === 'carry') return '常驻';
    return 's' + r.stage + ' ' + bufById(r.id).name.replace('row_expand_sub 临时量', 'rexp_tmp');
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 焦点：源码行 ⇄ 逻辑节点 ⇄ 硅片区域 ⇄ Vec 缓冲
  // ════════════════════════════════════════════════════════════════════════════
  function fromNodes(ids) {
    const f = { lines: [], regions: [], bufs: [], nodes: ids.slice() };
    ids.forEach(function (id) {
      const n = nodeById(id);
      f.lines = f.lines.concat(n.lines);
      f.regions = f.regions.concat(n.regions || []);
      f.bufs = f.bufs.concat(n.bufs || []);
    });
    return f;
  }

  function sectionFocus(k) {
    if (state.step === 1) return fromNodes(['f-pipe']);
    if (state.step === 2) {
      const f = fromNodes(SOFTMAX_NODES);
      f.regions = [];
      return f;
    }
    if (state.step === 3 && k.intruders.length) {
      const ids = k.intruders.map(function (r) { return r.id; });
      const nodes = nodesWhere(function (n) { return !n.frame && (n.bufs || []).some(function (b) { return ids.indexOf(b) >= 0; }); });
      const f = fromNodes(nodes);
      f.bufs = ids;
      f.lines.push(764);
      f.regions = ['cores'];
      return f;
    }
    return { lines: [], regions: [], bufs: [], nodes: [] };
  }

  function focusFor(h) {
    if (!h) return null;
    let f;
    if (h.type === 'node') f = fromNodes([h.id]);
    else if (h.type === 'line') {
      f = fromNodes(nodesWhere(function (n) { return n.lines.indexOf(h.no) >= 0; }));
      f.lines = [h.no];
    } else if (h.type === 'buf') {
      if (h.id === 'carry') return { lines: [], regions: ['ub'], bufs: ['carry'], nodes: ['f-pipe'] };
      f = fromNodes(nodesWhere(function (n) { return !n.frame && (n.bufs || []).indexOf(h.id) >= 0; }));
      f.bufs = [h.id];
      f.regions = [];
    } else {
      f = fromNodes(nodesWhere(function (n) { return (n.regions || []).indexOf(h.id) >= 0; }));
      f.regions = [h.id];
      if (h.id !== 'ub') f.bufs = [];
    }
    // 复用方案下，被省掉的两块指向 scores_scaled
    if (state.cfg.reuse) f.bufs = f.bufs.map(function (b) { return bufById(b) && bufById(b).alias ? 'scores_scaled' : b; });
    return f;
  }

  function applyFocus() {
    const k = shown();
    const f = focusFor(state.hover) || sectionFocus(k);
    document.querySelectorAll('.cl').forEach(function (el) {
      el.classList.toggle('is-hot', f.lines.indexOf(Number(el.dataset.line)) >= 0);
    });
    document.querySelectorAll('.sv-region').forEach(function (el) {
      el.classList.toggle('is-hot', f.regions.indexOf(el.dataset.region) >= 0);
    });
    const ub = document.getElementById('hpUb');
    if (ub) ub.classList.toggle('has-hot', f.bufs.length > 0);
    document.querySelectorAll('.sv-buf').forEach(function (el) {
      el.classList.toggle('is-hot', f.bufs.indexOf(el.dataset.buf) >= 0);
    });
    document.querySelectorAll('.lg-hit').forEach(function (el) {
      el.classList.toggle('is-hot', f.nodes.indexOf(el.dataset.node) >= 0);
    });
    document.querySelectorAll('.map-table tr[data-line]').forEach(function (el) {
      el.classList.toggle('is-hot', f.lines.indexOf(Number(el.dataset.line)) >= 0);
    });
    renderCaption(k);
  }

  function renderCaption(k) {
    const h = state.hover;
    let key = '', body = '';
    if (h && h.type === 'line') { key = 'line ' + h.no + ' → 逻辑 · 硅片'; body = lineCaption(h.no, k); }
    else if (h && h.type === 'node') {
      const n = nodeById(h.id);
      key = '逻辑 → line ' + n.lines[0] + (n.lines.length > 1 ? '–' + n.lines[n.lines.length - 1] : '') + ' · 硅片';
      body = nodeCaption(h.id, k);
    } else if (h && h.type === 'buf') {
      const b = bufById(h.id);
      if (b) { key = 'Vec 缓冲 → 逻辑 · line ' + b.line; body = lineCaption(b.line, k); }
      else { key = 'Vec 常驻区'; body = '常驻区 ' + kb(k.carry) + '：沿用 Lens 的估算口径（o_i / m_i / l_i），不随 stage 复制。'; }
    } else if (h && h.type === 'region') { key = '硅片 → 逻辑 · line ' + (nodesWhere(function (n) { return (n.regions || []).indexOf(h.id) >= 0; }).map(function (id) { return nodeById(id).lines[0]; }).join(' / ') || '—'); body = regionCaption(h.id, k); }
    if (!body) {
      key = '悬停源码行、逻辑节点或硅片区域，三处会一起亮起';
      if (k.verdict === 'silent') {
        body = '落进保留区的是 <b>stage ' + (k.stage - 1) + '</b> 的 ' + k.intruders.map(function (r) {
          return '<code>' + (bufById(r.id) || {}).name + '</code>';
        }).join('、') + '：都在最高地址，正好对上“只有最高地址的 tile 出 NaN”。';
      } else body = regionCaption('ub', k);
    }
    document.getElementById('hpCaption').innerHTML = '<span class="cap-k">' + key + '</span>' + body;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 右栏：五个可折叠章节，标题行实时给出结论
  // ════════════════════════════════════════════════════════════════════════════
  function meter(k) {
    const cls = k.bad ? 'is-bad' : (k.verdict === 'tight' ? 'is-warn' : 'is-good');
    return '<div class="meter"><span class="meter-k">Vec(UB) 高水位</span><span class="meter-v ' + cls + '">' + kb(k.hw)
      + ' <small>/ 184 KB 可用</small></span><div class="meter-bar"><span style="width:' + Math.min(100, k.hw / UB_NOMINAL * 100) + '%;'
      + (cls === 'is-bad' ? 'background:var(--red)' : '') + '"></span><i style="left:' + (UB_USABLE / UB_NOMINAL * 100) + '%"></i><em style="left:calc(100% - 1px)"></em></div></div>';
  }
  function next(n, label) {
    return '<button class="btn btn-primary" type="button" data-act="open" data-step="' + n + '">' + label + '</button>';
  }

  function body1() {
    const base = calc(BASE), changed = calc(CHANGED), isChanged = same(state.cfg, CHANGED);
    return '<h2 class="st-h">你只改了一个调度参数：<code>stage</code> 2 → 4</h2>'
      + '<p class="st-p">目的是让第 i+1 个头的 QK（AIC）与第 i 个头的 softmax（AIV）多重叠几轮。'
      + '在源码层面，这个改动<b>不改语义、不改数值</b>，读代码看不出任何风险。</p>'
      + '<p class="st-p">在源码 <code>764</code> 行直接点 stage 的值，或用下面的按钮。逻辑图里 softmax 段会多叠几层，Vec 地址条会跟着长高。</p>'
      + meter(current())
      + '<div class="st-actions">'
      + (isChanged ? '' : '<button class="btn btn-primary" type="button" data-act="apply-changed">把 stage 改成 4</button>')
      + (isChanged ? next(2, '展开：它落在哪里') : '<button class="btn" type="button" data-act="open" data-step="2">展开：它落在哪里</button>')
      + '</div>'
      + '<p class="note">stage 2：' + kb(base.hw) + ' · stage 4：' + kb(changed.hw) + '。数值按每个中间量各占一块估算，与 Lens 工作台口径一致。</p>';
  }

  function body2() {
    const k = current(), c = state.cfg;
    const rows = BUFS.map(function (b) {
      const size = bufSize(b, c);
      return '<tr data-line="' + b.line + '"><td>' + b.line + '</td><td><code>' + b.name + '</code></td><td class="num">'
        + (size ? bytes(size) : '复用') + '</td><td class="num">' + (size ? bytes(size * k.stage) : '—') + '</td></tr>';
    }).join('');
    return '<h2 class="st-h">stage 不是循环参数。它在 Vec 里把整条 softmax 链复制了 ' + k.stage + ' 份。</h2>'
      + '<p class="st-p">逻辑图里：pipeline 让相邻 ' + k.stage + ' 个头同时在飞，② 段因此叠成 ' + k.stage + ' 层。'
      + '硅片上：这 7 行源码的中间量全部落在 AIV 的 Vec(UB)，每多一级就多一份 <b>' + kb(k.per) + '</b>。</p>'
      + '<table class="map-table"><thead><tr><th>行</th><th>中间量</th><th>每级</th><th>× ' + k.stage + ' 级</th></tr></thead><tbody>' + rows
      + '</tbody><tfoot><tr><td></td><td>+ 常驻区</td><td class="num"></td><td class="num">' + kb(k.carry) + '</td></tr>'
      + '<tr><td></td><td>高水位</td><td class="num"></td><td class="num">' + kb(k.hw) + '</td></tr></tfoot></table>'
      + '<p class="note">① 打分和 ③ 加权求和的操作数在 AIC 一侧（L0A / L0B / L0C），不随 stage 复制；这条路的压力只在 Vec。</p>'
      + '<div class="st-actions">' + next(3, '展开：硬件怎么判') + '</div>';
  }

  function body3() {
    const k = current();
    const thresholds = '<div class="thresholds">'
      + '<div class="th-row is-hw"><span>本配置高水位</span><b class="' + (k.hw > UB_USABLE ? 'is-bad' : 'is-good') + '">' + kb(k.hw) + '</b></div>'
      + '<div class="th-row"><span>实际可用</span><b class="is-bad">184 KB</b><small>顶部约 8 KB 被运行时保留（ReserveBufferOp / pipe init）· pto-isa#170 · a2a3；随平台与 CANN 版本变化</small></div>'
      + '<div class="th-row"><span>分配器预算</span><b>192 KB</b><small>上游分配器按名义容量放行</small></div>'
      + '</div>';
    if (k.verdict !== 'silent') {
      return '<h2 class="st-h">' + (k.verdict === 'compile' ? '超过名义容量，分配器会直接报错' : '当前配置没有越过保留区') + '</h2>'
        + thresholds
        + '<div class="callout ' + (k.verdict === 'compile' ? '' : 'is-ok') + '">'
        + (k.verdict === 'compile' ? '这是能看见的失败：编译期就报出来。' : '距 184 KB 边界还有 ' + bytes(UB_USABLE - k.hw) + '。')
        + '</div>'
        + '<div class="st-actions">'
        + (same(state.cfg, BASE) ? '<button class="btn btn-primary" type="button" data-act="apply-changed">套用改动：stage = 4</button>' : '')
        + '<button class="btn" type="button" data-act="open" data-step="5">展开复查</button></div>';
    }
    const names = k.intruders.map(function (r) { return (bufById(r.id) || {}).name; });
    return '<h2 class="st-h">越过实际可用线 ' + bytes(k.hw - UB_USABLE) + '，但离分配器预算还有 ' + bytes(UB_NOMINAL - k.hw) + '：<span class="is-bad">没有任何一环会报错</span></h2>'
      + thresholds
      + '<p class="st-sec">如果没有在编码期拦下</p>'
      + '<ol class="timeline">'
      + '<li class="t-stop"><span class="t-k">编码期 · 本页</span><span class="t-v"><b>在这里拦下</b>：高水位对的是 184，不是 192</span></li>'
      + '<li class="t-pass"><span class="t-k">编译 · AllocateMemoryAddr</span><span class="t-v"><b>通过</b>：按 192 KB 预算，' + kb(k.hw) + ' 合法</span></li>'
      + '<li class="t-pass"><span class="t-k">上板运行</span><span class="t-v"><b>无报错</b>：没有越界异常，也不 hang</span></li>'
      + '<li class="t-fail"><span class="t-k">精度对比</span><span class="t-v"><b>输出出现 NaN</b>：stage ' + (k.stage - 1) + ' 的 ' + names.join('、') + ' 读写到保留区</span></li>'
      + '<li class="t-fail"><span class="t-k">定位</span><span class="t-v"><b>很容易误判</b>：改小 tile 或 stage=1 现象就消失，看起来像数值问题或偶发故障</span></li>'
      + '</ol>'
      + '<div class="callout">' + NUM_CORES + ' 个 block 共用同一份地址分配，所以这不是个别核偶发的问题。</div>'
      + '<p class="note">依据：<a href="' + ISSUE_URL + '" target="_blank" rel="noopener noreferrer">pto-isa#170</a>。时间线描述的是该 Issue 的失效方式，本页没有对这份源码做编译或上板。</p>'
      + '<div class="st-actions">' + next(4, '展开：怎么改') + '</div>';
  }

  function body4() {
    const chosen = chosenKey(state.cfg);
    const cards = Object.keys(OPTIONS).map(function (key) {
      const o = OPTIONS[key], k = calc(o.cfg);
      const cls = 'opt' + (state.preview === key ? ' is-preview' : '') + (chosen === key ? ' is-chosen' : '');
      return '<div class="' + cls + '" data-opt="' + key + '" tabindex="0">'
        + '<div class="opt-head"><span class="opt-id">' + key + '</span><span class="opt-title">' + o.title + '</span><span class="opt-sub">' + o.sub + '</span></div>'
        + '<div class="opt-res"><span>Vec <b class="' + (k.hw > UB_USABLE ? 'is-bad' : 'is-good') + '">' + kb(k.hw) + '</b></span>'
        + '<span>Right <b>' + pct(k.kDB / L0B_CAP) + '</b></span><span>压力转到 <b>' + o.to + '</b></span></div>'
        + '<p class="opt-cost">' + o.cost(k) + '</p>'
        + '<div class="opt-foot"><span class="opt-scope">' + o.scope + '</span>'
        + '<button class="btn' + (chosen === key ? '' : ' btn-primary') + '" type="button" data-act="choose" data-opt="' + key + '">'
        + (chosen === key ? '已采用' : '采用并复查') + '</button></div></div>';
    }).join('');
    return '<h2 class="st-h">三种改法都能回到 184 KB 以内，但压力会被推到不同地方</h2>'
      + '<p class="st-p">悬停卡片，逻辑图和地址条会切到这个方案的样子；地址条上的虚线是当前配置的高水位。</p>'
      + cards
      + '<p class="note">本页不推荐唯一答案：A 的代价要 profiling 才知道，B 要动运行时，C 要编译产物证明。</p>';
  }

  function body5() {
    const k = current(), before = calc(CHANGED), key = chosenKey(state.cfg);
    if (k.bad) {
      return '<h2 class="st-h">当前配置仍然越界</h2>'
        + '<div class="callout">高水位 ' + kb(k.hw) + '，超过 184 KB。先选一种改法。</div>'
        + '<div class="st-actions">' + next(4, '展开改法') + '</div>';
    }
    function row(name, a, b, cls, ev, evText) {
      return '<tr><td>' + name + '</td><td class="v">' + a + '</td><td class="v ' + (cls || '') + '">' + b + '</td><td><span class="ev ' + ev + '">' + evText + '</span></td></tr>';
    }
    const table = '<table class="cmp"><thead><tr><th>资源</th><th>改动后</th><th>' + (key ? '方案 ' + key : '当前') + '</th><th>依据</th></tr></thead><tbody>'
      + row('Vec(UB) 高水位', '<span class="is-bad">' + kb(before.hw) + '</span>', kb(k.hw), 'is-good', 'e-est', '估算 → 编译产物')
      + row('Right(L0B) 双缓冲', kb(before.kDB), kb(k.kDB), k.kDB !== before.kDB ? 'is-good' : '', 'e-est', '估算')
      + row('SEQ_TILE / page_size', before.cfg.seq + ' / ' + before.cfg.page, state.cfg.seq + ' / ' + state.cfg.page, k.pageOk ? (state.cfg.page !== before.cfg.page ? 'is-warn' : '') : 'is-bad', 'e-build', '运行时契约')
      + row('fa_fused 真实块', String(before.blocks), String(k.blocks), k.blocks !== before.blocks ? 'is-warn' : '', 'e-est', '估算')
      + row('流水驻留级数', String(before.stage), String(k.stage), k.stage !== before.stage ? 'is-warn' : '', 'e-board', '需 profiling')
      + row('中间量复用', '否', state.cfg.reuse ? '假设是' : '否', state.cfg.reuse ? 'is-warn' : '', 'e-build', '需编译产物')
      + '</tbody></table>';
    const checks = ['编译产物的内存映射里，Vec 高水位低于 <b>184 KB</b>，而不是只低于 192 KB',
      '上板结果与 torch 参考逐元素对比，没有 NaN / Inf',
      'Profiling 对比 stage 2 / 3 / 4 的 fa_fused 耗时，确认改法没把收益吃掉'];
    if (state.cfg.page !== BASE.page) checks.push('运行时 KV 分页同步改为 ' + state.cfg.page + '，并核对其它层与 block_table 生成');
    if (state.cfg.reuse) checks.push('编译产物中 scores_valid / scores 确实复用 scores_scaled 的地址');
    return '<h2 class="st-h">回到了 184 KB 以内。再看压力被推到了哪里</h2>'
      + meter(k) + table
      + '<p class="st-sec">本页能证明的到此为止，接下来要看</p>'
      + '<ul class="checklist">' + checks.map(function (t) { return '<li><span>' + t + '</span></li>'; }).join('') + '</ul>'
      + '<div class="st-actions"><button class="btn" type="button" data-act="open" data-step="4">换个方案</button>'
      + '<button class="btn" type="button" data-act="reset">从头再走一遍</button></div>';
  }

  // 每章标题行上的实时结论：收起时也能看到这条路走到了哪里
  function sectionSummary(n, k) {
    const c = state.cfg, key = chosenKey(c);
    if (n === 1) return c.stage === BASE.stage ? { t: '未改动 · stage=' + c.stage } : { t: 'stage ' + BASE.stage + ' → ' + c.stage, cls: 'is-warn' };
    if (n === 2) return { t: 'softmax 链 × ' + k.stage + ' · ' + kb(k.per * k.stage) };
    if (n === 3) {
      if (k.verdict === 'silent') return { t: '越界 ' + bytes(k.hw - UB_USABLE) + ' · 无报错', cls: 'is-bad' };
      if (k.verdict === 'compile') return { t: '超过 192 KB · 编译报错', cls: 'is-bad' };
      return { t: '余量 ' + bytes(UB_USABLE - k.hw), cls: 'is-good' };
    }
    if (n === 4) return key ? { t: '已采用 ' + key + ' · ' + OPTIONS[key].title, cls: 'is-good' } : { t: '3 个方案' };
    if (k.bad) return { t: '仍越界' , cls: 'is-bad' };
    return same(c, BASE) ? { t: '—' } : { t: '待编译产物 / 上板确认', cls: 'is-warn' };
  }

  function renderStory() {
    const k = current();
    const bodies = [body1, body2, body3, body4, body5];
    const html = SECTIONS.map(function (sec, i) {
      const n = i + 1, open = state.step === n, sum = sectionSummary(n, k);
      return '<section class="sec' + (open ? ' is-open' : '') + '" data-sec="' + n + '">'
        + '<button class="sec-head" type="button" data-act="toggle" data-step="' + n + '" aria-expanded="' + open + '" aria-controls="sec-body-' + n + '">'
        + '<span class="sec-no">0' + n + '</span>'
        + '<span class="sec-name">' + sec.name + '</span>'
        + '<span class="sec-q">' + sec.q + '</span>'
        + '<span class="sec-sum ' + (sum.cls || '') + '">' + sum.t + '</span>'
        + '<span class="sec-chev" aria-hidden="true"></span>'
        + '</button>'
        + (open ? '<div class="sec-body" id="sec-body-' + n + '">' + bodies[i]() + '</div>' : '')
        + '</section>';
    }).join('');
    document.getElementById('hpStory').innerHTML = html;
  }

  function renderStatus() {
    const k = current(), el = document.getElementById('hpStatus');
    let text, cls;
    if (k.verdict === 'silent') {
      text = '✕ Vec 高水位 ' + kb(k.hw) + ' > 实际可用 184 KB · 分配器按 192 KB 放行 → 编译、运行都不报错，输出静默 NaN';
      cls = 'is-bad';
    } else if (k.verdict === 'compile') {
      text = '✕ Vec 高水位 ' + kb(k.hw) + ' > 名义 192 KB · 编译期分配失败'; cls = 'is-bad';
    } else {
      text = (k.verdict === 'tight' ? '! ' : '✓ ') + 'Vec 高水位 ' + kb(k.hw) + ' / 184 KB 可用（' + pct(k.hw / UB_USABLE) + '）· stage=' + state.cfg.stage
        + ' · SEQ_TILE=' + state.cfg.seq;
      cls = k.verdict === 'tight' ? 'is-warn' : 'is-good';
    }
    el.textContent = text;
    el.className = 'hp-status-main ' + cls;
  }

  function render() {
    renderCode(); renderLogic(); renderSvg(); renderStory(); renderStatus(); applyFocus();
  }

  function openSection(n, scroll) {
    state.step = n; state.preview = null; state.hover = null;
    render();
    if (scroll && n) {
      const el = document.querySelector('.sec[data-sec="' + n + '"]');
      if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 事件
  // ════════════════════════════════════════════════════════════════════════════
  function setCfg(cfg) { state.cfg = Object.assign({}, cfg); state.preview = null; render(); }

  document.addEventListener('click', function (e) {
    const stageBtn = e.target.closest('[data-stage]');
    if (stageBtn) { setCfg(Object.assign({}, state.cfg, { stage: Number(stageBtn.dataset.stage) })); return; }
    const act = e.target.closest('[data-act]');
    if (!act) return;
    const a = act.dataset.act;
    if (a === 'toggle') { const n = Number(act.dataset.step); openSection(state.step === n ? 0 : n, false); }
    else if (a === 'open') { openSection(Number(act.dataset.step), true); }
    else if (a === 'apply-changed') { setCfg(CHANGED); }
    else if (a === 'choose') { state.step = 5; setCfg(OPTIONS[act.dataset.opt].cfg); }
    else if (a === 'reset') { state.step = 1; state.hover = null; setCfg(BASE); }
  });
  document.getElementById('hpReset').addEventListener('click', function () { state.step = 1; state.hover = null; setCfg(BASE); });

  // 悬停：代码行、表格行、逻辑节点、硅片区域、Vec 缓冲
  function hoverFrom(target) {
    const buf = target.closest('.sv-buf');
    if (buf) return { type: 'buf', id: buf.dataset.buf };
    const reg = target.closest('.sv-region');
    if (reg) return { type: 'region', id: reg.dataset.region };
    const node = target.closest('.lg-hit');
    if (node) return { type: 'node', id: node.dataset.node };
    const line = target.closest('.cl.is-mapped, .map-table tr[data-line]');
    if (line) return { type: 'line', no: Number(line.dataset.line) };
    return null;
  }
  document.addEventListener('mouseover', function (e) {
    const h = hoverFrom(e.target);
    const opt = e.target.closest('.opt[data-opt]');
    const preview = opt ? opt.dataset.opt : null;
    const hChanged = JSON.stringify(h) !== JSON.stringify(state.hover);
    if (preview !== state.preview) {
      state.preview = preview;
      state.hover = h;
      renderLogic(); renderSvg();
      document.querySelectorAll('.opt').forEach(function (el) { el.classList.toggle('is-preview', el.dataset.opt === preview); });
      applyFocus();
    } else if (hChanged) {
      state.hover = h;
      applyFocus();
    }
  });
  document.addEventListener('focusin', function (e) {
    const opt = e.target.closest('.opt[data-opt]');
    if (opt && opt.dataset.opt !== state.preview) { state.preview = opt.dataset.opt; renderLogic(); renderSvg(); applyFocus(); }
  });

  render();
})();
