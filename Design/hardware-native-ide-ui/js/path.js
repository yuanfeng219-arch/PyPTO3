/* ──────────────────────────────────────────────────────────────────────────────
 * Hardware-Native 主线
 *
 * 只讲一条路：把 fa_fused 的流水从 stage=2 调到 4 → 落到 AIV 的 Vec(UB) →
 * 越过实际可用的 184 KB、但没越过分配器按名义容量发的 192 KB → 编译、运行都不报错，
 * 输出静默 NaN（抽象自 pto-isa#170）→ 三种改法各把压力推到不同地方 → 复查。
 *
 * 三个视图共用一个焦点：源码行 ⇄ 算子逻辑图（循环与计算链）⇄ 硬件映射（区域与 Vec 缓冲）。
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
      title: '降低 pipeline stage', sub: 'stage = 3', scope: '改动 · 算子 1 行',
      to: '时间', cost: function () {
        return 'AIC / AIV 少一级重叠。<em>能少多少收益，只能靠 profiling 回答</em>，本页不给数字。';
      }
    },
    B: {
      cfg: { stage: 4, seq: 64, page: 64, reuse: false },
      title: '缩小 KV tile', sub: 'SEQ_TILE = 64', scope: '改动 · 算子 + 运行时 KV 分页',
      to: '运行时与 GM', cost: function (c) {
        return '<em>runtime.page_size 必须同步改为 64</em>（逻辑块 = 物理页），这是整个 KV cache 的分页参数；'
          + 'fa_fused 工作块数 ' + calc(CHANGED).blocks + ' → ' + c.blocks + '，work table 容量 ×2。';
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
  // 源码中的有效代码行
  // ════════════════════════════════════════════════════════════════════════════
  const SOURCE = [
    { no: 728, t: "with pl.spmd(" },
    { no: 729, t: "    NUM_CORES," },
    { no: 730, t: "    name_hint=\"fa_fused\"," },
    { no: 731, t: "    optimizations=[pl.split(pl.SplitMode.UP_DOWN)]," },
    { no: 732, t: "    deps=[work_tid, rope_grp_tids[0]]," },
    { no: 733, t: ") as fa_tid:" },
    { no: 734, t: "    fa_core = pl.get_block_idx()" },
    { no: 741, t: "    fa_total_blocks = pl.cast(pl.read(fa_total, [0, 0]), target_type=pl.INDEX)" },
    { no: 744, t: "    for fa_w in pl.range(fa_core, fa_total_blocks, NUM_CORES):" },
    { no: 745, t: "        fa_enc = pl.cast(pl.read(fa_work_table, [fa_w, 0]), target_type=pl.INDEX)" },
    { no: 746, t: "        fa_b = fa_enc // MAX_CTX_BLOCKS" },
    { no: 747, t: "        fa_p = fa_enc % MAX_CTX_BLOCKS" },
    { no: 748, t: "        fa_hg = 0" },
    { no: 749, t: "        fa_ctx_len = pl.read(seq_lens, [fa_b])" },
    { no: 751, t: "        sb = fa_p" },
    { no: 752, t: "        s0 = sb * SEQ_TILE" },
    { no: 753, t: "        valid_len = pl.min(SEQ_TILE, fa_ctx_len - s0)" },
    { no: 758, t: "        fa_pbid = pl.cast(pl.tensor.read(block_table, [fa_b * max_blocks_per_seq + sb]), pl.INDEX)" },
    { no: 764, t: "        for gp in pl.pipeline(GP_SIZE, stage=§):" },
    { no: 765, t: "            gi = fa_hg * GP_SIZE + gp" },
    { no: 766, t: "            kvh = gi" },
    { no: 767, t: "            q_pad_row_g = fa_b * NUM_KV_HEADS * Q_HEAD_PAD + gi * Q_HEAD_PAD" },
    { no: 768, t: "            q_padded = all_q_padded[q_pad_row_g : q_pad_row_g + Q_HEAD_PAD, :]" },
    { no: 769, t: "            g_base = (fa_b * NUM_KV_HEADS + gi) * MAX_CTX_BLOCKS * Q_HEAD_PAD" },
    { no: 770, t: "            cache_row = layer_cache_base + (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE" },
    { no: 773, t: "            k_tile = k_cache[cache_row : cache_row + SEQ_TILE, :]" },
    { no: 774, t: "            raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)" },
    { no: 775, t: "            scores_scaled = pl.mul(raw_scores, ATTN_SCALE)" },
    { no: 781, t: "            scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_BATCH, valid_len)" },
    { no: 782, t: "            scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)" },
    { no: 783, t: "            cur_mi = pl.row_max(scores)" },
    { no: 784, t: "            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))" },
    { no: 785, t: "            cur_li = pl.row_sum(exp_scores)" },
    { no: 786, t: "            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)" },
    { no: 789, t: "            v_tile = v_cache[cache_row : cache_row + SEQ_TILE, :]" },
    { no: 790, t: "            oi_tmp = pl.matmul(exp_scores_bf16, v_tile, out_dtype=pl.FP32)" },
    { no: 794, t: "            oi_tmp = pl.set_validshape(oi_tmp, Q_HEAD_BATCH, HEAD_DIM)" },
    { no: 796, t: "            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [g_base + sb * Q_HEAD_PAD, 0])" },
    { no: 797, t: "            all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [g_base + sb * Q_HEAD_PAD, 0])" },
    { no: 798, t: "            all_cur_li = pl.assemble(all_cur_li, cur_li, [g_base + sb * Q_HEAD_PAD, 0])" }
  ];

  // ════════════════════════════════════════════════════════════════════════════
  // 联动的单一真相源：算子逻辑节点 = 源码行 + 硅片区域 + Vec 缓冲
  // 循环框也是节点；源码行和硅片区域上的联动都从这张表反查。
  // ════════════════════════════════════════════════════════════════════════════
  const NODES = [
    { id: 'f-spmd', frame: true, lines: [728, 729, 730, 731, 732, 733, 734], regions: ['cores'] },
    // 731 行的 UP_DOWN 切分：逻辑图里不画，只负责把这一行连到 AIC / AIV 两侧和中间的交接通道
    { id: 'split', hidden: true, lines: [731], regions: ['aic', 'aiv', 'ch-c2v', 'ch-v2c'] },
    // 741–752：读块总数、解码 work table 项、算块起点，都属于“取块”这一层
    { id: 'f-stride', frame: true, lines: [741, 744, 745, 746, 747, 748, 749, 751, 752], regions: ['cores', 'gm-table'] },
    { id: 'f-pipe', frame: true, lines: [764], regions: ['ub'], bufs: ALL_BUFS },
    // 765–766 只是算本次迭代的头号：点亮 pipeline 框，不连带整份 Vec 缓冲
    { id: 'gp-idx', hidden: true, lines: [765, 766], show: ['f-pipe'] },
    { id: 'p-score', frame: true, lines: [], group: ['q', 'k', 'qk', 'scale'] },
    { id: 'p-softmax', frame: true, lines: [], group: ['valid', 'fill', 'max', 'exp', 'sum', 'cast'] },
    { id: 'p-weight', frame: true, lines: [], group: ['v', 'sv', 'out'] },
    { id: 'tbl', lines: [758], regions: ['gm-table', 'gm-kcache', 'gm-vcache'] },
    { id: 'q', lines: [768, 767], regions: ['l0a'] },
    { id: 'k', lines: [773, 770], regions: ['gm-kcache', 'l0b'] },
    { id: 'qk', lines: [774], regions: ['cube', 'l0c', 'ch-c2v'] },
    { id: 'scale', lines: [775], bufs: ['scores_scaled'] },
    { id: 'valid', lines: [781, 753], bufs: ['scores_valid'] },
    { id: 'fill', lines: [782], bufs: ['scores'] },
    { id: 'max', lines: [783], bufs: ['cur_mi'] },
    { id: 'exp', lines: [784], bufs: ['rexp', 'exp'] },
    { id: 'sum', lines: [785], bufs: ['cur_li'] },
    { id: 'cast', lines: [786], bufs: ['exp_bf16'], regions: ['ch-v2c'] },
    { id: 'v', lines: [789, 770], regions: ['gm-vcache', 'l0b'] },
    { id: 'sv', lines: [790, 794], regions: ['cube', 'l0c', 'ch-v2c'] },
    { id: 'out', lines: [796, 797, 798, 769], regions: ['gm-out'] }
  ];
  const SOFTMAX_NODES = ['scale', 'valid', 'fill', 'max', 'exp', 'sum', 'cast'];
  // 悬停 AIC / AIV 盒子时，指向在这一侧执行的运算，而不只是 spmd 声明
  const REGION_NODES = { aic: ['split', 'qk', 'sv'], aiv: ['split'].concat(SOFTMAX_NODES) };
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
      741: '在核内从 GM 读回真实块总数 <code>fa_total</code>，作为 grid-stride 的上界。',
      744: 'grid-stride 从 work table 取号：batch 16 的真实块 <b>' + k.blocks + '</b> 个，摊到 ' + NUM_CORES + ' 个 block，每个 block 约 ' + Math.ceil(k.blocks / NUM_CORES) + ' 块。',
      745: '按编号从 work table 取一项编码。',
      746: '解码出请求下标 <code>fa_b</code>。', 747: '解码出逻辑块号 <code>fa_p</code>。',
      748: '本例只有一个头组，头组偏移恒为 0。',
      749: '读这条请求的上下文长度。',
      751: 'work table 每项正好是一个真实块：<code>sb = fa_p</code>。',
      752: '这一块在序列里的起点：<code>s0 = sb × SEQ_TILE</code>（当前 SEQ_TILE = ' + c.seq + '）。',
      753: '<code>valid_len</code>：最后一块可能不满，后面的 set_validshape 只把这么多列标为有效。',
      758: 'block_table 把逻辑块翻成物理页。前提：<code>SEQ_TILE == page_size</code>（当前 ' + c.seq + ' / ' + c.page + '）。',
      764: '<code>stage=' + c.stage + '</code>：In-flight heads · <b>' + n + '</b>；UB residency · ' + n + ' 份 softmax 链，每份 ' + kb(k.per) + '，合计 <b>' + kb(k.per * n) + '</b>。',
      765: '本次迭代处理第 <code>gi</code> 个 KV 头（共 ' + GP_SIZE + ' 个）。', 766: 'KV 头号与 gi 相同。',
      767: 'Q 填充后每个头占 16 行，算出这个头在 <code>all_q_padded</code> 里的起始行。',
      768: '<code>q_padded [16, 128] BF16</code> 4 KB → L0A(Left)。5 行真实 Q 填到 16 行，对齐 Cube M。',
      769: '部分结果写回 GM 时的起始行。',
      770: '<code>cache_row</code> = 物理页 × 头 × BLOCK_SIZE：K 与 V 都从这一行开始读。',
      794: 'SV 输出 16 行里只有 5 行真实，标为有效后写回只写 5 行。',
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
      cores: '<b>pl.spmd × ' + NUM_CORES + '</b> · grid-stride 把 ' + k.blocks + ' 个真实块摊到 ' + NUM_CORES + ' 个 block。下方展开的是 block 0，其余 block 按相同布局独立分配片上存储，所以越界不是个别核的偶发问题。',
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
    const phase = {
      'p-score': '<b>① 打分</b>：Q 与 K 块在 Cube 上做 QK matmul，乘缩放系数后交给 Vector。操作数在 AIC 一侧，不随 stage 复制。',
      'p-softmax': '<b>② 局部 softmax</b>：掩码、行最大、exp、行和、转 BF16，全部在 AIV 的 Vec 里完成。<b>stage=' + k.cfg.stage + ' 时这一段同时存在 ' + k.stage + ' 份</b>，共 ' + kb(k.per * k.stage) + '。',
      'p-weight': '<b>③ 加权求和</b>：P 与 V 块在 Cube 上做 SV matmul，部分结果按块写回 GM，留给 online_softmax 合并。'
    };
    if (phase[id]) return phase[id];
    const n = nodeById(id);
    return n.lines.map(function (no) { return lineCaption(no, k); }).filter(Boolean).slice(0, 1).join('');
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 状态
  // ════════════════════════════════════════════════════════════════════════════
  const SECTIONS = [
    { name: '配置变更', q: 'Pipeline stage' },
    { name: '内存占用', q: 'UB residency' },
    { name: '诊断', q: 'Capacity constraint' },
    { name: '调整方案', q: '配置变更 alternatives' },
    { name: '验证', q: 'Evidence requirements' }
  ];
  // step = 当前展开的章节（0 = 全部收起）
  const state = { step: 3, cfg: Object.assign({}, CHANGED), preview: null, hover: null };

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
      let text;
      if (row.no === 764) {
        const parts = row.t.split('§');
        text = hl(parts[0], c) + stageControl(k) + hl(parts[1], c)
          + (c.stage !== BASE.stage ? '<span class="stage-was">原 stage=' + BASE.stage + '</span>' : '')
          + (k.bad ? '<button class="source-diagnostic" data-act="diagnose" type="button">⚠ UB 容量风险 · 查看诊断</button>' : '');
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
    const marker=(suffix,color)=>'<marker id="'+prefix+suffix+'" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" markerUnits="userSpaceOnUse" orient="auto"><path d="M1 1 L7 4 L1 7" fill="none" stroke="'+color+'" stroke-width="1.1" stroke-linecap="round" stroke-linejoin="round"/></marker>';
    return marker('Arrow','#777')+marker('ArrowHot',prefix==='lg'?'#e6e6e6':'#f2c45f');
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 算子逻辑图：spmd × 24 ⊃ grid-stride 取块 ⊃ pipeline × 8 头 ⊃ 三段计算链
  // ════════════════════════════════════════════════════════════════════════════
  // 逻辑图只讲“算什么、按什么顺序、循环几层”；形状、字节、落在哪块存储都交给硅片画布，悬停联动过去看。
  // 虚线框 = 循环作用域，实框 = 计算段，节点 = 一步运算（只写名字）
  const LG = {
    // 节点位置 [x, y, w]（viewBox 720 × 284，节点高 26，行距 40）
    q: [58, 132, 80], k: [148, 132, 80], qk: [58, 172, 80], scale: [148, 172, 80],
    valid: [258, 132, 84], fill: [352, 132, 84], max: [446, 132, 84],
    exp: [446, 172, 84], sum: [352, 172, 84], cast: [446, 212, 84],
    v: [572, 172, 92], sv: [572, 212, 92], out: [572, 252, 92]
  };
  const LG_H = 26;
  const LG_EDGES = [
    ['q', 'qk', 'M98 158 V 172'], ['k', 'qk', 'M188 158 V 165 H 118 V 172'], ['qk', 'scale', 'M138 185 H 148'],
    ['scale', 'valid', 'M228 185 H 243 V 145 H 258'], ['valid', 'fill', 'M342 145 H 352'], ['fill', 'max', 'M436 145 H 446'],
    ['max', 'exp', 'M488 158 V172'], ['exp', 'sum', 'M446 185 H436'], ['exp', 'cast', 'M488 198 V212'],
    ['cast', 'sv', 'M530 225 H572'], ['v', 'sv', 'M618 198 V212'], ['sv', 'out', 'M618 238 V252']
  ];
  const LG_LABEL = {
    q: 'Q', k: 'K 块', qk: 'Q · Kᵀ', scale: '× scale',
    valid: '有效区掩码', fill: '填充 −inf', max: '行最大',
    exp: 'exp(x − max)', sum: '行求和', cast: '转 BF16',
    v: 'V 块', sv: 'P · V', out: '写回部分结果'
  };
  // 三段计算：[id, x, w, 标题]
  const LG_PHASES = [['p-score', 48, 190, '① 打分'], ['p-softmax', 248, 296, '② 局部 SOFTMAX'], ['p-weight', 562, 112, '③ 加权求和']];
  const LG_PHASE_Y = 106, LG_PHASE_H = 184;

  function logicNode(id, badIds) {
    const p = LG[id], bad = badIds.indexOf(id) >= 0;
    if (['q','k','v','out'].includes(id)) {
      const shapes={q:'[16, 128]',k:'['+shown().cfg.seq+', 128]',v:'['+shown().cfg.seq+', 128]',out:'[16, 128]'};
      const color=id==='out'?'#ffbd1a':(id==='q'?'#72d7af':'#bca0f3');
      return '<g class="lg-hit" data-node="'+id+'">'+PathGridGlyphs.tensor(id,LG_LABEL[id],shapes[id],p[0],p[1],p[2],LG_H,color)+'</g>';
    }
    const categories = {qk:'matmul',sv:'matmul',scale:'gate',valid:'cache',fill:'reshape',max:'reduce',exp:'activation',sum:'reduce',cast:'reshape',out:'cache'};
    const icon = window.PtoOpIcons.find(i => i.id === categories[id]);
    return '<g class="lg-hit' + (bad ? ' is-bad' : '') + '" data-node="' + id + '">'
      + VisualSystem.node('op',id,LG_LABEL[id],p[0],p[1],{width:p[2],height:LG_H,icon:icon?.path,fontSize:id === 'exp' ? 8 : 10})+'</g>';

  }

  function renderLogic() {
    const k = shown(), c = k.cfg;
    const badIds = k.bad ? uniq(k.intruders.map(function (r) {
      return nodesWhere(function (n) { return (n.bufs || []).indexOf(r.id) >= 0 && !n.frame; })[0];
    }).filter(Boolean)) : [];
    // 页表一行放在取块之后、pipeline 之前，按代码顺序自上而下读；pipeline 整体下移 PIPE_DY
    const PIPE_DY = 26;
    let s = '<svg viewBox="0 0 720 350" preserveAspectRatio="xMidYMin" role="img" aria-label="fa_fused 的循环嵌套与计算链">'
      + '<defs>' + markers('lg') + '</defs>';

    // 三层循环作用域（虚线框）
    s += '<g class="lg-hit" data-node="f-spmd"><rect x="12" y="8" width="696" height="' + (312 + PIPE_DY) + '" class="lg-frame"/>'
      + txt(24, 25, 'PL.SPMD × ' + NUM_CORES, 'sv-title') + txt(124, 25, '每个 block 跑一份', 'sv-label') + '</g>';
    s += '<g class="lg-hit" data-node="f-stride"><rect x="24" y="36" width="672" height="' + (276 + PIPE_DY) + '" class="lg-frame"/>'
      + txt(36, 53, 'FOR fa_w · GRID-STRIDE', 'sv-title') + txt(210, 53, '逐个处理轮到的 KV 块', 'sv-label') + '</g>';
    s += '<g class="lg-hit" data-node="tbl"><rect x="36" y="62" width="168" height="26" class="lg-node"/>'
      + txt(44, 79, 'block_table → 物理页', 'sv-strong') + '</g>';
    s += txt(214, 79, '逻辑块 → 物理页；SEQ_TILE 必须等于 page_size', k.pageOk ? 'sv-label' : 'sv-label sv-bad');
    s += '<g transform="translate(0,' + PIPE_DY + ')">';
    s += '<g class="lg-hit" data-node="f-pipe"><rect x="36" y="78" width="648" height="230" class="lg-frame' + (k.bad ? ' is-bad' : '') + '"/>'
      + txt(48, 95, 'PL.PIPELINE × ' + GP_SIZE + ' · STAGE=' + c.stage, 'sv-title')
      + txt(250, 95, 'In-flight heads · ' + k.stage, 'sv-label') + '</g>';
    // 三段计算（实框）；softmax 段按在飞的头数叠虚线层
    for (let i = k.stage - 1; i >= 1; i--) {
      s += '<rect x="' + (248 + i * 5) + '" y="' + (LG_PHASE_Y + i * 5) + '" width="296" height="' + LG_PHASE_H + '" class="lg-ghost'
        + (k.bad && i === k.stage - 1 ? ' is-bad' : '') + '"/>';
    }
    LG_PHASES.forEach(function (p) {
      const bad = k.bad && p[0] === 'p-softmax';
      const title = p[3];
      s += '<g class="lg-hit" data-node="' + p[0] + '"><rect x="' + p[1] + '" y="' + LG_PHASE_Y + '" width="' + p[2] + '" height="' + LG_PHASE_H
        + '" class="lg-phase' + (bad ? ' is-bad' : '') + '"/>' + txt(p[1] + 10, LG_PHASE_Y + 16, title, 'sv-title') + '</g>';
    });

    s += LG_EDGES.map(function (e) {
      return '<path d="' + e[2] + '" class="lg-edge" data-from="' + e[0] + '" data-to="' + e[1] + '"/>';
    }).join('');
    Object.keys(LG).forEach(function (id) { s += logicNode(id, badIds); });
    s += '</g></svg>';
    document.getElementById('hpLogic').innerHTML = s;
    document.getElementById('hpLogicMeta').textContent = state.preview
      ? '预览 · 方案 ' + state.preview + ' · stage=' + c.stage + ' · SEQ_TILE=' + c.seq
      : 'fa_fused · SPMD × 24 → grid-stride → pipeline × 8；源码 stage 控制并发驻留级数。';
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 硅片画布（viewBox 720 × 470）
  // 自上而下：GM → pl.spmd 的 24 个 block（概览）→ 展开其中 block 0 的 AIC / AIV（细节）
  // ════════════════════════════════════════════════════════════════════════════
  const BAR = { x: 418, w: 56, top: 206, bottom: 426 };
  // 放大图压低一些，让引线从两条阈值标签下方穿过
  const INSET = { x: 560, w: 136, top: 206, bottom: 334, lo: 176 * 1024, hi: 192 * 1024 };
  const CORE = { x: 384, y: 96, pitch: 26, size: 18 };
  function ubY(addr) { return BAR.bottom - addr / UB_NOMINAL * (BAR.bottom - BAR.top); }
  function insetY(addr) { return INSET.bottom - (addr - INSET.lo) / (INSET.hi - INSET.lo) * (INSET.bottom - INSET.top); }

  function region(id, inner) { return '<g class="sv-region" data-region="' + id + '">' + inner + '</g>'; }

  function bufClass(r) {
    if (r.id === 'carry') return 'k-carry';
    return 'k-' + bufById(r.id).kind;
  }

  function renderSvg() {
    const k = shown(), c = k.cfg, ref = state.preview ? current() : null, bad = k.bad;
    let s = '<svg viewBox="0 0 720 510" preserveAspectRatio="xMidYMin" role="img" aria-label="fa_fused 在一个 AI Core block 上的落点">'
      + '<defs>' + markers('hp')
      + '<pattern id="hpHatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="5" height="5" fill="#2a1716"/><line x1="0" y1="0" x2="0" y2="5" stroke="#ff6b5b" stroke-width="1.4" opacity=".55"/></pattern>'
      + '<clipPath id="hpInsetClip"><rect x="' + INSET.x + '" y="' + INSET.top + '" width="' + INSET.w + '" height="' + (INSET.bottom - INSET.top) + '"/></clipPath>'
      + '</defs>';

    // GM shares the hardware module container and alignment edges.
    s += '<rect x="12" y="4" width="696" height="78" rx="12" fill="#282828"/>';
    s += '<rect x="12" y="8" width="696" height="74" class="sv-frame"/>' + txt(24, 25, 'GM · HBM', 'sv-title');
    s += region('gm-kcache', '<g transform="translate(24 32) scale(.7)">'+PathGridGlyphs.memory('k_cache · paged','页 = '+c.page+' 行',214,60)+'</g>');
    s += region('gm-vcache', '<g transform="translate(182 32) scale(.7)">'+PathGridGlyphs.memory('v_cache · paged','页 = '+c.page+' 行',214,60)+'</g>');
    s += region('gm-table', '<g transform="translate(340 32) scale(.7)">'+PathGridGlyphs.memory('block / work table',k.blocks+' 个真实块',243,60)+'</g>');
    s += region('gm-out', '<g transform="translate(518 32) scale(.7)">'+PathGridGlyphs.memory('all_oi_tmp / mi / li','每块的部分结果',254,60)+'</g>');

    // pl.spmd 的 24 个 block：概览条，block 0 向下展开
    s += region('cores', '<rect x="' + (CORE.x - 100) + '" y="94" width="' + (12 * CORE.pitch + 100) + '" height="40" class="sv-hit"/>'
      + txt(CORE.x - 8, 120, 'PL.SPMD × ' + NUM_CORES, 'sv-title', 'end')
      + '<g class="sv-cores' + (bad ? ' is-bad' : '') + '">' + cores() + '</g>');
    s += '<path d="M384 103 H374 V155 H172 V158 M374 155 H523 V158" class="sv-zoom"/>';


    s += '<g transform="translate(0 36)">';
    // block 0 外框
    s += '<rect x="12" y="122" width="696" height="340" class="sv-frame"/>'
      ;

    // Two aligned surfaces: compute/storage architecture and the same UB's address expansion.
    s += '<rect x="350" y="122" width="346" height="340" rx="16" fill="#282828"/>' + txt(370,144,'BLOCK 0 · UB','sv-title');
    s += '<foreignObject x="24" y="122" width="308" height="340"><div xmlns="http://www.w3.org/1999/xhtml" id="hpMemoryPattern" style="width:308px;height:340px;overflow:hidden"></div></foreignObject>';

    s += ubSvg(k, ref, bad);
    s += '</g>';
    s += '</svg>';
    document.getElementById('hpSvg').innerHTML = s;
    PathMemoryPattern.render(document.getElementById('hpMemoryPattern'), k, applyHardwareFocus);
    document.getElementById('hpChipMeta').textContent = state.preview
      ? '预览 · 方案 ' + state.preview + ' · ' + OPTIONS[state.preview].title + '（虚线 = 当前配置的高水位）'
      : '计算数据流与存储映射 · 搬运路径示意 · UB 地址按比例';
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
      const x=CORE.x+(i%12)*CORE.pitch,y=CORE.y+Math.floor(i/12)*20;
      out += '<g class="sv-core-unit' + (i===0?' is-expanded':'') + '">'
        + '<rect x="'+x+'" y="'+y+'" width="18" height="14" rx="3" class="sv-core'+(i===0?' is-zoom':'')+'"/>'
        + '<path d="M'+(x+4)+' '+(y-2)+'v2 M'+(x+9)+' '+(y-2)+'v2 M'+(x+14)+' '+(y-2)+'v2 M'+(x+4)+' '+(y+14)+'v2 M'+(x+9)+' '+(y+14)+'v2 M'+(x+14)+' '+(y+14)+'v2" class="sv-core-pins"/>'
        + '</g>';
    }
    return out;
  }

  function ubSvg(k, ref, bad) {
    const top=ubY(UB_USABLE), water=ubY(Math.min(k.hw,UB_NOMINAL));
    let s='<g class="sv-ub" id="hpUb">';
    s+=txt(376,170,'高水位','sv-label')+txt(376,190,kb(k.hw),'ub-value'+(bad?' sv-bad':''));
    s+=region('ub','<rect x="'+BAR.x+'" y="'+BAR.top+'" width="'+BAR.w+'" height="'+(BAR.bottom-BAR.top)+'" rx="2" fill="#202020"/>');
    k.layout.forEach(r=>{
      const y1=ubY(Math.min(r.hi,UB_NOMINAL)),y2=ubY(r.lo);
      s+='<rect x="'+BAR.x+'" y="'+y1+'" width="'+BAR.w+'" height="'+Math.max(.5,y2-y1)+'" class="sv-buf '+bufClass(r)+'" data-buf="'+r.id+'" data-stage="'+r.stage+'"/>';
    });
    for(let st=0;st<k.stage;st++){
      const lo=k.carry+st*k.per,hi=lo+k.per,yt=ubY(hi),yb=ubY(lo);
      s+='<line x1="'+(BAR.x-5)+'" y1="'+(yt+1)+'" x2="'+(BAR.x-5)+'" y2="'+(yb-1)+'" class="ub-stage-rail"/>';
      s+=txt(BAR.x-10,(yt+yb)/2+3,'stage '+st,'sv-label','end');
    }
    s+='<rect x="'+BAR.x+'" y="'+BAR.top+'" width="'+BAR.w+'" height="'+(top-BAR.top)+'" class="ub-reserve"/>';
    [0,64,128,192].forEach(n=>{const y=ubY(n*1024);s+='<path d="M'+(BAR.x+BAR.w)+' '+y+'h5" class="ub-axis"/>'+txt(BAR.x+BAR.w+10,y+3,n+' KB','sv-label');});
    s+='<path d="M'+BAR.x+' '+top+'H'+(BAR.x+BAR.w+5)+'" class="'+(bad?'sv-line-usable':'ub-axis')+'"/>';
    s+=txt(BAR.x+BAR.w+10,top+12,'184 可用','sv-text'+(bad?' sv-bad':''));
    s+='<path d="M'+BAR.x+' '+water+'H'+(BAR.x+BAR.w)+'" stroke="'+(bad?'#ff6b5b':'#dedede')+'" stroke-width="1.5"/>';
    if(ref&&ref.hw!==k.hw)s+='<path d="M'+BAR.x+' '+ubY(Math.min(ref.hw,UB_NOMINAL))+'H'+(BAR.x+BAR.w)+'" class="sv-line-ghost"/>';
    s+=txt(BAR.x-10,BAR.bottom-2,'常驻','sv-label','end');
    // Orthogonal detail correspondence, not a second dataflow or perspective projection.
    const bracket=BAR.x+BAR.w+4,rangeBottom=ubY(INSET.lo);
    s+='<path d="M'+bracket+' '+BAR.top+'h4V'+rangeBottom+'h-4 M'+(bracket+4)+' '+((BAR.top+rangeBottom)/2)+'H546V'+INSET.top+'H'+INSET.x+'" class="ub-detail-link"/>';
    s+=txt(INSET.x,170,'局部地址','sv-label')+txt(INSET.x,190,'176–192 KB','sv-strong');
    s+='<rect x="'+INSET.x+'" y="'+INSET.top+'" width="'+INSET.w+'" height="'+(INSET.bottom-INSET.top)+'" rx="2" fill="#202020"/>';
    s+='<g clip-path="url(#hpInsetClip)">';
    k.layout.filter(r=>r.hi>INSET.lo).forEach(r=>{
      const y1=insetY(Math.min(r.hi,INSET.hi)),y2=insetY(Math.max(r.lo,INSET.lo));
      s+='<rect x="'+INSET.x+'" y="'+y1+'" width="'+INSET.w+'" height="'+Math.max(.5,y2-y1)+'" class="sv-buf '+bufClass(r)+'" data-buf="'+r.id+'" data-stage="'+r.stage+'"/>';
    });
    s+='<rect x="'+INSET.x+'" y="'+INSET.top+'" width="'+INSET.w+'" height="'+(insetY(UB_USABLE)-INSET.top)+'" class="ub-reserve"/>';
    if(bad)s+='<rect x="'+INSET.x+'" y="'+insetY(Math.min(k.hw,INSET.hi))+'" width="'+INSET.w+'" height="'+(insetY(UB_USABLE)-insetY(Math.min(k.hw,INSET.hi)))+'" class="sv-over"/>';
    k.layout.filter(r=>r.hi>INSET.lo).forEach(r=>{
      const y1=insetY(Math.min(r.hi,INSET.hi)),y2=insetY(Math.max(r.lo,INSET.lo));
      if(y2-y1>9)s+=txt(INSET.x+5,(y1+y2)/2+2.5,shortName(r),'ub-buffer-name');
    });
    s+='</g>';
    s+=txt(INSET.x+INSET.w/2,INSET.top+24,'保留 8 KB','sv-text','middle');
    s+='<path d="M'+INSET.x+' '+insetY(UB_USABLE)+'H'+(INSET.x+INSET.w)+'" class="'+(bad?'sv-line-usable':'ub-axis')+'"/>';
    s+=txt(INSET.x,INSET.bottom+15,'176 KB','sv-label');
    s+=txt(INSET.x,389,bad?'越界':'可用余量','sv-label');
    s+=txt(INSET.x,407,bad?bytes(k.hw-UB_USABLE):bytes(UB_USABLE-k.hw),'ub-value'+(bad?' sv-bad':''));
    return s+'</g>';
  }

  function shortName(r) {
    if (r.id === 'carry') return '常驻';
    return 's' + r.stage + ' ' + bufById(r.id).name.replace('row_expand_sub 临时量', 'rexp_tmp');
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 焦点：源码行 ⇄ 逻辑节点 ⇄ 硅片区域 ⇄ Vec 缓冲
  // ════════════════════════════════════════════════════════════════════════════
  function fromNodes(ids) {
    // 计算段展开成它包含的节点
    ids = ids.reduce(function (acc, id) { return acc.concat([id], nodeById(id).group || []); }, []);
    const f = { lines: [], regions: [], bufs: [], nodes: ids.slice() };
    ids.forEach(function (id) {
      const n = nodeById(id);
      f.nodes = f.nodes.concat(n.show || []);
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
      f = fromNodes(REGION_NODES[h.id] || nodesWhere(function (n) { return (n.regions || []).indexOf(h.id) >= 0; }));
      f.regions = [h.id];
      if (h.id !== 'ub' && h.id !== 'aiv') f.bufs = [];
    }
    // 复用方案下，被省掉的两块指向 scores_scaled
    if (shown().cfg.reuse) f.bufs = f.bufs.map(function (b) { return bufById(b) && bufById(b).alias ? 'scores_scaled' : b; });
    return f;
  }

  function currentFocus() {
    const f = focusFor(state.hover) || sectionFocus(shown());
    if (state.hover && f.bufs.length) f.regions = uniq(f.regions.concat(['aiv', 'ub']));
    return f;
  }

  // Shared patterns lay out their routes asynchronously. Reapply only hardware
  // classes after layout, without scrolling source or changing caption geometry.
  function applyHardwareFocus() {
    const f = currentFocus();
    document.querySelectorAll('.sv-region').forEach(function (el) {
      el.classList.toggle('is-hot', f.regions.indexOf(el.dataset.region) >= 0);
    });
    const ub = document.getElementById('hpUb');
    if (ub) ub.classList.toggle('has-hot', f.bufs.length > 0);
    document.querySelectorAll('.sv-buf').forEach(function (el) {
      el.classList.toggle('is-hot', f.bufs.indexOf(el.dataset.buf) >= 0);
    });
  }

  function applyFocus() {
    const k = shown(), f = currentFocus();
    // 只有真正悬停时才把无关部分压暗；章节默认焦点不压暗，避免页面长期发灰
    document.body.classList.toggle('is-focusing', !!state.hover);
    document.querySelectorAll('.cl').forEach(function (el) {
      el.classList.toggle('is-hot', f.lines.indexOf(Number(el.dataset.line)) >= 0);
    });
    applyHardwareFocus();
    document.querySelectorAll('.lg-hit').forEach(function (el) {
      el.classList.toggle('is-hot', f.nodes.indexOf(el.dataset.node) >= 0);
    });
    document.querySelectorAll('.lg-edge').forEach(function (el) {
      el.classList.toggle('is-hot', f.nodes.indexOf(el.dataset.from) >= 0 || f.nodes.indexOf(el.dataset.to) >= 0);
    });
    document.querySelectorAll('.map-table tr[data-line]').forEach(function (el) {
      el.classList.toggle('is-hot', f.lines.indexOf(Number(el.dataset.line)) >= 0);
    });
    // 焦点从逻辑图、硅片或右栏表格发起时，把亮起的行滚进源码区；源码区内悬停或移开时不动，免得打断阅读
    if (state.hover && (state.hover.type !== 'line' || state.hover.fromTable)) revealLines();
    renderCaption(k);
  }

  // [796, 797, 798, 769] → "769 / 796–798"：只把源码里相邻的行合成区间
  function fmtLines(lines) {
    const shown = SOURCE.map(function (r) { return r.no; });
    const sorted = uniq(lines).sort(function (a, b) { return a - b; });
    const parts = [];
    sorted.forEach(function (no) {
      const last = parts[parts.length - 1];
      if (last && shown.indexOf(no) === shown.indexOf(last[1]) + 1) last[1] = no;
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
    const viewTop = box.scrollTop, viewBottom = viewTop + box.clientHeight;
    if (top >= viewTop && bottom <= viewBottom) return;
    // 整段放得下就居中整段，放不下就让第一行靠上
    const target = bottom - top < box.clientHeight ? (top + bottom - box.clientHeight) / 2 : top - 24;
    box.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
  }

  function renderCaption(k) {
    const h = state.hover;
    let key = '', body = '';
    if (h && h.type === 'line') { key = 'line ' + h.no + ' → 逻辑 · 硅片'; body = lineCaption(h.no, k); }
    else if (h && h.type === 'node') {
      key = '逻辑 → line ' + fmtLines(fromNodes([h.id]).lines) + ' · 硅片';
      body = nodeCaption(h.id, k);
    } else if (h && h.type === 'buf') {
      const b = bufById(h.id);
      if (b) { key = 'Vec 缓冲 → 逻辑 · line ' + b.line; body = lineCaption(b.line, k); }
      else { key = 'Vec 常驻区'; body = '常驻区 ' + kb(k.carry) + '：沿用 Lens 的估算口径（o_i / m_i / l_i），不随 stage 复制。'; }
    } else if (h && h.type === 'region') {
      key = '硅片 → 逻辑 · line ' + fmtLines(focusFor(h).lines);
      body = regionCaption(h.id, k);
    }
    if (!body) {
      if (k.verdict === 'silent') {
        body = '落进保留区的是 <b>stage ' + (k.stage - 1) + '</b> 的 ' + k.intruders.map(function (r) {
          return '<code>' + (bufById(r.id) || {}).name + '</code>';
        }).join('、') + '：都在最高地址，正好对上“只有最高地址的 tile 出 NaN”。';
      } else body = regionCaption('ub', k);
    }
    document.getElementById('hpCaption').innerHTML = (key ? '<span class="cap-k">' + key + '</span>' : '') + body;
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
    return '<h2 class="st-h">Pipeline 配置 · <code>stage</code> 2 → 4</h2>'
      + '<p class="st-p">增加 pipeline stage，以提高不同 head 的 QK（AIC）与 softmax（AIV）计算重叠度。'
      + '该参数不改变算法语义，但可能增加中间量的并发驻留与 UB 占用。</p>'
      + '<p class="st-p">逻辑图展示并发驻留的 softmax 实例；硬件映射展示对应的 UB 占用。</p>'
      + meter(current())
      + '<div class="st-actions">'
      + (isChanged ? '' : '<button class="btn btn-primary" type="button" data-act="apply-changed">应用 stage = 4</button>')
      + (isChanged ? next(2, '查看内存占用') : '<button class="btn" type="button" data-act="open" data-step="2">查看内存占用</button>')
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
    return '<h2 class="st-h">UB 驻留 · pipeline stage = ' + k.stage + '</h2>'
      + '<p class="st-p">Pipeline 中同时驻留 ' + k.stage + ' 个 head，softmax 对应 ' + k.stage + ' 层。'
      + '这些中间量分配于 AIV 的 Vec(UB)，每增加一级 stage，估算占用增加 <b>' + kb(k.per) + '</b>。</p>'
      + '<details class="story-detail"><summary>缓冲明细</summary><table class="map-table"><thead><tr><th>行</th><th>中间量</th><th>每级</th><th>× ' + k.stage + ' 级</th></tr></thead><tbody>' + rows
      + '</tbody><tfoot><tr><td></td><td>+ 常驻区</td><td class="num"></td><td class="num">' + kb(k.carry) + '</td></tr>'
      + '<tr><td></td><td>高水位</td><td class="num"></td><td class="num">' + kb(k.hw) + '</td></tr></tfoot></table></details>'
      + '<p class="note">① 打分和 ③ 加权求和的操作数在 AIC 一侧（L0A / L0B / L0C），不随 stage 复制；此处仅评估 Vec(UB) 的容量约束。</p>'
      + '<div class="st-actions">' + next(3, '查看诊断') + '</div>';
  }

  function body3() {
    const k=current(),excess=k.hw-UB_USABLE;
    const headline=k.bad?'UB 容量超限':'UB 容量满足约束';
    const explanation=k.verdict==='silent'
      ? '估算 footprint 超出 usable capacity，但仍低于 nominal capacity。需要通过 compiler memory map 核对 reserved region。'
      : k.verdict==='compile'?'估算 footprint 同时超出 nominal capacity。':'当前估算 footprint 低于 usable capacity；仍需 compiler artifact 与 device validation。';
    return '<h2 class="st-h">'+headline+'</h2>'
      +'<div class="diagnosis-primary"><span>'+(excess>0?'超出':'可用余量')+'</span><strong class="'+(k.bad?'is-bad':'')+'">'+bytes(Math.abs(excess))+'</strong></div>'
      +'<p class="st-p">'+explanation+'</p>'
      +'<dl class="diagnosis-metrics"><div><dt>估算占用</dt><dd>'+kb(k.hw)+'</dd></div><div><dt>可用容量</dt><dd>184 KB</dd></div></dl>'
      +'<details class="story-detail"><summary>容量依据</summary><p>Nominal capacity：192 KB；runtime reserved region：8 KB；usable capacity：184 KB。Reserved region 依据 pto-isa#170 的 a2a3 场景，可能随平台与 CANN 版本变化。</p></details>'
      +'<details class="story-detail"><summary>失效机制与 Issue 证据</summary><p>pto-isa#170 描述：按 nominal capacity 分配时，buffer 可能与 runtime reserved region 重叠，运行未触发异常但输出出现 NaN。</p><p>此处地址布局为估算，当前源码尚未编译或运行于设备，不能据此断言会出现同一失效。</p><a href="'+ISSUE_URL+'" target="_blank" rel="noopener noreferrer">Source · pto-isa#170 ↗</a></details>'
      +'<details class="story-detail"><summary>验证要求</summary><p>检查 compiler memory map 中的 allocation offset、buffer lifetime 和 reserved region；随后执行 Golden comparison 与 device profiling。</p></details>'
      +'<div class="st-actions">'+next(4,'比较调整方案')+'</div>';
  }

  function body4() {
    const chosen = chosenKey(state.cfg);
    const cards = Object.keys(OPTIONS).map(function (key) {
      const o = OPTIONS[key], k = calc(o.cfg);
      const cls = 'opt' + (state.preview === key ? ' is-preview' : '') + (chosen === key ? ' is-chosen' : '');
      return '<div class="' + cls + '" data-opt="' + key + '" tabindex="0">'
        + '<div class="opt-head"><span class="opt-id">' + key + '</span><span class="opt-title">' + o.title + '</span><span class="opt-sub">' + o.sub + '</span></div>'
        + '<div class="opt-res"><span>Vec <b class="' + (k.hw > UB_USABLE ? 'is-bad' : 'is-good') + '">' + kb(k.hw) + '</b></span>'
        + '<span>L0B 利用率 <b>' + pct(k.kDB / L0B_CAP) + '</b></span><span>主要代价 <b>' + o.to + '</b></span></div>'
        + '<details class="story-detail"><summary>代价与前提</summary><p class="opt-cost">' + o.cost(k) + '</p></details>'
        + '<div class="opt-foot"><span class="opt-scope">' + o.scope + '</span>'
        + '<button class="btn' + (chosen === key ? '' : ' btn-primary') + '" type="button" data-act="choose" data-opt="' + key + '">'
        + (chosen === key ? '已采用' : '采用并复查') + '</button></div></div>';
    }).join('');
    return '<h2 class="st-h">调整方案比较</h2>'
      + '<p class="st-p">地址条上的虚线是当前配置的高水位。</p>'
      + cards
      + '<p class="note">方案 A 需通过 profiling 评估性能影响；方案 B 需同步运行时分页配置；方案 C 需通过编译产物确认内存复用。</p>';
  }

  function body5() {
    const k = current(), before = calc(CHANGED), key = chosenKey(state.cfg);
    if (k.bad) {
      return '<h2 class="st-h">当前配置仍超出 UB 可用容量</h2>'
        + '<div class="callout">高水位 ' + kb(k.hw) + '，超过 184 KB。请选择调整方案。</div>'
        + '<div class="st-actions">' + next(4, '查看调整方案') + '</div>';
    }
    function row(name, a, b, cls, ev, evText) {
      return '<tr><td>' + name + '</td><td class="v">' + a + '</td><td class="v ' + (cls || '') + '">' + b + '</td><td><span class="ev ' + ev + '">' + evText + '</span></td></tr>';
    }
    const table = '<table class="cmp"><thead><tr><th>资源</th><th>改动后</th><th>' + (key ? '方案 ' + key : '当前') + '</th><th>依据</th></tr></thead><tbody>'
      + row('Vec(UB) 高水位', '<span class="is-bad">' + kb(before.hw) + '</span>', kb(k.hw), 'is-good', 'e-est', '估算 → 编译产物')
      + row('Right(L0B) 双缓冲', kb(before.kDB), kb(k.kDB), k.kDB !== before.kDB ? 'is-good' : '', 'e-est', '估算')
      + row('SEQ_TILE / page_size', before.cfg.seq + ' / ' + before.cfg.page, state.cfg.seq + ' / ' + state.cfg.page, k.pageOk ? (state.cfg.page !== before.cfg.page ? 'is-warn' : '') : 'is-bad', 'e-build', '运行时契约')
      + row('fa_fused 工作块数', String(before.blocks), String(k.blocks), k.blocks !== before.blocks ? 'is-warn' : '', 'e-est', '估算')
      + row('流水驻留级数', String(before.stage), String(k.stage), k.stage !== before.stage ? 'is-warn' : '', 'e-board', '需 profiling')
      + row('中间量复用', '否', state.cfg.reuse ? '假设是' : '否', state.cfg.reuse ? 'is-warn' : '', 'e-build', '需编译产物')
      + '</tbody></table>';
    const checks = ['编译产物的内存映射里，Vec 高水位低于 <b>184 KB</b>，而不是只低于 192 KB',
      '上板结果与 torch 参考逐元素对比，没有 NaN / Inf',
      'Profiling 对比 stage 2 / 3 / 4 的 fa_fused 耗时，量化 throughput 与 latency 的变化'];
    if (state.cfg.page !== BASE.page) checks.push('运行时 KV 分页同步改为 ' + state.cfg.page + '，并核对其它层与 block_table 生成');
    if (state.cfg.reuse) checks.push('编译产物中 scores_valid / scores 确实复用 scores_scaled 的地址');
    return '<h2 class="st-h">验证要求</h2>'
      + meter(k) + table
      + '<p class="st-sec">后续验证项</p>'
      + '<ul class="checklist">' + checks.map(function (t) { return '<li><span>' + t + '</span></li>'; }).join('') + '</ul>'
      + '<div class="st-actions"><button class="btn" type="button" data-act="open" data-step="4">重新选择方案</button>'
      + '<button class="btn" type="button" data-act="reset">重置配置</button></div>';
  }

  // 每章标题行上的实时结论：收起时也能看到这条路走到了哪里
  function sectionSummary(n, k) {
    const c = state.cfg, key = chosenKey(c);
    if (n === 1) return c.stage === BASE.stage ? { t: '未改动 · stage=' + c.stage } : { t: 'stage ' + BASE.stage + ' → ' + c.stage, cls: 'is-warn' };
    if (n === 2) return { t: kb(k.per * k.stage) };
    if (n === 3) {
      if (k.verdict === 'silent') return { t: '超出 · ' + bytes(k.hw - UB_USABLE), cls: 'is-bad' };
      if (k.verdict === 'compile') return { t: '超过 192 KB · 编译报错', cls: 'is-bad' };
      return { t: '余量 ' + bytes(UB_USABLE - k.hw), cls: 'is-good' };
    }
    if (n === 4) return key ? { t: '已采用方案 ' + key, cls: 'is-good' } : { t: '3 个方案' };
    if (k.bad) return { t: '容量超限' , cls: 'is-bad' };
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
      text = '✕ Vec 高水位 ' + kb(k.hw) + ' > usable capacity 184 KB · reserved-region overlap risk · 待 compiler artifact 验证';
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
    const diagnostic=document.getElementById('hpDiagnostic'),k=current();
    diagnostic.hidden=!k.bad;
    diagnostic.innerHTML=k.bad ? '<button type="button" data-act="diagnose"><strong>⚠ UB 容量风险</strong><span>高水位 '+kb(k.hw)+' / 可用 184 KB · 超出 '+bytes(k.hw-UB_USABLE)+'</span><b>查看诊断 →</b></button>' : '';

  }

  function openSection(n, scroll) {
    state.step = n; state.preview = null; state.hover = null;
    render();
    revealLines();
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
    const stageBtn = e.target.closest('.stage-ctl button[data-stage]');
    if (stageBtn) { setCfg(Object.assign({}, state.cfg, { stage: Number(stageBtn.dataset.stage) })); return; }
    const act = e.target.closest('[data-act]');
    if (!act) return;
    const a = act.dataset.act;
    if (a === 'diagnose') {
      document.getElementById('hpStoryPane').hidden=false;
      const toggle=document.getElementById('hpStoryToggle');toggle.setAttribute('aria-expanded','true');toggle.textContent='隐藏主线';
      openSection(3,true);syncResizeHandle();
    }
    else if (a === 'toggle') { const n = Number(act.dataset.step); openSection(state.step === n ? 0 : n, false); }
    else if (a === 'open') { openSection(Number(act.dataset.step), true); }
    else if (a === 'apply-changed') { setCfg(CHANGED); }
    else if (a === 'choose') { state.step = 5; setCfg(OPTIONS[act.dataset.opt].cfg); }
    else if (a === 'reset') { state.step = 1; state.hover = null; setCfg(BASE); }
  });

  const storyPane = document.getElementById('hpStoryPane');
  const storyToggle = document.getElementById('hpStoryToggle');
  storyToggle.addEventListener('click', function () {
    storyPane.hidden = !storyPane.hidden;
    storyToggle.setAttribute('aria-expanded', String(!storyPane.hidden));
    storyToggle.textContent = storyPane.hidden ? '显示主线' : '隐藏主线';
    syncResizeHandle();
  });

  // 拖动源码区边界；同时支持键盘微调。
  const codePane = document.getElementById('hpCodePane');
  const resizeHandle = document.getElementById('hpCodeResize');
  const mainPane = document.querySelector('.hp-main');
  function resizeBounds() {
    const width = mainPane.getBoundingClientRect().width;
    return window.innerWidth > 1280
      ? { min: 340, max: Math.max(340, width - (storyPane.hidden ? 560 : 920)) }
      : { min: 280, max: Math.max(280, width - 360) };
  }
  function setCodeWidth(width) {
    const bounds = resizeBounds();
    const value = Math.round(Math.max(bounds.min, Math.min(bounds.max, width)));
    mainPane.style.setProperty('--code-width', value + 'px');
    resizeHandle.setAttribute('aria-valuemin', bounds.min);
    resizeHandle.setAttribute('aria-valuemax', Math.round(bounds.max));
    resizeHandle.setAttribute('aria-valuenow', value);
  }
  resizeHandle.addEventListener('pointerdown', function (e) {
    if (e.button !== 0) return;
    e.preventDefault();
    const startX = e.clientX, startWidth = codePane.getBoundingClientRect().width;
    resizeHandle.setPointerCapture(e.pointerId);
    resizeHandle.classList.add('is-dragging');
    document.body.classList.add('is-resizing');
    function move(event) { setCodeWidth(startWidth + event.clientX - startX); }
    function end() {
      resizeHandle.classList.remove('is-dragging');
      document.body.classList.remove('is-resizing');
      resizeHandle.removeEventListener('pointermove', move);
      resizeHandle.removeEventListener('pointerup', end);
      resizeHandle.removeEventListener('pointercancel', end);
      resizeHandle.removeEventListener('lostpointercapture', end);
    }
    resizeHandle.addEventListener('pointermove', move);
    resizeHandle.addEventListener('pointerup', end);
    resizeHandle.addEventListener('pointercancel', end);
    resizeHandle.addEventListener('lostpointercapture', end);
  });
  resizeHandle.addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    setCodeWidth(codePane.getBoundingClientRect().width + (e.key === 'ArrowLeft' ? -20 : 20));
  });
  function syncResizeHandle() {
    const bounds = resizeBounds();
    resizeHandle.setAttribute('aria-valuemin', bounds.min);
    resizeHandle.setAttribute('aria-valuemax', Math.round(bounds.max));
    resizeHandle.setAttribute('aria-valuenow', Math.round(codePane.getBoundingClientRect().width));
  }
  window.addEventListener('resize', syncResizeHandle);
  syncResizeHandle();

  // 拖动中栏的横向边界，在两个画布间分配高度。
  const midPane = document.querySelector('.hp-mid');
  const logicPane = document.getElementById('hpLogicPane');
  const verticalHandle = document.getElementById('hpVerticalResize');
  function verticalBounds() {
    return { min: 140, max: Math.max(140, midPane.clientHeight - 240) };
  }
  function setLogicHeight(height) {
    const bounds = verticalBounds();
    const value = Math.round(Math.max(bounds.min, Math.min(bounds.max, height)));
    midPane.style.setProperty('--logic-height', value + 'px');
    verticalHandle.setAttribute('aria-valuemin', bounds.min);
    verticalHandle.setAttribute('aria-valuemax', bounds.max);
    verticalHandle.setAttribute('aria-valuenow', value);
  }
  verticalHandle.addEventListener('pointerdown', function (e) {
    if (e.button !== 0) return;
    e.preventDefault();
    const startY = e.clientY, startHeight = logicPane.getBoundingClientRect().height;
    verticalHandle.setPointerCapture(e.pointerId);
    verticalHandle.classList.add('is-dragging');
    document.body.classList.add('is-resizing-vertical');
    function move(event) { setLogicHeight(startHeight + event.clientY - startY); }
    function end() {
      verticalHandle.classList.remove('is-dragging');
      document.body.classList.remove('is-resizing-vertical');
      verticalHandle.removeEventListener('pointermove', move);
      verticalHandle.removeEventListener('pointerup', end);
      verticalHandle.removeEventListener('pointercancel', end);
      verticalHandle.removeEventListener('lostpointercapture', end);
    }
    verticalHandle.addEventListener('pointermove', move);
    verticalHandle.addEventListener('pointerup', end);
    verticalHandle.addEventListener('pointercancel', end);
    verticalHandle.addEventListener('lostpointercapture', end);
  });
  verticalHandle.addEventListener('keydown', function (e) {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
    e.preventDefault();
    setLogicHeight(logicPane.getBoundingClientRect().height + (e.key === 'ArrowUp' ? -20 : 20));
  });
  function syncVerticalHandle() {
    const bounds = verticalBounds();
    verticalHandle.setAttribute('aria-valuemin', bounds.min);
    verticalHandle.setAttribute('aria-valuemax', bounds.max);
    verticalHandle.setAttribute('aria-valuenow', Math.round(logicPane.getBoundingClientRect().height));
  }
  window.addEventListener('resize', syncVerticalHandle);
  syncVerticalHandle();
  // 悬停：代码行、表格行、逻辑节点、硅片区域、Vec 缓冲
  function hoverFrom(target) {
    const buf = target.closest('.sv-buf');
    if (buf) return { type: 'buf', id: buf.dataset.buf };
    const reg = target.closest('.sv-region');
    if (reg) return { type: 'region', id: reg.dataset.region };
    const node = target.closest('.lg-hit');
    if (node) return { type: 'node', id: node.dataset.node };
    const line = target.closest('.cl.is-mapped');
    if (line) return { type: 'line', no: Number(line.dataset.line) };
    const row = target.closest('.map-table tr[data-line]');
    if (row) return { type: 'line', no: Number(row.dataset.line), fromTable: true };
    return null;
  }
  document.addEventListener('mouseover', function (e) {
    if (e.target.closest('#hpCaption, .pane-info')) return;
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
  document.addEventListener('keydown', function(e) {
    if(e.key==='Escape') document.querySelectorAll('.pane-info[open]').forEach(el=>el.removeAttribute('open'));
  });
  document.addEventListener('click', function(e) {
    document.querySelectorAll('.pane-info[open]').forEach(el=>{if(!el.contains(e.target))el.removeAttribute('open');});
  });
})();
