/* ──────────────────────────────────────────────────────────────────────────────
 * Hardware-Native Lens · 交互
 * ────────────────────────────────────────────────────────────────────────────── */
(function () {
  'use strict';

  const M = window.HNL;
  const V = window.HNV;
  const esc = V.esc;
  const $ = function (sel, root) { return (root || document).querySelector(sel); };

  // 透镜：Tensor/Tile 合并为一组，其余各一组
  const LENSES = [
    { id: 'tensor', label: 'Tensor / Tile', members: ['tensor', 'tile'], desc: '形状 · dtype · 片上空间' },
    { id: 'scope', label: '任务与依赖', members: ['scope'], desc: '任务边界 · 任务句柄 · 显式依赖' },
    { id: 'parallel', label: '并行与同步', members: ['parallel'], desc: '并行调度 · 流水交接 · 原子累加' },
    { id: 'layout', label: '内部布局', members: ['layout'], desc: '填充 · 分页 · 对齐' },
    { id: 'precision', label: '数值与精度', members: ['precision'], desc: 'dtype · 转换 · 累加' }
  ];
  const LENS_OF = {};
  LENSES.forEach(function (L) { L.members.forEach(function (m) { LENS_OF[m] = L.id; }); });

  // 约束 → 代表性可视化面板
  const CONSTRAINT_CONCEPT = {
    'cube-m-fractal': 'layout:q-pad',
    'page-identity': 'layout:paged-kv',
    'l0b-capacity': 'tile:k-tile',
    'ub-usable-headroom': 'parallel:pipeline',
    'slab-1to1': 'layout:split-k',
    'down-chunk-align': 'layout:split-k',
    'splitk-divides-k': 'layout:split-k',
    'gp-head-cover': 'parallel:grid-stride',
    'spmd-even-split': 'parallel:spmd',
    'block-balance': 'parallel:grid-stride'
  };

  const STATUS_LABEL = { ok: '未触发', warn: '需关注', error: '未满足' };
  const CONSTRAINT_KIND = {
    'cube-m-fractal': '硬件布局模型', 'l0b-capacity': '容量估算',
    'ub-usable-headroom': '容量估算 · 保留区',
    'block-balance': '性能估算', 'page-identity': '运行时契约',
    'slab-1to1': '算子实现', 'down-chunk-align': '算子实现',
    'splitk-divides-k': '算子实现', 'gp-head-cover': '算子实现', 'spmd-even-split': '算子实现'
  };

  // ── 状态 ───────────────────────────────────────────────────────────────────
  const state = {
    cfg: Object.assign({}, M.DEFAULTS),
    lens: { tensor: true, scope: true, parallel: true, layout: true, precision: true },
    sel: { type: 'overview', id: null },
    anchor: null,
    issueHint: null,
    issueLine: null,
    inlineConfig: true,
    issueCores: 24,
    validExampleLength: 300,
    validExampleBlock: 'tail',
    bufferStage: 'qk',
    bufferStep: 'qk',
    bufferTile: 'full',
    bufferChunk: 0,
    bufferPlaying: false,
    bufferMode: 'separate',
    capacityPreview: false,
    capacityBaseline: null,
    filter: 'all',
    constraintReturn: null,
    d: null,
    results: {}
  };

  // ════════════════════════════════════════════════════════════════════════════
  // 代码着色
  // ════════════════════════════════════════════════════════════════════════════
  const KEYWORDS = ['with', 'for', 'in', 'as', 'def', 'return', 'assert', 'and', 'or', 'not',
    'if', 'elif', 'else', 'while', 'lambda', 'import', 'from', 'class', 'yield', 'pass',
    'break', 'continue', 'try', 'except', 'finally', 'raise', 'is', 'del', 'global', 'await', 'async'];
  const LANG_CONST = ['True', 'False', 'None'];
  const BUILTINS = ['range', 'len', 'min', 'max', 'int', 'float', 'str', 'bool', 'sum', 'abs',
    'enumerate', 'zip', 'print', 'list', 'dict', 'tuple'];
  const DTYPES = ['FP32', 'FP16', 'BF16', 'INT4', 'INT8', 'INT32', 'INT64', 'UINT8',
    'INDEX', 'TASK_ID', 'BOOL'];

  // 1 注释 · 2 字符串 · 3 装饰器 · 4 数字 · 5 标识符 · 6 空白 · 7 标点 · 8 运算符
  const TOKEN_RE = /(#[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(@[A-Za-z_][A-Za-z0-9_.]*)|(\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|([(){}\[\].,:;])|([^\s])/g;

  function fill(tpl, d) {
    return String(tpl).replace(/\{\{([A-Z_0-9]+)\}\}/g, function (_, k) {
      const v = d[k];
      if (v == null) return '?';
      return typeof v === 'number' && !Number.isInteger(v) ? String(Math.round(v * 100) / 100) : String(v);
    });
  }

  function symbolOf(name) {
    if (M.SYMBOLS[name]) return M.SYMBOLS[name];
    // pl.a.b.c → 尝试更短的前缀
    const parts = name.split('.');
    for (let i = parts.length - 1; i > 1; i--) {
      const p = parts.slice(0, i).join('.');
      if (M.SYMBOLS[p]) return M.SYMBOLS[p];
    }
    return null;
  }

  // ── 分词 ──────────────────────────────────────────────────────────────────
  function tokenize(line) {
    const toks = [];
    let m;
    TOKEN_RE.lastIndex = 0;
    while ((m = TOKEN_RE.exec(line)) !== null) {
      if (m[1] != null) toks.push({ k: 'cmt', v: m[1] });
      else if (m[2] != null) toks.push({ k: 'str', v: m[2] });
      else if (m[3] != null) toks.push({ k: 'dec', v: m[3] });
      else if (m[4] != null) toks.push({ k: 'num', v: m[4] });
      else if (m[5] != null) toks.push({ k: 'id', v: m[5] });
      else if (m[6] != null) toks.push({ k: 'ws', v: m[6] });
      else if (m[7] != null) toks.push({ k: 'punc', v: m[7] });
      else toks.push({ k: 'op', v: m[8] });
    }
    return toks;
  }

  function prevIdx(toks, i) { for (let j = i - 1; j >= 0; j--) if (toks[j].k !== 'ws') return j; return -1; }
  function nextIdx(toks, i) { for (let j = i + 1; j < toks.length; j++) if (toks[j].k !== 'ws') return j; return -1; }

  // ── 标识符分类 ────────────────────────────────────────────────────────────
  function classifyId(toks, i, inChain) {
    const v = toks[i].v;
    if (KEYWORDS.indexOf(v) >= 0) return 'tk-kw';
    if (LANG_CONST.indexOf(v) >= 0) return 'tk-lang';

    const p = prevIdx(toks, i);
    const n = nextIdx(toks, i);
    const isCall = n >= 0 && toks[n].k === 'punc' && toks[n].v === '(';

    // def foo(...)
    if (p >= 0 && toks[p].k === 'id' && toks[p].v === 'def') return 'tk-fndef';
    if (isCall) return 'tk-fn';
    if (!inChain && BUILTINS.indexOf(v) >= 0) return 'tk-fn';
    if (DTYPES.indexOf(v) >= 0) return 'tk-type';

    // 关键字实参：紧贴等号（PEP8 下赋值两侧有空格，实参没有）
    const adj = toks[i + 1];
    if (adj && adj.k === 'op' && adj.v === '=' && !(toks[i + 2] && toks[i + 2].k === 'op' && toks[i + 2].v === '=')) {
      return 'tk-param';
    }

    if (v === 'pl') return 'tk-mod';
    if (/^[A-Z][A-Z0-9_]*$/.test(v) && v.length > 1) return 'tk-const';
    if (/^[A-Z]/.test(v)) return 'tk-type';
    if (inChain) return 'tk-attr';
    return 'tk-var';
  }

  // 点号链：pl.Level.CORE_GROUP 整段作为一个可点击对象
  function chainAt(toks, i) {
    let j = i;
    const parts = [toks[i].v];
    const idx = [i];
    while (toks[j + 1] && toks[j + 1].k === 'punc' && toks[j + 1].v === '.'
      && toks[j + 2] && toks[j + 2].k === 'id') {
      parts.push(toks[j + 2].v);
      idx.push(j + 2);
      j += 2;
    }
    return { parts: parts, idx: idx, end: j };
  }

  function span(cls, text) { return '<span class="' + cls + '">' + esc(text) + '</span>'; }

  function highlight(line, resolved) {
    const toks = tokenize(line);
    let out = '';
    for (let i = 0; i < toks.length; i++) {
      const tk = toks[i];
      if (tk.k === 'ws') { out += esc(tk.v); continue; }
      if (tk.k === 'cmt') { out += span('tk-cmt', tk.v); continue; }
      if (tk.k === 'str') { out += span('tk-str', tk.v); continue; }
      if (tk.k === 'dec') { out += span('tk-dec', tk.v); continue; }
      if (tk.k === 'num') { out += span('tk-num', tk.v); continue; }
      if (tk.k === 'punc') { out += span('tk-punc', tk.v); continue; }
      if (tk.k === 'op') { out += span('tk-op', tk.v); continue; }

      // 标识符（可能是点号链）
      const ch = chainAt(toks, i);
      const cid = symbolOf(ch.parts.join('.'));
      const next = nextIdx(toks, i);
      const assignmentTarget = next >= 0 && toks[next].v === '=' && !(toks[next + 1] && toks[next + 1].v === '=');
      if (resolved && ch.parts.length === 1 && Object.prototype.hasOwnProperty.call(resolved, tk.v) && !assignmentTarget) {
        const hint = tk.v + ' = ' + resolved[tk.v] + '（当前配置代入；原源码为 ' + tk.v + '）';
        const value = '<span class="tk-num tk-config-value" title="' + esc(hint) + '">' + esc(resolved[tk.v]) + '</span>';
        out += cid ? '<span class="tk-link" data-concept="' + cid + '" role="link" tabindex="0" aria-label="' + esc(hint) + '" title="' + esc(hint) + '">' + value + '</span>' : value;
        continue;
      }
      let inner = '';
      for (let c = 0; c < ch.idx.length; c++) {
        if (c > 0) inner += span('tk-punc', '.');
        inner += span(classifyId(toks, ch.idx[c], c > 0), toks[ch.idx[c]].v);
      }
      out += cid
        ? '<span class="tk-link" data-concept="' + cid + '" role="link" tabindex="0">' + inner + '</span>'
        : inner;
      i = ch.end;
    }
    return out;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 约束求值
  // ════════════════════════════════════════════════════════════════════════════
  function evaluate() {
    state.d = M.derive(state.cfg);
    state.results = {};
    M.CONSTRAINTS.forEach(function (c) {
      state.results[c.id] = c.eval(state.d);
    });
  }

  function lineIssues() {
    // 行号 → 最严重的约束
    const map = {};
    M.CONSTRAINTS.forEach(function (c) {
      const res = state.results[c.id];
      c.lines.forEach(function (no) {
        const cur = map[no];
        const rank = { ok: 0, warn: 1, error: 2 };
        if (!cur || rank[res.status] > rank[cur.status]) {
          map[no] = { status: res.status, cid: c.id, title: c.title, detail: res.detail };
        }
      });
    });
    return map;
  }

  function statusSummary() {
    let ok = 0, warn = 0, error = 0;
    M.CONSTRAINTS.forEach(function (c) {
      const s = state.results[c.id].status;
      if (s === 'ok') ok++; else if (s === 'warn') warn++; else error++;
    });
    return { ok: ok, warn: warn, error: error };
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 编辑器
  // ════════════════════════════════════════════════════════════════════════════
  function buildScopeRanges() {
    // 按源码行号确定范围，避免补充代码后色带覆盖到无关计算。
    const scopeEnd = { 410: 1047, 411: 426, 434: 456, 438: 455, 615: 690, 728: 798, 764: 798 };
    const rows = [];
    SOURCE_ROWS.forEach(function (row, i) {
      if (row.entry.scope) {
        let end = i;
        const endLine = scopeEnd[row.entry.no] || row.entry.no;
        while (end + 1 < SOURCE_ROWS.length && (!SOURCE_ROWS[end + 1].entry.no || SOURCE_ROWS[end + 1].entry.no <= endLine)) end++;
        rows.push({ i: i, kind: row.entry.scope.kind, label: row.entry.scope.label, end: end });
      }
    });
    return rows;
  }

  let SOURCE_ROWS = [];
  let tileTimer = null;
  let tileStep = 0;
  let tilePlaying = false;

  function paintTileAnimation() {
    const host = $('[data-tile-animation]');
    if (!host) return;
    const count = Number(host.dataset.count), tk = Number(host.dataset.tk);
    host.querySelectorAll('[data-tile-chunk]').forEach(function (cell) {
      const i = Number(cell.dataset.tileChunk);
      cell.classList.toggle('is-current', i === tileStep);
      cell.classList.toggle('is-done', i < tileStep);
    });
    $('[data-tile-counter]', host).textContent = '第 ' + (tileStep + 1) + ' / ' + count + ' 块';
    $('[data-tile-location]', host).textContent = '当前 chunk：K [' + (tileStep * tk) + ', ' + ((tileStep + 1) * tk) + ')';
    $('[data-tile-acc]', host).textContent = tileStep + 1;
    $('[data-tile-progress]', host).style.width = ((tileStep + 1) / count * 100) + '%';
    $('[data-tile-description]', host).textContent = 'K = ' + (tileStep * tk) + '–' + ((tileStep + 1) * tk - 1)
      + '：A 列块 × B 行块 → ' + (tileStep === 0 ? 'matmul 初始化 C' : 'matmul_acc 加入 C');
    $('[data-tile-action="toggle"]', host).textContent = tilePlaying ? '暂停' : '播放';
  }

  function setTilePlaying(playing) {
    clearInterval(tileTimer);
    tileTimer = null;
    tilePlaying = playing;
    const host = $('[data-tile-animation]');
    if (playing && host) {
      tileTimer = setInterval(function () {
        if (!document.body.contains(host)) { setTilePlaying(false); return; }
        tileStep = (tileStep + 1) % Number(host.dataset.count);
        paintTileAnimation();
      }, 1600);
    }
    paintTileAnimation();
  }

  function conceptLines(id) {
    return SOURCE_ROWS.map(function (row) { return row.entry; }).filter(function (e) {
      return e.no && (e.focus === id || highlight(fill(e.t, state.d)).indexOf('data-concept="' + id + '"') !== -1);
    }).map(function (e) { return e.no; });
  }

  // 历史复现与当前算子的演示估算分开；不是当前设备的通用合法性判定。
  const ISSUE_HINTS = {
    capacity: { lines: [774, 790], label: '容量与分块', issue: 'pypto #630', url: 'https://github.com/hw-native-sys/pypto/issues/630' },
    capability: { lines: [651], label: '算子能力参考', issue: 'pypto #710', url: 'https://github.com/hw-native-sys/pypto/issues/710' },
    safety: { lines: [753, 781, 782], label: '有效形状', issue: 'pto-isa #97', url: 'https://github.com/hw-native-sys/pto-isa/issues/97' }
  };

  // 根据当前演示配置推导源码中可静态确定的值，不冒充运行时采样。
  function bufferLineValues(no, d) {
    const pad = d.Q_HEAD_PAD, seq = d.SEQ_TILE, head = d.HEAD_DIM;
    const values = {
      753: 'valid_len：运行时确定，≤ ' + seq,
      768: 'q_padded 逻辑块 [' + pad + ', ' + head + '] · BF16',
      770: 'cache_row：依赖 block_table 与运行时索引',
      773: 'K 逻辑块 [' + seq + ', ' + head + '] · BF16',
      774: 'QK 输出逻辑块 [' + pad + ', ' + seq + '] · FP32',
      781: '有效行 ' + d.Q_HEAD_BATCH + ' · 有效列 valid_len（运行时）',
      786: 'exp_scores 逻辑块 [' + pad + ', ' + seq + '] · BF16',
      789: 'V 逻辑块 [' + seq + ', ' + head + '] · BF16',
      790: 'PV 输出逻辑块 [' + pad + ', ' + head + '] · FP32'
    };
    return values[no] || '';
  }

  function rightBufferDiagram(d) {
    const seq = d.SEQ_TILE;
    const tile = state.bufferTile === 'full' ? seq : Math.min(Number(state.bufferTile), seq);
    const count = Math.ceil(seq / tile);
    const current = Math.min(state.bufferChunk, count - 1);
    const from = current * tile, to = Math.min(from + tile, seq);
    const pv = state.bufferStage === 'pv';
    const logicalKiB = seq * d.HEAD_DIM * 2 / 1024;
    const estimatedKiB = tile * d.HEAD_DIM * 2 * 2 / 1024;
    const fullKiB = seq * d.HEAD_DIM * 2 * 2 / 1024;
    const over = estimatedKiB > 64;
    const scale = Math.max(128, fullKiB, estimatedKiB);
    const pct = Math.min(100, estimatedKiB / scale * 100);
    const limit = 64 / scale * 100;
    const loop = pv
      ? 'output = 0  # FP32 累加\nfor s in range(0, SEQ_TILE, SEQ_CHUNK):\n    end = min(s + SEQ_CHUNK, SEQ_TILE)\n    output += P[:, s:end] @ V[s:end, :]'
      : 'for s in range(0, SEQ_TILE, SEQ_CHUNK):\n    end = min(s + SEQ_CHUNK, SEQ_TILE)\n    scores[:, s:end] = Q @ K[s:end, :]ᵀ';
    return '<div class="buffer-scene tile-case"><div class="buffer-scene-head"><b>容量与分块 · ' + (pv ? 'PV' : 'QK') + '</b><span>逻辑演示 · 910B 64 KiB 参考</span></div>'
      + '<div class="tile-case-controls"><label>SEQ_TILE <select data-buffer-seq>' + [64,128,256].map(function(n){return '<option value="' + n + '"' + (seq === n ? ' selected' : '') + '>' + n + '</option>';}).join('') + '</select></label>'
      + '<label>SEQ_CHUNK <select data-buffer-tile><option value="full"' + (state.bufferTile === 'full' ? ' selected' : '') + '>不分块 · ' + seq + '</option>' + [64,128,256].filter(function(n){return n < seq;}).map(function(n){return '<option value="' + n + '"' + (tile === n && state.bufferTile !== 'full' ? ' selected' : '') + '>' + n + '</option>';}).join('') + '</select></label>'
      + '<span class="tile-case-parameter">HEAD_DIM ' + d.HEAD_DIM + ' · BF16</span></div>'
      + '<div class="tile-case-stage" role="group" aria-label="选择算子"><button type="button" data-buffer-step="qk" aria-pressed="' + !pv + '">QK · L774</button><button type="button" data-buffer-step="pv" aria-pressed="' + pv + '">PV · L790</button><button type="button" data-goto="' + (pv ? 790 : 774) + '">定位源码 ↗</button></div>'
      + '<div class="tile-case-flow"><div class="tile-case-flow-head"><b>序列维 · ' + seq + '</b><span>' + count + ' 次循环 · 第 ' + (current + 1) + ' 次</span></div><div class="tile-chunks">' + Array.from({length:count},function(_,i){const l=i*tile,r=Math.min(l+tile,seq);return '<button type="button" data-tile-chunk="' + i + '" aria-pressed="' + (i === current) + '"><b>' + (i+1) + '</b><span>[' + l + ', ' + r + ')</span></button>';}).join('') + '</div><div class="tile-chunk-foot"><span>当前 ' + (pv ? 'V' : 'K') + ' [' + from + ', ' + to + ')</span><span>' + (pv ? '部分结果累加到输出' : '写入对应分数列') + '</span></div></div>'
      + '<div class="buffer-step-nav"><button type="button" data-tile-move="prev"' + (current===0 ? ' disabled' : '') + '>← 上一块</button><button class="buffer-play" type="button" data-buffer-play aria-pressed="' + state.bufferPlaying + '">' + (state.bufferPlaying ? 'Ⅱ 暂停' : current===count-1 && count>1 ? '↻ 重播' : '▶ 播放') + '</button><button type="button" data-tile-move="next"' + (current>=count-1 ? ' disabled' : '') + '>下一块 →</button></div>'
      + '<div class="tile-hardware"><div class="tile-hardware-head"><b>Right buffer · 双缓冲估算</b><span class="' + (over ? 'is-over' : 'is-fit') + '">' + estimatedKiB + ' / 64 KiB · ' + (over ? '超出 ' + (estimatedKiB - 64) + ' KiB' : estimatedKiB === 64 ? '达到限额' : '未超限') + '</span></div><div class="tile-buffer-pair"><div class="' + (current % 2 === 0 ? 'is-current' : '') + '"><b>Buffer 0</b><span>' + (estimatedKiB / 2) + ' KiB</span></div><div class="' + (current % 2 === 1 ? 'is-current' : '') + '"><b>Buffer 1</b><span>' + (estimatedKiB / 2) + ' KiB</span></div><small>当前块 → Buffer ' + (current % 2) + ' · 交替示意</small></div><div class="tile-capacity-track"><i class="tile-capacity-limit" style="left:' + limit + '%"></i><div class="tile-capacity-fill' + (over ? ' is-over' : '') + '" style="width:' + pct + '%"></div></div><div class="tile-capacity-scale"><span>0</span><span style="left:' + limit + '%">64 KiB</span><span>' + scale + ' KiB</span></div><div class="tile-formula">' + tile + ' × ' + d.HEAD_DIM + ' × 2 B × 2 = <b>' + estimatedKiB + ' KiB</b> <span>（两个完整块同时驻留的假设）</span></div>'
      + '<div class="tile-case-compare"><span>完整逻辑块：' + logicalKiB + ' KiB</span><span>不分块估算：' + fullKiB + ' KiB</span><span>当前分块估算：' + estimatedKiB + ' KiB</span></div></div>'
      + '<div class="tile-loop"><div class="tile-loop-head"><b>分块循环 · 伪代码</b><span>' + (pv ? 'PV 按序列维累加' : 'QK 按序列维拼接') + '</span></div><pre>' + esc(loop) + '</pre><small>QK 各段拼接并遮罩后统一 Softmax，再按段执行 PV。示意未改写源码。</small></div>'
      + '<div class="tile-disclaimer">示例估算，不代表编译分配或 950B 规格 · <a href="https://github.com/hw-native-sys/pypto/issues/630" target="_blank" rel="noopener noreferrer" title="DeepSeek 案例，PyPTO ca3b940 / pypto-lib 504243a；未注明设备">分块案例 #630 ↗</a> · <a href="https://github.com/hw-native-sys/pypto/issues/1595" target="_blank" rel="noopener noreferrer" title="Ascend 910B，PyPTO 567eb66">910B 限额依据 #1595 ↗</a></div></div>';
  }

  function validShapeDiagram(d) {
    const length = state.validExampleLength, tile = d.SEQ_TILE;
    const blocks = Math.ceil(length / tile);
    const injected = state.validExampleBlock === 'overflow';
    const selected = injected ? blocks + 1 : blocks ? (state.validExampleBlock === 'tail' ? blocks - 1 : Math.min(Number(state.validExampleBlock), blocks - 1)) : null;
    const start = selected === null ? null : selected * tile;
    const valid = start === null ? null : Math.min(tile, length - start);
    const bad = valid !== null && valid < 0;
    const validColumns = valid === null || bad ? 0 : valid;
    const colPercent = validColumns / tile * 100;
    const rowPercent = d.Q_HEAD_BATCH / d.Q_HEAD_PAD * 100;
    const blockOptions = Array.from({ length: blocks }, function (_, i) {
      return '<option value="' + i + '"' + (!injected && selected === i ? ' selected' : '') + '>块 ' + i + (i === blocks - 1 ? ' · 尾块' : '') + '</option>';
    }).join('');
    return '<div class="valid-case"><div class="buffer-scene-head"><b>有效形状 · 尾块</b><span>运行时输入示例</span></div>'
      + '<div class="valid-controls"><label>上下文长度 <input type="number" min="0" max="4096" step="1" value="' + length + '" data-valid-length></label>'
      + '<label>SEQ_TILE <select data-buffer-seq>' + [64, 128, 256].map(function (n) { return '<option value="' + n + '"' + (tile === n ? ' selected' : '') + '>' + n + '</option>'; }).join('') + '</select></label>'
      + '<label>工作表块 <select data-valid-block>' + (blocks ? blockOptions : '<option value="tail"' + (!injected ? ' selected' : '') + '>无有效块</option>') + '<option value="overflow"' + (injected ? ' selected' : '') + '>模拟越界块</option></select></label></div><small class="valid-input-error" data-valid-error aria-live="polite"></small>'
      + '<div class="valid-source-links"><button type="button" data-goto="753">valid_len · L753 ↗</button><button type="button" data-goto="781">set_validshape · L781 ↗</button><button type="button" data-goto="782">fillpad · L782 ↗</button></div>'
      + '<div class="valid-flow"><div><small>上下文</small><b>' + length + '</b></div><span>→</span><div><small>块起点 s0</small><b>' + (start === null ? '—' : start) + '</b></div><span>→</span><div class="' + (bad ? 'is-invalid' : '') + '"><small>valid_len</small><b>' + (valid === null ? '—' : valid) + '</b></div><span>→</span><div><small>物理 tile</small><b>' + d.Q_HEAD_PAD + ' × ' + tile + '</b></div></div>'
      + '<div class="valid-shape-panel"><div class="valid-shape-head"><b>scores · 物理形状 [' + d.Q_HEAD_PAD + ', ' + tile + ']</b><span>' + (valid === null ? '无块执行' : bad ? '无效 valid_len' : '有效 [' + d.Q_HEAD_BATCH + ', ' + valid + ']') + '</span></div>'
      + '<div class="valid-shape-grid' + (bad ? ' is-invalid' : '') + '" role="img" aria-label="' + (valid === null ? '无有效工作表块' : bad ? '模拟越界导致有效列数为负' : '物理 ' + d.Q_HEAD_PAD + ' 行 ' + tile + ' 列，有效 ' + d.Q_HEAD_BATCH + ' 行 ' + valid + ' 列') + '">' + (valid !== null && !bad ? '<div class="valid-shape-region" style="width:' + colPercent + '%;height:' + rowPercent + '%"><span>有效区域</span></div>' : '<div class="valid-shape-empty">' + (bad ? 'valid_len < 0' : '无块执行') + '</div>') + '</div>'
      + '<div class="valid-legend"><span><i></i> 有效区域</span><span><i></i> 填充区域 · PadValue.min</span></div></div>'
      + '<div class="valid-formula"><code>' + (valid === null ? '工作表无有效块' : 'min(' + tile + ', ' + length + ' − ' + start + ') = ' + valid) + '</code><b class="' + (bad ? 'is-invalid' : '') + '">' + (valid === null ? '不执行 set_validshape' : bad ? '违反 valid_len ≥ 0' : '0 ≤ valid_len ≤ ' + tile) + '</b></div>'
      + '<div class="valid-case-note">' + (injected ? '模拟异常工作表项；并非当前源码已验证的运行时故障。' : 'decode_layer 的工作表仅包含真实块；上述输入为演示值。') + ' <a href="https://github.com/hw-native-sys/pto-isa/issues/97" target="_blank" rel="noopener noreferrer">有效维度约束 #97 ↗</a></div></div>';
  }

  // ── Vec(UB) 预算 ─────────────────────────────────────────────────────────
  // 抽象自 pto-isa#170：名义 192 KB 的 Vec UB 顶部约 8 KB 被运行时保留，
  // 分配器按名义容量发预算，所以越过可用上限的配置在编码期之后就没人拦了。
  // 这里把地址空间画出来，配合旋钮上的就地预警，让「为什么这个值不行」一眼可见。
  function kbs(n) { return (n / 1024).toFixed(n % 1024 ? 1 : 0) + ' KB'; }

  // Vec 竞技场的地址布局：carry 在低地址，各级流水依次往高地址排
  function ubLayout(d) {
    const out = [];
    let at = 0;
    d.vecBuffers.forEach(function (b) {
      const from = at;
      at += b.size;
      out.push({
        name: b.name, kind: b.kind, stage: b.stage, size: b.size, from: from, to: at,
        crosses: at > d.UB_USABLE && from < d.UB_USABLE,
        inReserved: from >= d.UB_USABLE
      });
    });
    return out;
  }

  // 高水位那一级流水内部的中间量，逐个标出地址区间
  function ubPeakTiles(d, base) {
    const items = [
      { name: 'scores_scaled', line: 775, size: d.VEC_SCORE_B, dtype: 'FP32' },
      { name: 'scores_valid', line: 781, size: d.VEC_SCORE_B, dtype: 'FP32' },
      { name: 'scores', line: 782, size: d.VEC_SCORE_B, dtype: 'FP32' },
      { name: 'cur_mi', line: 783, size: d.VEC_ROW_B, dtype: 'FP32' },
      { name: 'sub_tmp', line: 784, size: d.VEC_SCORE_B, dtype: 'FP32' },
      { name: 'exp_scores', line: 784, size: d.VEC_SCORE_B, dtype: 'FP32' },
      { name: 'cur_li', line: 785, size: d.VEC_ROW_B, dtype: 'FP32' },
      { name: 'exp_scores_bf16', line: 786, size: d.VEC_SCORE_BF16_B, dtype: 'BF16' }
    ];
    let at = base;
    return items.map(function (it) {
      const from = at;
      at += it.size;
      return {
        name: it.name, line: it.line, dtype: it.dtype, size: it.size, from: from, to: at,
        crosses: at > d.UB_USABLE && from < d.UB_USABLE,
        inReserved: from >= d.UB_USABLE
      };
    });
  }

  // 原因可视化：把 Vec UB 的地址空间画成一条，上面标出当前高水位、可用上限与顶部保留带。
  // 只有一张图 —— 旋钮上的预警负责「哪个值不行」，这张图负责「为什么」。
  function ubBudgetBar(d) {
    const layout = ubLayout(d);
    const peak = layout[layout.length - 1];
    const tiles = ubPeakTiles(d, peak.from);
    const hit = tiles.filter(function (t) { return t.crosses || t.inReserved; })[0];
    const scale = Math.max(d.UB_NOMINAL, d.VEC_HW);
    const pct = function (n) { return (n / scale * 100).toFixed(2); };
    const segs = layout.map(function (b) {
      const cls = b.inReserved ? 'is-reserved' : b.crosses ? 'is-crosses' : (b.kind === 'carry' ? 'is-carry' : 'is-stage');
      return '<div class="ub-seg ' + cls + '" style="left:' + pct(b.from) + '%;width:' + pct(b.size) + '%" '
        + 'title="' + esc(b.name + ' · ' + kbs(b.size)) + '">'
        + '<span>' + (b.kind === 'carry' ? 'carry' : 's' + b.stage) + '</span></div>';
    }).join('');
    return '<div class="ub-map">'
      + '<div class="ub-map-head"><b>Vec UB 地址空间</b>'
      + '<span>0 → ' + kbs(scale) + (scale > d.UB_NOMINAL ? '（已超名义容量）' : ' 名义容量') + '</span></div>'
      + '<div class="ub-track">'
      + '<div class="ub-reserved-band" style="left:' + pct(d.UB_USABLE) + '%;width:' + pct(d.UB_RESERVED) + '%">'
      + '<span>保留 ' + kbs(d.UB_RESERVED) + '</span></div>'
      + segs
      + '<i class="ub-limit ub-limit--usable" style="left:' + pct(d.UB_USABLE) + '%"></i>'
      + '<i class="ub-limit ub-limit--nominal" style="left:' + pct(d.UB_NOMINAL) + '%"></i>'
      + '</div>'
      + '<div class="ub-scale"><span>0</span>'
      + '<span style="left:' + pct(d.UB_USABLE) + '%">' + kbs(d.UB_USABLE) + ' 可用上限</span></div>'
      + '<div class="ub-legend"><span><i class="is-carry"></i>跨块 carry</span>'
      + '<span><i class="is-stage"></i>流水各级</span>'
      + (hit ? '<span><i class="is-crosses"></i>越过上限的那块</span>' : '')
      + '<span><i class="is-reserved"></i>顶部保留 ' + kbs(d.UB_RESERVED) + '</span>'
      + '<span><i class="is-line-nominal"></i>名义上限 ' + kbs(d.UB_NOMINAL) + '</span></div>'
      + '<div class="ub-math">'
      + '<div><span>当前高水位</span><code>' + kbs(d.VEC_CARRY) + ' carry + '
      + d.FA_STAGE_EFF + ' × ' + kbs(d.VEC_PER_STAGE) + ' = ' + kbs(d.VEC_HW) + '</code></div>'
      + (d.FA_STAGE_CLAMPED ? '<div><span>实际驻留级数</span><code>min(stage ' + d.FA_STAGE
        + ', GP_SIZE ' + d.GP_SIZE + ') = ' + d.FA_STAGE_EFF + '</code></div>' : '')
      + '<div><span>分配器的预算</span><code>' + kbs(d.UB_NOMINAL) + '（名义容量）</code></div>'
      + '<div><span>真正能用的</span><code>' + kbs(d.UB_NOMINAL) + ' − ' + kbs(d.UB_RESERVED)
      + ' = ' + kbs(d.UB_USABLE) + '</code></div></div>'
      + '<p class="ub-why">这两个数不一样，就是全部原因。分配器拿名义 ' + kbs(d.UB_NOMINAL)
      + ' 做判断，高水位不超它就一路放行；而顶部那 ' + kbs(d.UB_RESERVED)
      + ' 已经被运行时占着（<code>ReserveBufferOp</code> / pipe init），落进去的 tile 读写的是别人的数据，'
      + '地址本身合法，越界中断也不会触发。'
      + (hit
        ? '当前越过上限的是 <code>' + esc(hit.name) + '</code>（这一级最后分配的那块），越出 '
          + d.VEC_OVER_USABLE_FMT + '。'
        : '当前距这条边界还有 ' + kbs(d.UB_USABLE - d.VEC_HW) + '。')
      + '</p>'
      + '<div class="ub-foot">'
      + '<button type="button" class="lineref" data-goto="764">定位 stage · L764</button>'
      + '<button type="button" class="lineref" data-goto="' + (hit ? hit.line : 786) + '">定位 '
      + esc(hit ? hit.name : 'exp_scores_bf16') + ' · L' + (hit ? hit.line : 786) + '</button>'
      + '<a href="https://github.com/hw-native-sys/pto-isa/issues/170" target="_blank" rel="noopener noreferrer">依据：pto-isa #170 ↗</a>'
      + '</div>'
      + '<p class="ub-note">保留区大小随平台与 CANN 版本变化，需按目标配置核对；'
      + '各中间量是否复用、是否就地，按编译产物核对 —— 此处按「每块各占一份、按出现顺序排布」估算。</p>'
      + '</div>';
  }

  function issueHintHtml(id) {
    const issue = ISSUE_HINTS[id];
    let body = '';
    if (id === 'capacity') {
      body = rightBufferDiagram(state.d);
    } else if (id === 'capability') {
      body = '<p>当前行为张量切片；其最终 lowering 和目标内存空间尚未确定，因此<strong>不能据此判定本行违规</strong>。</p>'
        + '<p><b>Issue 复现：</b>A5 上 <code>pl.slice</code> 被 lowering 为 <code>pto.textract</code> 后，Vec → Vec 组合被 PTOAS v0.15 拒绝；Issue 将其作为后端处理问题提出。</p>'
        + '<pre>q_lo = pl.slice(q_row, [1, HEAD_DIM_CFG // 2], [0, 0])\nq_hi = pl.slice(q_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])</pre>'
        + '<p class="issue-meta">设备：A5；版本：PyPTO <code>8b5d886</code>、pypto-lib <code>fc2b2aa</code>、PTOAS <code>v0.15</code>。支持情况会随 lowering 和版本变化。</p>';
    } else body = validShapeDiagram(state.d);
    const scene = id === 'capacity';
    const compact = scene || id === 'safety';
    const heads = { capacity: 'Right buffer · 源码场景' };
    return '<section class="issue-hint' + (scene ? ' issue-hint--scene' : '') + '" aria-label="' + issue.label + '">'
      + '<div class="issue-hint-head"><span>' + (heads[id] || issue.label + ' · 需结合目标验证') + '</span>'
      + (compact ? '<a class="issue-source-compact" href="' + issue.url + '" target="_blank" rel="noopener noreferrer">' + issue.issue + ' ↗</a>' : '')
      + '<button type="button" data-issue-close aria-label="关闭 Issue 参考">×</button></div>' + body
      + (compact ? '' : '<a href="' + issue.url + '" target="_blank" rel="noopener noreferrer">查看依据：' + issue.issue + ' ↗</a>') + '</section>';
  }

  function inlineConceptHtml(id) {
    const c = V.CONCEPTS[id];
    if (!c) return '';
    const lines = conceptLines(id);
    return '<section class="inline-concept inline-' + LENS_OF[c.lens] + '" aria-label="' + esc(c.title) + '详情">'
      + '<div class="inline-concept-head"><div><span class="inline-kind"><i></i>' + esc(c.kind) + '</span>'
      + '<h3>' + esc(c.title) + '</h3><p>' + esc(c.sub) + '</p></div><button class="inline-close" type="button" data-close-concept aria-label="关闭对象详情">×</button></div>'
      + '<div class="inline-content"><div class="inline-viz"><div class="inline-viz-label"><span>语义视图</span><span>示意图 · 随配置更新</span></div>' + c.viz(state.d) + '</div>'
      + '<dl class="insp-facts">' + c.facts(state.d).map(function (f) {
        return '<div><dt>' + esc(f[0]) + '</dt><dd>' + esc(f[1]) + '</dd></div>';
      }).join('') + '</dl>' + (['tile:q-acc', 'scope:deps', 'scope:atomic', 'tile:v-tile', 'parallel:handoff', 'parallel:pipeline', 'precision:cast'].indexOf(id) >= 0 && c.note ? '<p class="inline-note">' + esc(c.note) + '</p>' : '') + '</div>'
      + (lines.length > 1 ? '<div class="inline-locations"><span>代码落点</span>' + lines.map(function (no) {
        return '<button type="button" class="lineref" data-anchor="' + no + '"' + (no === state.anchor ? ' aria-current="location"' : '') + '>L' + no + '</button>';
      }).join('') + '</div>' : '') + '</section>';
  }

  function renderEditor() {
    const d = state.d;
    const resolved = state.inlineConfig ? {
      BATCH: d.BATCH, NUM_KV_HEADS: d.NUM_KV_HEADS, HEAD_DIM: d.HEAD_DIM,
      SEQ_TILE: d.SEQ_TILE, BLOCK_SIZE: d.SEQ_TILE, QKV_OK: d.QKV_OK,
      DOWN_TN: d.DOWN_TN, GP_SIZE: d.GP_SIZE, NUM_CORES: d.NUM_CORES,
      ROPE_CORES: d.ROPE_CORES, HIDDEN: d.HIDDEN, KV_HIDDEN: d.KV_HIDDEN,
      MAX_CTX_BLOCKS: d.MAX_CTX_BLOCKS, Q_HEAD_PAD: d.Q_HEAD_PAD,
      Q_HEAD_BATCH: d.Q_HEAD_BATCH, TK: d.TK, TN: d.TN
    } : null;
    const issues = lineIssues();
    const ranges = buildScopeRanges();
    const maxDepth = 4;
    let html = '';

    SOURCE_ROWS.forEach(function (row, i) {
      const e = row.entry;

      if (e.sec) {
        html += '<div class="ed-sec"><span class="ed-sec-label">' + esc(e.sec) + '</span>'
          + '<span class="ed-sec-hint">' + esc(e.hint || '') + '</span></div>';
        return;
      }
      if (e.fold) {
        html += '<div class="ed-fold"><span class="ed-fold-no">⋯</span>'
          + '<span class="ed-fold-body">' + e.fold + ' 行已折叠</span></div>';
        return;
      }

      // ribbons
      let rb = '';
      let depth = 0;
      ranges.forEach(function (rg) {
        if (i >= rg.i && i <= rg.end) {
          rb += '<i class="rb rb-' + rg.kind + (i === rg.i ? ' is-start' : '')
            + (i === rg.end ? ' is-end' : '') + '" style="--rb-depth:' + depth + '"></i>';
          depth++;
        }
      });
      for (let k = depth; k < maxDepth; k++) rb += '<i class="rb rb-void"></i>';

      // inlays
      const inlays = (e.inlay || []).slice();
      if (e.scope) inlays.unshift({ lens: 'scope', text: e.scope.label, strong: true });
      let chips = '';
      inlays.forEach(function (c) {
        const group = LENS_OF[c.lens] || c.lens;
        chips += '<span class="chip chip-' + c.lens + (c.strong ? ' is-strong' : '')
          + '" data-lens="' + group + '">' + esc(fill(c.text, d)) + '</span>';
      });
      if (state.issueHint === 'capacity') {
        const values = bufferLineValues(e.no, d);
        if (values) chips += '<span class="buffer-code-value" title="由当前演示配置推导，不是运行时采样">' + esc(values) + '</span>';
        if (e.no === 774 || e.no === 790) chips += '<span class="capacity-line-badge ' + (e.no === 774 ? 'capacity-code-qk' : 'capacity-code-pv') + '">' + (e.no === 774 ? 'QK · Right' : 'PV · Right') + '</span>';
      }
      Object.keys(ISSUE_HINTS).forEach(function (id) {
        if (ISSUE_HINTS[id].lines.indexOf(e.no) >= 0) chips += '<button class="issue-chip" type="button" data-issue-open="' + id + '" data-issue-at="' + e.no + '" aria-expanded="' + (state.issueHint === id && state.issueLine === e.no) + '">' + ISSUE_HINTS[id].label + '</button>';
      });

      const iss = issues[e.no];
      const cls = ['ed-line'];
      if (iss && iss.status !== 'ok') cls.push('has-' + iss.status);
      if (e.focus && state.sel.type === 'concept' && state.sel.id === e.focus) cls.push('is-linked');
      if (iss && state.sel.type === 'constraint' && state.sel.id === iss.cid) cls.push('is-flagged');
      if (state.issueHint === 'capacity' && (e.no === 774 || e.no === 790)) cls.push(e.no === 774 ? 'is-capacity-qk' : 'is-capacity-pv');
      if (state.issueHint === 'safety' && [749, 752, 753, 781, 782].indexOf(e.no) >= 0) cls.push('is-valid-shape-linked');
      if (state.issueHint === 'capacity' && ((state.bufferStep === 'load-k' && e.no === 773)
        || (state.bufferStep === 'load-v' && e.no === 789)
        || (state.bufferStep === 'qk' && e.no === 774)
        || (state.bufferStage === 'softmax' && e.no >= 783 && e.no <= 786)
        || (state.bufferStep === 'pv' && e.no === 790))) cls.push('is-buffer-active');

      html += '<div class="' + cls.join(' ') + '" data-no="' + e.no + '"'
        + (e.focus ? ' data-focus="' + e.focus + '"' : '') + '>'
        + '<span class="ed-no">' + e.no + '</span>'
        + '<span class="ed-mark">'
        + (iss && iss.status !== 'ok'
          ? '<button class="mark mark-' + iss.status + '" type="button" data-constraint="' + iss.cid
            + '" title="' + esc(iss.title) + '" aria-label="' + esc(iss.title) + '">'
            + (iss.status === 'error' ? '✕' : '!') + '</button>'
          : '')
        + '</span>'
        + '<span class="ed-rb">' + rb + '</span>'
        + '<code class="ed-code">' + highlight(fill(e.t, d), resolved) + '</code>'
        + '<span class="ed-inlay">' + chips + '</span>'
        + '</div>';
      if (state.issueHint && state.issueLine === e.no) html += issueHintHtml(state.issueHint);
      if (state.sel.type === 'concept' && e.no === state.anchor) html += inlineConceptHtml(state.sel.id);
    });

    $('#editorBody').innerHTML = html;
    document.querySelectorAll('.issue-navigation [data-issue-open]').forEach(function (button) {
      button.setAttribute('aria-expanded', String(button.dataset.issueOpen === state.issueHint));
    });
    const configToggle = $('[data-config-inline]');
    if (configToggle) configToggle.setAttribute('aria-pressed', String(state.inlineConfig));
    tileStep = 0;
    setTilePlaying(!!$('[data-tile-animation]') && !window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 左栏
  // ════════════════════════════════════════════════════════════════════════════
  function renderLensPanel() {
    let html = '';
    LENSES.forEach(function (L) {
      const on = state.lens[L.id];
      html += '<button class="lens-row' + (on ? ' is-on' : '') + '" type="button" data-lens-toggle="' + L.id + '"'
        + ' aria-pressed="' + on + '" title="' + esc(L.desc) + ' · 点击' + (on ? '隐藏' : '显示') + '代码标注">'
        + '<i class="lens-swatch lens-' + L.id + '"></i>'
        + '<span class="lens-text"><b>' + esc(L.label) + '</b></span>'
        + '</button>';
    });
    $('#lensPanel').innerHTML = html;
  }

  function renderCatalog() {
    const groups = [
      { label: '片上存储与布局', desc: '容量 · 分块 · 有效区域', ids: ['tile:q-acc', 'tile:k-tile', 'layout:q-pad', 'tile:v-tile', 'tile:x-chunk'] },
      { label: '计算单元与性能', desc: '交接 · 流水 · 精度 · 并行', ids: ['parallel:handoff', 'parallel:pipeline', 'precision:cast', 'scope:atomic', 'parallel:spmd', 'parallel:parallel', 'parallel:grid-stride'] },
      { label: '任务与运行时', desc: '任务边界 · 依赖 · 分页契约', ids: ['scope:manual', 'scope:at', 'scope:deps', 'layout:paged-kv', 'layout:split-k'] },
      { label: '模型数据', desc: '张量与层间接口', ids: ['tensor:hidden-states', 'tensor:normed', 'tensor:q-proj'] }
    ];
    let html = '';
    groups.forEach(function (group) {
      html += '<div class="cat-group"><span class="cat-head">' + esc(group.label) + '</span>'
        + '<span class="cat-description">' + esc(group.desc) + '</span>';
      group.ids.forEach(function (id) {
        const c = V.CONCEPTS[id];
        if (!c) return;
        const on = state.sel.type === 'concept' && state.sel.id === id;
        html += '<button class="cat-row' + (on ? ' is-active' : '') + '" type="button" data-concept="' + id + '">'
          + '<i class="lens-swatch lens-' + LENS_OF[c.lens] + '"></i><b>' + esc(c.title) + '</b></button>';
      });
      html += '</div>';
    });
    $('#catalog').innerHTML = html;
    $('#catalogCount').textContent = Object.keys(V.CONCEPTS).length + ' OBJECTS';
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 底部：场景与算子配置
  // ════════════════════════════════════════════════════════════════════════════
  // ── 旋钮就地预警 ─────────────────────────────────────────────────────────
  // 每条约束声明了自己依赖哪几个旋钮；反过来就知道「当前这个值让哪条约束不成立」。
  // 这是约束前移里最靠前的那一步：不等编译，设值的瞬间就在输入框下面说清楚。
  const RANK = { ok: 0, warn: 1, error: 2 };

  function knobAlerts(id) {
    return M.CONSTRAINTS
      .filter(function (c) {
        return c.knobs && c.knobs.indexOf(id) >= 0 && state.results[c.id].status !== 'ok';
      })
      .sort(function (a, b) {
        return RANK[state.results[b.id].status] - RANK[state.results[a.id].status];
      });
  }

  function knobAlertHtml(alerts, knobId) {
    return '<div class="knob-alerts" id="alert-' + knobId + '" aria-live="polite">' + alerts.map(function (c) {
      const res = state.results[c.id];
      return '<button type="button" class="knob-alert is-' + res.status + '" data-constraint="' + c.id + '">'
        + '<i class="dot dot-' + res.status + '"></i>'
        + '<span class="knob-alert-main"><b>' + esc(STATUS_LABEL[res.status]) + ' · ' + esc(c.title) + '</b>'
        + '<small>' + esc(res.detail) + '</small></span>'
        + '<em>看原因 ↗</em></button>';
    }).join('') + '</div>';
  }

  function renderKnobs() {
    let html = '';
    const groups = [
      { title: '工作负载', ids: ['BATCH'] },
      { title: '模型维度', note: '通常由模型决定；此处可调用于演示。', ids: ['NUM_KV_HEADS', 'HEAD_DIM'] },
      { title: '运行时', ids: ['PAGE_SIZE'] },
      { title: '算子策略', ids: ['SEQ_TILE', 'FA_STAGE', 'QKV_OK', 'DOWN_TN', 'GP_SIZE', 'NUM_CORES', 'ROPE_CORES'] }
    ];
    groups.forEach(function (group) {
      html += '<section class="knob-group" aria-label="' + esc(group.title) + '"><div class="knob-group-title">'
        + esc(group.title) + (group.note ? '<span>' + esc(group.note) + '</span>' : '') + '</div><div class="knob-group-fields">';
      group.ids.forEach(function (id) {
        const k = M.KNOBS.find(function (item) { return item.id === id; });
        const alerts = knobAlerts(k.id);
        // 未满足的在输入框下面展开成一条说明；需关注的只在标签旁留一个可点的小标记，
        // 否则默认配置里那两条「需关注」的估算会一直占着位置，真正的预警反而被淹掉
        const errors = alerts.filter(function (c) { return state.results[c.id].status === 'error'; });
        const warns = alerts.filter(function (c) { return state.results[c.id].status === 'warn'; });
        const worst = errors.length ? 'error' : (warns.length ? 'warn' : null);
        html += '<div class="knob' + (worst ? ' has-' + worst : '') + '">'
          + '<div class="knob-label"><label for="knob-' + k.id + '">' + esc(k.label) + '</label>'
          + (errors.length ? '<i class="knob-flag is-error" aria-hidden="true">!</i>' : '')
          + warns.map(function (c) {
            const label = STATUS_LABEL.warn;
            return '<button type="button" class="knob-flag is-warn" data-constraint="' + c.id + '"'
              + ' title="' + esc(label + ' · ' + c.title + '：' + state.results[c.id].detail) + '"'
              + ' aria-label="' + esc(label + ' · ' + c.title) + '">△</button>';
          }).join('')
          + '<span class="knob-help"><button type="button" class="knob-help-button" aria-label="' + esc(k.label) + '说明" aria-describedby="help-' + k.id + '">?</button>'
          + '<span class="knob-tooltip" role="tooltip" id="help-' + k.id + '">' + esc(k.note) + '</span></span></div>'
          + '<select id="knob-' + k.id + '" data-knob="' + k.id + '"'
          + (errors.length ? ' aria-invalid="true" aria-describedby="alert-' + k.id + '"' : '') + '>'
          + k.options.map(function (o) {
            return '<option value="' + o + '"' + (state.cfg[k.id] === o ? ' selected' : '') + '>' + o + '</option>';
          }).join('')
          + '</select>'
          + (errors.length ? knobAlertHtml(errors, k.id) : '')
          + '</div>';
      });
      html += '</div></section>';
    });
    $('#knobGrid').innerHTML = html;
  }

  // 面板头的判定计数 + 底部状态栏
  function renderStatus() {
    const sum = statusSummary();
    $('#problemCount').innerHTML =
      '<i class="dot dot-error"></i>' + sum.error
      + '<i class="dot dot-warn"></i>' + sum.warn
      + '<i class="dot dot-ok"></i>' + sum.ok;

    const st = sum.error ? 'error' : (sum.warn ? 'warn' : 'ok');
    const strip = $('#statusMain');
    strip.className = 'status-main is-' + st;
    strip.textContent = sum.error
      ? sum.error + ' 项示例规则未满足'
      : (sum.warn ? sum.warn + ' 项需要关注' : '全部 ' + sum.ok + ' 项示例检查通过');
  }

  // 当前选中对象携带的约束；没有则返回 null
  function relatedIds() {
    if (state.sel.type !== 'concept') return null;
    const ids = M.CONSTRAINTS
      .filter(function (k) { return CONSTRAINT_CONCEPT[k.id] === state.sel.id; })
      .map(function (k) { return k.id; });
    return ids.length ? ids : null;
  }

  // 右栏下段：约束区（始终列全 9 条，与当前对象相关的那几条高亮，其余压暗）
  function consHtml() {
    const sum = statusSummary();
    const total = M.CONSTRAINTS.length;
    const rel = relatedIds();
    const order = { error: 0, warn: 1, ok: 2 };
    const list = M.CONSTRAINTS.slice().sort(function (a, b) {
      return order[state.results[a.id].status] - order[state.results[b.id].status];
    });

    let html = '<div class="hn-sec-label">' + (rel ? '相关检查' : '约束与估算')
      + '<small>' + (rel ? rel.length + ' / ' + total : total + ' 条') + '</small></div>';
    html += '<p class="hn-cons-note">包含硬件布局模型、性能估算与当前算子／运行时契约；结果需按目标与编译产物复核。</p>';

    html += '<div class="hn-filters" role="group" aria-label="筛选约束与估算">'
      + [['all', '全部', total], ['error', '未满足', sum.error], ['warn', '需关注', sum.warn], ['ok', '未触发', sum.ok]]
        .map(function (it) {
          return '<button type="button" data-filter="' + it[0] + '" aria-pressed="'
            + (state.filter === it[0]) + '">' + it[1] + ' <span>' + it[2] + '</span></button>';
        }).join('')
      + '</div>';

    let rows = '';
    list.filter(function (c) {
      return state.filter === 'all' || state.results[c.id].status === state.filter;
    }).forEach(function (c) {
      const res = state.results[c.id];
      const on = state.sel.type === 'constraint' && state.sel.id === c.id;
      const isRel = !!rel && rel.indexOf(c.id) >= 0;
      const cls = ['prob'];
      if (on) cls.push('is-active');
      if (isRel) cls.push('is-related');
      if (rel && !isRel) cls.push('is-faded');
      rows += '<button class="' + cls.join(' ') + '" type="button" data-constraint="' + c.id + '">'
        + '<i class="dot dot-' + res.status + '"></i>'
        + '<span class="prob-main"><b>' + esc(c.title)
        + (isRel ? '<em class="prob-rel">相关</em>' : '') + '</b>'
        + '<small><span class="hn-kind">' + esc(CONSTRAINT_KIND[c.id]) + '</span> ' + esc(res.detail) + '</small></span>'
        + '<span class="prob-side"><code>' + esc(res.value) + '</code>'
        + '<em class="tag tag-' + res.status + '">' + STATUS_LABEL[res.status] + '</em></span>'
        + '</button>';
    });
    html += '<div class="hn-problem-list">'
      + (rows || '<p class="hn-empty">当前没有此状态的约束。</p>') + '</div>';
    return html;
  }

  function renderCons() {
    const host = $('#consSection');
    if (host) host.innerHTML = consHtml();
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 右栏：一列到底 —— 当前对象在上，约束区在下
  // ════════════════════════════════════════════════════════════════════════════
  function metric(label, value, note) {
    return '<div class="metric"><span>' + esc(label) + '</span><b>' + esc(value) + '</b>'
      + '<small>' + esc(note) + '</small></div>';
  }

  function overviewHtml() {
    const d = state.d;
    let html = '';
    html += '<div class="insp-metrics">'
      + metric('真实块 / 核', d.REAL_BLOCKS + ' / ' + d.NUM_CORES, d.SKEW_FMT + '× 偏差')
      + metric('Q 填充占比', d.PAD_WASTE_PCT + '%', d.Q_HEAD_BATCH + ' 行有效 / ' + d.Q_HEAD_PAD + ' 行物理')
      + metric('K tile 双缓冲估算', d.K_TILE_DB_KB + ' KB', '假设完整块双缓冲 / 示例容量 64 KiB')
      + metric('q_proj task', String(d.Q_TASK_TOTAL), d.Q_ON + ' N-tile × ' + d.QKV_OK + ' split-K')
      + '</div>';

    return html;
  }

  function constraintHtml(id) {
    const c = M.CONSTRAINTS.filter(function (k) { return k.id === id; })[0];
    if (!c) return null;
    const res = state.results[id];
    const d = state.d;
    const conceptId = CONSTRAINT_CONCEPT[id];
    const concept = conceptId ? V.CONCEPTS[conceptId] : null;
    let html = '';

    if (id === 'l0b-capacity') return capacityDecisionHtml(d);

    html += '<div class="insp-tag"><i class="dot dot-' + res.status + '"></i>'
      + STATUS_LABEL[res.status] + ' · ' + esc(CONSTRAINT_KIND[c.id]) + '</div>';
    html += '<h3 class="insp-title">' + esc(c.title) + '</h3>';
    html += '<pre class="insp-rule">' + esc(c.rule) + '</pre>';
    html += '<div class="insp-verdict is-' + res.status + '"><b>' + esc(res.value) + '</b>'
      + '<span>' + esc(res.detail) + '</span></div>';
    // 专属原因可视化优先；没有的才回落到所属语义对象的通用示意图
    if (c.viz === 'ub-budget') html += '<div class="insp-viz">' + ubBudgetBar(d) + '</div>';
    else if (concept) html += '<div class="insp-viz">' + concept.viz(d) + '</div>';
    if (concept) {
      html += '<p class="insp-sub insp-sub--link">相关对象：'
        + '<button class="inlink" type="button" data-concept="' + conceptId + '">'
        + esc(concept.title) + '</button></p>';
    }

    html += '<div class="insp-block"><span class="eyebrow">依据与适用范围</span>'
      + '<p class="insp-p">' + esc(c.origin) + '</p></div>';

    html += '<div class="insp-block"><span class="eyebrow">代码落点</span><div class="insp-lines">'
      + c.lines.map(function (no) {
        return '<button class="lineref" type="button" data-goto="' + no + '">L' + no + '</button>';
      }).join('') + '</div></div>';
    return html;
  }

  function capacityMemoryMap(kib, label) {
    const max = Math.max(128, kib);
    const width = 280, limit = width * 64 / max, block = width * kib / 2 / max;
    return '<div class="capacity-map" role="img" aria-label="' + esc(label) + '：两个 buffer 各 ' + (kib / 2) + ' KiB，合计 ' + kib + ' KiB；参考上限 64 KiB">'
      + '<div class="capacity-map-head"><span>片上 RIGHT BUFFER</span><span>910B 参考 · 64 KiB</span></div>'
      + '<svg viewBox="0 0 300 100" aria-hidden="true"><rect x="10" y="25" width="' + limit + '" height="38" rx="3" class="cm-space"/>'
      + '<rect x="10" y="32" width="' + block + '" height="24" class="cm-buf-a"/><rect x="' + (10 + block) + '" y="32" width="' + block + '" height="24" class="cm-buf-b"/>'
      + '<line x1="' + (10 + limit) + '" y1="12" x2="' + (10 + limit) + '" y2="75" class="cm-limit"/>'
      + '<text x="10" y="91" class="cm-label">0</text><text x="' + (10 + limit) + '" y="91" text-anchor="middle" class="cm-label">64 KiB</text>'
      + '<text x="' + (10 + block / 2) + '" y="48" text-anchor="middle" class="cm-block-label">0</text><text x="' + (10 + block * 1.5) + '" y="48" text-anchor="middle" class="cm-block-label">1</text></svg>'
      + '<div class="capacity-map-foot"><span>■ BUFFER 0　■ BUFFER 1</span><b class="' + (kib > 64 ? 'is-over' : 'is-fit') + '">' + kib + ' / 64 KiB ' + (kib > 64 ? '· 超出 ' + (kib - 64) : kib === 64 ? '· 达到上限' : '· 未超限') + '</b></div></div>';
  }

  function capacityDecisionHtml(d) {
    const used = d.K_TILE_DB / 1024;
    const over = used > 64;
    const candidate = [128, 64].find(function (n) { return n < d.SEQ_TILE && n * d.HEAD_DIM * 4 / 1024 <= 64; });
    const candidateKiB = candidate ? candidate * d.HEAD_DIM * 4 / 1024 : null;
    const pageMatches = candidate === d.PAGE_SIZE;
    const baseline = state.capacityBaseline;
    let html = '<div class="capacity-decision"><div class="capacity-kicker">RIGHT BUFFER · 示例估算</div>'
      + '<h3>' + (over ? '当前完整块的容量估算超限' : '当前完整块的容量估算未超限') + '</h3>'
      + capacityMemoryMap(used, '当前完整块估算')
      + '<p class="capacity-formula">' + d.SEQ_TILE + ' × ' + d.HEAD_DIM + ' × 2 B × 2 份 = ' + used + ' KiB</p>'
      + '<p>假设两个完整 BF16 K tile 同时驻留。64 KiB 为 910B 参考值；实际分配与目标设备需核实。</p>'
      + '<button type="button" class="capacity-link" data-goto="773">定位 K tile 源码 ↗</button>';
    if (baseline) {
      html += '<div class="capacity-applied"><b>已应用到演示配置</b><span>SEQ_TILE ' + baseline.seq + ' → ' + d.SEQ_TILE + '；源码和运行时均未修改。</span><span>分页契约：' + (d.SEQ_TILE === d.PAGE_SIZE ? '满足' : '不满足') + ' · page size ' + d.PAGE_SIZE + '</span></div>';
    }
    if (over && candidate) {
      html += '<div class="capacity-section"><div class="capacity-section-title">比较调整方向 <span>先预览，再决定</span></div>'
        + '<div class="capacity-option"><b><i>A</i> 缩小逻辑块</b><div class="capacity-path"><span>逻辑块 ' + d.SEQ_TILE + '</span><span>→</span><span class="capacity-path-new">逻辑块 ' + candidate + '</span><span>→</span><span>双缓冲 ' + candidateKiB + ' KiB</span></div><p>更小的逻辑块改变分页粒度。</p>'
        + '<small>可能影响分页契约；实际分配与性能待验证。</small><button type="button" data-capacity-action="preview">' + (state.capacityPreview ? '收起影响' : '预览影响') + '</button></div>';
      if (state.capacityPreview) {
        html += '<div class="capacity-preview"><b>方案 A · 配置预览</b>' + capacityMemoryMap(candidateKiB, '方案 A 预览') + '<div><span>容量估算</span><strong>' + candidateKiB + ' / 64 KiB' + (candidateKiB === 64 ? ' · 达到上限' : ' · 未超限') + '</strong></div>'
          + '<div><span>分页契约</span><strong class="' + (pageMatches ? 'capacity-good' : 'capacity-warn') + '">' + (pageMatches ? '满足' : '不满足') + ' · page size ' + d.PAGE_SIZE + '</strong></div>'
          + '<p>' + (pageMatches ? '预览值与当前 runtime.page_size 一致。' : '当前代码要求 SEQ_TILE 与 runtime.page_size 一致；请先确认运行时配置是否可调整。') + '</p>'
          + '<button type="button" data-capacity-action="apply" data-candidate="' + candidate + '">应用到演示配置</button><small>只改变本页参数，不修改源码或运行时。</small></div>';
      }
      html += '<div class="capacity-option"><b><i>B</i> 保持逻辑块，探索内部再分块</b><div class="capacity-path"><span>逻辑块 ' + d.SEQ_TILE + '</span><span>→</span><span class="capacity-path-new">分段 ' + candidate + ' × ' + Math.ceil(d.SEQ_TILE / candidate) + '</span><span>→</span><span>双缓冲 ' + candidateKiB + ' KiB</span></div><p>保留分页粒度，改变计算循环。</p>'
        + '<small>需要调整计算与结果处理；分块图为伪代码示意。</small><button type="button" data-capacity-action="chunk" data-candidate="' + candidate + '">查看分块示意 ↗</button></div></div>';
    }
    html += '<div class="capacity-section capacity-validation"><div class="capacity-section-title">验证闭环 <span>当前均未执行</span></div>'
      + '<ol><li><b>编译分配</b><span>重新编译，核对目标和实际 Right buffer 分配。</span></li>'
      + '<li><b>数值正确性</b><span>核对分段、合并与尾块处理，运行正确性测试。</span></li>'
      + '<li><b>设备性能</b><span>在目标设备测量，并与基线比较。</span></li></ol>'
      + '<p>本页不执行编译、测试或设备测量；估算未超限不代表验证通过。</p></div>'
      + '<details class="capacity-evidence"><summary>依据与适用范围</summary><p>双缓冲及完整块驻留是演示假设；真实生命周期、切分和其他空间占用需查看编译产物。</p>'
      + '<a href="https://github.com/hw-native-sys/pypto/issues/1595" target="_blank" rel="noopener noreferrer">910B 容量参考 #1595 ↗</a> · <a href="https://github.com/hw-native-sys/pypto/issues/630" target="_blank" rel="noopener noreferrer">独立分块案例 #630 ↗</a></details></div>';
    return html;
  }

  // keepScroll：改旋钮引起的重算不该把右栏滚回顶部；换选中对象才回顶部
  function renderRight(keepScroll) {
    let body = null;
    let title = '总览';
    if (state.sel.type === 'concept') { body = ''; title = '约束与估算'; }
    else if (state.sel.type === 'constraint') { body = constraintHtml(state.sel.id); title = '当前约束'; }
    if (body == null) {
      state.sel = { type: 'overview', id: null };
      body = overviewHtml();
      title = '总览';
    }
    const host = $('#rightBody');
    const top = keepScroll ? host.scrollTop : 0;
    $('#rightTitle').textContent = title;
    if (state.sel.type === 'constraint') {
      host.innerHTML = '<div class="hn-obj"><button class="hn-back" type="button" data-constraint-back>← 返回约束列表</button>' + body + '</div>';
    } else {
      host.innerHTML = '<div class="hn-obj">' + body + '</div>'
        + '<section class="hn-cons-sec" id="consSection" aria-label="约束与估算"></section>';
      renderCons();
    }
    host.scrollTop = top;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 选择与渲染编排
  // ════════════════════════════════════════════════════════════════════════════
  function select(type, id, opts) {
    const active = document.activeElement;
    const focusAttr = active && active.hasAttribute
      ? (active.hasAttribute('data-constraint') ? 'data-constraint'
        : active.hasAttribute('data-concept') ? 'data-concept' : null)
      : null;
    const focusValue = focusAttr && active.getAttribute(focusAttr);
    // 重渲染后焦点要落回「点击来源那一栏」的同名元素，而不是任意第一个
    const focusHost = focusAttr && active.closest('#rightBody, #hnExplorer, #editorBody');
    const focusHostId = focusHost ? focusHost.id : null;

    if (type === 'constraint' && state.sel.type !== 'constraint') {
      state.constraintReturn = { sel: state.sel, anchor: state.anchor, scrollTop: $('#rightBody').scrollTop };
    }
    state.sel = { type: type, id: id };
    const lines = type === 'concept' ? conceptLines(id) : [];
    state.anchor = lines.indexOf(opts && opts.anchor) >= 0 ? opts.anchor : (lines[0] || null);
    // 选中语义对象会换掉「约束区在讲什么」，筛选随之复位；
    // 在约束之间跳转只是同一个列表内的导航，筛选保留。
    if (type === 'concept' && !(opts && opts.preserveFilter)) state.filter = 'all';
    const inspectorToggle = $('[data-ide-toggle="inspector"]');
    if (type === 'constraint' && inspectorToggle.getAttribute('aria-expanded') === 'false') inspectorToggle.click();
    renderEditor();
    renderCatalog();
    renderRight();
    if (type === 'constraint' && focusHostId === 'rightBody') {
      $('[data-constraint-back]').focus({ preventScroll: true });
    } else if (focusAttr && focusValue) {
      const q = '[' + focusAttr + '="' + focusValue + '"]';
      const scope = focusHostId ? $('#' + focusHostId) : null;
      const replacement = (scope && $(q, scope)) || document.querySelector(q);
      if (replacement) replacement.focus({ preventScroll: true });
    }
    if (opts && opts.scroll) scrollToSelection();
  }

  function scrollToSelection() {
    let target = null;
    if (state.sel.type === 'constraint') {
      const c = M.CONSTRAINTS.filter(function (k) { return k.id === state.sel.id; })[0];
      if (c) target = $('#editorBody .ed-line[data-no="' + c.lines[0] + '"]');
    } else if (state.sel.type === 'concept') {
      target = $('#editorBody .ed-line[data-no="' + state.anchor + '"]');
    }
    if (target) target.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  function gotoLine(no) {
    const el = $('#editorBody .ed-line[data-no="' + no + '"]');
    if (el) {
      el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      el.classList.add('is-ping');
      setTimeout(function () { el.classList.remove('is-ping'); }, 900);
    }
  }

  function applyLensAttrs() {
    LENSES.forEach(function (L) {
      document.documentElement.setAttribute('data-lens-' + L.id, state.lens[L.id] ? 'on' : 'off');
    });
  }

  function renderAll() {
    evaluate();
    renderEditor();
    renderLensPanel();
    renderCatalog();
    renderKnobs();
    renderStatus();
    renderRight(true);
    const changed = M.KNOBS.filter(function (k) { return state.cfg[k.id] !== M.DEFAULTS[k.id]; }).length;
    $('#configChanges').textContent = changed ? '已修改 ' + changed + ' 项' : '默认配置';
    $('#envShape').textContent = state.d.BATCH + '×' + state.d.HIDDEN + ' · ' + state.d.NUM_KV_HEADS + ' KV heads';
  }

  // ════════════════════════════════════════════════════════════════════════════
  // 事件
  // ════════════════════════════════════════════════════════════════════════════
  const BUFFER_STEPS = ['qk', 'pv'];
  let bufferTimer = null;

  function stopBufferPlayback() {
    state.bufferPlaying = false;
    if (bufferTimer !== null) clearTimeout(bufferTimer);
    bufferTimer = null;
  }

  function setBufferStep(step) {
    state.bufferStep = step;
    state.bufferStage = step === 'load-k' || step === 'qk' ? 'qk' : step === 'softmax' ? 'softmax' : 'pv';
  }

  function bufferChunkCount() {
    const seq = state.d.SEQ_TILE;
    const tile = state.bufferTile === 'full' ? seq : Math.min(Number(state.bufferTile), seq);
    return Math.ceil(seq / tile);
  }

  function refreshBufferScene() {
    const host = $('#editorBody');
    if (!host) return;
    const top = host.scrollTop, left = host.scrollLeft;
    const playHadFocus = !!(document.activeElement && document.activeElement.matches('[data-buffer-play]'));
    renderEditor();
    host.scrollTop = top; host.scrollLeft = left;
    const play = $('[data-buffer-play]', host);
    if (playHadFocus && play) play.focus({ preventScroll: true });
  }

  function scheduleBufferStep() {
    bufferTimer = setTimeout(function () {
      bufferTimer = null;
      if (!state.bufferPlaying || state.issueHint !== 'capacity') { stopBufferPlayback(); return; }
      if (state.bufferChunk >= bufferChunkCount() - 1) {
        stopBufferPlayback();
        refreshBufferScene();
        return;
      }
      state.bufferChunk++;
      refreshBufferScene();
      scheduleBufferStep();
    }, 1800);
  }

  function animateCapacityChange(previous) {
    if (state.issueHint !== 'capacity' || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const host = $('#editorBody');
    if (!host) return;
    const valid = function (d) { return d && d.HEAD_DIM === 128 && (d.SEQ_TILE === 128 || d.SEQ_TILE === 256); };
    const oldKnown = valid(previous), newKnown = valid(state.d);
    const duration = 1100;
    if (oldKnown && newKnown) {
      const before = previous.SEQ_TILE === 128 ? 25 : 50;
      const after = state.d.SEQ_TILE === 128 ? 25 : 50;
      [['.capacity-qk', false], ['.capacity-pv', true], ['.capacity-shared', false]].forEach(function (item) {
        const node = $(item[0], host);
        if (!node || !node.animate) return;
        const from = { width: before + '%', left: item[1] ? before + '%' : '0%' };
        const to = { width: after + '%', left: item[1] ? after + '%' : '0%' };
        if (item[1]) {
          from.backgroundColor = before === 50 ? '#703b3a' : '#51436d';
          to.backgroundColor = after === 50 ? '#703b3a' : '#51436d';
        }
        node.animate([from, to], { duration: duration, easing: 'ease-in-out' });
      });
      if (after === 50 && before !== after) {
        host.querySelectorAll('.capacity-limit').forEach(function (node) {
          if (node.animate) node.animate([{ opacity: 1 }, { opacity: 0.25 }, { opacity: 1 }], { duration: 750, delay: 450 });
        });
      }
    } else if (newKnown) {
      const chart = $('.capacity-comparison', host);
      if (chart && chart.animate) chart.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500 });
    }
    [774, 790].forEach(function (line) {
      const node = $('.ed-line[data-no="' + line + '"]', host);
      if (node && node.animate) node.animate([{ filter: 'brightness(1)' }, { filter: 'brightness(1.7)' }, { filter: 'brightness(1)' }], { duration: duration });
    });
  }

  function wire() {
    document.addEventListener('input', function (ev) {
      const input = ev.target.closest('[data-valid-length]');
      if (!input) return;
      const scene = input.closest('.valid-case');
      const error = $('[data-valid-error]', scene);
      const value = input.valueAsNumber;
      if (!Number.isInteger(value) || value < 0 || value > 4096) {
        input.setAttribute('aria-invalid', 'true');
        if (error) error.textContent = '请输入 0 至 4096 之间的整数。';
        return;
      }
      input.removeAttribute('aria-invalid');
      if (error) error.textContent = '';
      state.validExampleLength = value;
      if (state.validExampleBlock !== 'overflow') state.validExampleBlock = 'tail';
      const template = document.createElement('template');
      template.innerHTML = validShapeDiagram(state.d);
      ['[data-valid-block]', '.valid-flow', '.valid-shape-panel', '.valid-formula', '.valid-case-note'].forEach(function (selector) {
        const current = $(selector, scene), updated = $(selector, template.content);
        if (current && updated) current.replaceWith(updated);
      });
    });
    document.addEventListener('click', function (ev) {
      const capacityAction = ev.target.closest('[data-capacity-action]');
      if (capacityAction) {
        const action = capacityAction.dataset.capacityAction;
        if (action === 'preview') {
          state.capacityPreview = !state.capacityPreview;
          renderRight(true);
          const button = $('[data-capacity-action="preview"]');
          if (button) button.focus({ preventScroll: true });
        } else if (action === 'apply') {
          state.capacityBaseline = { seq: state.d.SEQ_TILE };
          state.cfg.SEQ_TILE = Number(capacityAction.dataset.candidate);
          state.capacityPreview = false;
          state.bufferTile = 'full';
          renderAll();
          gotoLine(176);
        } else if (action === 'chunk') {
          stopBufferPlayback();
          state.issueHint = 'capacity';
          state.issueLine = 774;
          state.bufferStage = 'qk';
          state.bufferStep = 'qk';
          state.bufferTile = Number(capacityAction.dataset.candidate);
          state.bufferChunk = 0;
          renderEditor();
          const target = $('#editorBody .ed-line[data-no="774"]');
          if (target) target.scrollIntoView({ block: 'start', behavior: 'smooth' });
        }
        return;
      }
      const configInline = ev.target.closest('[data-config-inline]');
      if (configInline) {
        const host = $('#editorBody'), top = host.scrollTop, left = host.scrollLeft;
        state.inlineConfig = !state.inlineConfig;
        renderEditor();
        host.scrollTop = top; host.scrollLeft = left;
        configInline.focus({ preventScroll: true });
        return;
      }
      const playButton = ev.target.closest('[data-buffer-play]');
      if (playButton) {
        if (state.bufferPlaying) stopBufferPlayback();
        else {
          if (state.bufferChunk >= bufferChunkCount() - 1) state.bufferChunk = 0;
          state.bufferPlaying = true;
          scheduleBufferStep();
        }
        refreshBufferScene();
        return;
      }
      const tileChunk = ev.target.closest('[data-tile-chunk]');
      const tileMove = ev.target.closest('[data-tile-move]');
      if (tileChunk || tileMove) {
        stopBufferPlayback();
        const target = tileChunk ? Number(tileChunk.dataset.tileChunk) : state.bufferChunk + (tileMove.dataset.tileMove === 'next' ? 1 : -1);
        state.bufferChunk = Math.max(0, Math.min(bufferChunkCount() - 1, target));
        refreshBufferScene();
        const button = $('[data-tile-chunk="' + state.bufferChunk + '"]', $('#editorBody'));
        if (button) button.focus({ preventScroll: true });
        return;
      }
      const stageButton = ev.target.closest('[data-buffer-step]');
      const modeButton = ev.target.closest('[data-buffer-mode]');
      if (stageButton || modeButton) {
        stopBufferPlayback();
        const host = $('#editorBody');
        const top = host.scrollTop, left = host.scrollLeft;
        const attr = stageButton ? 'data-buffer-step' : 'data-buffer-mode';
        const value = (stageButton || modeButton).getAttribute(attr);
        if (stageButton) {
          setBufferStep(value);
          state.bufferChunk = 0;
        } else state.bufferMode = value;
        renderEditor();
        host.scrollTop = top; host.scrollLeft = left;
        const back = $('[' + attr + '="' + value + '"]', host);
        if (back) back.focus({ preventScroll: true });
        return;
      }
      const issueLine = ev.target.closest('[data-issue-line]');
      if (issueLine && state.issueHint === 'capacity') {
        stopBufferPlayback();
        state.issueLine = Number(issueLine.dataset.issueLine);
        state.bufferStage = state.issueLine === 790 ? 'pv' : 'qk';
        state.bufferStep = state.bufferStage;
        state.bufferChunk = 0;
        renderEditor();
        const destination = $('#editorBody .ed-line[data-no="' + state.issueLine + '"]');
        if (destination) destination.scrollIntoView({ block: 'start', behavior: 'smooth' });
        const link = $('#editorBody [data-issue-at="' + state.issueLine + '"]');
        if (link) link.focus({ preventScroll: true });
        return;
      }
      const issueOpen = ev.target.closest('[data-issue-open]');
      if (issueOpen) {
        stopBufferPlayback();
        const id = issueOpen.getAttribute('data-issue-open');
        const line = Number(issueOpen.dataset.issueAt) || ISSUE_HINTS[id].lines[0];
        const closeCurrent = state.issueHint === id && state.issueLine === line;
        state.issueHint = closeCurrent ? null : id;
        state.issueLine = closeCurrent ? null : line;
        if (id === 'capacity' && !closeCurrent) { state.bufferStage = line === 790 ? 'pv' : 'qk'; state.bufferStep = state.bufferStage; state.bufferChunk = 0; }
        renderEditor();
        document.querySelectorAll('[data-issue-open]').forEach(function (button) {
          button.setAttribute('aria-expanded', String(button.dataset.issueOpen === state.issueHint && (!button.dataset.issueAt || Number(button.dataset.issueAt) === state.issueLine)));
        });
        const target = issueOpen.closest('.issue-navigation') ? issueOpen : $('#editorBody [data-issue-open="' + id + '"][data-issue-at="' + line + '"]');
        if (target) target.focus({ preventScroll: true });
        if (state.issueHint) {
          const targetLine = $('#editorBody .ed-line[data-no="' + state.issueLine + '"]');
          if (targetLine) targetLine.scrollIntoView({ block: 'start', behavior: 'smooth' });
        }
        return;
      }
      if (ev.target.closest('[data-issue-close]')) {
        stopBufferPlayback();
        const id = state.issueHint;
        const line = state.issueLine;
        state.issueHint = null;
        state.issueLine = null;
        renderEditor();
        document.querySelectorAll('[data-issue-open]').forEach(function (button) { button.setAttribute('aria-expanded', 'false'); });
        const target = $('#editorBody [data-issue-open="' + id + '"][data-issue-at="' + line + '"]');
        if (target) target.focus({ preventScroll: true });
        return;
      }
      const tileAction = ev.target.closest('[data-tile-action]');
      if (tileAction) {
        const host = tileAction.closest('[data-tile-animation]');
        const action = tileAction.dataset.tileAction;
        if (action === 'toggle') setTilePlaying(!tilePlaying);
        else {
          setTilePlaying(false);
          const count = Number(host.dataset.count);
          tileStep = (tileStep + (action === 'next' ? 1 : count - 1)) % count;
          paintTileAnimation();
        }
        return;
      }
      const close = ev.target.closest('[data-close-concept]');
      if (close) {
        const id = state.sel.id;
        select('overview', null);
        const item = $('#catalog [data-concept="' + id + '"]');
        if (item) item.focus({ preventScroll: true });
        return;
      }
      const anchor = ev.target.closest('[data-anchor]');
      if (anchor) {
        const no = Number(anchor.getAttribute('data-anchor'));
        select('concept', state.sel.id, { anchor: no, scroll: true });
        const replacement = $('[data-anchor="' + no + '"]');
        if (replacement) replacement.focus({ preventScroll: true });
        return;
      }
      const filter = ev.target.closest('[data-filter]');
      if (filter) {
        state.filter = filter.getAttribute('data-filter');
        renderCons();
        const back = $('[data-filter="' + state.filter + '"]');
        if (back) back.focus({ preventScroll: true });
        return;
      }
      const scenario = ev.target.closest('[data-scenario]');
      if (scenario) {
        const presets = { default: {}, capacity: { SEQ_TILE: 256 }, alignment: { QKV_OK: 4 }, parallel: { ROPE_CORES: 24 } };
        const scenarioId = scenario.getAttribute('data-scenario');
        state.cfg = Object.assign({}, M.DEFAULTS, presets[scenarioId]);
        state.capacityPreview = false;
        state.capacityBaseline = null;
        if (scenarioId === 'capacity') {
          state.issueHint = 'capacity';
          state.issueLine = 774;
          state.bufferStage = 'qk';
          state.bufferStep = 'qk';
          state.bufferTile = 'full';
          state.bufferChunk = 0;
        }
        renderAll();
        if (scenarioId === 'capacity') {
          select('constraint', 'l0b-capacity');
          gotoLine(774);
        }
        pulse();
        return;
      }
      const lensBtn = ev.target.closest('[data-lens-toggle]');
      if (lensBtn) {
        const id = lensBtn.getAttribute('data-lens-toggle');
        state.lens[id] = !state.lens[id];
        applyLensAttrs();
        renderLensPanel();
        $('[data-lens-toggle="' + id + '"]').focus();
        return;
      }
      const constraintBack = ev.target.closest('[data-constraint-back]');
      if (constraintBack) {
        const id = state.sel.id;
        const previous = state.constraintReturn || { sel: { type: 'overview', id: null }, anchor: null, scrollTop: 0 };
        state.constraintReturn = null;
        select(previous.sel.type, previous.sel.id, { anchor: previous.anchor, preserveFilter: true });
        $('#rightBody').scrollTop = previous.scrollTop;
        const row = $('#rightBody [data-constraint="' + id + '"]');
        if (row) row.focus({ preventScroll: true });
        return;
      }
      const consBtn = ev.target.closest('[data-constraint]');
      if (consBtn) {
        select('constraint', consBtn.getAttribute('data-constraint'), { scroll: true });
        return;
      }
      const conBtn = ev.target.closest('[data-concept]');
      if (conBtn) {
        const sourceLine = conBtn.closest('.ed-line');
        select('concept', conBtn.getAttribute('data-concept'), { anchor: sourceLine ? Number(sourceLine.dataset.no) : null, scroll: true });
        return;
      }
      const go = ev.target.closest('[data-goto]');
      if (go) { gotoLine(Number(go.getAttribute('data-goto'))); return; }

      if (ev.target.closest('#resetAll')) { reset(); return; }
      if (ev.target.closest('#backOverview')) { select('overview', null); return; }

      const line = ev.target.closest('#editorBody .ed-line');
      if (line && line.getAttribute('data-focus')) {
        select('concept', line.getAttribute('data-focus'), { anchor: Number(line.dataset.no), scroll: true });
      }
    });

    document.addEventListener('keydown', function (ev) {
      if (ev.key !== 'Enter' && ev.key !== ' ') return;
      const link = ev.target.closest && ev.target.closest('.tk-link');
      if (link) { ev.preventDefault(); const line = link.closest('.ed-line'); select('concept', link.getAttribute('data-concept'), { anchor: line ? Number(line.dataset.no) : null, scroll: true }); }
    });

    document.addEventListener('change', function (ev) {
      const validBlock = ev.target.closest('[data-valid-block]');
      if (validBlock) {
        const host = $('#editorBody'), top = host.scrollTop, left = host.scrollLeft;
        const value = validBlock.value;
        const blocks = Math.ceil(state.validExampleLength / state.d.SEQ_TILE);
        state.validExampleBlock = value !== 'overflow' && Number(value) === blocks - 1 ? 'tail' : value;
        renderEditor();
        host.scrollTop = top; host.scrollLeft = left;
        const select = $('[data-valid-block]', host);
        if (select) select.focus({ preventScroll: true });
        return;
      }
      const tileSelect = ev.target.closest('[data-buffer-tile]');
      if (tileSelect) {
        stopBufferPlayback();
        const host = $('#editorBody'), top = host.scrollTop, left = host.scrollLeft;
        state.bufferTile = tileSelect.value;
        state.bufferChunk = 0;
        renderEditor();
        host.scrollTop = top; host.scrollLeft = left;
        const select = $('[data-buffer-tile]', host);
        if (select) select.focus({ preventScroll: true });
        return;
      }
      const bufferSeq = ev.target.closest('[data-buffer-seq]');
      if (bufferSeq) {
        stopBufferPlayback();
        const host = $('#editorBody'), top = host.scrollTop, left = host.scrollLeft;
        const previous = state.d;
        state.capacityBaseline = null;
        state.capacityPreview = false;
        state.cfg.SEQ_TILE = Number(bufferSeq.value);
        if (state.bufferTile !== 'full' && Number(state.bufferTile) >= state.cfg.SEQ_TILE) state.bufferTile = 'full';
        state.bufferChunk = 0;
        renderAll();
        host.scrollTop = top; host.scrollLeft = left;
        animateCapacityChange(previous);
        const select = $('[data-buffer-seq]', host);
        if (select) select.focus({ preventScroll: true });
        pulse();
        return;
      }
      const issueCores = ev.target.closest('[data-issue-cores]');
      if (issueCores) {
        state.issueCores = Number(issueCores.value);
        renderEditor();
        $('[data-issue-cores]').focus({ preventScroll: true });
        return;
      }
      const knob = ev.target.closest('[data-knob]');
      if (!knob) return;
      const id = knob.getAttribute('data-knob');
      stopBufferPlayback();
      const previous = state.d;
      state.capacityBaseline = null;
      state.capacityPreview = false;
      state.cfg[id] = Number(knob.value);
      if (id === 'SEQ_TILE') {
        if (state.bufferTile !== 'full' && Number(state.bufferTile) >= state.cfg.SEQ_TILE) state.bufferTile = 'full';
        state.bufferChunk = 0;
      }
      renderAll();
      if (id === 'SEQ_TILE') animateCapacityChange(previous);
      $('[data-knob="' + id + '"]').focus();
      pulse();
    });
    document.addEventListener('visibilitychange', function () {
      if (document.hidden && state.bufferPlaying) { stopBufferPlayback(); refreshBufferScene(); }
    });
  }

  function pulse() {
    const el = $('#statusStrip');
    el.classList.add('is-pulse');
    setTimeout(function () { el.classList.remove('is-pulse'); }, 600);
  }

  function reset() {
    stopBufferPlayback();
    state.issueHint = null;
    state.issueLine = null;
    state.issueCores = 24;
    state.validExampleLength = 300;
    state.validExampleBlock = 'tail';
    state.inlineConfig = true;
    state.bufferStage = 'qk';
    state.bufferStep = 'qk';
    state.bufferTile = 'full';
    state.bufferChunk = 0;
    state.bufferMode = 'separate';
    state.capacityPreview = false;
    state.capacityBaseline = null;
    state.cfg = Object.assign({}, M.DEFAULTS);
    state.lens = { tensor: true, scope: true, parallel: true, layout: true, precision: true };
    state.sel = { type: 'overview', id: null };
    state.anchor = null;
    state.filter = 'all';
    state.constraintReturn = null;
    applyLensAttrs();
    renderAll();
    $('#rightBody').scrollTop = 0;
  }

  // ── 启动 ───────────────────────────────────────────────────────────────────
  function init() {
    SOURCE_ROWS = M.SOURCE.map(function (e) { return { entry: e }; });
    applyLensAttrs();
    wire();
    renderAll();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
