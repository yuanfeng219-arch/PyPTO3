/* Run → Compilation 内容区。

   信息架构来自参考稿 pypto_compilation_redesign_v6.html：
     摘要 → 需要关注 → 工作区（Kernel 列表 + Source → 关键 Pass → Kernel｜编译 IR 全流程）
   视觉语法走项目已有 token，不引入第二套色板；没有第二份 Object Inspector，
   Pass 下钻就在主视图里就地展开。

   ── 编译数据来源──────────────────────────────────────────────
   window.PTO_IR_KERNELS
     .kernels[]        45 个 kernel：type / mem(按存储空间字节) / reuse / intent /
                       split / diags / perf(窄搬运提示) / passes[{i,st,n}]
     .limits / .spaces 各存储空间上限（L0A=Left / L0B=Right / L0C=Acc / UB=Vec / L1=Mat）
     .passNames        42 个 pass 名，kernels[].passes[].i 是这个数组的下标
   window.PTO_IR_PIPELINE
     .strata[]         7 个编译层（前端 / 规范化张量 / 层级化 / Tile / 双核 Kernel / 物理内存 / 运行时）
     .passes[]         每个 pass 的中文 desc、gain / lose、函数与算子计数、IR 变更 hunks
   window.PTO_DECODE_LAYER_SOURCE
     decode_layer.py 全文，用 name_hint="<kernel>" 定位 kernel 的源码块

   Numerical Validation 不从上述 artifact 推断。只有调用方显式提供
   runContext().numericalValidation 时才展示证据；否则始终是 Not Collected。
   文件底部暴露的 numericalFixtures 仅用于交互验收，不会自动贴到 Run 上。

   ── 三个「变化」概念的区别（参考稿第十节明确要求）───────────────────
   1. 「Pass 改变了整体 IR」      → PIPE.passes[i].hunks / f / o（所有 kernel 共用）
   2. 「Pass 对当前 Kernel 有变化」→ K.kernels[].passes[i].st / .n（逐 kernel，真实）
   3. 「Pass 对解释当前 Finding 有价值」→ EXPLAIN 权重表（下面，人工策展的排序函数，
      不是数据；它只决定「哪几个 Pass 值得画进关键链」，不改变任何事实）
   三者不混用：2 决定节点是否高亮，3 决定哪些进入横向关键链。

   ── 已知缺口（没有伪造，明确说明）──────────────────────────────────
   - 最终 Kernel 的「设备代码」在数据源里不存在（inventory 里的 PTO-ISA .pt 未加载）。
     Kernel 阶段因此展示的是真实的资源/切分/复用事实，不是伪代码。
   - `k_seed` / `v_seed` 的 name_hint 块只有 2 行；`decode_fwd_layers` 是编排根，
     定位不到独立源码块，此时 Source 阶段会明确说明，而不是编一段源码。
   - k.passes[i].n 是「该 pass 在该 kernel 上产生的变更条数」，数据源没有给出
     变更内容本身，所以 Pass 阶段展示的 IR 前后差异来自 PIPE.passes[i].hunks
     （那是整个 IR 的 diff，不是这个 kernel 的 diff）—— UI 上已分别标注。 */
(function () {
  'use strict';

  const K = window.PTO_IR_KERNELS;
  const PIPE = window.PTO_IR_PIPELINE;
  const SOURCE = window.PTO_DECODE_LAYER_SOURCE;
  if (!K) return;

  const LIM = K.limits || {};
  const SPACES = K.spaces || [];
  const PASSNAMES = K.passNames || [];
  const PASSMETA = (PIPE && PIPE.passes) || [];
  const STRATA = (PIPE && PIPE.strata) || [];
  const SRC_LINES = SOURCE ? SOURCE.split('\n') : [];
  const SRC_FILE = 'decode_layer.py';

  /* 存储空间在硬件上的叫法与参考稿一致：Left=L0A / Right=L0B / Acc=L0C / Vec=UB / Mat=L1 */
  const SPACE_LABEL = { Vec: 'UB', Mat: 'L1', Acc: 'L0C', Left: 'L0A', Right: 'L0B', Bias: 'Bias' };

  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.prototype.slice.call((r || document).querySelectorAll(s));
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const kb = b => (b >= 1024 ? (b / 1024).toFixed(b >= 10240 ? 0 : 1) + ' KB' : b + ' B');
  const pct = (a, b) => (b ? Math.round((a / b) * 100) : 0);
  const runStamp = () => (String(K.source || '').match(/\d{8}_\d{6}/) || [null])[0];
  const trace = () => window.PTO_RUN_TRACE || null;
  const runContext = () => window.PTO_RUN_CONTEXT || {};

  /* ---------- 派生量（与 ir-compile-guard.js 口径一致，避免同一界面两套算法） ---------- */
  function worstMem(k) {
    let w = 0, space = null;
    for (const s of SPACES) {
      const p = pct((k.mem || {})[s] || 0, LIM[s]);
      if (p > w) { w = p; space = s; }
    }
    return { p: w, space };
  }
  function reuseGain(k) {
    const b = (k.reuse && k.reuse.before.b) || 0;
    return b ? Math.round((1 - k.reuse.after.b / b) * 100) : 0;
  }
  function diagOf(k) {
    const c = { error: 0, warn: 0, perf: 0 };
    (k.diags || []).forEach(d => { c[d.sev] = (c[d.sev] || 0) + 1; });
    return c;
  }
  const perfHintsOf = k => (k.perf || []).length;

  /* 三个 Finding 的真实判定 */
  function findingSets() {
    const mem = K.kernels.filter(k => {
      const w = worstMem(k);
      return w.space === 'Right' && w.p >= 100;
    });
    const intent = K.kernels.filter(k => k.intent && k.intent.demoted > 0);
    const perf = K.kernels.filter(k => perfHintsOf(k) > 0);
    return { mem, intent, perf };
  }
  const FINDINGS = [
    { id: 'mem', tone: 'warn', icon: '!', kind: 'mem', title: 'L0B 达到平台上限',
      route: 'resources', routeLabel: '去 Resources 验证实际资源压力',
      next: 'L0B 在编译阶段已经达到平台上限，但是否形成真实瓶颈仍需结合运行时数据判断。' },
    { id: 'intent', tone: 'warn', icon: '!', kind: 'intent', title: '流水线意图发生变化',
      route: 'performance', routeLabel: '在「执行」中查看性能证据',
      next: '当前只有编译阶段的意图信号，是否真的拖慢执行，需要映射到 Task 和运行时数据继续验证。' },
    { id: 'perf', tone: 'info', icon: '◇', kind: 'perf', title: '存在编译性能提示',
      route: 'performance', routeLabel: '在「执行」中查看性能证据',
      next: '编译器在这里给出了静态提示，需要映射到实际 Task 与运行期时间才能判断是否形成瓶颈。' }
  ];

  /* 解释价值权重：人工策展，只回答「这个 Pass 多能解释当前 Kernel」。
     key 必须是 passNames 里真实存在的名字；表里没有的 pass 不会进入关键链。 */
  const EXPLAIN = {
    mem: { AutoTileMatmulL0: 90, MemoryReuse: 80, AllocateMemoryAddr: 80, InitMemRef: 60,
           ConvertTensorToTileOps: 55, OutlineIncoreScopes: 35, OptimizeOrchTensors: 30,
           LowerPipelineLoops: 42, CanonicalizeIOOrder: 38 },
    intent: { SkewCrossCorePipeline: 90, LowerPipelineLoops: 80, SplitVectorKernel: 70,
              ExpandMixedKernel: 70, InjectGMPipeBuffer: 60, ConvertTensorToTileOps: 45, UnrollLoops: 40,
              CanonicalizeIOOrder: 38, InitMemRef: 28 },
    perf: { LowerVectorTransfer: 90, ConvertTensorToTileOps: 70, ResolveBackendOpLayouts: 60,
            CanonicalizeTileSlice: 50, InferTileMemorySpace: 45, OptimizeOrchTensors: 40,
            LowerPipelineLoops: 42, CanonicalizeIOOrder: 38, MemoryReuse: 32, AllocateMemoryAddr: 32 },
    none: { ConvertTensorToTileOps: 70, AutoTileMatmulL0: 60, MemoryReuse: 60,
            AllocateMemoryAddr: 60, OutlineIncoreScopes: 40, SkewCrossCorePipeline: 40,
            LowerPipelineLoops: 42, CanonicalizeIOOrder: 38, InitMemRef: 28 }
  };

  /* ---------- Numerical Validation data model ----------
     这是与 Structural Verification 正交的证据维度：前者由编译诊断得出，
     后者只接收「每个 Pass 后的 IR 在 host 上执行并与 Golden 比对」的显式结果。 */
  const EMPTY_NUMERICAL_VALIDATION = Object.freeze({
    status: 'not_collected', tolerance: null, passes: [], firstDivergentPass: null
  });

  function normalizeNumericalValidation(input) {
    if (!input || !['pass', 'fail', 'not_collected'].includes(input.status)) {
      return EMPTY_NUMERICAL_VALIDATION;
    }
    const passes = Array.isArray(input.passes) ? input.passes.map((p, order) => {
      const index = Number.isInteger(p.index) ? p.index : PASSNAMES.indexOf(p.name);
      return {
        index, order, name: p.name || PASSNAMES[index] || ('Pass ' + index),
        status: ['match', 'mismatch', 'not_checked'].includes(p.status) ? p.status : 'not_checked',
        maxAbs: p.maxAbs, maxRel: p.maxRel, mismatchCount: p.mismatchCount,
        comparedCount: p.comparedCount || p.total
      };
    }).filter(p => p.index >= 0) : [];
    return {
      status: input.status,
      tolerance: input.tolerance && { rtol: input.tolerance.rtol, atol: input.tolerance.atol },
      passes,
      firstDivergentPass: input.firstDivergentPass == null ? null : input.firstDivergentPass,
      passed: input.passed,
      total: input.total
    };
  }

  function fixturePasses(firstDivergentIndex) {
    return PASSNAMES.map((name, index) => {
      const mismatch = firstDivergentIndex != null && index >= firstDivergentIndex;
      const first = index === firstDivergentIndex;
      return {
        index, name, status: mismatch ? 'mismatch' : 'match',
        maxAbs: first ? 0.028 : mismatch ? 0.031 : 0,
        maxRel: first ? 0.014 : mismatch ? 0.016 : 0,
        mismatchCount: first ? 182 : mismatch ? 211 : 0,
        comparedCount: 4096
      };
    });
  }

  /* 显式 demo fixtures：只有 useNumericalFixture() 被调用时才进入视图。 */
  const NUMERICAL_FIXTURES = Object.freeze({
    all_match: Object.freeze({
      status: 'pass', tolerance: { rtol: 0.05, atol: 0.05 },
      passes: fixturePasses(null), passed: PASSNAMES.length, total: PASSNAMES.length,
      firstDivergentPass: null
    }),
    compiler_semantic_error: Object.freeze({
      status: 'fail', tolerance: { rtol: 0.05, atol: 0.05 },
      passes: fixturePasses(PASSNAMES.indexOf('ExpandMixedKernel')),
      passed: PASSNAMES.indexOf('ExpandMixedKernel'), total: PASSNAMES.length,
      firstDivergentPass: 'ExpandMixedKernel'
    })
  });

  /* 每个 kernel 归到哪一个 Finding —— 决定它的关键链用哪套「解释价值」权重 */
  function kindOf(k) {
    const w = worstMem(k);
    if (w.space === 'Right' && w.p >= 100) return 'mem';
    if (k.intent && k.intent.demoted > 0) return 'intent';
    if (perfHintsOf(k) > 0) return 'perf';
    return 'none';
  }

  /* ---------- 源码定位 ---------- */
  const rxEsc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  function srcCandidates(name) {
    const out = [name];
    out.push(name.replace(/_(spmd|aic|aiv)$/, ''));
    out.push(name.replace(/_\d+$/, ''));
    return out.filter((v, i, a) => v && a.indexOf(v) === i);
  }
  function locateBlock(name) {
    if (!SRC_LINES.length) return null;
    let hit = -1, used = null;
    for (const c of srcCandidates(name)) {
      const i = SRC_LINES.findIndex(l => l.indexOf('name_hint="' + c + '"') >= 0);
      if (i >= 0) { hit = i; used = c; break; }
    }
    if (hit < 0) {
      for (const c of srcCandidates(name)) {
        const rx = new RegExp('^\\s*def\\s+' + rxEsc(c) + '\\s*\\(');
        const i = SRC_LINES.findIndex(l => rx.test(l));
        if (i >= 0) { hit = i; used = c; break; }
      }
    }
    if (hit < 0) return null;

    /* pl.at(...) 往往跨多行：块首是上一处 `with pl.at(`，块体从 `) as x:` 之后开始 */
    let open = -1;
    for (let j = hit; j >= Math.max(0, hit - 40); j--) {
      if (SRC_LINES[j].indexOf('with pl.at(') >= 0) { open = j; break; }
    }
    let as = -1;
    for (let j = hit; j < Math.min(SRC_LINES.length, hit + 10); j++) {
      if (/\)\s*as\s+[A-Za-z_]\w*\s*:/.test(SRC_LINES[j])) { as = j; break; }
    }
    if (open < 0) open = as >= 0 ? as : hit;
    const head = as >= 0 ? as : open;
    const ind = (SRC_LINES[open].match(/^\s*/) || [''])[0].length;
    let end = head;
    for (let j = head + 1; j < SRC_LINES.length && j < head + 80; j++) {
      const l = SRC_LINES[j];
      if (!l.trim()) { end = j; continue; }
      if ((l.match(/^\s*/) || [''])[0].length <= ind) break;
      end = j;
    }
    return { from: open, to: end, key: used, hint: hit };
  }

  function srcSnippet(name, block, maxLines) {
    if (!block) return null;
    const from = block.from, to = Math.min(block.to, from + maxLines - 1);
    const out = [];
    for (let i = from; i <= to; i++) {
      let line = esc(SRC_LINES[i]);
      if (i === block.hint) line = '<span class="kc-hl">' + line + '</span>';
      else if (/pl\.pipeline\(/.test(SRC_LINES[i])) line = '<span class="kc-hl-warn">' + line + '</span>';
      out.push(line);
    }
    if (block.to > to) out.push('<span class="kc-dim">… 共 ' + (block.to - from + 1) + ' 行</span>');
    return out.join('\n');
  }

  /* 关键 Pass：先按解释价值排序取前 3，再按真实先后顺序展示（编译链是时间序） */
  function keyPasses(k) {
    const w = EXPLAIN[kindOf(k)] || EXPLAIN.none;
    const cand = (k.passes || []).filter(p => p.st === 'changed' || p.st === 'born');
    if (!cand.length) return [];
    const scored = cand.map(p => {
      const name = PASSNAMES[p.i] || ('Pass ' + p.i);
      return { i: p.i, name, st: p.st, n: p.n, w: w[name] || 0, score: (w[name] || 0) * 1000 + Math.min(p.n, 999) };
    });
    const curated = scored.filter(s => s.w > 0);
    const top = (curated.length ? curated : scored)
      .slice().sort((a, b) => b.score - a.score).slice(0, 3);
    /* 策展表没覆盖到的 Kernel（例如 SPMD 包壳）：用剩下的真实变更 Pass 按变更量补到 3 个。
       补进来的每个节点仍然是「真的改过这个 Kernel」的 Pass，只是排序依据换成了变更量。 */
    if (top.length < 3) {
      const rest = scored.filter(s => top.indexOf(s) < 0).sort((a, b) => b.n - a.n);
      while (top.length < 3 && rest.length) top.push(rest.shift());
    }
    const targets = [firstDivergentIndex(numericalValidation()), st.expandedPass]
      .filter((index, at, all) => index >= 0 && all.indexOf(index) === at);
    targets.forEach(index => {
      if (top.some(p => p.i === index)) return;
      const kernelPass = (k.passes || []).find(p => p.i === index);
      top.push({
        i: index, name: PASSNAMES[index] || ('Pass ' + index),
        st: kernelPass ? kernelPass.st : 'observed', n: kernelPass ? kernelPass.n : 0,
        w: Number.MAX_SAFE_INTEGER, score: Number.MAX_SAFE_INTEGER, numerical: true
      });
    });
    return top.sort((a, b) => a.i - b.i);
  }

  /* ---------- IR 前后差异（来自 PIPE.passes[i].hunks，是整个 IR 的 diff） ---------- */
  function irExcerpt(meta, side, maxLines) {
    if (!meta || !meta.hunks || !meta.hunks.length) return null;
    const h = meta.hunks[0];
    const arr = (side === 'before' ? h.b : h.a) || [];
    if (!arr.length) return null;
    return arr.slice(0, maxLines).map(l => esc(String(l).replace(/^\s+/, m => m))).join('\n');
  }
  /* hunks 缺失时用真实的 IR 规模变化顶上，不编造代码 */
  function irScaleFacts(meta) {
    if (!meta) return [];
    const f = meta.f || {}, o = meta.o || {};
    const out = [];
    const fn = [];
    if (f.aic) fn.push('AIC ' + f.aic);
    if (f.aiv) fn.push('AIV ' + f.aiv);
    if (f.incore) fn.push('InCore ' + f.incore);
    if (f.group) fn.push('Group ' + f.group);
    if (f.spmd) fn.push('SPMD ' + f.spmd);
    if (fn.length) out.push(['函数', fn.join(' · ')]);
    const op = [];
    if (o.te != null) op.push('tensor ' + o.te);
    if (o.ti != null) op.push('tile ' + o.ti);
    if (o.mr != null) op.push('memref ' + o.mr);
    if (o.al != null) op.push('alloc ' + o.al);
    if (op.length) out.push(['算子', op.join(' · ')]);
    if (meta.l) out.push(['IR 行数', String(meta.l)]);
    return out;
  }

  /* ---------- state ---------- */
  const previewParams = new URLSearchParams(window.location.search);
  const previewFixture = NUMERICAL_FIXTURES[previewParams.get('numericalFixture')] || null;
  const previewEntry = previewFixture && previewParams.get('from') === 'correctness'
    ? {
        from: 'correctness',
        finding: '数值精度 · 输出不一致',
        intent: '正在定位编译语义分歧'
      }
    : null;
  const st = {
    kernel: null, filter: 'issues', findOn: null, expandedPass: null,
    /* 工作区页签：kernel = Kernel 列表 + 详情；trace = 借用来的编译 IR 全流程 */
    pane: 'kernel', artifact: null,
    numericalOverride: previewFixture, entryContext: previewEntry, numericalFocusKey: null
  };
  let host = null;

  function kernelNames() { return K.kernels.map(k => k.name); }
  function issueSet() {
    const s = findingSets();
    const set = {};
    ['mem', 'intent', 'perf'].forEach(id => s[id].forEach(k => { set[k.name] = true; }));
    return set;
  }
  function byName(n) { return K.kernels.find(k => k.name === n) || null; }
  function current() { return byName(st.kernel) || K.kernels[0] || null; }

  function numericalValidation() {
    if (st.numericalOverride) return normalizeNumericalValidation(st.numericalOverride);
    return normalizeNumericalValidation(runContext().numericalValidation);
  }
  function compilationEntryContext() {
    return st.entryContext || runContext().compilationEntry || null;
  }
  function firstDivergentIndex(nv) {
    if (!nv || nv.firstDivergentPass == null) return -1;
    if (Number.isInteger(nv.firstDivergentPass)) return nv.firstDivergentPass;
    const row = nv.passes.find(p => p.name === nv.firstDivergentPass);
    return row ? row.index : PASSNAMES.indexOf(nv.firstDivergentPass);
  }
  function numericalPassAt(index) {
    return numericalValidation().passes.find(p => p.index === index) || null;
  }
  function fmtNumber(value) {
    if (value == null || value === '') return '—';
    return typeof value === 'number' ? String(value) : value;
  }

  function listRows() {
    const set = issueSet();
    let rows = K.kernels.slice();
    if (st.filter === 'issues') rows = rows.filter(k => set[k.name]);
    const order = { mem: 0, intent: 1, perf: 2, none: 3 };
    rows.sort((a, b) => {
      const d = order[kindOf(a)] - order[kindOf(b)];
      if (d) return d;
      return worstMem(b).p - worstMem(a).p || a.name.localeCompare(b.name);
    });
    return rows;
  }

  /* ---------- 渲染 ---------- */
  function summaryHTML() {
    const changed = PASSMETA.filter(p => p.d === null || p.d > 0).length;
    let e = 0;
    K.kernels.forEach(k => { e += diagOf(k).error; });
    const pass = e === 0;
    const s = findingSets();
    const findingCount = FINDINGS.filter(f => (s[f.id] || []).length).length;
    const meta = [PASSMETA.length + ' Pass', K.kernels.length + ' 个生成 Kernel', K.target].filter(Boolean).join(' · ');
    return '<section class="kc-summary">' +
      '<div class="kc-summary__outcome"><div class="kc-summary__title"><span>编译</span>' +
        '<b class="' + (pass ? 'is-ok' : 'is-bad') + '"><i>' + (pass ? '✓' : '×') + '</i>' + (pass ? 'PASS' : 'FAIL') + '</b></div>' +
        '<small>' + esc(meta) + '</small>' +
        '<div class="kc-summary__evidence"><b class="' + (pass ? 'is-ok' : 'is-bad') + '">' +
          (pass ? '结构校验 · PASS' : '结构校验 · FAIL') + '</b>' +
          '<span>·</span><span>' + (findingCount ? findingCount + ' 项发现需要复核' : '没有需要复核的发现') + '</span></div></div>' +
      '<div class="kc-summary__metrics" aria-label="编译摘要指标">' +
        '<span>L0B peak <b class="is-warn">' + esc(String(Math.max.apply(null, K.kernels.map(k => worstMem(k).space === 'Right' ? worstMem(k).p : 0)))) + '%</b></span>' +
        '<span>意图变化 <b>' + esc(String(s.intent.length)) + '</b></span>' +
        '<span>性能提示 <b>' + esc(String(s.perf.reduce((n, k) => n + perfHintsOf(k), 0))) + '</b></span>' +
        '<span>IR 变化 <b>' + esc(changed + ' / ' + PASSMETA.length) + '</b></span>' +
      '</div>' +
    '</section>';
  }

  function contextHTML() {
    const c = compilationEntryContext();
    if (!c || c.from !== 'correctness') return '';
    /* 与 task-history 的统一跨域 Context Banner 同构：来源 → 意图 → 失效模式。 */
    const finding = c.finding || '数值精度 · 输出不一致';
    const intent = c.intent || '正在定位编译语义分歧';
    return '<section class="kf-cv-context" aria-label="跨域上下文">' +
      '<span class="kf-cv-context-from">来自「正确性」</span>' +
      '<div class="kf-cv-context-intent"><b>' + esc(finding) + '</b><i>→</i><b>' + esc(intent) + '</b></div>' +
      (c.failureMode ? '<p class="kf-cv-context-focus">失效模式 · <b>' + esc(c.failureMode) + '</b></p>' : '') +
      '</section>';
  }

  function toleranceText(nv) {
    if (!nv.tolerance) return '未记录容差';
    return 'rtol ' + fmtNumber(nv.tolerance.rtol) + ' · atol ' + fmtNumber(nv.tolerance.atol);
  }

  function numericalWindow(nv) {
    const first = firstDivergentIndex(nv);
    if (first < 0) return [];
    const at = nv.passes.findIndex(p => p.index === first);
    if (at < 0) return [];
    return nv.passes.slice(Math.max(0, at - 3), Math.min(nv.passes.length, at + 4));
  }

  function numericalValidationHTML() {
    const nv = numericalValidation();
    const checked = nv.passes.filter(p => p.status !== 'not_checked');
    const matched = nv.passed != null ? nv.passed : checked.filter(p => p.status === 'match').length;
    const total = nv.total != null ? nv.total : checked.length;
    const first = firstDivergentIndex(nv);
    let tone = 'idle', status = 'NOT COLLECTED', message = '本次 Run 没有逐 Pass 数值证据。';
    if (nv.status === 'pass') {
      tone = 'ok'; status = matched + ' / ' + total + ' PASS';
      message = '编译器未引入语义分歧，可继续查看「正确性」或「执行」。';
    } else if (nv.status === 'fail') {
      tone = 'bad'; status = 'FIRST DIVERGENCE · ' + (PASSNAMES[first] || nv.firstDivergentPass || '—');
      message = 'Host execution 在该 Pass 之后首次偏离 Golden。';
    }
    const rows = nv.status === 'fail' ? numericalWindow(nv).map(p => {
      const isFirst = p.index === first;
      const label = isFirst ? 'FIRST DIVERGENCE' : p.status === 'match' ? 'MATCH' : p.status === 'mismatch' ? 'MISMATCH' : 'NOT CHECKED';
      return '<button type="button" class="kc-nv-pass is-' + (isFirst ? 'first' : p.status) +
        '" data-kc-pass="' + p.index + '"' + (isFirst ? ' aria-current="true"' : '') + '>' +
        '<span>' + esc(p.name) + '</span><b>' + esc(label) + '</b><i>›</i></button>';
    }).join('') : '';
    return '<section class="kc-nv is-' + tone + '" aria-labelledby="kcNvTitle">' +
      '<div class="kc-nv-head"><div><span class="kc-nv-kicker">编译语义</span>' +
        '<h2 id="kcNvTitle">数值校验</h2></div>' +
        '<div class="kc-nv-result"><b>' + esc(status) + '</b><span>' + esc(message) + '</span></div>' +
        '<small>' + esc(toleranceText(nv)) + '</small></div>' +
      (rows ? '<div class="kc-nv-passes" aria-label="Pass numerical validation sequence">' + rows + '</div>' : '') +
      (nv.status === 'fail' ? '<p class="kc-nv-note">展示首个分歧附近的校验窗口。选择一个 Pass，可在既有 IR diff 中查看。</p>' : '') +
    '</section>';
  }

  function findingsHTML() {
    const s = findingSets();
    return '<div class="kc-findings">' + FINDINGS.map(f => {
      const hits = s[f.id];
      if (!hits.length) return '';
      let desc, count;
      if (f.id === 'mem') {
        const top = hits.slice().sort((a, b) => worstMem(b).p - worstMem(a).p)[0];
        desc = top.name + ' · ' + kb(top.mem.Right) + ' / ' + kb(LIM.Right) +
          '。编译通过，当前尚不能判断是否影响运行性能。';
        if (hits.length > 1) desc += ' 另有 ' + (hits.length - 1) + ' 个 Kernel 同样打满。';
        count = hits.length + ' Kernel';
      } else if (f.id === 'intent') {
        const names = hits.slice(0, 3).map(k => k.name).join(' · ');
        desc = hits.length + ' 个 Kernel 的 pl.pipeline 在编译过程中被降级为顺序执行：' + names + '。';
        count = hits.length + ' Kernel';
      } else {
        /* 计数和「最小内层」都取自 kernels[].perf（逐 kernel 的窄搬运记录），
           不混用 perfHints —— 两者口径不同，混在一起会对不上。 */
        const recs = hits.reduce((a, k) => a.concat(k.perf || []), []);
        const minInner = recs.reduce((a, r) => Math.min(a, r.innermost), Infinity);
        desc = '检测到窄数据搬运 / 向量化机会' +
          (isFinite(minInner) ? '（最小内层 ' + minInner + ' 元素，目标 ' + K.perfMinInnermost + ' 元素）' : '') +
          '，需要在「执行」中查看性能证据。';
        count = hits.length + ' Kernel · ' + recs.length + ' 处';
      }
      const top = hits.slice().sort((a, b) => worstMem(b).p - worstMem(a).p)[0];
      const runtime = f.route === 'performance' && top ?
        '<button type="button" class="kc-finding-action" data-kc-runtime="' + esc(top.name) + '" data-kc-finding="' + esc(f.id) + '">在「执行」中验证影响 →</button>' : '';
      return '<div class="kc-finding is-' + f.tone + (st.findOn === f.id ? ' is-on' : '') + '">' +
        '<button type="button" class="kc-finding-main" data-kc-find="' + f.id + '">' +
        '<span class="kc-fico">' + esc(f.icon) + '</span>' +
        '<span class="kc-fbody"><span class="kc-ftitle"><b>' + esc(f.title) + '</b><em>' + esc(count) + '</em></span>' +
        '<small>' + esc(desc) + '</small></span><span class="kc-fchev">›</span></button>' + runtime + '</div>';
    }).join('') + '</div>';
  }

  function secondarySignalsHTML(count) {
    return '<details class="kc-secondary"><summary><span>其他信号</span><small>' + count + ' 项</small><i>展开</i></summary>' +
      findingsHTML() + '</details>';
  }

  /* Orchestration 这类长类型名放不进 38px 的 chip，给一个显示缩写，完整名进 title */
  const TYPE_ABBR = { Orchestration: 'Orch' };

  function itemHTML(k) {
    const kind = kindOf(k);
    const w = worstMem(k);
    const c = diagOf(k);
    const bits = [];
    if (w.space) bits.push((SPACE_LABEL[w.space] || w.space) + ' ' + w.p + '%');
    if (k.intent && k.intent.demoted) bits.push('pipeline 降级 ×' + k.intent.demoted);
    if (perfHintsOf(k)) bits.push(perfHintsOf(k) + ' 条性能提示');
    const dc = c.error + c.warn + c.perf;
    if (dc) bits.push(dc + ' 条诊断');
    if (!bits.length && k.reuse && k.reuse.before.b) bits.push('复用 −' + reuseGain(k) + '%');
    const tyCls = k.type === 'AIC' ? 'is-aic' : k.type === 'AIV' ? 'is-aiv' : 'is-other';
    return '<button type="button" class="kc-item' + (k.name === st.kernel ? ' is-on' : '') +
      '" data-kc-kernel="' + esc(k.name) + '" title="' + esc(k.name + ' · ' + (k.type || '') + '') + '">' +
      '<span class="kc-ty ' + tyCls + '" title="' + esc(k.type || '') + '">' +
      esc(TYPE_ABBR[k.type] || k.type || '—') + '</span>' +
      '<span><span class="kc-iname">' + esc(k.name) + '</span>' +
      '<span class="kc-imeta">' + esc(bits.join(' · ')) + '</span></span>' +
      '<span class="kc-istate is-' + (kind === 'none' ? 'ok' : 'warn') + '">' +
      (kind === 'none' ? '✓ 正常' : '● 关注') + '</span></button>';
  }

  function listHTML() {
    const rows = listRows();
    if (!rows.length) return '<p class="kc-list-empty">没有命中的 kernel。</p>';
    return rows.map(itemHTML).join('');
  }

  /* --- Source 阶段 --- */
  function srcStageHTML(k) {
    const block = locateBlock(k.name);
    const range = block ? SRC_FILE + ':' + (block.from + 1) + '–' + (block.to + 1) : SRC_FILE;
    const snippet = block ? srcSnippet(k.name, block, 22) : null;
    const tags = ['PyPTO DSL'];
    if (block && block.key !== k.name) tags.push('name_hint=' + block.key);
    if (k.intent && k.intent.declared.length) {
      tags.push(k.intent.declared.map(d => '声明 pl.pipeline(stage=' + d + ')').join(' '));
    }
    const inner = snippet
      ? '<div class="kc-code"><div class="kc-code-head"><span>' + esc(SRC_FILE) + '</span><span>Source</span></div>' +
        '<div class="kc-code-body">' + snippet + '</div></div>'
      : '<div class="kc-code"><div class="kc-code-head"><span>' + esc(SRC_FILE) + '</span><span>Source</span></div>' +
        '<div class="kc-src-missing">该 Kernel 是编排/合成结果，源码里没有独立的 name_hint 块。<br>' +
        '它由 ' + esc(SRC_FILE) + ' 中的多个 scope 组合而成，此处不展示片段以免误导。</div></div>';

    return '<section class="kc-stage is-source">' +
      '<div class="kc-stage-head"><span class="kc-icon">&lt;/&gt;</span>' +
      '<span class="kc-kind">源码</span><b>' + esc(range) + '</b></div>' +
      '<div class="kc-desc">' + esc(sourceSummary(k, block)) + '</div>' +
      '<div class="kc-meta">' + tags.map(t => '<span>' + esc(t) + '</span>').join('') + '</div>' +
      inner + '</section>';
  }

  function sourceSummary(k, block) {
    if (!block) return '该 Kernel 没有独立的源码块，它是编排层合成的结果。';
    const kind = kindOf(k);
    if (kind === 'mem') return '这段源码描述 ' + k.name + ' 的矩阵分块计算与结果写回，' +
      '后面的 L0 切分与 Buffer 分配都从它推导出来。';
    if (kind === 'intent') return '这段源码声明了流水线执行意图，是后续「意图是否被兑现」的分析起点。';
    if (kind === 'perf') return '这段源码包含数据搬运与向量计算，后续会被降低成 AIV 执行路径。';
    return '这段源码是 ' + k.name + ' 的直接来源，关键编译意图均已兑现。';
  }

  /* --- Pass 阶段 --- */
  function numericalPassEvidenceHTML(k, index) {
    const nv = numericalValidation();
    const row = nv.passes.find(p => p.index === index);
    if (!row || row.status === 'not_checked') return '';
    const pos = nv.passes.indexOf(row);
    let previous = null;
    for (let i = pos - 1; i >= 0; i--) {
      if (nv.passes[i].status !== 'not_checked') { previous = nv.passes[i]; break; }
    }
    const first = firstDivergentIndex(nv) === index;
    const label = s => s === 'match' ? 'MATCH' : s === 'mismatch' ? 'MISMATCH' : 'NOT CHECKED';
    const block = locateBlock(k.name);
    const source = block ? SRC_FILE + ':' + (block.from + 1) + '–' + (block.to + 1) : SRC_FILE + ' · composed kernel';
    return '<section class="kc-pass-validation' + (first ? ' is-first' : '') + '" aria-label="Pass numerical evidence">' +
      '<div class="kc-pass-validation__head"><span>数值校验</span>' +
        (first ? '<b>FIRST DIVERGENCE</b>' : '') + '</div>' +
      '<dl>' +
        '<div><dt>上一个 Pass</dt><dd class="is-ok">' + esc(previous ? label(previous.status) : '—') + '</dd></div>' +
        '<div><dt>当前 Pass</dt><dd class="' + (row.status === 'mismatch' ? 'is-bad' : 'is-ok') + '">' + esc(label(row.status)) + '</dd></div>' +
        '<div><dt>max_abs_diff</dt><dd>' + esc(fmtNumber(row.maxAbs)) + '</dd></div>' +
        '<div><dt>max_rel_diff</dt><dd>' + esc(fmtNumber(row.maxRel)) + '</dd></div>' +
        '<div><dt>不一致元素</dt><dd>' + esc(fmtNumber(row.mismatchCount) + ' / ' + fmtNumber(row.comparedCount)) + '</dd></div>' +
        '<div><dt>容差</dt><dd>' + esc(toleranceText(nv)) + '</dd></div>' +
      '</dl>' +
      '<p><span>源码映射</span><code>' + esc(source) + '</code></p>' +
    '</section>';
  }

  function passStageHTML(k, p, kind) {
    const meta = PASSMETA[p.i];
    const nvPass = numericalPassAt(p.i);
    const firstDivergence = firstDivergentIndex(numericalValidation()) === p.i;
    const warn = p.name === 'SkewCrossCorePipeline' || p.name === 'AllocateMemoryAddr' ||
      p.name === 'AutoTileMatmulL0';
    const before = irExcerpt(meta, 'before', 7);
    const after = irExcerpt(meta, 'after', 7);
    const facts = irScaleFacts(meta);
    let code;
    if (before || after) {
      code = '<div class="kc-code">' +
        '<div class="kc-code-head"><span>IR · 整体变更 hunk</span><span>' + esc(p.name) + '</span></div>' +
        '<div class="kc-code-body">' +
        (before ? '<span class="kc-dim">变化前</span>\n' + before : '<span class="kc-dim">变化前 · 该 hunk 无删除行</span>') +
        '\n\n<span class="kc-dim">→</span>\n\n' +
        (after ? '<span class="kc-dim">变化后</span>\n' + after : '<span class="kc-dim">变化后 · 该 hunk 无新增行</span>') +
        '</div></div>';
    } else {
      code = '<div class="kc-code">' +
        '<div class="kc-code-head"><span>IR 规模变化</span><span>' + esc(p.name) + '</span></div>' +
        '<div class="kc-code-body">' +
        '<span class="kc-dim">该 Pass 的 dump 没有行级 diff，以下是整体 IR 的真实计数</span>\n\n' +
        (facts.length ? facts.map(f => f[0].padEnd(8, ' ') + f[1]).join('\n') : '无可用计数') +
        '</div></div>';
    }

    const expanded = st.expandedPass === p.i;
    const state = p.st === 'born' ? '诞生' : p.st === 'observed' ? '校验目标' : '有变更';
    return '<article class="kc-transform' + (warn ? ' is-warn' : '') + (firstDivergence ? ' is-first-divergence' : '') + (expanded ? ' is-expanded' : '') + '">' +
      '<button type="button" class="kc-transform-toggle" data-kc-pass="' + p.i + '" aria-expanded="' + expanded + '">' +
        '<span class="kc-transform-dot"></span><span class="kc-transform-copy">' +
          '<span class="kc-kind">' + (firstDivergence ? '首个数值分歧 Pass' : '相关变换') + '</span><b>' + esc(p.name) + '</b>' +
          '<small>' + esc((meta && meta.s ? meta.s + ' ' + stratumName(meta.s) + ' · ' : '') + state +
            (p.st === 'observed' ? '' : ' · ' + p.n + ' 处变更') +
            (nvPass ? ' · ' + (nvPass.status === 'match' ? 'MATCH' : nvPass.status === 'mismatch' ? 'MISMATCH' : 'NOT CHECKED') : '')) + '</small>' +
        '</span><span class="kc-transform-chev">' + (expanded ? '−' : '+') + '</span></button>' +
      (expanded ? '<div class="kc-transform-detail">' + numericalPassEvidenceHTML(k, p.i) + '<p class="kc-desc">' +
        esc((meta && meta.desc) || '该 Pass 改变了当前 Kernel 的中间表示。') + '</p><div class="kc-meta">' +
          (p.st === 'born' ? '<span class="is-ok">在此 Pass 诞生</span>' : p.st === 'observed' ? '<span>Pass 变化 · 全局 IR 证据</span>' : '<span class="is-warn">变更 ' + p.n + ' 处</span>') +
          (meta && meta.gain && meta.gain.length ? '<span>' + esc('收益 ' + meta.gain.join(' · ')) + '</span>' : '') +
        '</div>' + code + '</div>' : '') + '</article>';
  }

  function transformationsStageHTML(k, keys, kind) {
    const expanded = keys.some(p => st.expandedPass === p.i);
    return '<section class="kc-stage is-transformations' + (expanded ? ' has-expanded' : '') + '">' +
      '<div class="kc-stage-head"><span class="kc-kind">编译变换</span>' +
        '<b>与当前 Kernel 相关的编译变化</b></div>' +
      '<div class="kc-transform-rail">' + keys.map(p => passStageHTML(k, p, kind)).join('') + '</div></section>';
  }

  function stratumName(id) {
    const s = STRATA.find(x => x.id === id);
    return s ? s.name : id;
  }

  function traceKernel(name) {
    const D = trace();
    return D && (D.kernels || []).find(k => k.name === name) || null;
  }
  function kernelArtifacts(name) {
    const tk = traceKernel(name);
    return tk && Array.isArray(tk.files) ? tk.files.map(f => ({ path: f[0], size: f[1] })) : [];
  }
  function artifactKind(path) {
    if (/\.pto$/i.test(path)) return 'PTO-ISA';
    if (/\.o$/i.test(path)) return 'Binary metadata';
    return 'Generated C++';
  }
  function artifactPanelHTML() {
    const a = st.artifact;
    if (!a) return '';
    const choices = kernelArtifacts(a.kernel || '').map(f => '<button type="button" class="' +
      (f.path === a.path ? 'is-on' : '') + '" data-kc-artifact="' + esc(f.path) + '" data-kc-size="' + f.size + '">' +
      esc(f.path.split('/').pop()) + '</button>').join('');
    const binary = /\.o$/i.test(a.path);
    const body = binary
      ? '<p class="kc-artifact-unavailable">二进制文件不在浏览器内展开。可用信息：' + esc(a.path) + ' · ' + esc(kb(a.size)) + '</p>'
      : a.loading
        ? '<p class="kc-artifact-unavailable">正在读取当前 Run 的产物…</p>'
        : a.error
          ? '<p class="kc-artifact-unavailable">无法读取该产物：' + esc(a.error) + '</p>'
          : '<pre class="kc-artifact-code">' + esc(a.content || '') + '</pre>';
    return '<section class="kc-artifact-drawer" aria-label="生成产物查看器">' +
      '<header><div><span>' + esc(artifactKind(a.path)) + '</span><b>' + esc(a.path) + '</b><small>' + esc(kb(a.size)) + ' · 当前 Run</small></div>' +
      '<button type="button" data-kc-artifact-close aria-label="关闭产物查看器">×</button></header>' +
      '<nav class="kc-artifact-choices" aria-label="当前 Kernel 生成产物">' + choices + '</nav>' + body + '</section>';
  }
  function artifactsRawHTML() {
    const inv = (K.inventory || []).filter(i => i.g === 'codegen' || i.g === 'compile');
    if (!inv.length) return '';
    const currentName = current() && current().name;
    const item = i => {
      if (i.k === 'ir') return '<button type="button" data-kc-tab="trace">' + esc(i.label) + '<code>' + esc(i.where || i.meta || '') + '</code></button>';
      if (i.g === 'codegen' && currentName && kernelArtifacts(currentName).length) {
        return '<button type="button" data-kc-code="' + esc(currentName) + '">' + esc(i.label) + '<code>' + esc(i.where || i.meta || '') + '</code></button>';
      }
      const file = (i.where || '').indexOf('/') >= 0 && !/\/$/.test(i.where || '') ? i.where : '';
      return file
        ? '<button type="button" data-kc-raw-file="' + esc(file) + '">' + esc(i.label) + '<code>' + esc(i.where || i.meta || '') + '</code></button>'
        : '<span><b>' + esc(i.label) + '</b><code>' + esc(i.where || i.meta || '') + '</code></span>';
    };
    return '<details class="kc-raw-artifacts"><summary>产物与原始输出 <small>按需查看当前 Run 的编译证据</small></summary>' +
      '<div>' + inv.map(item).join('') + '</div></details>';
  }

  /* --- Kernel 阶段 --- */
  function kernelStageHTML(k) {
    const declared = km => (km.intent && km.intent.declared.length)
      ? km.intent.declared.map(d => 'pipeline(stage=' + d + ')').join(' ')
      : (km.intent && km.intent.l0 && km.intent.l0.length
          ? km.intent.l0.map(l => 'L0 split ' + l.from + '→' + l.to + ' step ' + l.step).join(' · ')
          : '—');
    const c = diagOf(k);
    const dc = c.error + c.warn + c.perf;
    const artifacts = kernelArtifacts(k.name);
    const w = worstMem(k);
    const memories = SPACES.map(s => {
      const v = (k.mem || {})[s];
      return v ? (SPACE_LABEL[s] || s) + ' ' + kb(v) + ' / ' + kb(LIM[s]) : '';
    }).filter(Boolean).join(' · ') || '—';
    const reuse = k.reuse && k.reuse.after.b != null
      ? k.reuse.before.n + ' → ' + k.reuse.after.n + ' buffers · ' + kb(k.reuse.before.b) + ' → ' + kb(k.reuse.after.b)
      : '—';
    const diagnostics = dc ? dc + ' 条（' + [c.error && c.error + 'E', c.warn && c.warn + 'W',
      c.perf && c.perf + 'P'].filter(Boolean).join(' ') + '）' : '无';
    const hint = perfHintsOf(k) ? '最小内层 ' + k.perf[0].innermost + ' 元素 · ' + k.perf[0].bytes +
      ' B · 目标 ' + K.perfMinInnermost + ' B' : '';

    return '<section class="kc-stage is-kernel">' +
      '<div class="kc-stage-head"><span class="kc-kind">生成 Kernel</span><b>' + esc(k.name) + '</b></div>' +
      '<div class="kc-desc">' + esc(kernelSummary(k)) + '</div>' +
      '<dl class="kc-kernel-facts">' +
        '<div><dt>类型</dt><dd><span class="kc-type-label">' + esc((k.type || '—') + ' Kernel') + '</span></dd></div>' +
        '<div><dt>Split</dt><dd>' + esc(k.split || '—') + '</dd></div>' +
        '<div><dt>内存</dt><dd class="' + (w.p >= 100 ? 'is-warn' : '') + '">' + esc(memories) + '</dd></div>' +
        '<div><dt>复用</dt><dd>' + esc(reuse) + '</dd></div>' +
        '<div><dt>Pipeline</dt><dd class="' + (k.intent && k.intent.demoted ? 'is-warn' : '') + '">' +
          esc(declared(k) + (k.intent && k.intent.demoted ? ' · demoted ×' + k.intent.demoted + ' → sequential' : '')) + '</dd></div>' +
        '<div><dt>诊断</dt><dd class="' + (c.error ? 'is-bad' : dc ? 'is-warn' : '') + '">' + esc(diagnostics) + '</dd></div>' +
        (hint ? '<div><dt>性能提示</dt><dd class="is-warn">' + esc(hint) + '</dd></div>' : '') +
      '</dl>' +
      (artifacts.length ? '<button type="button" class="kc-code-action" data-kc-code="' + esc(k.name) + '">查看生成代码 <span>→</span></button>' : '') +
      '</section>';
  }

  function kernelSummary(k) {
    const kind = kindOf(k);
    if (kind === 'mem') return '最终生成合法 AIC Kernel，但片上内存已经达到平台上限。';
    if (kind === 'intent') return '编译完成，但声明的流水线意图未被完全兑现，已降级为顺序执行。';
    if (kind === 'perf') return '编译完成，没有阻塞，但编译器标记了更宽搬运 / 向量化的潜在机会。';
    return '编译完成，没有需要优先处理的阻塞或风险信号。';
  }

  function detailHTML() {
    const k = current();
    if (!k) return '<p class="kc-list-empty">没有可展示的 Kernel。</p>';
    const block = locateBlock(k.name);
    const range = block ? SRC_FILE + ':' + (block.from + 1) + '–' + (block.to + 1) : SRC_FILE;
    const kind = kindOf(k);
    const keys = keyPasses(k);

    const facts = [];
    const w = worstMem(k);
    if (w.space) facts.push([(SPACE_LABEL[w.space] || w.space) + ' ' + w.p + '%', w.p >= 70 ? 'is-warn' : '']);
    if (k.intent && k.intent.demoted) facts.push(['意图未兑现 ×' + k.intent.demoted, 'is-warn']);
    if (perfHintsOf(k)) facts.push([perfHintsOf(k) + ' 条性能提示', 'is-warn']);
    const c = diagOf(k);
    if (c.error + c.warn + c.perf) facts.push([(c.error + c.warn + c.perf) + ' 条诊断', c.error ? 'is-bad' : 'is-warn']);
    if (!facts.length) facts.push(['资源正常 · 意图兑现', '']);

    const track = [srcStageHTML(k), '<div class="kc-arrow">→</div>',
      transformationsStageHTML(k, keys, kind), '<div class="kc-arrow">→</div>',
      kernelStageHTML(k)];

    const F = FINDINGS.find(f => f.kind === kind) || FINDINGS[0];
    const nv = numericalValidation();
    const nvFirst = firstDivergentIndex(nv);
    const nextText = nv.status === 'fail' && nvFirst >= 0
      ? (PASSNAMES[nvFirst] || nv.firstDivergentPass) + ' 首次引入数值语义偏差。检查该 Pass 的 Before / After IR 与 Source mapping。'
      : F.next;

    const runtimeAction = runContext().runId !== 'run_109' && traceKernel(k.name)
      ? '<button type="button" data-kc-runtime="' + esc(k.name) + '" data-kc-finding="' + esc(kind) + '">在 Runtime 中验证影响 →</button>'
      : '';
    return '<div class="kc-dhead"><div><h3>' + esc(k.name) + '</h3>' +
      '<p>' + esc(range) + ' · 沿编译解释链理解当前 Kernel</p></div>' +
      '<div class="kc-dfacts">' + facts.map(f =>
        '<span class="' + f[1] + '">' + esc(f[0]) + '</span>').join('<i>·</i>') + '</div></div>' +
      '<div class="kc-flow">' +
        '<div class="kc-flow-top"><div class="kc-flow-title">' +
          '<b>Source → 相关编译变换 → 生成的 Kernel</b>' +
          '<small>相关编译变化按解释价值筛选，不表示严格因果溯源。点击 transformation 查看描述、收益与全局 IR diff；左右拖动查看完整链路。</small></div></div>' +
        '<div class="kc-journey-scroll" data-kc-scroll>' +
          '<div class="kc-journey-track" data-kc-track>' +
          track.join('') + '</div></div></div>' +
      '<div class="kc-next"><b>下一步</b><span>' + esc(nextText) + '</span>' +
      runtimeAction + '</div>' + artifactPanelHTML();
  }

  /* ---------- 装配 ---------- */
  function shellHTML() {
    const rows = listRows();
    const findingCount = FINDINGS.filter(f => (findingSets()[f.id] || []).length).length;
    const semanticHandoff = runContext().runId === 'run_109' && numericalValidation().status === 'fail';
    return '<section class="kc" data-kc>' +
      (semanticHandoff ? '' : summaryHTML() + contextHTML()) +
      numericalValidationHTML() +
      (semanticHandoff ? secondarySignalsHTML(findingCount) :
        '<div class="kc-sect"><div><h2>需要关注</h2>' +
          '<p>把底层编译信号转成可定位、可解释、可继续验证的发现</p></div>' +
          '<span>' + findingCount + ' 项发现</span></div>' + findingsHTML()) +
      /* 工作区两个页签：Kernel 列表 + 详情 / 编译 IR 全流程（借用 #kgTrace）。
         页签名已经写明是「Kernel 工作区」，面板头也写着「Kernel 列表」，
         所以这里不再重复一个小节标题。 */
      '<nav class="kc-tabs" role="tablist" aria-label="编译工作区">' +
        '<button type="button" role="tab" class="kc-tab' + (st.pane === 'kernel' ? ' is-on' : '') + '"' +
          ' data-kc-tab="kernel" aria-selected="' + (st.pane === 'kernel') + '">Kernel 工作区</button>' +
        '<button type="button" role="tab" class="kc-tab' + (st.pane === 'trace' ? ' is-on' : '') + '"' +
          ' data-kc-tab="trace" aria-selected="' + (st.pane === 'trace') + '">编译 IR 全流程</button>' +
      '</nav>' +
      '<section class="kc-pane" data-kc-pane="kernel">' +
        '<div class="kc-work">' +
          '<section class="kc-list-panel"><div class="kc-lhead"><b>Kernel 列表</b>' +
            '<span class="kc-filter">' +
              '<button type="button" data-kc-filter="issues" class="' + (st.filter === 'issues' ? 'is-on' : '') + '">需要关注</button>' +
              '<button type="button" data-kc-filter="all" class="' + (st.filter === 'all' ? 'is-on' : '') + '">全部 ' + K.kernels.length + '</button>' +
            '</span></div>' +
            '<div class="kc-list" data-kc-list>' + listHTML() + '</div></section>' +
          '<section class="kc-detail" data-kc-detail>' + detailHTML() + '</section>' +
        '</div>' +
      '</section>' +
      '<section class="kc-pane" data-kc-pane="trace"></section>' +
      artifactsRawHTML() +
      '</section>';
  }

  /* 页签切换只改 class，不重画 —— 页签 2 里挂的是从 kernelGuard 借来的 #kgTrace
     实体节点，重画会把它冲掉。 */
  function showPane() {
    if (!host) return;
    $$('[data-kc-pane]', host).forEach(p => p.classList.toggle('is-on', p.dataset.kcPane === st.pane));
    $$('[data-kc-tab]', host).forEach(b => {
      const on = b.dataset.kcTab === st.pane;
      b.classList.toggle('is-on', on);
      b.setAttribute('aria-selected', String(on));
    });
    if (st.pane === 'trace') attachTrace();
  }

  /* 把 compile guard 的编译 IR 全流程搬进页签 2。同一时刻只有一个宿主：
     这里 appendChild(e) 之后它就不在 #kernelGuard 里了，归还见 release()。 */
  function attachTrace() {
    const box = $('[data-kc-pane="trace"]', host);
    if (!box) return;
    const G = window.PTO_GUARD;
    const el = G && G.traceEl ? G.traceEl() : null;
    if (!el) {
      box.innerHTML = '<p class="kc-pane-missing">编译 IR 全流程需要 compile guard 的 Pass 数据，本次视图没有加载。</p>';
      return;
    }
    if (el.parentElement !== box) {
      box.innerHTML = '';              // 清掉上一次的占位文案
      box.appendChild(el);
    }
    // 两个页签看的是同一个 Kernel：切过来时把 guard 的选中态对齐，
    // activate() 会顺带把还没画过的河流图补上。
    if (G.select && st.kernel && byName(st.kernel)) G.select(st.kernel);
    else if (G.activate) G.activate();
  }

  function paint() {
    if (!host) return;
    host.innerHTML = shellHTML();
    showPane();
  }
  function paintList() {
    const box = $('[data-kc-list]', host);
    if (box) box.innerHTML = listHTML();
  }
  function paintDetail() {
    const box = $('[data-kc-detail]', host);
    if (box) box.innerHTML = detailHTML();
  }

  function focusExpandedPass(scrollBlock) {
    requestAnimationFrame(() => {
      const target = $('.kc-transform.is-expanded', host);
      if (!target) return;
      const journey = $('[data-kc-scroll]', host);
      if (journey) journey.scrollLeft = Math.max(0, target.offsetLeft - 18);
      if (scrollBlock && target.scrollIntoView) target.scrollIntoView({ block: 'center', inline: 'nearest' });
      const button = $('.kc-transform-toggle', target);
      if (button && button.focus) button.focus({ preventScroll: true });
    });
  }

  function openPass(index, opts) {
    if (!Number.isInteger(index) || index < 0 || index >= PASSNAMES.length) return false;
    opts = opts || {};
    st.pane = 'kernel';
    st.expandedPass = opts.toggle && st.expandedPass === index ? null : index;
    paintDetail();
    if (st.expandedPass != null) focusExpandedPass(!!opts.scroll);
    return true;
  }

  function pickKernel(name, opts) {
    if (!byName(name)) return;
    st.kernel = name;
    st.expandedPass = null;
    opts = opts || {};
    if (!issueSet()[name]) st.filter = 'all';
    paint();
    if (opts.scroll) {
      const on = $('.kc-item.is-on', host);
      if (on && on.scrollIntoView) on.scrollIntoView({ block: 'nearest' });
    }
  }

  function markFind(id) {
    st.findOn = st.findOn === id ? null : id;
    if (st.findOn) {
      const set = findingSets();
      const hits = set[st.findOn] || [];
      if (hits.length) {
        const top = hits.slice().sort((a, b) => worstMem(b).p - worstMem(a).p)[0];
        st.kernel = top.name;
        st.filter = issueSet()[top.name] ? st.filter : 'all';
      }
    }
    st.pane = 'kernel';          // 命中结果在列表里，先切回工作区页签
    st.expandedPass = null;
    paint();
    const box = $('.kc-work', host);
    if (box && box.scrollIntoView) box.scrollIntoView({ block: 'nearest' });
  }

  function findingIdFor(kernel, kind) {
    const found = (runContext().findings || []).find(f =>
      (f.affectedObjects || []).some(o => o.kind === 'kernel' && o.id === kernel));
    return found ? found.id : kind;
  }
  function routeRuntime(kernel, kind) {
    if (!traceKernel(kernel)) return;
    const context = {
      runId: runContext().runId || runStamp(), from: 'compilation',
      findingId: findingIdFor(kernel, kind), kernelName: kernel
    };
    window.dispatchEvent(new CustomEvent('pto:compilation-runtime', { detail: context }));
  }
  function findingKernelForRuntime(kind, fallback) {
    if (kind !== 'perf') return fallback;
    const byRunFinding = (runContext().findings || []).find(f =>
      (f.affectedObjects || []).some(o => o.kind === 'kernel' && traceKernel(o.id)));
    return (byRunFinding && byRunFinding.affectedObjects.find(o => o.kind === 'kernel').id) || fallback;
  }
  async function openArtifact(path, size, kernel) {
    const binary = /\.o$/i.test(path);
    st.artifact = { path, size, kernel: kernel || (st.artifact && st.artifact.kernel) || '', loading: !binary, content: '', error: '' };
    paintDetail();
    if (binary) return;
    try {
      const url = '../../Data/' + encodeURIComponent(K.source) + '/' + path.split('/').map(encodeURIComponent).join('/');
      const response = await fetch(url);
      if (!response.ok) throw new Error('HTTP ' + response.status);
      st.artifact = { path, size, kernel: st.artifact.kernel, loading: false, content: await response.text(), error: '' };
    } catch (error) {
      st.artifact = { path, size, kernel: st.artifact.kernel, loading: false, content: '', error: error && error.message ? error.message : 'unavailable' };
    }
    paintDetail();
  }

  function onClick(e) {
    const tb = e.target.closest('[data-kc-tab]');
    if (tb) { st.pane = tb.dataset.kcTab === 'trace' ? 'trace' : 'kernel'; showPane(); return; }
    const fid = e.target.closest('[data-kc-find]');
    if (fid) { markFind(fid.dataset.kcFind); return; }
    const runtime = e.target.closest('[data-kc-runtime]');
    if (runtime) {
      routeRuntime(findingKernelForRuntime(runtime.dataset.kcFinding, runtime.dataset.kcRuntime), runtime.dataset.kcFinding || kindOf(current()));
      return;
    }
    const kid = e.target.closest('[data-kc-kernel]');
    if (kid) { pickKernel(kid.dataset.kcKernel, { scroll: true }); return; }
    const flt = e.target.closest('[data-kc-filter]');
    if (flt) {
      st.filter = flt.dataset.kcFilter;
      $$('[data-kc-filter]', host).forEach(b => b.classList.toggle('is-on', b.dataset.kcFilter === st.filter));
      paintList();
      return;
    }
    const pass = e.target.closest('[data-kc-pass]');
    if (pass) {
      const scroll = $('[data-kc-scroll]', host);
      const left = scroll ? scroll.scrollLeft : 0;
      const index = Number(pass.dataset.kcPass);
      const fromNumericalSummary = !!pass.closest('.kc-nv');
      openPass(index, { toggle: !fromNumericalSummary, scroll: fromNumericalSummary });
      requestAnimationFrame(() => {
        const next = $('[data-kc-scroll]', host);
        if (next && !fromNumericalSummary) next.scrollLeft = left;
      });
      return;
    }
    const code = e.target.closest('[data-kc-code]');
    if (code) {
      const files = kernelArtifacts(code.dataset.kcCode);
      if (files.length) openArtifact(files[0].path, files[0].size, code.dataset.kcCode);
      return;
    }
    if (e.target.closest('[data-kc-artifact-close]')) { st.artifact = null; paintDetail(); return; }
    const artifact = e.target.closest('[data-kc-artifact]');
    if (artifact) { openArtifact(artifact.dataset.kcArtifact, Number(artifact.dataset.kcSize || 0)); return; }
    const raw = e.target.closest('[data-kc-raw-file]');
    if (raw) { openArtifact(raw.dataset.kcRawFile, 0, ''); return; }
    const rt = e.target.closest('[data-kc-route]');
    if (rt) {
      const btn = document.querySelector('[data-th-tab="' + rt.dataset.kcRoute + '"]');
      if (btn) btn.click();
      return;
    }
  }

  function focusKnownDivergenceOnce() {
    const nv = numericalValidation();
    const index = firstDivergentIndex(nv);
    const key = [runContext().runId || '', nv.status, index].join(':');
    if (index < 0 || st.numericalFocusKey === key) return;
    st.numericalFocusKey = key;
    st.pane = 'kernel';
    st.expandedPass = index;
  }

  function applyNumericalContext(validation, entry) {
    st.numericalOverride = validation || null;
    st.entryContext = entry || null;
    st.numericalFocusKey = null;
    focusKnownDivergenceOnce();
    if (host) {
      paint();
      if (st.expandedPass != null) focusExpandedPass(true);
    }
    return numericalValidation();
  }

  /* ---------- 对外 ---------- */
  window.PTO_COMPILATION = {
    /* root 由 task-history 传入（#runTabPanel），事件用委托，重复 render 不会叠加监听 */
    render(root) {
      host = root;
      if (!host) return false;
      if (!st.kernel || !byName(st.kernel)) {
        const rows = listRows();
        st.kernel = rows.length ? rows[0].name : (kernelNames()[0] || null);
      }
      focusKnownDivergenceOnce();
      if (!host.dataset.kcBound) {
        host.addEventListener('click', onClick);
        host.dataset.kcBound = '1';
      }
      paint();
      if (runContext().runId === 'run_109' && st.expandedPass != null) focusExpandedPass(true);
      return true;
    },
    /* Execution 页签的「在 Compilation 查看」落到这里 */
    selectKernel(name) {
      if (!host) return false;
      pickKernel(String(name), { scroll: true });
      return true;
    },
    selectPass(name) {
      if (!host) return false;
      const index = PASSNAMES.indexOf(String(name));
      if (index < 0) return false;
      openPass(index, { toggle: false, scroll: true });
      return true;
    },
    /* Correctness 可将它已有的 compiler evidence 原样传入。若已知首个
       divergence，会在现有 transformation / IR diff 中自动展开，不创建新 viewer。 */
    setNumericalContext(validation, entry) {
      return applyNumericalContext(validation, entry);
    },
    clearNumericalContext() {
      return applyNumericalContext(null, null);
    },
    useNumericalFixture(name, options) {
      const fixture = NUMERICAL_FIXTURES[name];
      if (!fixture) return false;
      const opts = options || {};
      applyNumericalContext(fixture, opts.fromCorrectness === false ? null : {
        from: 'correctness',
        finding: opts.finding || '数值精度 · 输出不一致',
        intent: opts.intent || '正在定位编译语义分歧'
      });
      return true;
    },
    numericalFixtures: NUMERICAL_FIXTURES,
    /* task-history 的 syncPanel() 靠它判断这块面板现在归谁：是本视图自绘的
       .kc，还是从 stage 2 搬来的 DOM。判错就会把新视图冲掉。 */
    owns(node) { return !!node && host === node; },
    /* task-history 在重写 #runTabPanel 之前必须先调这个：页签 2 里挂的
       #kgTrace 是从 kernelGuard 借来的实体节点，被 innerHTML 冲掉就成了游离
       节点，归还后再也长不回 stage 2 的 Kernel Guard 里。 */
    release() {
      if (window.PTO_GUARD && window.PTO_GUARD.traceHome) window.PTO_GUARD.traceHome();
      host = null;
    },
    /* 供调试与验收用：当前状态快照 */
    state() {
      return { kernel: st.kernel, filter: st.filter, findOn: st.findOn,
        pane: st.pane, expandedPass: st.expandedPass,
        numericalValidation: numericalValidation(), entryContext: compilationEntryContext(),
        keyPasses: (current() ? keyPasses(current()).map(p => p.name) : []) };
    },
    ready: true
  };
})();
