/* Compile guard — kernel-centric view (stage 2).
   Primary axis is the developer's kernels, not the compiler's passes.
   Data: window.PTO_IR_KERNELS (real) + window.PTO_IR_PIPELINE (pass metadata). */
(function () {
  'use strict';

  const K = window.PTO_IR_KERNELS;
  const PIPE = window.PTO_IR_PIPELINE;
  if (!K) return;

  const LIMITS = K.limits;
  const PASSNAMES = K.passNames;
  const PASSMETA = PIPE ? PIPE.passes : [];
  const STRATA = PIPE ? PIPE.strata : [];

  const TYPE_C = {
    AIC: '#CE5622', AIV: '#E09258', Group: '#BC7440',
    Orchestration: '#4E7C90', Spmd: '#6E8288', InCore: '#A2814F', Unknown: '#7E8A90'
  };
  const SPACE_LABEL = { Vec: 'UB (Vec)', Mat: 'L1 (Mat)', Acc: 'L0C (Acc)', Left: 'L0A (Left)', Right: 'L0B (Right)' };

  // The four dimensions a kernel developer actually decides on.
  const DIMS = {
    mem:    { label: '内存水位', hint: '片上 buffer 峰值 / 平台上限 —— 超限直接编译失败' },
    intent: { label: '意图兑现', hint: '我声明的 pipeline / split / 切分，编译器兑现了吗' },
    diag:   { label: '诊断',     hint: 'Error / Warning / PerfHint —— 唯一能直接导向改代码的维度' },
    gain:   { label: '优化收益', hint: 'MemoryReuse 为这个 kernel 省下多少片上空间' }
  };

  const st = { dim: 'mem', sel: null, pass: null, kpass: null, fact: 0, onlyIssues: false };

  const $ = (s, r) => (r || document).querySelector(s);
  // Quotes must be escaped too: this output also lands in HTML attributes
  // (data-kg-tip, aria-label), and text like attrs["arg_directions"] would
  // otherwise close the attribute early and truncate the value.
  const esc = (s) => String(s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const kb = (b) => b >= 1024 ? (b / 1024).toFixed(b >= 10240 ? 0 : 1) + 'KB' : b + 'B';
  const pct = (a, b) => b ? Math.round((a / b) * 100) : 0;

  /* ---------- derived per-kernel scores ---------- */
  function ubPct(k) { return pct(k.mem.Vec || 0, LIMITS.Vec); }
  function worstPct(k) {
    let w = 0, sp = null;
    for (const s of K.spaces) {
      const p = pct(k.mem[s] || 0, LIMITS[s]);
      if (p > w) { w = p; sp = s; }
    }
    return { p: w, space: sp };
  }
  function diagCount(k) {
    const c = { error: 0, warn: 0, perf: 0 };
    k.diags.forEach(d => { c[d.sev] = (c[d.sev] || 0) + 1; });
    return c;
  }
  function gainPct(k) {
    const b = k.reuse.before.b;
    return b ? Math.round((1 - k.reuse.after.b / b) * 100) : 0;
  }
  function intentIssues(k) {
    const out = [];
    if (k.intent.demoted > 0) out.push({ bad: true, t: 'pipeline 降级 ×' + k.intent.demoted });
    if (k.intent.declared.length) out.push({ bad: false, t: 'pipeline ×' + k.intent.declared.length });
    if (k.intent.l0.length) out.push({ bad: false, t: 'L0 切分 ×' + k.intent.l0.length });
    if (k.split) out.push({ bad: false, t: 'split ' + k.split });
    return out;
  }
  function score(k) {
    if (st.dim === 'mem') return worstPct(k).p;
    if (st.dim === 'gain') return gainPct(k);
    if (st.dim === 'diag') { const c = diagCount(k); return c.error * 1000 + c.warn * 100 + c.perf; }
    return k.intent.demoted * 1000 + k.intent.l0.length * 10 + k.intent.declared.length;
  }
  function hasIssue(k) {
    const c = diagCount(k);
    return c.error > 0 || c.warn > 0 || c.perf > 0 || k.intent.demoted > 0 || worstPct(k).p >= 70;
  }

  let els = null;

  /* ---------- mount ---------- */
  function mount() {
    const anchor = $('#passStrip');
    if (!anchor || !anchor.parentNode) return false;

    const root = document.createElement('div');
    root.className = 'kf-kg';
    root.id = 'kernelGuard';
    // Global facts first (summary → operator trace), then the list with its
    // own controls. The sort/filter bar belongs to the list, not to the
    // whole-operator numbers above it.
    root.innerHTML =
      '<div class="kf-kg-summary" id="kgSummary"></div>' +
      '<div class="kf-kg-lineage" id="kgLineage"></div>' +
      '<div class="kf-kg-trace" id="kgTrace"></div>' +
      '<div class="kf-kg-bar">' +
        '<div class="kf-kg-seg" id="kgDims" role="group" aria-label="关注维度">' +
          Object.keys(DIMS).map(d =>
            '<button type="button" data-kg-dim="' + d + '" aria-pressed="' + (d === st.dim) + '"' +
            ' title="' + esc(DIMS[d].hint) + '">' + DIMS[d].label + '</button>').join('') +
        '</div>' +
        '<button class="kf-kg-toggle" id="kgIssues" type="button" aria-pressed="false"><i></i>只看有问题的</button>' +
        '<span class="kf-kg-spacer"></span>' +
        '<span class="kf-kg-target">' + esc(K.target) + '</span>' +
      '</div>' +
      '<p class="kf-kg-hint" id="kgHint"></p>' +
      '<div class="kf-kg-list" id="kgList" aria-label="Kernel 列表"></div>';

    anchor.parentNode.insertBefore(root, anchor.nextSibling);

    const tip = document.createElement('div');
    tip.className = 'kf-kg-tip';
    document.body.appendChild(tip);

    els = {
      root, tip, list: $('#kgList', root), trace: $('#kgTrace', root),
      dims: $('#kgDims', root), issues: $('#kgIssues', root),
      summary: $('#kgSummary', root), hint: $('#kgHint', root),
      lineage: $('#kgLineage', root)
    };
    return true;
  }

  /* ---------- summary ---------- */
  function renderSummary() {
    const withMem = K.kernels.filter(k => Object.keys(k.mem).length);
    const peak = K.kernels.reduce((a, k) => Math.max(a, worstPct(k).p), 0);
    const peakK = K.kernels.find(k => worstPct(k).p === peak);
    let e = 0, w = 0, p = 0;
    K.kernels.forEach(k => { const c = diagCount(k); e += c.error; w += c.warn; p += c.perf; });
    const tb = K.kernels.reduce((a, k) => a + k.reuse.before.b, 0);
    const ta = K.kernels.reduce((a, k) => a + k.reuse.after.b, 0);
    const demoted = K.kernels.filter(k => k.intent.demoted > 0).length;

    els.summary.innerHTML =
      card('Kernel', K.kernels.length, withMem.length + ' 个占用片上', e ? '' : 'ok') +
      card('峰值水位', peak + '%', peakK ? peakK.name : '', peak >= 90 ? 'bad' : peak >= 70 ? 'warn' : 'ok') +
      card('诊断', (e ? e + 'E ' : '') + (w ? w + 'W ' : '') + p + 'P', e ? '有阻塞' : '无阻塞', e ? 'bad' : w ? 'warn' : 'ok') +
      card('意图未兑现', demoted, demoted ? 'pipeline 被降级' : '全部兑现', demoted ? 'warn' : 'ok') +
      card('复用收益', (tb ? Math.round((1 - ta / tb) * 100) : 0) + '%', kb(tb) + ' → ' + kb(ta), 'ok');
  }
  const card = (t, v, s, tone) =>
    '<div class="kf-kg-card' + (tone ? ' is-' + tone : '') + '"><dt>' + t + '</dt><dd>' + v + '</dd><small>' + esc(s) + '</small></div>';

  /* ---------- kernel list ---------- */
  function renderList() {
    els.hint.textContent = DIMS[st.dim].hint;

    let rows = K.kernels.slice();
    if (st.onlyIssues) rows = rows.filter(hasIssue);
    rows.sort((a, b) => score(b) - score(a) || a.name.localeCompare(b.name));

    if (!rows.length) { els.list.innerHTML = '<p class="kf-kg-empty">没有命中的 kernel。</p>'; return; }

    els.list.innerHTML = rows.map(k => {
      const w = worstPct(k);
      const c = diagCount(k);
      const g = gainPct(k);
      const tone = w.p >= 90 ? 'bad' : w.p >= 70 ? 'warn' : 'ok';

      const badges =
        (c.error ? '<span class="kf-kg-badge is-error">' + c.error + ' E</span>' : '') +
        (c.warn ? '<span class="kf-kg-badge is-warn">' + c.warn + ' W</span>' : '') +
        (c.perf ? '<span class="kf-kg-badge is-perf">' + c.perf + ' P</span>' : '') +
        (k.intent.demoted ? '<span class="kf-kg-badge is-warn">降级</span>' : '');

      // the metric shown on the right follows the active dimension
      const right =
        st.dim === 'gain' ? (k.reuse.before.b ? '省 ' + g + '%' : '—')
        : st.dim === 'diag' ? (c.error + c.warn + c.perf) + ' 条'
        : st.dim === 'intent' ? (intentIssues(k)[0] ? intentIssues(k)[0].t : '—')
        : (w.space ? kb(k.mem[w.space]) + ' / ' + kb(LIMITS[w.space]) : '—');

      const open = k.name === st.sel;
      return '<div class="kf-kg-item' + (open ? ' is-open' : '') + '">' +
        '<button type="button" class="kf-kg-row' + (open ? ' is-sel' : '') + '"' +
        ' data-kg-k="' + esc(k.name) + '" aria-expanded="' + open + '">' +
        '<span class="kf-kg-type" style="--kg-c:' + (TYPE_C[k.type] || TYPE_C.Unknown) + '">' + k.type.slice(0, 3).toUpperCase() + '</span>' +
        '<span class="kf-kg-name">' + esc(k.name) + '</span>' +
        '<span class="kf-kg-badges">' + badges + '</span>' +
        '<span class="kf-kg-metric">' + esc(right) + '</span>' +
        '<span class="kf-kg-gauge is-' + tone + '"><i style="width:' + Math.min(w.p, 100) + '%"></i>' +
          '<b style="left:' + Math.min(w.p, 100) + '%"></b></span>' +
        '<span class="kf-kg-pctv">' + (w.p || 0) + '%</span>' +
      '</button>' +
      (open ? detailHtml(k) : '') +
      '</div>';
    }).join('');
  }

  /* ---------- detail (rendered inline, under the selected row) ---------- */
  function detailHtml(k) {
    /* ① memory */
    const memRows = K.spaces.filter(s => k.mem[s]).map(s => {
      const p = pct(k.mem[s], LIMITS[s]);
      const tone = p >= 90 ? 'bad' : p >= 70 ? 'warn' : 'ok';
      return '<div class="kf-kg-mem is-' + tone + '">' +
        '<span class="kf-kg-mem-l">' + SPACE_LABEL[s] + '</span>' +
        '<span class="kf-kg-mem-bar"><i style="width:' + Math.min(p, 100) + '%"></i></span>' +
        '<span class="kf-kg-mem-v">' + kb(k.mem[s]) + ' / ' + kb(LIMITS[s]) + '<em>' + p + '%</em></span>' +
      '</div>';
    }).join('') || '<p class="kf-kg-none">该 kernel 没有片上分配（编排 / SPMD 层）。</p>';

    /* ② intent */
    const it = k.intent;
    let intentRows = '';
    if (it.declared.length) {
      const ok = it.declared.length - it.demoted;
      intentRows += intentRow(
        '声明了 ' + it.declared.length + ' 个 pl.pipeline（stage ' + [...new Set(it.declared)].join('/') + '）',
        it.demoted ? it.demoted + ' 个被 SkewCrossCorePipeline 降级为 Sequential，' + ok + ' 个按声明展开'
                   : '全部按声明展开为多级流水',
        it.demoted ? 'warn' : 'ok');
    }
    if (it.l0.length) {
      const t = it.l0[0];
      intentRows += intentRow('matmul 自动 L0 切分 ×' + it.l0.length,
        'K 轴 ' + t.to + ' 按步长 ' + t.step + ' 切成 ' + Math.ceil((t.to - t.from) / t.step) + ' 段，stage=' + t.stage + ' 做 ping-pong', 'ok');
    }
    if (k.split) intentRows += intentRow('声明了 pl.split(' + k.split + ')', 'AIV 双核切分已生效，tpush/tpop 带 split 标记', 'ok');
    if (!intentRows) intentRows = '<p class="kf-kg-none">该 kernel 没有流水 / 切分 / split 声明。</p>';

    /* ③ diagnostics */
    const diagRows = k.diags.length ? k.diags.map(d =>
      '<div class="kf-kg-diag is-' + d.sev + '">' +
        '<span class="kf-kg-diag-i">' + ({ error: '✕', warn: '!', perf: '⚡' }[d.sev] || '·') + '</span>' +
        '<div><b>' + d.code + (d.mock ? '<em class="kf-kg-mock">MOCK</em>' : '') + '</b>' +
        '<small>' + esc(d.msg) + '</small></div>' +
      '</div>').join('') : '<p class="kf-kg-none">无诊断。</p>';

    /* ④ gain */
    const g = gainPct(k);
    const gainBlock = k.reuse.before.b
      ? '<div class="kf-kg-gain">' +
          '<div><dt>alloc 条数</dt><dd>' + k.reuse.before.n + ' → ' + k.reuse.after.n + '</dd></div>' +
          '<div><dt>片上字节</dt><dd>' + kb(k.reuse.before.b) + ' → ' + kb(k.reuse.after.b) + '</dd></div>' +
          '<div class="' + (g > 0 ? 'is-ok' : 'is-warn') + '"><dt>MemoryReuse</dt><dd>' + (g > 0 ? '省 ' + g + '%' : '未省') + '</dd></div>' +
        '</div>' + (g === 0 ? '<p class="kf-kg-none">复用没有生效 —— 可能本就无重叠机会，也可能写法挡住了合并，值得看一眼。</p>' : '')
      : '<p class="kf-kg-none">无片上分配，不涉及复用。</p>';

    return '<div class="kf-kg-panel">' +
      sec('① 内存水位', memRows) +
      sec('② 意图兑现', intentRows) +
      sec('③ 诊断', diagRows) +
      sec('④ 优化收益', gainBlock) +
      sec('这个 kernel 的编译轨迹', kernelStrip(k)) +
    '</div>';
  }
  const sec = (t, body) => '<section class="kf-kg-sec"><h4>' + t + '</h4>' + body + '</section>';
  const intentRow = (t, s, tone) =>
    '<div class="kf-kg-intent is-' + tone + '"><b>' + esc(t) + '</b><small>' + esc(s) + '</small></div>';

  /* ---------- compile trace ----------
     A kernel's trace is a sequence of TYPED EVENTS, not a magnitude series —
     "改写了 2016 行" is almost all indentation, "改写了 88 行" is a codegen
     prerequisite, so line count is the wrong encoding. Milestones below carry
     a real consequence; everything else is a plain touch. */

  // Passes with a developer-visible consequence, keyed by pass name so the map
  // survives pipeline reordering. `f` returns null when it didn't apply here.
  const MILESTONES = {
    OutlineIncoreScopes:    () => ['struct', '从 scope 外提成独立 kernel'],
    OutlineClusterScopes:   () => ['struct', '外提成 SPMD 分发函数'],
    ConvertTensorToTileOps: () => ['struct', '张量算子 → tile 算子，值类型换成 TileType'],
    FlattenTileNdTo2D:      () => ['struct', 'tile 压到 2D'],
    AutoTileMatmulL0: (k) => k.intent.l0.length
      ? ['intent', 'matmul 按 L0 容量切分：K 轴 ' + k.intent.l0[0].to + ' / 步长 ' + k.intent.l0[0].step +
                   ' → ' + Math.ceil((k.intent.l0[0].to - k.intent.l0[0].from) / k.intent.l0[0].step) + ' 段'] : null,
    ExpandMixedKernel:      () => ['struct', 'Cube / Vector 拆成两个 kernel，靠 tpush / tpop 通信'],
    SplitVectorKernel: (k) => k.split ? ['intent', 'split ' + k.split + ' 标记落到 tpush / tpop'] : null,
    SkewCrossCorePipeline: (k) => k.intent.demoted
      ? ['bad', '声明的 pl.pipeline 有 ' + k.intent.demoted + ' 个被降级为 Sequential']
      : ['intent', '跨核流水错位为 prologue / steady / epilogue'],
    LowerPipelineLoops:     () => ['intent', 'pipeline 按 stage 展开成多份克隆，供 ping-pong'],
    InitMemRef: (k) => ['mem', '绑定片上 buffer：' + k.reuse.before.n + ' 块 / ' + kb(k.reuse.before.b)],
    MemoryReuse: (k) => k.reuse.before.b
      ? [gainPct(k) > 0 ? 'mem' : 'bad',
         gainPct(k) > 0
           ? '生命周期复用合并到 ' + k.reuse.after.n + ' 块 / ' + kb(k.reuse.after.b) + '，省 ' + gainPct(k) + '%'
           : '没有可合并的 buffer，复用未生效'] : null,
    AllocateMemoryAddr: (k) => {
      const w = worstPct(k);
      return w.space ? ['mem', '分配物理地址，峰值 ' + SPACE_LABEL[w.space] + ' ' + w.p + '%'] : null;
    },
    DeriveCallDirections:     () => ['rt', '推导调用参数方向，供运行时依赖跟踪'],
    MaterializeRuntimeScopes: () => ['rt', '插入显式 PTO2_SCOPE'],
    InlineFunctions:          () => ['struct', '内联函数体拼接进调用点']
  };

  function eventsOf(k) {
    return k.passes.map(p => {
      if (p.st === 'absent') return { i: p.i, kind: 'absent' };
      if (p.st === 'same') return { i: p.i, kind: 'same' };
      if (p.st === 'born') {
        const m = MILESTONES[PASSNAMES[p.i]];
        const r = m && m(k);
        return { i: p.i, kind: 'birth', text: r ? r[1] : '此 kernel 在这一步诞生', milestone: true };
      }
      const m = MILESTONES[PASSNAMES[p.i]];
      const r = m && m(k);
      if (r) return { i: p.i, kind: r[0], text: r[1], milestone: true };
      return { i: p.i, kind: 'touch', text: '改写了这个 kernel' };
    });
  }

  /* ---------- operator-level compile trace ----------
     The trace sits above the kernel list, so it describes the WHOLE operator's
     journey through the pipeline — 45 kernels are an output of that journey,
     not its subject. Per-kernel traces live inside each expanded row. */

  const bornAt = (i) => K.kernels.filter(k => k.passes[i] && k.passes[i].st === 'born').length;
  const touchedAt = (i) => K.kernels.filter(k => k.passes[i] && (k.passes[i].st === 'changed' || k.passes[i].st === 'born')).length;

  // Operator-level milestones, keyed by pass name. Numbers are aggregated from
  // the real per-kernel data or from the global per-pass metrics in PIPE.
  const OP_MILESTONES = {
    InlineFunctions:        () => ['struct', 'Inline 函数体拼接进每个调用点，此后无 Inline 函数'],
    UnrollLoops:            () => ['struct', '展开 Unroll 循环，循环变量替换为常量'],
    ConvertToSSA:           () => ['struct', '转入 SSA，循环进位量变成显式 iter_arg / yield 链'],
    FlattenCallExpr:        () => ['struct', '摊平嵌套调用，一句一调用'],
    OutlineIncoreScopes:    () => ['struct', bornAt(11) + ' 个 InCore scope 外提成独立 kernel —— 程序从一棵树变成调用图'],
    OutlineClusterScopes:   () => ['struct', bornAt(12) + ' 个 Cluster scope 外提成 SPMD 分发函数'],
    ConvertTensorToTileOps: (P) => ['struct', '整套算子词汇替换：tensor ' + P[12].o.te + '→' + P[13].o.te +
                                              '，tile ' + P[12].o.ti + '→' + P[13].o.ti],
    FlattenTileNdTo2D:      () => ['struct', 'ND tile 压到 2D'],
    AutoTileMatmulL0:       () => ['intent', K.kernels.filter(k => k.intent.l0.length).length + ' 个 kernel 的 matmul 按 L0 容量切成 K 循环'],
    ExpandMixedKernel:      (P) => ['struct', '混合 InCore 裂成 ' + P[22].f.aic + ' AIC + ' + P[22].f.aiv + ' AIV + ' +
                                              P[22].f.group + ' Group，靠 tpush / tpop 通信'],
    SplitVectorKernel:      () => ['intent', K.kernels.filter(k => k.split).length + ' 个 AIV kernel 落上 split 标记'],
    SkewCrossCorePipeline:  () => {
      const n = K.kernels.filter(k => k.intent.demoted > 0).length;
      return n ? ['bad', n + ' 个 kernel 声明的 pl.pipeline 被降级为 Sequential']
               : ['intent', '跨核流水错位为 prologue / steady / epilogue'];
    },
    LowerPipelineLoops:     (P) => ['intent', 'pipeline 按 stage 展开成克隆，tile 算子 ' + P[26].o.ti + '→' + P[27].o.ti],
    CanonicalizeIOOrder:    () => ['struct', 'SeqStmts 内拓扑重排为 [标量… load… 计算… store…]'],
    InitMemRef:             (P) => ['mem', '绑定 ' + P[30].o.mr + ' 个 MemRef 并插入 tile.alloc；同时作废 SSA'],
    MemoryReuse:            () => {
      const tb = K.kernels.reduce((a, k) => a + k.reuse.before.b, 0);
      const ta = K.kernels.reduce((a, k) => a + k.reuse.after.b, 0);
      const nb = K.kernels.reduce((a, k) => a + k.reuse.before.n, 0);
      const na = K.kernels.reduce((a, k) => a + k.reuse.after.n, 0);
      return ['mem', 'alloc ' + nb + '→' + na + ' 条，' + kb(tb) + '→' + kb(ta) +
                     '，省 ' + Math.round((1 - ta / tb) * 100) + '%'];
    },
    AllocateMemoryAddr:     () => {
      const peak = K.kernels.reduce((a, k) => Math.max(a, worstPct(k).p), 0);
      const pk = K.kernels.find(k => worstPct(k).p === peak);
      return [peak >= 90 ? 'bad' : 'mem', '分配物理地址；峰值 ' + peak + '%（' + (pk ? pk.name : '') + ' ' +
              (pk && worstPct(pk).space ? SPACE_LABEL[worstPct(pk).space] : '') + '）'];
    },
    DeriveCallDirections:     () => ['rt', '推导每个调用的实参方向，写进 attrs["arg_directions"]'],
    MaterializeRuntimeScopes: () => ['rt', '编排函数体与 for / if 分支插入显式 RuntimeScopeStmt']
  };

  function opEvents() {
    const P = PASSMETA;
    return P.map(p => {
      const changed = p.d === null ? true : p.d > 0;
      if (!changed) return { i: p.i, kind: 'same', touched: 0 };
      const m = OP_MILESTONES[p.name];
      const r = m && m(P);
      const touched = touchedAt(p.i);
      if (r) return { i: p.i, kind: r[0], text: r[1], touched, milestone: true };
      return { i: p.i, kind: 'touch', text: p.desc || '改写了 IR', touched };
    });
  }

  const KIND_LABEL = {
    struct: '结构变换', intent: '意图相关', bad: '意图被破坏',
    mem: '内存', rt: '运行时', touch: '普通改动',
    same: '未改动', birth: '诞生', absent: '尚不存在'
  };
  const MILESTONE_KINDS = ['birth', 'struct', 'intent', 'bad', 'mem', 'rt'];

  function strataBands() {
    return STRATA.map(s => {
      const n = s.to - s.from + 1;
      return '<span style="--kg-c: var(--irp-' + s.id.toLowerCase() + '); flex:' + n + ' 1 0">' +
        (n > 3 ? s.id + ' ' + s.name : n > 1 ? s.id : '') + '</span>';
    }).join('');
  }

  function legendFor(ev) {
    const present = [];
    ['struct', 'intent', 'bad', 'mem', 'rt', 'touch', 'same', 'birth', 'absent']
      .forEach(kd => { if (ev.some(e => e.kind === kd)) present.push(kd); });
    return '<div class="kf-kg-legend">' + present.map(kd => {
      const s = ev.find(e => e.kind === kd);
      const strat = STRATA.find(x => s.i >= x.from && s.i <= x.to);
      return '<span><i class="is-' + kd + (MILESTONE_KINDS.indexOf(kd) >= 0 ? ' is-milestone' : '') +
        '" style="--kg-c: var(--irp-' + (strat ? strat.id.toLowerCase() : 's3') + ')"></i>' + KIND_LABEL[kd] + '</span>';
    }).join('') + '</div>';
  }

  /* ---------- kernel lineage ----------------------------------------------
     "Where did these 45 functions come from" is a cross-pass question: no
     single pass visual can answer it. Only 5 of the 42 passes change the
     function inventory; everything else rewrites bodies. Two curves, sharing
     the operator trace's 42-column geometry so the two read as one instrument:
       函数      how many functions exist        2 → 1 → 39 → 43 → 45
       待外提作用域  named scopes not yet outlined   27 → 38 → 4 → 0
     Both come from K.lineage, diffed off the real dumps. */
  const LIN = K.lineage || [];
  const scopesOf = (l) => (l.at || 0) + (l.spmd || 0);
  const LIN_EVENTS = LIN.filter((l, i) => {
    if (!i) return true;
    const p = LIN[i - 1];
    return l.born.length || l.gone.length || scopesOf(l) !== scopesOf(p);
  });
  const LIN_WHY = {
    0:  '源码里 23 处 pl.at + 4 处 pl.spmd，每处一个 name_hint',
    1:  '_decode_layer 被内联进 decode_fwd_layers，函数数反而先减少',
    2:  'pl.unroll 展平循环，同一个 name_hint 被复制成多份实例',
    11: '每个 pl.at 作用域外提成独立函数，签名由编译器推导',
    12: '每处 pl.spmd 再生成一个 launcher 包装函数',
    22: 'pl.split(UP_DOWN) 把混合核拆成 Cube / Vector 两个 kernel'
  };

  function renderLineage() {
    if (!els.lineage || !LIN.length) return;
    const maxF = Math.max(...LIN.map(l => l.n));
    const maxS = Math.max(...LIN.map(scopesOf));
    const evIdx = new Set(LIN_EVENTS.map(l => l.i));

    const bars = (pick, max, cls) => LIN.map(l => {
      const v = pick(l);
      return '<i class="' + cls + (evIdx.has(l.i) ? ' is-ev' : '') + '"' +
        ' style="height:' + (max ? Math.max(v ? 8 : 0, v / max * 100) : 0) + '%"' +
        ' data-kg-lin="' + l.i + '"' +
        ' title="' + esc(String(l.i).padStart(2, '0') + ' ' + PASSNAMES[l.i] + ' · ' + v) + '"></i>';
    }).join('');

    const chips = LIN_EVENTS.map(l => {
      const prev = l.i ? LIN[l.i - 1] : null;
      const dF = prev ? l.n - prev.n : l.n;
      const dS = prev ? scopesOf(l) - scopesOf(prev) : scopesOf(l);
      const delta = [
        dF ? (dF > 0 ? '+' : '') + dF + ' 函数' : '',
        dS ? (dS > 0 ? '+' : '') + dS + ' 作用域' : ''
      ].filter(Boolean).join(' · ');
      return '<button type="button" class="kf-kg-linchip' + (st.pass === l.i ? ' is-sel' : '') + '"' +
        ' data-kg-lin="' + l.i + '">' +
        '<em>' + String(l.i).padStart(2, '0') + '</em>' +
        '<b>' + esc(PASSNAMES[l.i]) + '</b>' +
        '<span>' + esc(delta || '—') + '</span>' +
      '</button>';
    }).join('');

    // detail for whichever event pass is currently selected in the trace
    const cur = LIN_EVENTS.find(l => l.i === st.pass);
    let detail = '';
    if (cur) {
      const prev = cur.i ? LIN[cur.i - 1] : null;
      const names = (arr, label, cls) => arr.length
        ? '<div class="kf-kg-linlist"><h6 class="' + cls + '">' + label + ' · ' + arr.length + '</h6>' +
          '<div>' + arr.map(n => '<code>' + esc(n) + '</code>').join('') + '</div></div>'
        : '';
      detail =
        '<div class="kf-kg-lindet">' +
          '<p>' + esc(LIN_WHY[cur.i] || '') + '</p>' +
          '<div class="kf-kg-linnum">' +
            '<span>函数 <b>' + (prev ? prev.n + ' → ' : '') + cur.n + '</b></span>' +
            '<span>待外提作用域 <b>' +
              (prev ? scopesOf(prev) + ' → ' : '') + scopesOf(cur) + '</b></span>' +
          '</div>' +
          names(cur.born, '新诞生', 'is-born') +
          names(cur.gone, '消失', 'is-gone') +
        '</div>';
    }

    els.lineage.innerHTML =
      '<div class="kf-kg-thead">' +
        '<h4>KERNEL 诞生谱系 · <b>' + scopesOf(LIN[0]) + ' 个命名作用域 → ' +
          LIN[LIN.length - 1].n + ' 个函数</b></h4>' +
        '<span class="kf-kg-tmeta">00 是起点，其后 ' + (LIN_EVENTS.length - 1) +
          ' 步动过函数构成，其余 ' + (LIN.length - LIN_EVENTS.length) +
          ' 个 pass 只改写函数体</span>' +
      '</div>' +
      '<div class="kf-kg-linlegend">' +
        '<span><i class="is-fn"></i>函数数 · 峰值 ' + maxF + '</span>' +
        '<span><i class="is-sc"></i>待外提作用域 · 峰值 ' + maxS + '</span>' +
        '<span class="kf-kg-linnote">柱子与下方 pass 轨迹逐列对齐</span>' +
      '</div>' +
      '<div class="kf-kg-lintrack">' +
        '<div class="kf-kg-linrow">' + bars(l => l.n, maxF, 'is-fn') + '</div>' +
        '<div class="kf-kg-linrow">' + bars(scopesOf, maxS, 'is-sc') + '</div>' +
      '</div>' +
      '<div class="kf-kg-linchips">' + chips + '</div>' +
      detail;
  }

  function renderTrace() {
    const ev = opEvents();
    const miles = ev.filter(e => e.milestone);

    const slots = ev.map(e => {
      const s = STRATA.find(x => e.i >= x.from && e.i <= x.to);
      const what = e.kind === 'same' ? '未改动'
        : e.text + (e.touched ? '\n影响 ' + e.touched + ' 个 kernel' : '');
      return '<button type="button" class="kf-kg-slot is-' + e.kind + (e.milestone ? ' is-milestone' : '') +
        (e.i === st.pass ? ' is-sel' : '') + '"' +
        ' data-kg-p="' + e.i + '" data-kg-tip="' + esc(String(e.i).padStart(2, '0') + ' ' + PASSNAMES[e.i] + '|' + what) + '"' +
        ' style="--kg-c: var(--irp-' + (s ? s.id.toLowerCase() : 's0') + ')"' +
        ' aria-label="' + esc(PASSNAMES[e.i] + ' — ' + what) + '">' +
        '<i></i><em>' + String(e.i).padStart(2, '0') + '</em></button>';
    }).join('');

    // Selected pass → operator-level detail from the global per-pass record.
    const p = st.pass !== null ? PASSMETA[st.pass] : null;
    let detail = '';
    if (p) {
      const e = ev[st.pass];
      const prev = st.pass > 0 ? PASSMETA[st.pass - 1] : null;
      const strat = STRATA.find(x => p.i >= x.from && p.i <= x.to);
      const props =
        p.gain.map(x => '<span class="kf-kg-prop">+' + x + '</span>').join('') +
        p.lose.map(x => '<span class="kf-kg-prop is-lost">−' + x + '</span>').join('') ||
        '<span class="kf-kg-prop is-none">不声明任何 produced / invalidated 性质</span>';
      const dl = (a, b) => (prev && b !== undefined && a - b !== 0 ? (a - b > 0 ? '+' : '') + (a - b) : '±0');
      const hk = p.hunks && p.hunks[0];

      detail =
        '<div class="kf-kg-pdetail">' +
          '<div class="kf-kg-pdhead">' +
            '<span class="kf-kg-pdi" style="background: var(--irp-' + (strat ? strat.id.toLowerCase() : 's0') + ')">' +
              String(p.i).padStart(2, '0') + '</span>' +
            '<b>' + esc(p.name) + '</b>' +
            '<span class="kf-kg-dtag">' + esc(p.c) + '</span>' +
            '<span class="kf-kg-pdk">' + (e && e.touched ? '影响 ' + e.touched + ' / ' + K.kernels.length + ' 个 kernel' : '未改动 IR') + '</span>' +
          '</div>' +
          '<p class="kf-kg-pdesc">' + esc(p.desc || '') + '</p>' +
          '<div class="kf-kg-pstats">' +
            stat('IR 行数', p.l, dl(p.l, prev && prev.l)) +
            stat('函数', Object.keys(p.f).reduce((a, x) => a + p.f[x], 0),
                 dl(Object.keys(p.f).reduce((a, x) => a + p.f[x], 0),
                    prev && Object.keys(prev.f).reduce((a, x) => a + prev.f[x], 0))) +
            stat('tensor 算子', p.o.te, dl(p.o.te, prev && prev.o.te)) +
            stat('tile 算子', p.o.ti, dl(p.o.ti, prev && prev.o.ti)) +
            stat('MemRef', p.o.mr, dl(p.o.mr, prev && prev.o.mr)) +
          '</div>' +
          '<div class="kf-kg-props">' + props + '</div>' +
          visualFor(p) +
          (hk ? '<details class="kf-kg-raw"><summary>查看这一步的 IR diff</summary>' +
                  '<div class="kf-kg-hunk"><div class="kf-kg-hunk-h">整网 IR diff · @ line ' + hk.at + '</div>' +
                    (hk.b.length ? '<pre class="is-before">' + esc(hk.b.join('\n')) + '</pre>' : '') +
                    (hk.a.length ? '<pre class="is-after">' + esc(hk.a.join('\n')) + '</pre>' : '') +
                  '</div></details>' : '') +
        '</div>';
    }

    els.trace.innerHTML =
      '<div class="kf-kg-thead">' +
        '<h4>编译 IR 全流程 · <b>' + esc(K.source.replace(/_\d{8}_\d{6}$/, '')) + '</b></h4>' +
        '<span class="kf-kg-tmeta">' + miles.length + ' 个关键事件 · ' +
          ev.filter(e => e.kind !== 'same').length + ' / ' + ev.length + ' 个 pass 改动了 IR</span>' +
      '</div>' +
      '<div class="kf-kg-track">' +
        '<div class="kf-kg-strata">' + strataBands() + '</div>' +
        '<div class="kf-kg-slots">' + slots + '</div>' +
      '</div>' +
      legendFor(ev) +
      detail;
  }
  const stat = (t, v, d) =>
    '<div><dt>' + t + '</dt><dd>' + v + '<em>' + d + '</em></dd></div>';

  /* What the pass DID to the computation, drawn. The IR diff is demoted to a
     collapsed <details> below it — reading code should be the fallback, not
     the primary way to understand a pass. */
  function visualFor(p) {
    const VIS = window.PTO_PASS_VISUAL;
    if (!VIS) return '';
    // Memory visuals need a concrete kernel: the expanded one, else the peak.
    const focus = (st.sel && K.kernels.find(x => x.name === st.sel && x.bufs && x.bufs.b32.length)) ||
      K.kernels.slice().filter(x => x.bufs && x.bufs.b32.length)
        .sort((a, b) => worstPct(b).p - worstPct(a).p)[0] || null;
    const r = VIS.render(p.name, {
      pass: p,
      prev: p.i > 0 ? PASSMETA[p.i - 1] : null,
      kernels: K.kernels,
      limits: LIMITS,
      focus,
      worstSpace: focus ? worstPct(focus).space : null,
      born: K.kernels.filter(k => k.passes[p.i] && k.passes[p.i].st === 'born').length,
      facts: K.facts || {},
      factSel: st.fact
    });
    if (!r) return '';
    return '<figure class="kf-kg-visbox">' + (r.html || r.svg) +
      (r.caption ? '<figcaption>' + esc(r.caption) + '</figcaption>' : '') + '</figure>';
  }

  /* ---------- per-kernel trace (inside an expanded row) ---------- */
  function kernelStrip(k) {
    const ev = eventsOf(k);
    const slots = ev.map(e => {
      const s = STRATA.find(x => e.i >= x.from && e.i <= x.to);
      const what = e.kind === 'absent' ? '尚不存在' : e.kind === 'same' ? '未改动' : e.text;
      return '<button type="button" class="kf-kg-slot is-' + e.kind + (e.milestone ? ' is-milestone' : '') +
        (e.i === st.kpass ? ' is-sel' : '') + '"' +
        ' data-kg-kp="' + e.i + '" data-kg-tip="' + esc(String(e.i).padStart(2, '0') + ' ' + PASSNAMES[e.i] + '|' + what) + '"' +
        ' style="--kg-c: var(--irp-' + (s ? s.id.toLowerCase() : 's0') + ')"' +
        ' aria-label="' + esc(PASSNAMES[e.i] + ' — ' + what) + '"><i></i><em>' +
        String(e.i).padStart(2, '0') + '</em></button>';
    }).join('');

    const sel = st.kpass !== null ? k.passes.find(x => x.i === st.kpass) : null;
    const hunk = sel && sel.h
      ? '<div class="kf-kg-hunk"><div class="kf-kg-hunk-h">' +
          String(sel.i).padStart(2, '0') + ' ' + esc(PASSNAMES[sel.i]) + ' · @ line ' + sel.h.at + '</div>' +
          (sel.h.b.length ? '<pre class="is-before">' + esc(sel.h.b.join('\n')) + '</pre>' : '') +
          (sel.h.a.length ? '<pre class="is-after">' + esc(sel.h.a.join('\n')) + '</pre>' : '') +
        '</div>'
      : '<p class="kf-kg-none">点击色块查看该 pass 对这个 kernel 做了什么。</p>';

    return '<div class="kf-kg-track is-mini">' +
        '<div class="kf-kg-strata">' + strataBands() + '</div>' +
        '<div class="kf-kg-slots">' + slots + '</div>' +
      '</div>' + hunk;
  }


  /* ---------- events ---------- */
  function wire() {
    els.dims.addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      st.dim = b.dataset.kgDim;
      Array.prototype.forEach.call(els.dims.querySelectorAll('button'),
        x => x.setAttribute('aria-pressed', String(x === b)));
      renderList();
    });
    els.issues.addEventListener('click', () => {
      st.onlyIssues = !st.onlyIssues;
      els.issues.setAttribute('aria-pressed', String(st.onlyIssues));
      renderList();
    });
    els.list.addEventListener('click', e => {
      // per-kernel strip inside an expanded row
      const kp = e.target.closest('[data-kg-kp]');
      if (kp) { st.kpass = +kp.dataset.kgKp; renderList(); return; }

      const b = e.target.closest('.kf-kg-row'); if (!b) return;
      const name = b.dataset.kgK;
      st.sel = (st.sel === name) ? null : name;   // click again to collapse
      st.pass = null; st.kpass = null;
      renderList(); renderTrace(); renderLineage(); syncHead();
      if (st.sel) {
        const row = els.list.querySelector('.kf-kg-item.is-open');
        if (row) row.scrollIntoView({ block: 'nearest' });
      }
    });
    // the lineage strip drives the same selection as the trace below it, so
    // clicking an event there opens that pass's detail in one place
    els.lineage.addEventListener('click', e => {
      const b = e.target.closest('[data-kg-lin]'); if (!b) return;
      st.pass = +b.dataset.kgLin; st.fact = 0;
      renderTrace(); renderLineage();
      const d = els.trace.querySelector('.kf-kg-pdetail');
      if (d) d.scrollIntoView({ block: 'nearest' });
    });

    els.trace.addEventListener('click', e => {
      const fb = e.target.closest('[data-kg-fact]');
      if (fb) { st.fact = +fb.dataset.kgFact; renderTrace(); renderLineage(); return; }
      const b = e.target.closest('[data-kg-p]'); if (!b) return;
      const f = e.target.closest('[data-kg-fact]');
      if (f) { st.fact = +f.dataset.kgFact; renderTrace(); renderLineage(); return; }
      st.pass = +b.dataset.kgP; st.fact = 0;
      renderTrace(); renderLineage();
    });

    els.trace.addEventListener('mousemove', e => {
      const b = e.target.closest('.kf-kg-slot');
      if (!b) { els.tip.style.opacity = '0'; return; }
      const [head, body] = (b.dataset.kgTip || '|').split('|');
      els.tip.innerHTML = '<b>' + esc(head) + '</b><br>' + esc(body);
      els.tip.style.opacity = '1';
      const r = els.tip.getBoundingClientRect();
      els.tip.style.left = Math.min(e.clientX + 12, window.innerWidth - r.width - 8) + 'px';
      els.tip.style.top = Math.max(e.clientY - r.height - 10, 8) + 'px';
    });
    els.trace.addEventListener('mouseleave', () => { els.tip.style.opacity = '0'; });

    // same tooltip for the per-kernel strip
    els.list.addEventListener('mousemove', e => {
      const b = e.target.closest('.kf-kg-slot');
      if (!b) { els.tip.style.opacity = '0'; return; }
      const [head, body] = (b.dataset.kgTip || '|').split('|');
      els.tip.innerHTML = '<b>' + esc(head) + '</b><br>' + esc(body);
      els.tip.style.opacity = '1';
      const r = els.tip.getBoundingClientRect();
      els.tip.style.left = Math.min(e.clientX + 12, window.innerWidth - r.width - 8) + 'px';
      els.tip.style.top = Math.max(e.clientY - r.height - 10, 8) + 'px';
    });
    els.list.addEventListener('mouseleave', () => { els.tip.style.opacity = '0'; });
    const run = document.getElementById('runCompile');
    if (run) run.addEventListener('click', sweep);
  }

  function syncHead() {
    const n = $('#activePassName');
    if (n && st.sel) n.textContent = st.sel;
  }

  /* ---------- compile sweep ---------- */

  // demo-v2.js drives its own 5-step animation on the same button and finishes
  // at a similar time, so don't race it on wall clock: set the real result and
  // hold the claim for a short window against any later writer.
  function claimStatus(text, cls) {
    const el = $('#compileStatus');
    if (!el) return;
    const apply = () => { if (el.textContent !== text) { el.textContent = text; el.className = cls; } };
    apply();
    if (typeof MutationObserver !== 'function') return;
    const mo = new MutationObserver(apply);
    mo.observe(el, { childList: true, characterData: true, subtree: true, attributes: true });
    setTimeout(() => mo.disconnect(), 3000);
  }

  let sweeping = false;
  async function sweep() {
    if (sweeping) return;
    sweeping = true;
    const rows = Array.prototype.slice.call(els.list.querySelectorAll('.kf-kg-row'));
    for (const r of rows) {
      r.classList.add('is-checking');
      await new Promise(x => setTimeout(x, 26));
      r.classList.remove('is-checking');
      r.classList.add('is-checked');
    }
    let e = 0, w = 0, p = 0;
    K.kernels.forEach(k => { const c = diagCount(k); e += c.error; w += c.warn; p += c.perf; });
    const peak = K.kernels.reduce((a, k) => Math.max(a, worstPct(k).p), 0);

    claimStatus(
      e ? e + ' 个 kernel 阻塞' : K.kernels.length + ' / ' + K.kernels.length + ' Kernel 通过',
      'kf-state-chip ' + (e ? 'danger' : 'good'));

    const sum = $('#guardSummary');
    if (sum) sum.textContent = '峰值水位 ' + peak + '% · ' + e + ' error / ' + w + ' warning / ' + p + ' perf hint';
    sweeping = false;
  }

  /* ---------- boot ---------- */
  function boot() {
    if (!mount()) return;
    wire();
    renderSummary(); renderList();
    // open on the worst offender — the developer's actual entry point
    const worst = K.kernels.slice().sort((a, b) => worstPct(b).p - worstPct(a).p)[0];
    if (worst) { st.sel = worst.name; renderList(); }
    renderTrace(); renderLineage(); syncHead();
  }

  // Let other panels drill into one kernel here (the run detail heatmap does).
  window.PTO_GUARD = {
    select(name) {
      if (!els || !K.kernels.some(k => k.name === name)) return false;
      st.sel = name; st.pass = null; st.kpass = null;
      st.onlyIssues = false;                 // never hide the row we were asked for
      els.issues.setAttribute('aria-pressed', 'false');
      renderList(); renderTrace(); renderLineage(); syncHead();
      const row = els.list.querySelector('.kf-kg-item.is-open');
      if (row) row.scrollIntoView({ block: 'center', behavior: 'smooth' });
      return true;
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
