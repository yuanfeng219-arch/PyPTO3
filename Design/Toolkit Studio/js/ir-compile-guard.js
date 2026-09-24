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
      // KERNEL 诞生谱系暂时下线：不挂容器，renderLineage() 自己会因为 els.lineage
      // 为空而 return。谱系的数据（K.lineage）与渲染代码都留着，把这一行加回来
      // 就恢复。
      '<div class="kf-kg-trace" id="kgTrace"></div>' +
      // 维度栏（内存水位 / 意图兑现 / 诊断 / 优化收益 + 只看有问题的 + 目标平台）
      // 与它所切换的 45 行 Kernel 列表一并暂时下线。两段渲染代码（renderList /
      // kernelStrip）与 st.dim / st.sel 状态都留着，把下面两行容器加回来即恢复：
      //   '<div class="kf-kg-bar">…</div><p class="kf-kg-hint" id="kgHint"></p>'
      //   '<div class="kf-kg-list" id="kgList" aria-label="Kernel 列表"></div>'
      '';

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
    // Kernel 列表已下线：容器不存在时整个函数是 no-op（st.sel / st.dim 仍照常
    // 维护，pass 可视化还要靠 st.sel 挑聚焦 kernel）
    if (!els.list) return;
    if (els.hint) els.hint.textContent = DIMS[st.dim].hint;

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
      if (!changed) return { i: p.i, kind: 'same', touched: 0, changed: false };
      const m = OP_MILESTONES[p.name];
      const r = m && m(P);
      const touched = touchedAt(p.i);
      if (r) return { i: p.i, kind: r[0], text: r[1], touched, milestone: true, changed: true };
      return { i: p.i, kind: 'touch', text: p.desc || '改写了 IR', touched, changed: true };
    });
  }

  /* ---------- validation events -----------------------------------------
     The trace is a Run diagnostic surface. Circle versus vertical bar shows
     whether a Pass changed IR; its color carries validation only. Callers can
     supply compilation, correctness, or execution events with one common
     shape and a pass (or flow) anchor. */
  const VALIDATION_ORDER = {
    not_collected: 0, pass: 1, unresolved: 2, warning: 3, error: 4, first_divergence: 5
  };
  const VALIDATION_META = {
    pass: { label: 'MATCH', color: 'var(--foreground-muted)' },
    warning: { label: 'WARNING', color: 'var(--foreground-muted)' },
    error: { label: 'MISMATCH', color: 'var(--warning)' },
    first_divergence: { label: 'FIRST DIVERGENCE', color: 'var(--danger)' },
    unresolved: { label: 'UNRESOLVED', color: 'var(--foreground-muted)' },
    not_collected: { label: 'NOT COLLECTED', color: 'var(--foreground-muted)' }
  };
  function validationStatus(value) {
    const key = String(value || '').toLowerCase();
    if (['pass', 'match', 'passed', 'ok'].includes(key)) return 'pass';
    if (['warning', 'warn'].includes(key)) return 'warning';
    if (['error', 'fail', 'failed', 'mismatch'].includes(key)) return 'error';
    if (['first_divergence', 'first-divergence', 'first divergence'].includes(key)) return 'first_divergence';
    if (['unresolved', 'pending'].includes(key)) return 'unresolved';
    return 'not_collected';
  }
  function validationAnchor(value) {
    if (value === 'all' || value === 'run') return value;
    if (Number.isInteger(value)) return value;
    const byName = PASSNAMES.indexOf(value);
    return byName >= 0 ? byName : null;
  }
  function fallbackValidationEvents() {
    const nv = window.PTO_RUN_CONTEXT && window.PTO_RUN_CONTEXT.numericalValidation;
    if (!nv) return [];
    const first = validationAnchor(nv.firstDivergentPass);
    if (Array.isArray(nv.passes) && nv.passes.length) {
      return nv.passes.map(p => ({
        id: 'numerical:' + (p.index == null ? p.name : p.index), domain: 'compilation', type: 'numerical',
        status: p.index === first ? 'first_divergence' : validationStatus(p.status),
        severity: p.index === first || p.status === 'mismatch' ? 'error' : 'info',
        anchor: validationAnchor(p.index == null ? p.name : p.index), title: p.name,
        summary: p.index === first ? 'Host IR execution 在该 Pass 后首次偏离 Golden。' : '', evidence: p
      }));
    }
    return [{
      id: 'numerical:flow', domain: 'compilation', type: 'numerical',
      status: validationStatus(nv.status), severity: nv.status === 'fail' ? 'error' : 'info', anchor: 'all',
      title: '数值校验', summary: nv.summary || '本次 Run 没有逐 Pass 数值证据。', evidence: nv
    }];
  }
  function validationEvents() {
    const supplied = window.PTO_RUN_CONTEXT && window.PTO_RUN_CONTEXT.validationEvents;
    const source = Array.isArray(supplied) ? supplied : fallbackValidationEvents();
    return source.map((event, order) => Object.assign({}, event, {
      id: event.id || 'validation:' + order,
      domain: event.domain || 'compilation', type: event.type || 'numerical',
      status: validationStatus(event.status),
      anchor: validationAnchor(event.anchor)
    })).filter(event => event.anchor === 'all' || event.anchor === 'run' || event.anchor !== null);
  }
  function validationsAt(index, events) {
    const matches = events.filter(event => event.anchor === 'all' || event.anchor === index);
    if (!matches.length) return { status: 'not_collected', events: [] };
    return matches.reduce((current, event) =>
      VALIDATION_ORDER[event.status] > VALIDATION_ORDER[current.status]
        ? { status: event.status, events: matches } : current,
      { status: 'not_collected', events: matches });
  }
  function validationCounts(events) {
    return PASSMETA.reduce((counts, pass) => {
      const status = validationsAt(pass.i, events).status;
      counts[status] = (counts[status] || 0) + 1;
      return counts;
    }, {});
  }
  function validationMessage(result) {
    const event = (result.events || []).find(item => item.status === result.status) || result.events[0];
    return event && event.summary ? event.summary :
      result.status === 'not_collected' ? '本次 Run 未采集该 Pass 的验证结果。' :
      result.status === 'pass' ? '已采集的验证结果匹配。' : '';
  }
  /* ---------- Pass flow --------------------------------------------------
     IR form labels stay neutral context. Node colour, rings and labels only
     express the Validation status of the current Run. */
  function passRiver(ev, validation) {
    const padL = 76, padR = 28;
    const W = Math.max(1140, PASSMETA.length * 26 + padL + padR);
    const irY = 18, bandY = 52, bandH = 56, nodeY = bandY + 34, H = bandY + bandH + 32;
    const x = (index) => padL + ((index + 0.5) / PASSMETA.length) * (W - padL - padR);
    const irLabels = {
      S0: 'Tensor IR', S1: 'SSA / Tensor', S2: 'Structured IR', S3: 'Tile IR',
      S4: 'AIC / AIV Kernel', S5: 'MemRef / 物理内存', S6: 'Runtime IR'
    };
    let svg = `<svg class="kf-kg-river" viewBox="0 0 ${W} ${H}" style="min-width:${W}px" role="img" aria-label="IR Pass 验证流程">`;
    svg += '<defs><linearGradient id="kgRiverBand" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0%" stop-color="var(--foreground)" stop-opacity=".055"/>' +
      '<stop offset="100%" stop-color="var(--foreground)" stop-opacity=".015"/></linearGradient>' +
      '<linearGradient id="kgRiverSpine" gradientUnits="userSpaceOnUse" x1="' + padL + '" y1="0" x2="' + (W - padR) + '" y2="0">' +
      '<stop offset="0%" stop-color="var(--foreground)" stop-opacity=".05"/><stop offset="12%" stop-color="var(--foreground)" stop-opacity=".2"/>' +
      '<stop offset="88%" stop-color="var(--foreground)" stop-opacity=".2"/><stop offset="100%" stop-color="var(--foreground)" stop-opacity=".05"/></linearGradient></defs>';
    STRATA.forEach(stratum => {
      const start = Math.max(0, stratum.from), end = Math.min(PASSMETA.length - 1, stratum.to);
      if (end < start) return;
      const left = x(start) - 11, right = x(end) + 11;
      svg += `<rect class="kf-kg-river-band" x="${left}" y="${bandY}" width="${right - left}" height="${bandH}" rx="10" fill="url(#kgRiverBand)" stroke="var(--border-default)" stroke-opacity=".8"/>`;
      svg += `<text class="kf-kg-river-band-label" x="${(left + right) / 2}" y="${bandY + 16}" text-anchor="middle">${esc(stratum.id + ' · ' + stratum.name)}</text>`;
      const irLeft = x(start) - 9, irRight = x(end) + 9;
      svg += `<rect class="kf-kg-river-irbox" x="${irLeft}" y="${irY - 10}" width="${Math.max(46, irRight - irLeft - 4)}" height="19" rx="9"/>`;
      svg += `<text class="kf-kg-river-ir" x="${irLeft + 10}" y="${irY + 3}">${esc(irLabels[stratum.id] || stratum.name)}</text>`;
    });
    svg += `<text class="kf-kg-river-label" x="${padL - 12}" y="${irY + 3}" text-anchor="end">IR 形态</text>`;
    svg += `<line class="kf-kg-river-spine" x1="${padL}" y1="${nodeY}" x2="${W - padR}" y2="${nodeY}" stroke="url(#kgRiverSpine)"/>`;
    svg += `<text class="kf-kg-river-label" x="${padL - 12}" y="${nodeY + 4}" text-anchor="end">验证序 →</text>`;
    PASSMETA.forEach((pass, index) => {
      const item = ev[index], result = validationsAt(pass.i, validation), meta = VALIDATION_META[result.status];
      const selected = item.i === st.pass, nodeX = x(index);
      const changed = !!item.changed;
      const hasError = result.status === 'first_divergence' || result.status === 'error';
      const markerColor = hasError ? meta.color : 'var(--border-strong)';
      const cls = ['kf-kg-river-node', 'is-' + result.status, selected ? 'is-sel' : ''].filter(Boolean).join(' ');
      const title = esc(`${String(pass.i).padStart(2, '0')} ${pass.name}`);
      const tip = `${String(pass.i).padStart(2, '0')} ${pass.name}|${meta.label} · ${validationMessage(result) || meta.label}`;
      const label = result.status === 'first_divergence' ? 'FIRST' : String(pass.i).padStart(2, '0');
      svg += `<g class="${cls}" data-kg-p="${pass.i}" data-kg-tip="${esc(tip)}" tabindex="0" role="button" aria-label="${title} · ${meta.label}">`;
      if (changed) {
        svg += `<circle class="kf-kg-river-glow" cx="${nodeX}" cy="${nodeY}" r="${result.status === 'first_divergence' ? 15 : 12}" fill="${markerColor}"/>`;
        svg += `<circle class="kf-kg-river-ring" cx="${nodeX}" cy="${nodeY}" r="${result.status === 'first_divergence' ? 10.5 : 8.5}" fill="none" stroke="${markerColor}" stroke-width="${result.status === 'first_divergence' ? 2 : 1.2}"/>`;
        svg += `<circle class="kf-kg-river-dot" cx="${nodeX}" cy="${nodeY}" r="${result.status === 'first_divergence' ? 6.8 : 5.8}" fill="${hasError ? markerColor : 'var(--surface-3)'}" stroke="${markerColor}" stroke-width="1.4"/>`;
      } else {
        svg += `<line class="kf-kg-river-tick" x1="${nodeX}" y1="${nodeY - 6}" x2="${nodeX}" y2="${nodeY + 6}" stroke="${markerColor}"/>`;
      }
      svg += `<text class="kf-kg-river-index" x="${nodeX}" y="${result.status === 'first_divergence' ? nodeY + 30 : nodeY + 25}" text-anchor="middle">${label}</text><title>${title}</title></g>`;
    });
    return svg + '</svg>';
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
    const validation = validationEvents();
    const counts = validationCounts(validation);

    // Selected pass → operator-level detail from the global per-pass record.
    const p = st.pass !== null ? PASSMETA[st.pass] : null;
    let detail = '';
    if (p) {
      const e = ev[st.pass];
      const result = validationsAt(p.i, validation);
      const resultMeta = VALIDATION_META[result.status];
      const prev = st.pass > 0 ? PASSMETA[st.pass - 1] : null;
      const props =
        p.gain.map(x => '<span class="kf-kg-prop">+' + x + '</span>').join('') +
        p.lose.map(x => '<span class="kf-kg-prop is-lost">−' + x + '</span>').join('') ||
        '<span class="kf-kg-prop is-none">不声明任何 produced / invalidated 性质</span>';
      const dl = (a, b) => (prev && b !== undefined && a - b !== 0 ? (a - b > 0 ? '+' : '') + (a - b) : '±0');
      const hk = p.hunks && p.hunks[0];

      /* 「影响 39 / 45 个 kernel」是个计数，看不出是谁。这里把它展开成名单：
         只有在这个 Pass 里诞生（born）或真的被改写（changed）的 Kernel 才在列，
         按变更条数从多到少排。 */
      const changedKs = K.kernels
        .map(k => ({ k, kp: (k.passes || [])[p.i] }))
        .filter(x => x.kp && (x.kp.st === 'changed' || x.kp.st === 'born'))
        .sort((a, b) => b.kp.n - a.kp.n || a.k.name.localeCompare(b.k.name));
      const changedList = changedKs.length
        ? '<div class="kf-kg-pchg">' +
            '<div class="kf-kg-pchg-head"><b>这个 Pass 有变更的 Kernel</b>' +
              '<span>' + changedKs.length + ' / ' + K.kernels.length + ' 个</span></div>' +
            '<div class="kf-kg-pchg-list">' + changedKs.map(x => {
              const born = x.kp.st === 'born';
              return '<span class="kf-kg-pchg-item' + (born ? ' is-born' : '') + '">' +
                '<i>' +
                  esc(x.k.type.slice(0, 3).toUpperCase()) + '</i>' +
                '<b>' + esc(x.k.name) + '</b>' +
                '<em>' + (born ? '在此 Pass 诞生' : '变更 ' + x.kp.n + ' 处') + '</em>' +
              '</span>';
            }).join('') + '</div>' +
          '</div>'
        : '';

      detail =
        '<div class="kf-kg-pdetail">' +
          '<div class="kf-kg-pdhead">' +
            '<span class="kf-kg-pdi is-' + result.status + '" style="background:' + resultMeta.color + '">' +
              String(p.i).padStart(2, '0') + '</span>' +
            '<b>' + esc(p.name) + '</b>' +
            '<span class="kf-kg-dtag">' + esc(p.c) + '</span>' +
            '<span class="kf-kg-vtag is-' + result.status + '">' + resultMeta.label + '</span>' +
            '<span class="kf-kg-pdk">' + (e && e.touched ? '影响 ' + e.touched + ' / ' + K.kernels.length + ' 个 kernel' : '无 kernel 变更') + '</span>' +
          '</div>' +
          '<p class="kf-kg-vsummary">' + esc(validationMessage(result)) + '</p>' +
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
          changedList +
          '<div class="kf-kg-props">' + props + '</div>' +
          visualFor(p) +
          (hk ? '<details class="kf-kg-raw"><summary>查看这一步的 IR diff</summary>' +
                  irDiff(hk, '整网 IR diff') + '</details>' : '') +
        '</div>';
    }

    els.trace.innerHTML =
      '<div class="kf-kg-thead">' +
        '<h4>编译 IR 全流程 · <b>' + esc(K.source.replace(/_\d{8}_\d{6}$/, '')) + '</b></h4>' +
        '<span class="kf-kg-tmeta">' +
          (counts.first_divergence ? '1 FIRST DIVERGENCE · ' : '') +
          (counts.error || 0) + ' MISMATCH · ' + (counts.pass || 0) + ' MATCH · ' +
          (counts.warning ? counts.warning + ' WARNING · ' : '') +
          (counts.unresolved ? counts.unresolved + ' UNRESOLVED · ' : '') +
          (counts.not_collected || 0) + ' NOT COLLECTED</span>' +
      '</div>' +
      '<div class="kf-kg-river-wrap">' + passRiver(ev, validation) + '</div>' +
      detail;
  }
  const stat = (t, v, d) =>
    '<div><dt>' + t + '</dt><dd>' + v + '<em>' + d + '</em></dd></div>';

  /* ---------- IR diff：业界通用的左右对照 --------------------------------
     原来是两块上下叠的 <pre>（前一块整段红、后一块整段绿），行与行对不上，
     得自己拿眼睛找哪一行变了。改成 split view：
       · LCS 对齐行 —— 未改动的行左右同排，改动的行左右成对，纯增 / 纯删留空
       · 成对改动行再做词级 LCS，把真正变了的 token 高亮出来
     hunk 最大 11 行、每行几十个 token，O(n·m) 的朴素实现完全够用。 */
  function lcsTable(a, b) {
    const m = a.length, n = b.length;
    const d = [];
    for (let i = 0; i <= m; i++) d.push(new Uint16Array(n + 1));
    for (let i = m - 1; i >= 0; i--) {
      for (let j = n - 1; j >= 0; j--) {
        d[i][j] = a[i] === b[j] ? d[i + 1][j + 1] + 1 : Math.max(d[i + 1][j], d[i][j + 1]);
      }
    }
    return d;
  }

  function diffSeq(a, b) {
    const d = lcsTable(a, b), out = [];
    let i = 0, j = 0;
    while (i < a.length && j < b.length) {
      if (a[i] === b[j]) { out.push({ t: 'same', a: i, b: j }); i++; j++; }
      else if (d[i + 1][j] >= d[i][j + 1]) { out.push({ t: 'del', a: i }); i++; }
      else { out.push({ t: 'add', b: j }); j++; }
    }
    while (i < a.length) { out.push({ t: 'del', a: i }); i++; }
    while (j < b.length) { out.push({ t: 'add', b: j }); j++; }
    return out;
  }

  // 行相似度：token 多重集交集。整行都被改写的 pass（比如 ConvertTensorToTileOps
  // 把 15 行全换掉）在行级 LCS 下一个锚点都没有，这时"按位配对"会把
  // valid_len = pl.dynamic(...) 这种纯删除行也拉去和别人凑一对，整块错位。
  function lineSim(x, y) {
    const tk = (s) => {
      const m = new Map();
      (s.match(IR_TOKEN) || []).forEach(t => { if (!/^\s+$/.test(t)) m.set(t, (m.get(t) || 0) + 1); });
      return m;
    };
    const A = tk(x), B = tk(y);
    let inter = 0, na = 0, nb = 0;
    A.forEach((c, t) => { na += c; inter += Math.min(c, B.get(t) || 0); });
    B.forEach(c => { nb += c; });
    return na + nb ? (2 * inter) / (na + nb) : 0;
  }

  // 改动块内按相似度对齐（带空位的 Needleman–Wunsch），相似度低于阈值就不配对，
  // 让它作为纯删 / 纯增单独占一行。块最大 11 行，O(n·m) 足够。
  const PAIR_MIN = 0.3;
  function alignBlock(dels, adds, before, after) {
    const m = dels.length, n = adds.length;
    const best = [];
    for (let i = 0; i <= m; i++) best.push(new Float32Array(n + 1));
    const sim = [];
    for (let i = 0; i < m; i++) {
      sim.push(new Float32Array(n));
      for (let j = 0; j < n; j++) sim[i][j] = lineSim(before[dels[i]], after[adds[j]]);
    }
    for (let i = m - 1; i >= 0; i--) {
      for (let j = n - 1; j >= 0; j--) {
        const pair = sim[i][j] >= PAIR_MIN ? sim[i][j] + best[i + 1][j + 1] : -Infinity;
        best[i][j] = Math.max(pair, best[i + 1][j], best[i][j + 1]);
      }
    }
    const rows = [];
    let i = 0, j = 0;
    while (i < m && j < n) {
      const pair = sim[i][j] >= PAIR_MIN ? sim[i][j] + best[i + 1][j + 1] : -Infinity;
      if (pair >= best[i + 1][j] && pair >= best[i][j + 1]) { rows.push({ t: 'chg', a: dels[i++], b: adds[j++] }); }
      else if (best[i + 1][j] >= best[i][j + 1]) rows.push({ t: 'chg', a: dels[i++] });
      else rows.push({ t: 'chg', b: adds[j++] });
    }
    while (i < m) rows.push({ t: 'chg', a: dels[i++] });
    while (j < n) rows.push({ t: 'chg', b: adds[j++] });
    return rows;
  }

  function diffRows(before, after) {
    const rows = [];
    let dels = [], adds = [];
    const flush = () => {
      if (dels.length || adds.length) alignBlock(dels, adds, before, after).forEach(r => rows.push(r));
      dels = []; adds = [];
    };
    diffSeq(before, after).forEach(op => {
      if (op.t === 'same') { flush(); rows.push({ t: 'same', a: op.a, b: op.b }); }
      else if (op.t === 'del') dels.push(op.a);
      else adds.push(op.b);
    });
    flush();
    return rows;
  }

  // 词级高亮：IR 行动辄两百字符，只标"整行变了"等于没标
  const IR_TOKEN = /[A-Za-z_][\w]*|\d+|\s+|[^\s\w]/g;
  function markPair(x, y) {
    const ax = x.match(IR_TOKEN) || [x];
    const by = y.match(IR_TOKEN) || [y];
    const ops = diffSeq(ax, by);
    // 相邻的差异 token 合成一段 <mark>：逐 token 包会把一行炸成几十个小色块，
    // 反而看不出改了什么。
    const build = (side) => {
      const src = side === 'del' ? ax : by;
      let out = '', run = '';
      const flush = () => { if (run) { out += '<mark>' + esc(run) + '</mark>'; run = ''; } };
      ops.forEach(op => {
        if (op.t === 'same') { flush(); out += esc(src[side === 'del' ? op.a : op.b]); }
        else if (op.t === side) run += src[side === 'del' ? op.a : op.b];
      });
      flush();
      return out;
    };
    return [build('del'), build('add')];
  }

  function irDiff(hk, title) {
    const rows = diffRows(hk.b || [], hk.a || []);
    let oldNo = hk.at, newNo = hk.at;
    let added = 0, removed = 0;
    const L = [], R = [];
    rows.forEach(r => {
      const bl = r.a !== undefined ? hk.b[r.a] : null;
      const al = r.b !== undefined ? hk.a[r.b] : null;
      let lh = bl === null ? '' : esc(bl);
      let rh = al === null ? '' : esc(al);
      if (r.t === 'chg' && bl !== null && al !== null) { const p = markPair(bl, al); lh = p[0]; rh = p[1]; }
      if (r.t !== 'same') { if (bl !== null) removed++; if (al !== null) added++; }
      const lc = bl === null ? 'nil' : (r.t === 'same' ? 'same' : 'del');
      const rc = al === null ? 'nil' : (r.t === 'same' ? 'same' : 'add');
      L.push('<div class="kf-kg-dl is-' + lc + '"><i>' + (bl === null ? '' : oldNo++) + '</i><code>' + lh + '</code></div>');
      R.push('<div class="kf-kg-dl is-' + rc + '"><i>' + (al === null ? '' : newNo++) + '</i><code>' + rh + '</code></div>');
    });
    return '<div class="kf-kg-diff">' +
      '<div class="kf-kg-diff-h"><span>' + esc(title) + ' · @ line ' + hk.at + '</span>' +
        '<span class="kf-kg-diff-n"><b class="is-del">−' + removed + '</b><b class="is-add">+' + added + '</b></span></div>' +
      '<div class="kf-kg-diff-body">' +
        '<div class="kf-kg-diff-side" data-kg-diff-pane><div class="kf-kg-diff-cap">变更前</div>' + L.join('') + '</div>' +
        '<div class="kf-kg-diff-side" data-kg-diff-pane><div class="kf-kg-diff-cap">变更后</div>' + R.join('') + '</div>' +
      '</div></div>';
  }

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
    const validation = validationEvents();
    const slots = ev.map(e => {
      const result = validationsAt(e.i, validation);
      const what = VALIDATION_META[result.status].label + ' · ' + (validationMessage(result) || '未采集逐 Pass 证据');
      return '<button type="button" class="kf-kg-slot is-' + result.status +
        (e.i === st.kpass ? ' is-sel' : '') + '"' +
        ' data-kg-kp="' + e.i + '" data-kg-tip="' + esc(String(e.i).padStart(2, '0') + ' ' + PASSNAMES[e.i] + '|' + what) + '"' +
        ' aria-label="' + esc(PASSNAMES[e.i] + ' — ' + what) + '"><i></i><em>' +
        String(e.i).padStart(2, '0') + '</em></button>';
    }).join('');

    const sel = st.kpass !== null ? k.passes.find(x => x.i === st.kpass) : null;
    const hunk = sel && sel.h
      ? irDiff(sel.h, String(sel.i).padStart(2, '0') + ' ' + PASSNAMES[sel.i])
      : '<p class="kf-kg-none">点击色块查看该 pass 对这个 kernel 做了什么。</p>';

    return '<div class="kf-kg-track is-mini">' +
        '<div class="kf-kg-slots">' + slots + '</div>' +
      '</div>' + hunk;
  }


  /* ---------- events ---------- */
  function wire() {
    if (els.dims) els.dims.addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      st.dim = b.dataset.kgDim;
      Array.prototype.forEach.call(els.dims.querySelectorAll('button'),
        x => x.setAttribute('aria-pressed', String(x === b)));
      renderList();
    });
    if (els.issues) els.issues.addEventListener('click', () => {
      st.onlyIssues = !st.onlyIssues;
      els.issues.setAttribute('aria-pressed', String(st.onlyIssues));
      renderList();
    });
    if (els.list) els.list.addEventListener('click', e => {
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
    if (els.lineage) els.lineage.addEventListener('click', e => {
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
    els.trace.addEventListener('keydown', e => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      const b = e.target.closest('.kf-kg-river-node');
      if (!b) return;
      e.preventDefault();
      st.pass = +b.dataset.kgP; st.fact = 0;
      renderTrace(); renderLineage();
    });

    els.trace.addEventListener('mousemove', e => {
      const b = e.target.closest('.kf-kg-slot, .kf-kg-river-node');
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
    if (els.list) els.list.addEventListener('mousemove', e => {
      const b = e.target.closest('.kf-kg-slot');
      if (!b) { els.tip.style.opacity = '0'; return; }
      const [head, body] = (b.dataset.kgTip || '|').split('|');
      els.tip.innerHTML = '<b>' + esc(head) + '</b><br>' + esc(body);
      els.tip.style.opacity = '1';
      const r = els.tip.getBoundingClientRect();
      els.tip.style.left = Math.min(e.clientX + 12, window.innerWidth - r.width - 8) + 'px';
      els.tip.style.top = Math.max(e.clientY - r.height - 10, 8) + 'px';
    });
    if (els.list) els.list.addEventListener('mouseleave', () => { els.tip.style.opacity = '0'; });
    // 左右两栏各自横向滚动，但要一起走——IR 行很长，两边错开就失去对照意义。
    // diff 每次重渲染都是新 DOM，所以用捕获阶段的全局监听，不逐个挂。
    let syncing = false;
    document.addEventListener('scroll', e => {
      const pane = e.target instanceof Element ? e.target.closest('[data-kg-diff-pane]') : null;
      if (!pane || syncing) return;
      syncing = true;
      Array.prototype.forEach.call(pane.parentNode.children, other => {
        if (other !== pane) other.scrollLeft = pane.scrollLeft;
      });
      syncing = false;
    }, true);

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
  let visualsReady = false;
  function activateVisuals() {
    if (visualsReady) return;
    visualsReady = true;
    renderTrace();
    renderLineage();
  }

  async function sweep() {
    if (sweeping) return;
    sweeping = true;
    const rows = els.list ? Array.prototype.slice.call(els.list.querySelectorAll('.kf-kg-row')) : [];
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
    syncHead();
    const stage = document.querySelector('.kf-stage[data-stage="2"]');
    if (stage) {
      const activateIfVisible = () => {
        if (stage.classList.contains('is-active')) activateVisuals();
      };
      new MutationObserver(activateIfVisible).observe(stage, { attributes: true, attributeFilter: ['class'] });
      activateIfVisible();
    }
  }

  // Let other panels drill into one kernel here (the run detail heatmap does).
  window.PTO_GUARD = {
    activate: activateVisuals,
    refreshValidation() {
      if (!els) return false;
      if (visualsReady) { renderTrace(); renderLineage(); }
      return true;
    },
    /* #kgTrace 会被 Run → Compilation 借去当独立页签（见 ir-compilation-view.js）。
       用引用而不是 querySelector，节点被搬走、或被 innerHTML 冲成游离节点后
       仍能拿回来；归还用 appendChild —— kernelGuard 的末子节点就是它。 */
    traceEl() { return els && els.trace ? els.trace : null; },
    traceHome() {
      const home = document.getElementById('kernelGuard');
      if (els && els.trace && home && els.trace.parentElement !== home) home.appendChild(els.trace);
    },
    select(name) {
      if (!els || !K.kernels.some(k => k.name === name)) return false;
      st.sel = name; st.pass = null; st.kpass = null;
      st.onlyIssues = false;                 // never hide the row we were asked for
      if (els.issues) els.issues.setAttribute('aria-pressed', 'false');
      const hadVisuals = visualsReady;
      activateVisuals();
      renderList();
      if (hadVisuals) { renderTrace(); renderLineage(); }
      syncHead();
      const row = els.list && els.list.querySelector('.kf-kg-item.is-open');
      if (row) row.scrollIntoView({ block: 'center', behavior: 'smooth' });
      return true;
    },
    selectPass(name) {
      if (!els) return false;
      const index = Number.isInteger(name) ? name : PASSNAMES.indexOf(String(name));
      if (!Number.isInteger(index) || index < 0 || index >= PASSMETA.length) return false;
      st.pass = index; st.fact = 0;
      activateVisuals();
      renderTrace(); renderLineage();
      const detail = els.trace && els.trace.querySelector('.kf-kg-pdetail');
      if (detail) detail.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      return true;
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
