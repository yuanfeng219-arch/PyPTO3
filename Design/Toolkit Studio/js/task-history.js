/* Task history — engineering records for model / operator work.

   Model (mirrors how artifacts land on disk):
     任务 Task   stable target, e.g. Data/_jit_<operator>_*   — identity
       └ 运行 Run   one timestamped execution, e.g. _20260625_184941
           └ 产物   passes_dump / kernels / orchestration / ptoas /
                    dfx_outputs / report / debug
   Artifacts belong to a RUN, never to a task. A task's verdict is its latest
   run's verdict. Artifact buttons carry data-step / data-open-runs so
   demo-v2.js keeps owning stage switching. */
(function () {
  'use strict';

  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.prototype.slice.call((r || document).querySelectorAll(s));
  const esc = (s) => String(s).replace(/[&<>"']/g,
    c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const VERDICT = {
    blocked: ['编译阻塞', 'bad'], passed: ['编译通过', 'ok'], warn: ['有告警', 'warn'],
    trusted: ['可信基线', 'ok'], running: ['执行中', 'run'], purged: ['产物已清理', 'idle']
  };
  const DIRLABEL = {
    passes_dump: 'IR 快照', kernels: 'Kernel 源码', orchestration: '编排代码',
    ptoas: '汇编', dfx_outputs: '运行时 Trace', report: '编译报告', debug: '调试'
  };

  /* ---------- facts from the real run ----------
     Everything here is read off ir-kernels-data.js, which is generated from
     the run directory (passes_dump + report/). No shape is invented. */
  function liveRun() {
    const K = window.PTO_IR_KERNELS;
    if (!K) return null;
    const pct = (a, b) => (b ? (a / b) * 100 : 0);

    // worst-loaded memory space per kernel
    const worst = (k) => {
      let w = -1, sp = null, u = 0;
      for (const s of K.spaces) {
        const p = pct(k.mem[s] || 0, K.limits[s]);
        if (p > w) { w = p; sp = s; u = k.mem[s] || 0; }
      }
      return { p: w, sp, u, lim: K.limits[sp] };
    };

    const kmem = K.kernels.map(k => {
      const w = worst(k);
      return { n: k.name, t: k.type, sp: w.sp, u: w.u, lim: w.lim,
               p: Math.round(w.p), diag: k.diags.length };
    }).filter(x => x.u > 0).sort((a, b) => b.p - a.p || b.u - a.u);

    // per-space rollup: how many kernels live there and how hot it gets
    const spaces = K.spaces.map(s => {
      const on = K.kernels.filter(k => (k.mem[s] || 0) > 0);
      const peak = on.reduce((a, k) => Math.max(a, pct(k.mem[s], K.limits[s])), 0);
      return { sp: s, n: on.length, lim: K.limits[s], peak: Math.round(peak) };
    }).filter(x => x.n).sort((a, b) => b.peak - a.peak);

    // functions that never allocate on chip (orchestration / spmd / group
    // wrappers) — shown as a muted group so the grid accounts for all 45
    const noMem = K.kernels.filter(k => !worst(k).u).map(k => ({ n: k.name, t: k.type }));

    const types = {};
    K.kernels.forEach(k => { types[k.type] = (types[k.type] || 0) + 1; });

    const nb = K.kernels.reduce((a, k) => a + k.reuse.before.n, 0);
    const na = K.kernels.reduce((a, k) => a + k.reuse.after.n, 0);
    const bb = K.kernels.reduce((a, k) => a + k.reuse.before.b, 0);
    const ba = K.kernels.reduce((a, k) => a + k.reuse.after.b, 0);
    const noGain = K.kernels.filter(k => k.reuse.before.b > 0 && k.reuse.after.b >= k.reuse.before.b).length;

    const demotedK = K.kernels.filter(k => k.intent.demoted > 0)
      .map(k => ({ n: k.name, d: k.intent.demoted, decl: k.intent.declared.length }));
    const l0K = K.kernels.filter(k => k.intent.l0.length)
      .map(k => ({ n: k.name, tiles: k.intent.l0.length }));

    const m = (K.source || '').match(/^_jit_(.+)_(\d{8})_(\d{6})$/);
    return {
      op: m ? m[1] : (K.source || 'unknown'),
      stamp: m ? m[2] + '_' + m[3] : '',
      time: m ? m[2].replace(/(\d{4})(\d\d)(\d\d)/, '$1-$2-$3') + ' ' +
                m[3].replace(/(\d\d)(\d\d)(\d\d)/, '$1:$2:$3') : '',
      dir: K.source,
      passes: K.passNames.length, kernels: K.kernels.length, types,
      kmem, spaces, noMem,
      atLimit: kmem.filter(x => x.p >= 100).length,
      peak: kmem.length ? kmem[0].p : 0, peakK: kmem.length ? kmem[0].n : '',
      peakSp: kmem.length ? kmem[0].sp : '',
      errors: 0, warns: 0,
      hints: (K.perfHints || []).slice(), minInner: K.perfMinInnermost || 512,
      hintKernels: K.kernels.filter(k => k.diags.length).length,
      reuse: { nb, na, bb, ba, noGain, pct: bb ? Math.round((1 - ba / bb) * 100) : 0 },
      declared: K.kernels.reduce((a, k) => a + k.intent.declared.length, 0),
      demotedK, l0K,
      target: K.target,
      inventory: K.inventory || []
    };
  }

  const ART = (over) => Object.assign({
    overview: { k: 'overview', label: '任务契约', step: 0 },
    source: { k: 'source', label: '算子源码', explorer: true },
    compile: { k: 'compile', label: '编译 IR 全流程', step: 2, primary: true },
    correct: { k: 'correct', label: '正确性比对', step: 3 },
    baseline: { k: 'baseline', label: '环境与基线', step: 4 }
  }, over);

  function buildTasks() {
    const r = liveRun();
    const tasks = [];

    /* --- the one task with a real run directory on disk --- */
    tasks.push({
      id: 'task_decode', kind: '算子', model: 'qwen3-14b',
      op: r ? r.op : 'decode_fwd_layers', branch: 'kernel/decode-layer',
      title: 'Decode Layer 融合算子',
      runs: [
        {
          id: r ? r.stamp : '20260625_184941', live: !!r,
          verdict: r && r.errors ? 'blocked' : 'passed',
          time: r ? r.time : '2026-06-25 18:49:41',
          target: r ? r.target : 'Ascend 910B',   // no real compile duration on disk
          dir: r ? 'Data/' + r.dir : '',
          inventory: r ? r.inventory : [],
          artifacts: [
            Object.assign({}, ART().overview, { meta: 'hidden [16,5120] FP32 · 40Q / 8KV', tone: 'ok' }),
            Object.assign({}, ART().source, { meta: 'decode_layer.py · 在资源管理器中打开', tone: 'ok' }),
            Object.assign({}, ART().compile, {
              meta: r ? r.passes + ' pass · ' + r.kernels + ' kernel · 峰值 ' + r.peak + '%' : '',
              tone: r && r.peak >= 90 ? 'warn' : 'ok' }),
            Object.assign({}, ART().correct, { meta: '待运行 · 未产出比对结果', tone: 'idle' }),
            Object.assign({}, ART().baseline, { meta: 'env:8da1bf09 · 4 / 4 门禁', tone: 'ok' })
          ],
          // live runs render measured sections instead of summary chips
          signals: []
        },
        { id: '20260624_101233', verdict: 'purged', time: '2026-06-24 10:12:33',
          duration: '1m03s', target: 'Ascend 910B', purged: true,
          note: '产物目录已清理，仅保留结论：AutoTileMatmulL0 前的版本，L0B 尚未打满。' },
        { id: '20260620_093015', verdict: 'purged', time: '2026-06-20 09:30:15',
          duration: '3m58s', target: 'Ascend 910B', purged: true,
          note: '产物目录已清理，仅保留结论：首次跑通全部 42 个 pass。' }
      ]
    });

    /* --- archive tasks (no run directory in Data/) --- */
    tasks.push({
      id: 'task_rmsrope', kind: '算子', model: 'qwen3-32b', op: 'rmsnorm_rope',
      branch: 'kernel/rmsnorm-rope', title: 'RMSNorm + RoPE 融合内核',
      runs: [{
        id: '20260618_100712', verdict: 'trusted', time: '2026-06-18 10:07:12',
        duration: '1m52s', target: 'Ascend 910B', archived: true,
        artifacts: [
          Object.assign({}, ART().overview, { meta: 'L14 · RMSNorm + RoPE', tone: 'ok' }),
          Object.assign({}, ART().source, { meta: 'rmsnorm_rope.py · 在资源管理器中打开', tone: 'ok' }),
          Object.assign({}, ART().compile, { meta: '编译通过 · 7 约束', tone: 'ok', primary: false }),
          Object.assign({}, ART().correct, { meta: '3 / 3 oracle · 16 / 16 match', tone: 'ok' }),
          Object.assign({}, ART().baseline, { meta: '已签发可信基线', tone: 'ok' })
        ],
        signals: [['ok', '端到端 +8% vs 基线'], ['ok', 'UB 61% · 预算内'], ['ok', '最大绝对误差 0.0004883']]
      }]
    });

    tasks.push({
      id: 'task_dsv4', kind: '模型', model: 'deepseek-v4-flash', op: 'decode_fwd',
      branch: 'model/v4-flash-npu', title: 'DeepSeek V4 Flash · 整网融合替换',
      runs: [{
        id: '20260611_163004', verdict: 'warn', time: '2026-06-11 16:30:04',
        duration: '12m04s', target: 'Ascend 910B', archived: true,
        artifacts: [
          Object.assign({}, ART().overview, { meta: 'MoE · CSA / HCA', tone: 'ok' }),
          Object.assign({}, ART().source, { label: '模型源码', meta: 'torch_npu 融合替换后', tone: 'ok' }),
          Object.assign({}, ART().compile, { meta: '整网 61 kernel', tone: 'warn', primary: false }),
          Object.assign({}, ART().correct, { meta: '逐层 checkpoint 通过', tone: 'ok' }),
          Object.assign({}, ART().baseline, { meta: '待复核', tone: 'warn' })
        ],
        signals: [['warn', 'MoE 路由分支未覆盖'], ['ok', '融合替换命中 18 处']]
      }]
    });

    tasks.push({
      id: 'task_paged', kind: '算子', model: 'qwen3-14b', op: 'fa_fused',
      branch: 'exp/paged-block', title: 'Paged Attention · block 调度实验',
      runs: [{
        id: '20260529_091210', verdict: 'purged', time: '2026-05-29 09:12:10',
        duration: '3m41s', target: 'Ascend 910B', purged: true,
        note: '实验分支，产物已清理。结论：已被 affine fallback 方案取代。'
      }]
    });

    return tasks;
  }

  const TASKS = buildTasks();
  const latest = (t) => t.runs[0];
  const FILTERS = [['all', '全部'], ['op', '算子'], ['model', '模型'], ['live', '产物在库']];

  const st = { task: TASKS[0].id, run: TASKS[0].runs[0].id, artifact: 'compile', filter: 'all' };
  let els = null;

  function matches(t) {
    if (st.filter === 'all') return true;
    if (st.filter === 'op') return t.kind === '算子';
    if (st.filter === 'model') return t.kind === '模型';
    return t.runs.some(r => r.live);
  }

  /* ---------- mount ---------- */
  function mount() {
    const host = $('[data-side-view="workflow"]');
    if (!host) return false;
    const root = document.createElement('div');
    root.className = 'kf-th';
    root.id = 'taskHistory';
    root.innerHTML =
      '<div class="kf-th-head">' +
        '<span class="kf-eyebrow">TASK HISTORY</span><h2>任务与运行记录</h2>' +
        '<p>任务是稳定的工程目标，每次执行是它的一次运行；产物归属于运行。</p>' +
      '</div>' +
      '<div class="kf-th-filters" id="thFilters">' +
        FILTERS.map(f => '<button type="button" data-th-filter="' + f[0] + '"' +
          ' aria-pressed="' + (f[0] === st.filter) + '">' + f[1] + '</button>').join('') +
      '</div>' +
      '<div class="kf-th-list" id="thList"></div>';
    host.insertBefore(root, host.firstChild);
    // Run detail goes at the top of stage 0; the recipe / contract form below
    // it stays as the "start a new task" affordance.
    let detail = null;
    const stage0 = $('.kf-stage[data-stage="0"]');
    if (stage0) {
      detail = document.createElement('section');
      detail.className = 'kf-rd';
      detail.id = 'runDetail';
      stage0.insertBefore(detail, stage0.firstChild);
    }

    els = { root, detail, list: $('#thList', root), filters: $('#thFilters', root) };
    return true;
  }

  /* ---------- render ---------- */
  function runRow(t, r) {
    const v = VERDICT[r.verdict] || VERDICT.purged;
    const on = r.id === st.run && t.id === st.task;
    return '<button type="button" class="kf-th-run' + (on ? ' is-sel' : '') +
      (r.purged ? ' is-purged' : '') + '" data-th-run="' + r.id + '" data-th-of="' + t.id + '">' +
      '<span class="kf-th-rdot is-' + v[1] + '"></span>' +
      '<code>' + esc(r.id) + '</code>' +
      '<span class="kf-th-rv is-' + v[1] + '">' + v[0] + '</span>' +
      '<small>' + esc(r.time) + (r.duration ? ' · ' + esc(r.duration) : '') + '</small>' +
      (r.live ? '<em>产物在库</em>' : '') +
    '</button>';
  }

  /* ---------- run detail (main area, top of stage 0) ----------
     A run detail page has to answer, without another click: did it pass, where
     is it tight, what did the compiler complain about, and what came out.
     Artifacts are the last section, not the whole page. */
  const GROUPS = [
    ['input',   '输入', '这次运行消费了什么'],
    ['compile', '编译产物', '编译器走过的 IR 与它自己的结论'],
    ['codegen', '生成代码', '最终落盘的可执行物'],
    ['runtime', '运行时证据', '上设备之后采集到的'],
    ['repro',   '复现', '把这次运行原样跑回来']
  ];
  const OPEN = {                       // artifacts that have a viewer wired up
    step2: { step: 2, k: 'compile' },
    explorer: { explorer: true, k: 'source' }
  };
  // dfx_outputs/ is what the execution timeline on this page is built from, so
  // those three artifacts scroll to it rather than claiming to have no viewer.
  const TL_ARTS = { swimlane: 1, deps: 1, namemap: 1 };

  const kb = (b) => b >= 1048576 ? (b / 1048576).toFixed(1) + ' MB'
                  : b >= 1024 ? Math.round(b / 1024) + ' KB' : b + ' B';
  const tone = (p) => p >= 100 ? 'bad' : p >= 80 ? 'warn' : p >= 50 ? 'mid' : 'ok';

  /* ---------- header ------------------------------------------------------
     The left pane already carries identity (title, kind, model, verdict, run id
     and time) on both the task row and the run row, so repeating it here is
     pure noise. The header keeps only what the left pane cannot show — branch,
     target, and the run's own findings — and leads with the numbers a kernel
     developer acts on: how long it took on device, how busy the cores were,
     how much of the wall time is an unavoidable dependency chain, and what is
     at risk. Scale counts (42 passes, 45 functions) and the compiler's own
     wins (58% reuse) are stated by the sections that own them further down. */
  function headline(t, r, L) {
    const bits = [t.op, t.branch, r.target].filter(Boolean);
    return '<div class="kf-rd-head">' +
      '<div class="kf-rd-id">' +
        '<h2>' + esc(t.title) + '</h2>' +
        '<code>' + bits.map(esc).join(' · ') + '</code>' +
      '</div>' +
      '<div class="kf-rd-run"><code>' + esc(r.id) + '</code></div>' +
    '</div>';
  }

  /* one line saying what, if anything, needs attention */
  function verdictLine(r, L) {
    const v = VERDICT[r.verdict] || VERDICT.passed;
    const flags = [];
    if (L) {
      if (L.atLimit) flags.push(L.atLimit + ' 个 kernel 打满 ' + spLabel(L.kmem[0].sp));
      if (L.hints.length) flags.push(L.hints.length + ' 条性能提示');
      if (L.demotedK.length) flags.push(L.demotedK.length + ' 处流水被降级');
    }
    return '<p class="kf-rd-verdictline is-' + v[1] + '">' +
      '<b>' + v[0] + '</b>' +
      (!L ? '<span>归档记录，没有逐项数据可核。</span>'
        : flags.length
          ? '<span>无 Error。需要关注 ' + flags.length + ' 处：' + flags.map(esc).join(' · ') + '</span>'
          : '<span>无 Error，各项检查均无待关注项。</span>') +
    '</p>';
  }

  const usFmt = (v) => v >= 1000 ? (v / 1000).toFixed(2) + ' ms' : Math.round(v) + ' µs';

  function kpis(r, L) {
    const P = (window.PTO_RUN_TRACE || {}).perf;
    const tiles = [];

    if (P) {
      const aic = P.occ.AIC || { pct: 0, n: 0 }, aiv = P.occ.AIV || { pct: 0, n: 0 };
      const gap = Math.round(aic.pct - aiv.pct);
      tiles.push(
        { v: Math.round(P.span), u: ' µs', l: '设备执行时间', t: 'ok',
          s: '实测 · ' + P.chain.n + ' 步关键链' },
        { v: Math.round(aic.pct), u: '%', l: 'AIC 核占用', t: aic.pct >= 60 ? 'warn' : 'ok',
          s: aiv ? 'AIV 仅 ' + Math.round(aiv.pct) + '% · 相差 ' + gap + ' 个百分点' : '' },
        { v: Math.round(P.chain.workPct), u: '%', l: '关键链执行占比', t: P.chain.workPct < 50 ? 'warn' : 'ok',
          s: usFmt(P.chain.work) + ' 在算 · ' + usFmt(P.chain.wait) + ' 在等核' }
      );
    }
    if (L) {
      tiles.push(
        { v: L.atLimit, u: '', l: '内存零余量的 kernel', t: L.atLimit ? 'bad' : 'ok',
          s: L.atLimit ? spLabel(L.kmem[0].sp) + ' 打满 · 改 shape 即溢出' : '各空间均有余量' },
        { v: L.hints.length, u: '', l: '性能提示', t: L.hints.length ? 'warn' : 'ok',
          s: L.hints.length ? '命中 ' + L.hintKernels + ' 个 kernel · 带源码行号' : '无' },
        { v: L.demotedK.length, u: '', l: '流水被降级', t: L.demotedK.length ? 'warn' : 'ok',
          s: L.demotedK.length ? L.demotedK.map(k => k.n).join(' · ') : '声明的流水全部保留' }
      );
    }
    if (!tiles.length) return '';
    return '<div class="kf-rd-kpis">' + tiles.map(t =>
      '<div class="kf-rd-kpi is-' + t.t + '">' +
        '<b>' + t.v + '<i>' + t.u + '</i></b>' +
        '<span>' + esc(t.l) + '</span>' +
        '<small>' + esc(t.s) + '</small>' +
      '</div>').join('') + '</div>';
  }

  /* memory water level — one square per kernel, filled by utilisation.
     Squares are grouped by the space that is TIGHTEST for that kernel, because
     only two spaces ever become the bottleneck here (Vec and L0B/Right) and a
     percentage is only comparable against its own limit. Utilisation is binned
     rather than continuous: five bands read faster than a gradient, and the
     legend can then state exactly what each shade means. */
  // IR-level space name first, hardware buffer in parens. Note ir-compile-guard.js
  // uses the reverse order ('UB (Vec)'); keep the two in sync if either changes.
  const SPACE_LABEL = { Vec: 'Vec (UB)', Mat: 'Mat (L1)', Acc: 'Acc (L0C)',
                       Left: 'Left (L0A)', Right: 'Right (L0B)' };
  const spLabel = (s) => SPACE_LABEL[s] || s;

  const BINS = [
    { k: 'b1', lo: 0,   hi: 25,  l: '< 25%' },
    { k: 'b2', lo: 25,  hi: 50,  l: '25 – 50%' },
    { k: 'b3', lo: 50,  hi: 75,  l: '50 – 75%' },
    { k: 'b4', lo: 75,  hi: 100, l: '75 – 100%' },
    { k: 'b5', lo: 100, hi: Infinity, l: '打满上限' }
  ];
  const binOf = (p) => BINS.find(b => p >= b.lo && p < b.hi) || BINS[BINS.length - 1];

  function memBlock(r) {
    if (!r.kmem.length) return '';

    // group by the kernel's own tightest space, hottest space first
    const byspace = {};
    r.kmem.forEach(k => { (byspace[k.sp] = byspace[k.sp] || []).push(k); });
    const groups = Object.keys(byspace)
      .map(sp => ({ sp, ks: byspace[sp].slice().sort((a, b) => b.p - a.p || b.u - a.u) }))
      .sort((a, b) => b.ks[0].p - a.ks[0].p);

    const counts = {};
    r.kmem.forEach(k => { const b = binOf(k.p).k; counts[b] = (counts[b] || 0) + 1; });
    const legend = '<div class="kf-rd-hleg">' +
      BINS.map(b => '<span><i class="is-' + b.k + '"></i>' + b.l +
        '<em>' + (counts[b.k] || 0) + '</em></span>').join('') +
      '<span class="kf-rd-hlegn">每个方块 = 1 个 kernel · 取它最紧的那块空间</span></div>';

    const cell = (k) => {
      const b = binOf(k.p);
      const tip = k.n + ' · ' + k.t + '\n' + spLabel(k.sp) + ' ' + kb(k.u) + ' / ' + kb(k.lim) +
        '（' + k.p + '%）' + (k.p >= 100 ? '\n已打满，零余量' : '\n余量 ' + kb(k.lim - k.u)) +
        (k.diag ? '\n' + k.diag + ' 条性能提示' : '') + '\n点击在编译卫士中打开';
      return '<button type="button" class="kf-rd-cell is-' + b.k + '"' +
        ' data-th-kernel="' + esc(k.n) + '" title="' + esc(tip) + '">' +
        '<b>' + esc(k.n) + '</b>' +
        '<i>' + k.p + '<u>%</u></i>' +
        '<small>' + kb(k.u) + ' / ' + kb(k.lim) + '</small>' +
      '</button>';
    };

    const grid = groups.map(g => {
      const peak = g.ks[0].p, full = g.ks.filter(k => k.p >= 100).length;
      return '<div class="kf-rd-hgrp">' +
        '<div class="kf-rd-hh"><b>' + esc(spLabel(g.sp)) + '</b>' +
          '<span>上限 ' + kb(g.ks[0].lim) + ' · ' + g.ks.length + ' 个 kernel · 峰值 ' + peak + '%' +
          (full ? ' · <em>' + full + ' 个打满</em>' : '') + '</span></div>' +
        '<div class="kf-rd-heat">' + g.ks.map(cell).join('') + '</div>' +
      '</div>';
    }).join('');

    const none = (r.noMem && r.noMem.length)
      ? '<div class="kf-rd-hgrp is-none">' +
          '<div class="kf-rd-hh"><b>无片上分配</b><span>' + r.noMem.length +
            ' 个 · 编排 / SPMD 壳 / Group，不占片上内存</span></div>' +
          '<div class="kf-rd-heat">' + r.noMem.map(k =>
            '<div class="kf-rd-cell is-none" title="' + esc(k.n + ' · ' + k.t) + '">' +
              '<b>' + esc(k.n) + '</b><small>' + esc(k.t) + '</small></div>').join('') +
          '</div>' +
        '</div>'
      : '';

    return '<section class="kf-rd-sec">' +
      '<div class="kf-rd-h">内存水位<small>按每个 kernel 最紧的那块空间着色 · 来自 report/memory_after_AllocateMemoryAddr.txt</small></div>' +
      legend + grid + none +
    '</section>';
  }

  /* perf hints — real rows out of report/perf_hints.log, with source lines */
  function hintBlock(r) {
    if (!r.hints.length) return '';
    const rows = r.hints.map(h =>
      '<tr>' +
        '<td><code>' + esc(h.file + ':' + h.line) + '</code></td>' +
        '<td>' + esc(h.op) + '</td>' +
        '<td class="is-bad">' + h.bytes + ' B</td>' +
        '<td>&ge; ' + h.want + ' B</td>' +
        '<td>' + esc(h.dtype) + '</td>' +
        '<td>' + esc(h.mem) + '</td>' +
      '</tr>').join('');
    return '<section class="kf-rd-sec">' +
      '<div class="kf-rd-h">性能提示 · ' + r.hints.length + ' 条' +
        '<small>TileInnermostDimGranularity · 最内维搬运量低于 L2 行宽 ' + r.minInner + 'B</small></div>' +
      '<div class="kf-rd-tblwrap"><table class="kf-rd-tbl">' +
        '<thead><tr><th>源码位置</th><th>算子</th><th>最内维</th><th>要求</th><th>dtype</th><th>空间</th></tr></thead>' +
        '<tbody>' + rows + '</tbody></table></div>' +
    '</section>';
  }

  /* what the optimiser promised vs. what it delivered */
  function intentBlock(r) {
    const dem = r.demotedK.length
      ? r.demotedK.map(k => '<div class="kf-rd-line is-warn"><code>' + esc(k.n) + '</code>' +
          '<span>声明 ' + k.decl + ' 处流水 · 被 SkewCrossCorePipeline 降级 ' + k.d + ' 处为串行</span></div>').join('')
      : '<div class="kf-rd-line is-ok"><span>声明的流水全部保留</span></div>';
    const l0 = r.l0K.length
      ? '<div class="kf-rd-line is-ok"><span>AutoTileMatmulL0 为 ' + r.l0K.length +
        ' 个 kernel 生成了 L0 分块</span></div>' : '';
    const reuseW = r.reuse.bb ? Math.round(r.reuse.ba / r.reuse.bb * 100) : 0;
    return '<section class="kf-rd-sec">' +
      '<div class="kf-rd-h">意图兑现与优化收益<small>声明的调度是否活到最后 · MemoryReuse 前后</small></div>' +
      '<div class="kf-rd-two">' +
        '<div class="kf-rd-col">' + dem + l0 +
          '<div class="kf-rd-line is-dim"><span>全算子共声明 ' + r.declared + ' 处 pl.pipeline</span></div>' +
        '</div>' +
        '<div class="kf-rd-col">' +
          '<div class="kf-rd-reuse">' +
            '<div class="kf-rd-rbar"><span>复用前</span><i class="is-before" style="width:100%"></i>' +
              '<b>' + r.reuse.nb + ' 块 · ' + kb(r.reuse.bb) + '</b></div>' +
            '<div class="kf-rd-rbar"><span>复用后</span><i class="is-after" style="width:' + reuseW + '%"></i>' +
              '<b>' + r.reuse.na + ' 块 · ' + kb(r.reuse.ba) + '</b></div>' +
          '</div>' +
          '<div class="kf-rd-line is-dim"><span>' + r.reuse.noGain +
            ' 个 kernel 没有复用空间可省</span></div>' +
        '</div>' +
      '</div>' +
    '</section>';
  }

  function artBlock(r) {
    return '<section class="kf-rd-sec">' +
      '<div class="kf-rd-h">产物清单<small>产物目录 <code>' + esc(r.dir || '') + '</code></small></div>' +
      GROUPS.map(function (G) {
        const items = (r.inventory || []).filter(i => i.g === G[0]);
        if (!items.length) return '';
        return '<div class="kf-rd-grp">' +
          '<div class="kf-rd-glabel">' + G[1] + '<small>' + G[2] + '</small></div>' +
          '<div class="kf-rd-arts">' + items.map(i => {
            const o = i.open ? OPEN[i.open] : null;
            const tl = !o && TL_ARTS[i.k] && window.PTO_RUN_TRACE;
            const attr = o
              ? (o.explorer ? ' data-th-view="explorer" data-th-art="source"'
                            : ' data-step="' + o.step + '" data-th-art="' + o.k + '"')
              : tl ? ' data-th-scroll="timeline"' : '';
            const on = o && o.k === st.artifact;
            const tag = o ? '<span class="kf-rd-art-go">打开 &rarr;</span>'
                          : tl ? '<span class="kf-rd-art-go">看时间线 &uarr;</span>'
                          : '<span class="kf-rd-art-go is-off">在库</span>';
            return '<' + (o || tl ? 'button type="button"' : 'div') +
              ' class="kf-rd-art' + (o || tl ? '' : ' is-static') + (on ? ' is-sel' : '') + '"' + attr + '>' +
              '<b>' + esc(i.label) + '</b>' +
              '<span class="kf-rd-art-meta">' + esc(i.meta) + '</span>' +
              '<code>' + esc(i.where) + '</code>' + tag +
            '</' + (o || tl ? 'button' : 'div') + '>';
          }).join('') + '</div>' +
        '</div>';
      }).join('') +
    '</section>';
  }

  function renderDetail() {
    if (!els.detail) return;
    const t = TASKS.find(x => x.id === st.task);
    const r = t && t.runs.find(x => x.id === st.run);
    if (!t || !r) {
      els.detail.innerHTML = '<p class="kf-rd-empty">在左侧选择一次运行，查看它产出的全部工件。</p>';
      return;
    }
    const v = VERDICT[r.verdict] || VERDICT.purged;

    const LX = r.live ? liveRun() : null;
    const head = headline(t, r, LX);

    if (r.purged) {
      els.detail.innerHTML = head +
        '<p class="kf-rd-note">' + esc(r.note || '产物目录已清理。') + '</p>' +
        '<p class="kf-rd-note is-dim">该次运行的工件不在工作区中，只能查看结论。</p>';
      return;
    }

    const L = LX;
    if (L) {
      const gap = '<p class="kf-rd-note is-dim">正确性比对：这次运行没有产出 oracle 输出，需要单独跑验证。</p>';
      els.detail.innerHTML = head + verdictLine(r, L) + kpis(r, L) +
        '<div id="runTimeline"></div>' +
        memBlock(L) + hintBlock(L) + intentBlock(L) +
        artBlock(Object.assign({ dir: r.dir }, L)) + gap;
      // run-timeline.js owns its own state, so it re-renders from scratch here
      if (window.PTO_TIMELINE) window.PTO_TIMELINE.mount($('#runTimeline', els.detail));
      return;
    }

    // archived runs: no directory in the workspace, only the recorded verdict
    const sig = (r.signals && r.signals.length)
      ? '<div class="kf-rd-signals">' + r.signals.map(s =>
          '<span class="is-' + s[0] + '"><i></i>' + esc(s[1]) + '</span>').join('') + '</div>'
      : '';
    const body = r.artifacts
      ? '<section class="kf-rd-sec"><div class="kf-rd-h">产物' +
          '<small>归档记录，目录不在当前工作区</small></div>' +
        '<div class="kf-rd-arts">' + r.artifacts.map(a =>
          '<button type="button" class="kf-rd-art' + (a.k === st.artifact ? ' is-sel' : '') + '"' +
            ' data-th-art="' + a.k + '"' +
            (a.explorer ? ' data-th-view="explorer"' : ' data-step="' + a.step + '"') + '>' +
            '<b>' + esc(a.label) + '</b>' +
            '<span class="kf-rd-art-meta">' + esc(a.meta) + '</span>' +
            '<span class="kf-rd-art-go">打开 &rarr;</span>' +
          '</button>').join('') + '</div></section>'
      : '';
    els.detail.innerHTML = head + verdictLine(r, null) + sig + body;
  }

  function render() {
    const rows = TASKS.filter(matches);
    if (!rows.length) { els.list.innerHTML = '<p class="kf-th-empty">没有符合条件的任务。</p>'; return; }

    els.list.innerHTML = rows.map(t => {
      const open = t.id === st.task;
      const head = latest(t);
      const v = VERDICT[head.verdict] || VERDICT.purged;
      const liveN = t.runs.filter(r => r.live).length;
      const sel = open ? (t.runs.find(r => r.id === st.run) || head) : null;

      return '<div class="kf-th-item' + (open ? ' is-open' : '') + '">' +
        '<button type="button" class="kf-th-row" data-th-task="' + t.id + '" aria-expanded="' + open + '">' +
          '<span class="kf-th-dot is-' + v[1] + '"></span>' +
          '<span class="kf-th-title">' + esc(t.title) + '</span>' +
          '<span class="kf-th-verdict is-' + v[1] + '">' + v[0] + '</span>' +
          '<span class="kf-th-sub">' + esc(t.kind) + ' · ' + esc(t.model) + ' · ' + esc(t.op) + '</span>' +
          '<span class="kf-th-time">' + t.runs.length + ' 次运行 · 最近 ' + esc(head.time.slice(5, 16)) + '</span>' +
        '</button>' +
        (open
          ? '<div class="kf-th-runs">' +
              '<div class="kf-th-arts-h">运行历史 · ' + t.runs.length + ' 次' +
                (liveN ? ' · ' + liveN + ' 次产物在库' : '') + '</div>' +
              t.runs.map(r => runRow(t, r)).join('') +
            '</div>' +
            ''
          : '') +
      '</div>';
    }).join('');
  }

  /* ---------- events ---------- */
  function wire() {
    // the detail panel lives outside the side pane, so it needs its own
    // listener; data-step is still handled by demo-v2 document delegation.
    if (els.detail) els.detail.addEventListener('click', e => {
      // a hot cell in the memory heatmap drills into that kernel in the
      // compile guard, which is where the buffer detail actually lives
      const kc = e.target.closest('[data-th-kernel]');
      if (kc) {
        const b = $('#stepNav [data-step="2"]'); if (b) b.click();
        const name = kc.dataset.thKernel;
        setTimeout(() => { if (window.PTO_GUARD) window.PTO_GUARD.select(name); }, 60);
        return;
      }
      const sc = e.target.closest('[data-th-scroll]');
      if (sc) { const el = $('#runTimeline', els.detail);
        if (el) el.scrollIntoView({ block: 'start', behavior: 'smooth' }); return; }
      const art = e.target.closest('[data-th-art]');
      if (!art) return;
      st.artifact = art.dataset.thArt;
      render(); renderDetail();
      if (art.dataset.thView === 'explorer') { const b = $('#activityExplorer'); if (b) b.click(); }
    });

    els.filters.addEventListener('click', e => {
      const b = e.target.closest('[data-th-filter]'); if (!b) return;
      st.filter = b.dataset.thFilter;
      $$('[data-th-filter]', els.filters).forEach(x => x.setAttribute('aria-pressed', String(x === b)));
      render();
    });

    els.list.addEventListener('click', e => {
      const art = e.target.closest('[data-th-art]');
      if (art) {
        st.artifact = art.dataset.thArt;
        render(); renderDetail();
        // stage 1 belongs to the explorer activity view (EXPLORER_STEP in
        // demo-v2.js), so route source artifacts through the rail button.
        if (art.dataset.thView === 'explorer') { const b = $('#activityExplorer'); if (b) b.click(); }
        return;                       // demo-v2 handles data-step itself
      }
      const run = e.target.closest('[data-th-run]');
      if (run) {
        st.task = run.dataset.thOf; st.run = run.dataset.thRun;
        render(); renderDetail(); toOverview();
        return;
      }
      const row = e.target.closest('[data-th-task]');
      if (!row) return;
      const id = row.dataset.thTask;
      if (st.task === id) { st.task = null; }
      else { st.task = id; const t = TASKS.find(x => x.id === id); st.run = t ? latest(t).id : null; }
      render(); renderDetail();
      if (st.task) toOverview();
    });

    // keep the artifact highlight in sync when the stage changes elsewhere
    const stages = $$('.kf-stage');
    if (stages.length && typeof MutationObserver === 'function') {
      const mo = new MutationObserver(() => {
        const i = stages.findIndex(s => s.classList.contains('is-active'));
        const t = TASKS.find(x => x.id === st.task);
        if (i < 0 || !t) return;
        const r = t.runs.find(x => x.id === st.run);
        const a = r && r.artifacts && r.artifacts.find(x => x.step === i);
        if (a && a.k !== st.artifact) { st.artifact = a.k; render(); renderDetail(); }
      });
      stages.forEach(s => mo.observe(s, { attributes: true, attributeFilter: ['class'] }));
    }
  }

  // demo-v2.js owns stage switching, so reach stage 0 through its own button.
  function toOverview() { const b = $('#stepNav [data-step="0"]'); if (b) b.click(); }

  // demo-v2.js rewrites the pane header on every activity-view switch with the
  // old route wording; take it back whenever the workflow pane is showing.
  function claimPaneHeader() {
    const title = $('#sidePaneTitle'), meta = $('#sidePaneMeta');
    if (!title || typeof MutationObserver !== 'function') return;
    const fix = () => {
      const showing = !$('[data-side-view="workflow"]').hidden;
      if (!showing) return;
      if (title.textContent !== '任务与运行') title.textContent = '任务与运行';
      const want = TASKS.length + ' 个任务';
      if (meta && meta.textContent !== want) meta.textContent = want;
    };
    new MutationObserver(fix).observe(title, { childList: true, characterData: true, subtree: true });
    if (meta) new MutationObserver(fix).observe(meta, { childList: true, characterData: true, subtree: true });
    new MutationObserver(fix).observe($('[data-side-view="workflow"]'), { attributes: true, attributeFilter: ['hidden'] });
    fix();
  }

  // Stage 0 is now the run detail page and carries its own identity header,
  // so the generic pane header ('定义目标 / recipe · decode_layer') is dead
  // chrome there. Hidden by class so demo-v2 keeps owning the .hidden flag.
  function hideStageHeaderOnDetail() {
    const hdr = $('#stageTitle') && $('#stageTitle').closest('.pto-ide-frame__pane-header');
    const stage0 = $('.kf-stage[data-stage="0"]');
    if (!hdr || !stage0 || typeof MutationObserver !== 'function') return;
    const fix = () => hdr.classList.toggle('kf-rd-nohdr', stage0.classList.contains('is-active'));
    new MutationObserver(fix).observe(stage0, { attributes: true, attributeFilter: ['class'] });
    fix();
  }

  function boot() { if (!mount()) return; wire(); claimPaneHeader(); hideStageHeaderOnDetail(); render(); renderDetail(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
