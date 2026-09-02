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

  /* ---------- facts from the real run ---------- */
  function liveRun() {
    const K = window.PTO_IR_KERNELS;
    if (!K) return null;
    const pct = (a, b) => b ? Math.round((a / b) * 100) : 0;
    const worst = (k) => {
      let w = 0, sp = null;
      for (const s of K.spaces) { const p = pct(k.mem[s] || 0, K.limits[s]); if (p > w) { w = p; sp = s; } }
      return { p: w, sp };
    };
    let e = 0, wn = 0, pf = 0;
    K.kernels.forEach(k => k.diags.forEach(d => {
      if (d.sev === 'error') e += 1; else if (d.sev === 'warn') wn += 1; else pf += 1;
    }));
    const peak = K.kernels.reduce((a, k) => Math.max(a, worst(k).p), 0);
    const peakK = K.kernels.find(k => worst(k).p === peak);
    const tb = K.kernels.reduce((a, k) => a + k.reuse.before.b, 0);
    const ta = K.kernels.reduce((a, k) => a + k.reuse.after.b, 0);
    const m = (K.source || '').match(/^_jit_(.+)_(\d{8})_(\d{6})$/);
    return {
      op: m ? m[1] : (K.source || 'unknown'),
      stamp: m ? m[2] + '_' + m[3] : '',
      time: m ? m[2].replace(/(\d{4})(\d\d)(\d\d)/, '$1-$2-$3') + ' ' +
                m[3].replace(/(\d\d)(\d\d)(\d\d)/, '$1:$2:$3') : '',
      dir: K.source,
      passes: K.passNames.length, kernels: K.kernels.length,
      peak, peakK: peakK ? peakK.name : '',
      errors: e, warns: wn, hints: (K.perfHints || []).length,
      hintKernels: K.kernels.filter(k => k.diags.length).length,
      reuse: tb ? Math.round((1 - ta / tb) * 100) : 0,
      demoted: K.kernels.filter(k => k.intent.demoted > 0).length,
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
          duration: '4m12s', target: r ? r.target : 'Ascend 910B',
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
          signals: r ? [
            ['ok', '编译通过 · 无 Error'],
            ['warn', r.hints + ' 条性能提示 · 命中 ' + r.hintKernels + ' 个 kernel'],
            ['warn', '峰值水位 ' + r.peak + '%（' + r.peakK + '）'],
            ['warn', r.demoted + ' 个 kernel 流水被降级'],
            ['ok', 'MemoryReuse 省 ' + r.reuse + '%']
          ] : []
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
    els = { root, list: $('#thList', root), filters: $('#thFilters', root) };
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
      '<small>' + esc(r.time) + ' · ' + esc(r.duration) + '</small>' +
      (r.live ? '<em>产物在库</em>' : '') +
    '</button>';
  }

  function runBody(t, r) {
    if (r.purged) return '<p class="kf-th-note">' + esc(r.note || '产物已清理。') + '</p>';

    const arts = '<div class="kf-th-arts">' +
      '<div class="kf-th-arts-h">这次运行的产物</div>' +
      (r.artifacts || []).map(a =>
        '<button type="button" class="kf-th-art is-' + a.tone +
          (a.k === st.artifact ? ' is-sel' : '') + (a.primary ? ' is-primary' : '') + '"' +
          ' data-th-art="' + a.k + '"' +
          (a.explorer ? ' data-th-view="explorer"' : ' data-step="' + a.step + '"') + '>' +
          '<i></i><b>' + esc(a.label) + '</b><small>' + esc(a.meta) + '</small>' +
          (a.primary ? '<em>实测</em>' : '') +
        '</button>').join('') +
    '</div>';

    const sig = (r.signals && r.signals.length)
      ? '<div class="kf-th-signals">' + r.signals.map(s =>
          '<span class="is-' + s[0] + '">' + esc(s[1]) + '</span>').join('') + '</div>'
      : '';

    const inv = (r.inventory && r.inventory.length)
      ? '<div class="kf-th-inv">' +
          '<div class="kf-th-arts-h">产物目录 · <code>' + esc(r.dir) + '</code></div>' +
          r.inventory.map(i => '<span><b>' + i.n + '</b>' +
            (DIRLABEL[i.d] || i.d) + '<code>' + esc(i.d) + '</code></span>').join('') +
        '</div>'
      : (r.archived ? '<p class="kf-th-note">归档记录，产物目录不在当前工作区。</p>' : '');

    return arts + sig + inv;
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
            (sel ? runBody(t, sel) : '') +
            '<dl class="kf-th-meta">' +
              '<div><dt>分支</dt><dd>' + esc(t.branch) + '</dd></div>' +
              '<div><dt>目标</dt><dd>' + esc(head.target) + '</dd></div>' +
              '<div><dt>算子</dt><dd>' + esc(t.op) + '</dd></div>' +
            '</dl>'
          : '') +
      '</div>';
    }).join('');
  }

  /* ---------- events ---------- */
  function wire() {
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
        render();
        // stage 1 belongs to the explorer activity view (EXPLORER_STEP in
        // demo-v2.js), so route source artifacts through the rail button.
        if (art.dataset.thView === 'explorer') { const b = $('#activityExplorer'); if (b) b.click(); }
        return;                       // demo-v2 handles data-step itself
      }
      const run = e.target.closest('[data-th-run]');
      if (run) { st.task = run.dataset.thOf; st.run = run.dataset.thRun; render(); return; }
      const row = e.target.closest('[data-th-task]');
      if (!row) return;
      const id = row.dataset.thTask;
      if (st.task === id) { st.task = null; }
      else { st.task = id; const t = TASKS.find(x => x.id === id); st.run = t ? latest(t).id : null; }
      render();
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
        if (a && a.k !== st.artifact) { st.artifact = a.k; render(); }
      });
      stages.forEach(s => mo.observe(s, { attributes: true, attributeFilter: ['class'] }));
    }
  }

  function boot() { if (!mount()) return; wire(); render(); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
