/* =============================================================
 * Tuning Console
 *
 * Two real on-device runs, switchable from the topbar case chip, opened as a
 * working surface for the tuning loop:
 *   E2E  -> L2 schedule -> L1/L0 core pipeline -> compiler lowering -> ISA / layout
 *
 * Timing and utilization numbers come from data.js, which build-data.cjs
 * derives from the run's own artifacts.  E2E model-stage labels can be
 * supplied by JSON; when absent, the screen marks its fallback rule grouping.
 *
 * Shared patterns used:
 *   ide-frame        page shell, panes, bottom dock, status strip
 *   workbench-shell  split resize kernel (through ide-frame)
 *   swimlane-task    every timed task bar + its hover tooltip + colormap
 * ============================================================= */
(function () {
  'use strict';

  const RUNS = window.TUNING_RUNS;
  const CASES = window.TUNING_CASES;
  const SW = window.PtoSwimlaneTaskPattern;
  const requestedEmbedView = new URLSearchParams(window.location.search).get('embed');
  const EMBED_VIEW = ['l1', 'l2'].includes(requestedEmbedView) ? requestedEmbedView : null;
  const requestedCase = new URLSearchParams(window.location.search).get('case');
  const initialCase = CASES.some((c) => c.id === requestedCase) ? requestedCase : CASES[0].id;

  /* The active case. Everything derived from it is rebuilt by loadCase(),
   * because the two dumps do not carry the same artifacts: one has host
   * STRACE spans and two ranks, the other has neither. */
  let D = RUNS[initialCase];
  let CYC_PER_US = D.case.clockHz ? D.case.clockHz / 1e6 : null;
  let TRACE_MATCH = {};
  let findingById = {};
  let tasksOf = {};
  const hasE2E = () => !!D.e2e;
  const multiRank = () => D.case.ranks.length > 1;

  /* ------------------------------------------------------------- state */
  const S = {
    rank: D.defaultRank,
    view: 'e2e',
    task: D.derived.worstHandoff,
    finding: null,
    chainStep: null,          /* 'C2:1' -- which ladder rung the reader is on */
    findingLevel: 'all',
    focus: null,               /* 'finding' | 'task' | 'hint' | 'pass' */
    laneFilter: 'all',
    colorMode: 'semantic',
    colorOn: true,          /* off => every bar goes neutral grey */
    scopeReturn: null,      /* window + focus to restore when drilling back up */
    folded: {},             /* inspector sections the reader has collapsed */
    overlay: 'sched',
    critOnly: false,
    pathOnly: 'off',       /* 'off' | 'obs' | 'cpm' -- which path the filter shows */
    focusEvidence: false,
    scrollToLane: null,
    t0: 0, t1: 0,
    compilerTab: 'passes',
    pass: 17,
    passMode: 'overview',
    hintSite: null,
    hintModule: 'all',
    dockMode: 'sched',
    termTab: 'problems',
    ledger: [],
    tile: null,
  };

  const R = () => D.ranks[S.rank];
  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const num = (v, d) => (v == null ? '—' : Number(v).toFixed(d == null ? 2 : d));
  const us = (v, d) => (v == null ? '—' : Number(v).toFixed(d == null ? 1 : d) + ' us');
  const kb = (b) => (b == null ? '—' : b >= 1024 ? (b / 1024).toFixed(b % 1024 ? 1 : 0) + ' KB' : b + ' B');
  const pct = (v, d) => (v == null ? '—' : Number(v).toFixed(d == null ? 1 : d) + '%');
  /* title= needs a real newline; a literal one inside a string breaks the file */
  const NL = String.fromCharCode(10);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const LEVELS = [
    { id: 'e2e', label: 'E2E', hint: '端到端与 rank 分解' },
    { id: 'l2', label: 'L2 调度', hint: '任务放置、依赖、关键路径' },
    { id: 'l1', label: 'L1 / L0', hint: '单核流水与片上预算' },
    { id: 'compiler', label: '编译器', hint: 'Pass、流水深度、搬运粒度' },
    { id: 'isa', label: 'ISA / 布局', hint: '布局与指令层证据' },
  ];
  /* role -> how the reader should read this rung. 'stop' is deliberately a
   * first-class rung: a chain that cannot go further says so here instead of
   * ending on a guess. */
  const ROLE = {
    observe: { label: '现象', hint: '这一层看到了什么' },
    descend: { label: '下探', hint: '往下一层追什么' },
    root: { label: '落点', hint: '可以直接验证的地方' },
    stop: { label: '止步', hint: '本 dump 到此为止' },
  };

  const LEVEL_LABEL = {};
  LEVELS.forEach((l) => { LEVEL_LABEL[l.id] = l.label; });

  /* Colormap: the shared pattern owns every task color decision, including the
   * aic / aiv / aicpu lane-kind colors. No page-local palette. */
  const CMAP = SW.createTaskColormap();

  /* Task colour has one decision point. With colouring off every bar goes to
   * a flat neutral, which hands the contrast budget to the occupancy band,
   * the idle wash, the critical path and the evidence markers. */
  function taskColor(t) {
    /* must be an opaque hex: the pattern lightens/alpha-blends baseColor and
     * an rgba() token comes back out as white. --surface-4 is the quiet
     * neutral in both themes. */
    if (!S.colorOn) return cssVar('--surface-4');
    return S.colorMode === 'engine'
      ? CMAP.colorForTask({ laneKind: t.kind }, 'engine')
      : CMAP.colorForTask({ colorKey: t.callable, label: t.callable }, 'semantic');
  }

  /* ------------------------------------------------- derived per case */
  const curTask = () => tasksOf[S.rank][S.task] || R().tasks[0];
  let ledgerSeq = 0;

  function loadCase(id) {
    D = RUNS[id];
    CYC_PER_US = D.case.clockHz ? D.case.clockHz / 1e6 : null;
    /* the tab title names the case, so it has to follow the switch */
    document.title = 'Tuning Console · ' + id;

    /* Which invocation does each rank's device trace correspond to?
     * Reconcile the trace span against the host-reported device_wall.sched.
     * Without host spans there is nothing to reconcile against. */
    TRACE_MATCH = {};
    Object.keys(D.ranks).forEach((rank) => {
      if (!D.e2e || !D.e2e[rank]) { TRACE_MATCH[rank] = null; return; }
      const span = D.ranks[rank].swimlane.spanUs;
      let best = null;
      Object.keys(D.e2e[rank]).forEach((inv) => {
        const sp = D.e2e[rank][inv]['chip.run.runner_run.device_wall.sched'];
        if (!sp) return;
        const diff = Math.abs(sp.us - span);
        if (!best || diff < best.diff) best = { inv: +inv, hostUs: sp.us, diff: diff };
      });
      TRACE_MATCH[rank] = best;
    });

    findingById = {};
    D.findings.forEach((f) => { findingById[f.id] = f; });

    tasksOf = {};
    Object.keys(D.ranks).forEach((rank) => {
      tasksOf[rank] = {};
      D.ranks[rank].tasks.forEach((t) => { tasksOf[rank][t.tag] = t; });
    });

    /* state that only makes sense inside one case */
    S.rank = D.defaultRank;
    S.finding = null;
    S.chainStep = null;
    S.focus = null;
    S.focusEvidence = false;
    S.scopeReturn = null;
    S.findingLevel = 'all';
    S.laneFilter = 'all';
    S.critOnly = false;
    S.pathOnly = 'off';
    S.task = tasksOf[S.rank][D.derived.worstHandoff]
      ? D.derived.worstHandoff : R().tasks[0].tag;
    S.hintSite = D.tileSites.length ? D.tileSites[0].key : null;
    S.pass = D.passes.length ? D.passes[Math.min(17, D.passes.length - 1)].idx : 0;
    S.passMode = 'overview';
    S.t0 = 0;
    S.t1 = R().swimlane.spanUs;
    /* the E2E tab stays selectable: its absence page is the explanation */

    /* the ledger belongs to the case: a baseline for one run is not a
     * baseline for the other */
    S.ledger.length = 0;
    ledgerSeq = 0;
    S.ledger.push({
      id: 'B0',
      state: 'baseline',
      title: '基线锁定 · ' + D.case.program,
      findingId: null,
      hypothesis: '固定 shape / dtype / 平台 / 卡数 / 工具链，作为后续所有对比的唯一基准。',
      change: D.case.runDir + '（' + D.case.capturedAt + '，platform ' + D.case.toolchain.platform + '）',
      correctness: D.case.params.length
        ? 'distributed_meta.json 记录 ' + D.case.params.length + ' 个绑定参数，schema ' + D.case.metaSchema
        : '本 dump 无 distributed_meta.json：绑定参数未记录，正确性基准缺口',
      perf: hasE2E()
        ? D.case.ranks.map((r) => r + ' device_wall '
            + us(D.e2e[r][TRACE_MATCH[r].inv]['chip.run.runner_run.device_wall'].us)).join(' / ')
          + '（inv=' + TRACE_MATCH[D.defaultRank].inv + '）'
        : '本 dump 无 host STRACE log：device_wall 不可得，基线只能用 trace span '
          + us(R().swimlane.spanUs),
      keep: '保留为基线',
    });
  }
  const openExperiment = () => S.ledger.find((r) => r.state === 'open') || null;

  /* ============================================================ tables */
  function table(cols, rows, opts) {
    const o = opts || {};
    const wrap = el('div', 'tc-table-scroll');
    if (o.tall) wrap.dataset.tall = 'true';
    const t = el('table', 'tc-table');
    const thead = el('thead');
    const tr = el('tr');
    cols.forEach((c) => {
      const th = el('th', c.num ? 'num' : null, c.label);
      if (c.width) th.style.width = c.width;
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    t.appendChild(thead);
    const tb = el('tbody');
    rows.forEach((row) => {
      const r = el('tr');
      if (row.__selected) r.className = 'is-selected';
      if (row.__subject) r.classList.add('is-subject');
      cols.forEach((c) => {
        const td = el('td', [c.num ? 'num' : null, c.mono ? 'mono' : null].filter(Boolean).join(' ') || null);
        const v = c.cell ? c.cell(row) : row[c.key];
        if (v instanceof Node) td.appendChild(v);
        else td.innerHTML = v == null ? '—' : String(v);
        r.appendChild(td);
      });
      if (o.onPick) r.addEventListener('click', () => o.onPick(row));
      else r.style.cursor = 'default';
      tb.appendChild(r);
    });
    t.appendChild(tb);
    wrap.appendChild(t);
    return wrap;
  }

  function bar(ratio, tone) {
    const b = el('span', 'tc-bar');
    const i = el('i');
    i.style.width = clamp(ratio * 100, 0, 100).toFixed(2) + '%';
    if (tone) i.dataset.tone = tone;
    b.appendChild(i);
    return b;
  }

  function tiles(items) {
    const g = el('div', 'tc-tiles');
    items.forEach((it) => {
      const n = el('div', 'tc-tile');
      if (it.tone) n.dataset.tone = it.tone;
      n.appendChild(el('span', 'k', it.k));
      n.appendChild(el('span', 'v', it.v));
      if (it.u) n.appendChild(el('span', 'u', it.u));
      g.appendChild(n);
    });
    return g;
  }

  /* ============================================== active finding context
   * S.finding stays active while the reader works, independent of what the
   * inspector happens to be showing. Everything below answers one question:
   * "which things on this screen are the evidence for the active finding?" */
  const activeFinding = () => (S.finding ? findingById[S.finding] : null);
  /* the queue's own order, chains first: a hygiene item must never take a
   * slot in a "top N" list while an attributed chain is left out */
  const topChains = (n) => D.findings.filter((f) => f.kind !== 'hygiene').slice(0, n);

  /* When the reader steps onto a rung of a chain, the marked objects are that
   * rung's, not the whole chain's -- otherwise walking down to the compiler
   * layer still leaves L2 tasks numbered on screen. */
  function activeStep() {
    const f = activeFinding();
    if (!f || !S.chainStep || S.chainStep.indexOf(f.id + ':') !== 0) return null;
    return (f.chain || [])[Number(S.chainStep.split(':')[1])] || null;
  }
  function activeSubjects() {
    const st = activeStep();
    if (st) return st.subjects;
    const f = activeFinding();
    return f ? f.subjects : null;
  }

  function subjectTaskSet() {
    const s = activeSubjects();
    const set = {};
    if (s) s.tasks.forEach((t, i) => { set[t] = i + 1; });
    return set;
  }
  function subjectSiteSet() {
    const s = activeSubjects();
    const set = {};
    if (s) s.sites.forEach((x, i) => { set[x] = i + 1; });
    return set;
  }
  function subjectLaneSet() {
    const s = activeSubjects();
    const set = {};
    if (s) s.lanes.forEach((x, i) => { set[x] = i + 1; });
    return set;
  }

  /* Jump to one piece of evidence. The inspector deliberately stays on the
   * finding: it is the argument, the stage is where that argument shows up.
   * The object's own detail is still one click away on the canvas. */
  function gotoChip(chip) {
    const f = activeFinding();
    if (f) S.focus = 'finding';
    if (chip.kind === 'task') {
      const t = tasksOf[S.rank][chip.id];
      S.task = chip.id;
      if (!f) S.focus = 'task';
      const home = (activeSubjects() || f.subjects).view;
      if (t && (home === 'l2' || S.view === 'l2')) {
        S.view = 'l2';
        const pad = Math.max(40, t.span * 0.35);
        setWindow(t.start - pad, t.end + pad);
      } else {
        S.view = 'l1';
      }
    } else if (chip.kind === 'site') {
      S.view = 'compiler';
      S.compilerTab = D.depthSites.some((s) => s.key === chip.id) ? 'depth' : 'granularity';
      S.hintSite = chip.id;
      if (!f) S.focus = 'hint';
    } else if (chip.kind === 'lane') {
      S.view = 'l2';
      S.laneFilter = chip.id.indexOf('AIC') === 0 ? 'aic' : 'aiv';
      S.scrollToLane = chip.id;
    } else if (chip.kind === 'rank') {
      S.rank = chip.id;
      S.view = 'e2e';
      S.t0 = 0; S.t1 = R().swimlane.spanUs;
      if (!tasksOf[S.rank][S.task]) S.task = R().tasks[0].tag;
    } else if (chip.kind === 'phase') {
      S.view = 'l2';
      S.overlay = 'sched';
      S.dockMode = 'sched';
    }
    render();
  }

  /* the bar itself, pinned to the top of the centre stage */
  function findingBar(stage) {
    const f = activeFinding();
    if (!f) return;
    const bar = el('div', 'tc-findingbar');
    bar.dataset.sev = f.severity;
    if (f.kind === 'hygiene') bar.classList.add('is-hygiene');

    /* Which rung of the chain the reader is standing on. Without this the bar
     * always speaks for the first layer, and a four-layer chain reads as one
     * flat observation again. */
    const chain = f.chain || [];
    const rung = activeStep();
    const stepIdx = rung ? chain.indexOf(rung) : -1;
    const chips = rung ? rung.chips : f.chips;

    const hd = el('div', 'hd');
    hd.appendChild(el('span', 'id', f.id));
    hd.appendChild(el('span', 'ti', rung ? rung.headline : f.title));
    hd.appendChild(el('span', 'mt', rung
      ? (LEVEL_LABEL[rung.level] || rung.level) + ' · '
        + (ROLE[rung.role] || { hint: '' }).hint
      : f.metric + (f.cost ? ' · ' + f.cost.share + '% of makespan' : ' · 无归因')));
    const acts = el('div', 'acts');
    const onHomeView = f.subjects.view === S.view
      && (!f.subjects.tab || f.subjects.tab === S.compilerTab);
    if (!onHomeView) {
      acts.appendChild(btn('去证据所在页 · ' + LEVEL_LABEL[f.subjects.view], {
        size: 'sm', variant: 'solid',
        on: () => { applyFocus(f); S.view = f.subjects.view; render(); },
      }));
    } else if (chips.length) {
      acts.appendChild(btn('聚焦证据', {
        size: 'sm', selected: S.focusEvidence,
        title: '把非证据对象压暗，只留这条瓶颈牵涉到的部分',
        on: () => { S.focusEvidence = !S.focusEvidence; render(); },
      }));
    }
    acts.appendChild(btn('退出', {
      size: 'sm', variant: 'ghost',
      title: '清除当前瓶颈上下文，回到自由浏览',
      on: () => { S.finding = null; S.focusEvidence = false; S.focus = null; render(); },
    }));
    hd.appendChild(acts);
    bar.appendChild(hd);

    /* the ladder, walkable from the stage itself */
    if (chain.length) {
      const rungs = el('div', 'tc-rungs');
      rungs.appendChild(el('span', 'lead', '链'));
      chain.forEach((st, i) => {
        if (i) rungs.appendChild(el('span', 'arrow', '→'));
        const b = el('button', 'tc-rung' + (i === stepIdx ? ' is-current' : ''));
        b.type = 'button';
        b.dataset.role = st.role;
        b.title = st.headline;
        b.appendChild(el('span', 'lv', LEVEL_LABEL[st.level] || st.level));
        b.appendChild(el('span', 'rl', (ROLE[st.role] || { label: st.role }).label));
        b.addEventListener('click', () => { applyStep(f, st); render(); });
        rungs.appendChild(b);
      });
      bar.appendChild(rungs);
    }

    if (chips.length) {
      const row = el('div', 'tc-evchips');
      row.appendChild(el('span', 'lead', '证据 ' + chips.length));
      chips.forEach((chip, i) => {
        const b = el('button', 'tc-evchip');
        b.type = 'button';
        const isCurrent = (chip.kind === 'task' && chip.id === S.task)
          || (chip.kind === 'site' && chip.id === S.hintSite)
          || (chip.kind === 'rank' && chip.id === S.rank);
        if (isCurrent) b.classList.add('is-current');
        b.appendChild(el('span', 'mk', String(i + 1)));
        b.appendChild(el('span', 'nm', chip.label));
        if (chip.value) b.appendChild(el('span', 'vl', chip.value));
        b.addEventListener('click', () => gotoChip(chip));
        row.appendChild(b);
      });
      bar.appendChild(row);
    } else if (f.subjects.absent) {
      const row = el('div', 'tc-evchips');
      row.appendChild(el('span', 'lead', '证据 0 · 缺席项'));
      bar.appendChild(row);
    }

    stage.appendChild(bar);
  }

  function sectionHead(title, sub, right) {
    const h = el('div', 'tc-section-head');
    h.appendChild(el('h2', null, title));
    if (sub) h.appendChild(el('span', 'sub', sub));
    if (right) { h.appendChild(el('span', 'spacer')); h.appendChild(right); }
    return h;
  }

  function btn(label, opts) {
    const o = opts || {};
    const b = el('button', ['btn', o.variant ? 'btn-' + o.variant : null, o.size ? 'btn-' + o.size : null,
      o.selected ? 'is-selected' : null].filter(Boolean).join(' '), label);
    b.type = 'button';
    if (o.title) b.title = o.title;
    if (o.disabled) b.disabled = true;
    if (o.on) b.addEventListener('click', o.on);
    return b;
  }

  function group(cls, items, current, onPick) {
    const g = el('div', cls);
    items.forEach((it) => {
      const cls2 = cls.indexOf('segmented') === 0 ? 'segmented-control-item' : 'tab-control-item';
      const b = el('button', cls2 + (it.id === current ? ' is-selected' : ''), it.label);
      b.type = 'button';
      if (it.hint) b.title = it.hint;
      b.setAttribute('aria-pressed', it.id === current ? 'true' : 'false');
      b.addEventListener('click', () => onPick(it.id));
      g.appendChild(b);
    });
    return g;
  }

  function field(label, control) {
    const f = el('label', 'tc-field');
    f.appendChild(el('span', null, label));
    f.appendChild(control);
    return f;
  }

  function select(options, current, onChange) {
    const s = el('select');
    options.forEach((o) => {
      const opt = el('option', null, o.label);
      opt.value = o.id;
      if (o.id === current) opt.selected = true;
      s.appendChild(opt);
    });
    s.addEventListener('change', () => onChange(s.value));
    return s;
  }

  /* ====================================================== canvas basics */
  function fitCanvas(canvas, cssW, cssH) {
    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width = Math.max(1, Math.round(cssW * dpr));
    canvas.height = Math.max(1, Math.round(cssH * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    return ctx;
  }
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function drawTimeRuler(ctx, x0, w, y, t0, t1, opts) {
    const o = opts || {};
    const span = t1 - t0;
    const stepRaw = span / 8;
    const mag = Math.pow(10, Math.floor(Math.log10(stepRaw)));
    const step = [1, 2, 5, 10].map((m) => m * mag).find((v) => v >= stepRaw) || mag * 10;
    ctx.save();
    if (o.dense) {
      const bandH = 26;
      const bandY = y;
      const label = (t) => step >= 1000
        ? (t / 1000).toFixed(1) + 'ms'
        : Math.round(t) + 'us';
      ctx.fillStyle = cssVar('--surface-3');
      ctx.fillRect(x0, bandY, w, bandH);
      ctx.strokeStyle = cssVar('--border-subtle');
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, bandY + bandH - 0.5);
      ctx.lineTo(x0 + w, bandY + bandH - 0.5);
      ctx.stroke();
      const minor = step / 10;
      const firstMinor = Math.ceil(t0 / minor) * minor;
      for (let t = firstMinor; t <= t1 + minor * 0.001; t += minor) {
        const x = x0 + ((t - t0) / span) * w;
        const isMajor = Math.abs(t / step - Math.round(t / step)) < 0.0001;
        ctx.strokeStyle = isMajor ? cssVar('--border-default') : cssVar('--border-subtle');
        ctx.globalAlpha = isMajor ? 0.9 : 0.62;
        ctx.beginPath();
        ctx.moveTo(Math.round(x) + 0.5, bandY + (isMajor ? 14 : 18));
        ctx.lineTo(Math.round(x) + 0.5, bandY + bandH - 4);
        ctx.stroke();
        if (isMajor) {
          ctx.globalAlpha = 1;
          ctx.font = '500 11px ' + cssVar('--font-sans');
          ctx.fillStyle = cssVar('--foreground-secondary');
          ctx.textBaseline = 'middle';
          ctx.textAlign = t === t0 ? 'left' : 'center';
          ctx.fillText(label(t), t === t0 ? x + 3 : x, bandY + 8);
        }
      }
      ctx.restore();
      return;
    }
    ctx.font = '500 11px ' + cssVar('--font-sans');
    ctx.fillStyle = cssVar('--foreground-muted');
    ctx.strokeStyle = cssVar('--border-subtle');
    ctx.lineWidth = 1;
    ctx.textBaseline = 'alphabetic';
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
      const x = x0 + ((t - t0) / span) * w;
      ctx.beginPath();
      ctx.moveTo(Math.round(x) + 0.5, y + 4);
      ctx.lineTo(Math.round(x) + 0.5, y + 10);
      ctx.stroke();
      ctx.textAlign = 'left';
      ctx.fillText(Math.round(t) + '', Math.round(x) + 3, y + 2);
    }
    ctx.restore();
  }

  /* task object handed to the shared pattern (tooltip + bar) */
  function barTask(t, block, laneName) {
    return {
      label: t.callable,
      displayName: t.callable,
      rawName: t.tag + ' · ' + t.callable,
      colorKey: t.callable,
      lane: laneName,
      laneKind: t.kind,
      laneId: laneName,
      totalCycle: Math.round((block ? block[1] : t.span) * CYC_PER_US),
      clcCycle: block ? Math.round((block[1] - (t.setupMean || 0)) * CYC_PER_US) : null,
      status: /_wait$/.test(t.callable) ? 'wait' : (t.kind === 'mix' ? 'overlap' : 'ok'),
      dominantCounter: t.kind.toUpperCase() + ' · ' + t.blockCount + ' blk / ' + t.coreCount + ' core',
      wrapId: 'ring ' + t.ring,
    };
  }

  /* one shared tooltip per canvas host, created by the pattern */
  function attachTooltip(host, canvas, resolve) {
    const hover = SW.initHoverTooltip({
      root: canvas,
      targets: [canvas],
      appendTo: host,
      bounds: host,
      durationUnit: 'cyc',
      getTask: (target, event) => resolve(event),
    });
    if (!hover) return null;
    let last = null;
    canvas.addEventListener('pointermove', (event) => {
      const task = resolve(event);
      if (!task) { SW.hideTooltip(hover.tooltip); last = null; return; }
      const key = task.rawName + '|' + task.totalCycle;
      if (key !== last) {
        last = key;
        SW.showTooltip(hover.tooltip, task, event, { bounds: host, target: canvas, durationUnit: 'cyc' });
      }
    });
    canvas.addEventListener('pointerleave', () => { last = null; });
    return hover;
  }

  /* ======================================================= E2E view */
  const SPAN_TREE = [
    ['chip.run', 0],
    ['chip.run.bind', 1],
    ['chip.run.bind.args', 2],
    ['chip.run.bind.prebuilt', 2],
    ['chip.run.runner_run', 1],
    ['chip.run.runner_run.device_wall', 2],
    ['chip.run.runner_run.device_wall.preamble', 3],
    ['chip.run.runner_run.device_wall.graph_build', 3],
    ['chip.run.runner_run.device_wall.orch', 3],
    ['chip.run.runner_run.device_wall.sched', 3],
    ['chip.run.runner_run.device_wall.post_orch', 3],
    ['chip.run.validate', 1],
  ];

  /* ------------------------------------------------- function summary
   * Sigma = N x mean is exact. Comparing against N x median says which of
   * the three causes is in play without inventing a verdict model. */
  function renderFuncSummary(stage) {
    const rank = R();
    const sec = el('section');
    const top = rank.scopes.slice(0, 12);
    sec.appendChild(sectionHead('函数汇总 · 慢在哪一项',
      'Σ = 重复 × 宽度 × 均值 · 前 ' + top.length + ' / ' + rank.scopes.length,
      el('span', 'tc-readout', 'Σ 大 ≠ 拖慢墙钟 —— 末列才是')));
    const obs = {};
    rank.cpath.segments.forEach((sg) => { obs[sg.tag] = 1; });
    sec.appendChild(table([
      { label: 'scope', cell: (r) => esc(r.name), mono: true },
      { label: 'Σ core-time', cell: (r) => num(r.coreTime, 0), mono: true, num: true },
      { label: '块数', cell: (r) => String(r.cost.blocks), num: true },
      { label: '重复', cell: (r) => num(r.cost.repeat, r.cost.repeat % 1 ? 1 : 0), num: true },
      { label: '宽度', cell: (r) => r.cost.width + ' 核', num: true },
      { label: '均值', cell: (r) => num(r.cost.mean, 2), mono: true, num: true },
      { label: 'p90/中位', cell: (r) => (r.cost.spread == null ? '—'
        : '<span class="' + (r.cost.spread > 1.5 ? 'bad' : r.cost.spread > 1.2 ? 'warn' : '') + '">'
          + num(r.cost.spread, 2) + '</span>'), num: true },
      { label: '偏离中位', cell: (r) => (r.cost.skew >= 0 ? '+' : '') + num(r.cost.skew, 0), mono: true, num: true },
      { label: '主因', cell: (r) => causeOf(r, rank) },
      /* the column that answers what this layer cannot */
      { label: '在路径上', cell: (r) => pathCell(r, obs, rank) },
    ], top, {
      onPick: (r) => { S.view = 'l2'; S.focus = 'scope'; S.task = r.tags[0]; render(); },
    }));
    sec.appendChild(el('p', 'tc-note',
      '块数 = 重复 × 宽度，Σ = 块数 × 均值，都是恒等式。'
      + '重复 = launch 次数 × 波数（同一批核跑了几轮），宽度 = 一次铺开占几个核 —— '
      + '把两者混成一个「次数」会把「宽」误读成「调用多」。'
      + '偏离中位为负 = 中位高于均值，少数快块把均值拉低了，不是长尾。'
      + '「主因」的倍数是相对本 run 所有 scope 的中位数。'
      + '末列来自「路径归责」那一层 —— 本层自己证明不了一个 scope 是否拖慢墙钟。'));
    stage.appendChild(sec);
  }

  /* Which factor carries the cost, measured against this run's own median
   * rather than an absolute threshold, and reported as the multiple so the
   * reader can see how lopsided it is instead of trusting a label. */
  function causeOf(sc, rank) {
    const midOf = (f) => {
      const v = rank.scopes.map(f).slice().sort((a, b) => a - b);
      return v[Math.floor(v.length / 2)] || 1;
    };
    const medMid = midOf((x) => x.cost.med);
    const repMid = midOf((x) => x.cost.repeat);
    const slowX = sc.cost.med / medMid;
    const manyX = sc.cost.repeat / repMid;
    const wobbly = sc.cost.spread != null && sc.cost.spread > 1.5;
    const x = (n) => ' ×' + num(n, n >= 10 ? 0 : 1);
    let main;
    if (slowX < 2 && manyX < 2) main = '<span class="muted">无突出项</span>';
    else if (slowX >= manyX * 3) main = '单次慢' + x(slowX);
    else if (manyX >= slowX * 3) main = '次数多' + x(manyX);
    else main = '两者兼有';
    return main + (wobbly ? ' <span class="warn">+ 波动</span>' : '');
  }

  /* wall-clock relevance comes from the path layer, not from this table */
  function pathCell(sc, obs, rank) {
    const onObs = sc.tags.some((t) => obs[t]);
    if (sc.onCrit) return '<span class="bad">依赖关键路径</span>';
    if (onObs) return '<span class="warn">观测路径</span>';
    return '<span class="muted">都不在 · slack ' + num(sc.minSlack, 0) + '</span>';
  }

  /* ================================================= E2E model projection
   *
   * The raw dump deliberately has no model-stage taxonomy: it describes
   * calls, ranks and device traces.  This adapter keeps that boundary clear.
   * A future JSON can provide `e2eRuntime.operators[tag].stage` to replace the
   * fallback rule below; until then the stage label is shown as “规则归类”.
   * Timings always stay trace-derived, only the grouping is mocked/inferred.
   */
  const E2E_STAGES = [
    { id: 'input', label: '输入 / 嵌入', hint: 'token、position、embedding' },
    { id: 'attention', label: 'Attention', hint: 'QKV、attention、softmax' },
    { id: 'communication', label: '通信 / 同步', hint: 'collective、wait、all-to-all' },
    { id: 'moe', label: 'MoE 路由', hint: 'gate、expert、route' },
    { id: 'ffn', label: 'FFN / 输出', hint: 'projection、matmul、norm' },
    { id: 'runtime', label: '运行时 / 其他', hint: '未匹配的运行时任务' },
  ];

  function e2eStageOf(task) {
    const supplied = D.e2eRuntime && D.e2eRuntime.operators
      && D.e2eRuntime.operators[task.tag];
    if (supplied && supplied.stage) return { id: supplied.stage, source: 'json' };
    const name = String(task.callable || '').toLowerCase();
    if (/wait|allgather|all_reduce|allreduce|reduce_scatter|a2a|collective|comm/.test(name)) return { id: 'communication', source: 'rule' };
    if (/embed|token|position|rope/.test(name)) return { id: 'input', source: 'rule' };
    if (/attn|attention|softmax|qk|q_proj|k_proj|v_proj|qkv|fa_/.test(name)) return { id: 'attention', source: 'rule' };
    if (/expert|gate|router|route|moe/.test(name)) return { id: 'moe', source: 'rule' };
    if (/proj|matmul|mlp|norm|ffn|down|up_|out_/.test(name)) return { id: 'ffn', source: 'rule' };
    return { id: 'runtime', source: 'rule' };
  }

  function e2eProjection() {
    const ranks = Object.keys(D.ranks);
    const stages = {};
    E2E_STAGES.forEach((s) => { stages[s.id] = { id: s.id, label: s.label, hint: s.hint, ranks: {} }; });
    const operators = {};
    let ruleCount = 0;
    ranks.forEach((rank) => {
      D.ranks[rank].tasks.forEach((task) => {
        const classified = e2eStageOf(task);
        const stage = stages[classified.id] || stages.runtime;
        if (classified.source === 'rule') ruleCount += 1;
        const existing = stage.ranks[rank] || { coreUs: 0, count: 0, max: null, source: classified.source };
        existing.coreUs += task.span;
        existing.count += 1;
        if (!existing.max || task.span > existing.max.span) existing.max = task;
        stage.ranks[rank] = existing;

        const op = operators[task.callable] || {
          name: task.callable, stage: stage.id, ranks: {}, pathRanks: [], source: classified.source,
        };
        const onCritical = D.ranks[rank].critical.tags.indexOf(task.tag) >= 0;
        const stat = op.ranks[rank] || { coreUs: 0, count: 0, max: null, onCritical: false };
        stat.coreUs += task.span;
        stat.count += 1;
        stat.onCritical = stat.onCritical || onCritical;
        if (!stat.max || task.span > stat.max.span) stat.max = task;
        op.ranks[rank] = stat;
        if (onCritical && op.pathRanks.indexOf(rank) < 0) op.pathRanks.push(rank);
        operators[task.callable] = op;
      });
    });
    return { ranks: ranks, stages: E2E_STAGES.map((s) => stages[s.id]), operators: Object.keys(operators).map((k) => operators[k]), ruleCount: ruleCount };
  }

  function e2eJump(rank, task) {
    S.rank = rank;
    S.task = task.tag;
    S.focus = 'task';
    S.view = 'l1';
    render();
  }

  function renderE2ERuntimeOverview(stage, projection) {
    const rankRows = projection.ranks.map((rank) => {
      const match = TRACE_MATCH[rank];
      const spans = D.e2e[rank][match.inv];
      return { rank: rank, dev: spans['chip.run.runner_run.device_wall'].us, trace: D.ranks[rank].swimlane.spanUs };
    });
    const slow = rankRows.slice().sort((a, b) => b.dev - a.dev)[0];
    const fast = rankRows.slice().sort((a, b) => a.dev - b.dev)[0];
    const sec = el('section');
    sec.appendChild(sectionHead('运行总览', '模型 → 阶段 → 算子 → rank / 卡',
      el('span', 'tc-readout', '时延 / trace：测量 · 阶段：' + (projection.ruleCount ? '规则归类' : 'JSON'))));
    const grid = el('div', 'tc-e2e-overview');
    [
      ['模型', D.case.model, D.case.level ? 'L' + D.case.level + ' · ' + D.case.backend : D.case.backend],
      ['卡数', String(projection.ranks.length), D.case.device + ' · ' + D.case.numCores + ' cores / card'],
      ['全局尾时延', num(slow.dev, 1) + ' us', slow.rank + ' · traced inv=' + TRACE_MATCH[slow.rank].inv],
      ['卡间偏斜', num(slow.dev - fast.dev, 1) + ' us', slow.rank + ' vs ' + fast.rank],
    ].forEach((item, i) => {
      const card = el('div', 'tc-e2e-overview-card');
      if (i === 2) card.dataset.tone = 'warn';
      card.appendChild(el('span', 'k', item[0]));
      card.appendChild(el('strong', 'v', item[1]));
      card.appendChild(el('span', 'u', item[2]));
      grid.appendChild(card);
    });
    sec.appendChild(grid);
    stage.appendChild(sec);
  }

  function e2eStageLegend(projection) {
    const legend = el('div', 'tc-e2e-legend');
    projection.stages.filter((s) => projection.ranks.some((rank) => s.ranks[rank])).forEach((s) => {
      const key = el('span', 'tc-e2e-legend-key');
      key.dataset.stage = s.id;
      key.appendChild(el('i'));
      key.appendChild(el('span', null, s.label));
      legend.appendChild(key);
    });
    return legend;
  }

  function e2ePackedTasks(tasks) {
    const lanes = [];
    return tasks.slice().sort((a, b) => a.start - b.start || b.span - a.span).map((task) => {
      let lane = lanes.findIndex((end) => end <= task.start);
      if (lane < 0) { lane = lanes.length; lanes.push(task.end); }
      else lanes[lane] = task.end;
      return { task: task, lane: lane };
    });
  }

  /* Inspired by profiler timelines: rows are ranks, x is device time, and
   * every mark is a real trace task.  The model-stage colour is a grouping
   * layer, never a replacement for the underlying timing. */
  function renderE2ETraceAtlas(stage, projection) {
    const sec = el('section');
    sec.appendChild(sectionHead('执行时间地图', '共用任务色语义 · 点击任一 span 下钻到 L1'));
    sec.appendChild(e2eStageLegend(projection));
    const atlas = el('div', 'tc-e2e-atlas');
    projection.ranks.forEach((rank) => {
      const rankData = D.ranks[rank];
      const packed = e2ePackedTasks(rankData.tasks);
      const laneCount = Math.max.apply(null, packed.map((x) => x.lane)) + 1;
      const row = el('div', 'tc-e2e-atlas-row');
      const label = el('button', 'tc-e2e-atlas-label' + (rank === S.rank ? ' is-armed' : ''));
      label.type = 'button';
      const deviceWall = D.e2e[rank][TRACE_MATCH[rank].inv]['chip.run.runner_run.device_wall'].us;
      label.appendChild(el('strong', null, rank));
      label.appendChild(el('span', null, 'wall ' + num(deviceWall, 0) + ' · trace ' + num(rankData.swimlane.spanUs, 0)));
      label.appendChild(el('small', null, pct(rankData.occupancy.aicUtil, 0) + ' AIC · ' + pct(rankData.occupancy.aivUtil, 0) + ' AIV'));
      label.addEventListener('click', () => { S.rank = rank; S.focus = null; render(); });
      row.appendChild(label);
      const track = el('div', 'tc-e2e-atlas-track');
      track.style.height = Math.max(64, laneCount * 16 + 12) + 'px';
      [0, 25, 50, 75, 100].forEach((p) => {
        const tick = el('i', 'tick');
        tick.style.left = p + '%';
        if (p < 100) tick.appendChild(el('span', null, num(rankData.swimlane.spanUs * p / 100, 0)));
        track.appendChild(tick);
      });
      packed.forEach((entry) => {
        const task = entry.task;
        const stageInfo = e2eStageOf(task);
        const mark = el('button', 'tc-e2e-trace-mark');
        mark.type = 'button';
        mark.dataset.stage = stageInfo.id;
        if (rankData.critical.tags.indexOf(task.tag) >= 0) mark.dataset.critical = 'true';
        mark.style.left = clamp(task.start / rankData.swimlane.spanUs * 100, 0, 100).toFixed(3) + '%';
        mark.style.width = Math.max(0.55, task.span / rankData.swimlane.spanUs * 100).toFixed(3) + '%';
        mark.style.top = (entry.lane * 16 + 8) + 'px';
        mark.title = task.callable + ' · ' + num(task.span, 2) + ' us · ' + (rankData.critical.tags.indexOf(task.tag) >= 0 ? '依赖关键路径' : 'trace task');
        mark.setAttribute('aria-label', mark.title);
        mark.addEventListener('click', () => e2eJump(rank, task));
        track.appendChild(mark);
      });
      row.appendChild(track);
      atlas.appendChild(row);
    });
    sec.appendChild(atlas);
    stage.appendChild(sec);
  }

  /* A 100% composition bar separates “where the device spent trace work”
   * from the wall-clock view above.  Each segment opens its longest scope. */
  function renderE2EStageComposition(stage, projection) {
    const sec = el('section');
    sec.appendChild(sectionHead('阶段工作构成', '各卡独立归一 · segment 宽度 = trace core-time'));
    const chart = el('div', 'tc-e2e-composition');
    projection.ranks.forEach((rank) => {
      const cells = projection.stages.filter((s) => s.ranks[rank]);
      const total = cells.reduce((sum, s) => sum + s.ranks[rank].coreUs, 0) || 1;
      const row = el('div', 'tc-e2e-composition-row');
      const label = el('span', 'rank');
      label.appendChild(el('strong', null, rank));
      label.appendChild(el('small', null, num(total, 0) + ' us work'));
      row.appendChild(label);
      const barHost = el('div', 'tc-e2e-composition-bar');
      cells.forEach((s) => {
        const cell = s.ranks[rank];
        const segment = el('button', 'tc-e2e-composition-segment');
        segment.type = 'button';
        segment.dataset.stage = s.id;
        segment.style.flexGrow = cell.coreUs;
        segment.title = s.label + ' · ' + num(cell.coreUs, 1) + ' us / ' + cell.count + ' tasks · 下钻到 ' + cell.max.callable;
        segment.setAttribute('aria-label', segment.title);
        if (D.ranks[rank].critical.tags.indexOf(cell.max.tag) >= 0) segment.dataset.critical = 'true';
        segment.addEventListener('click', () => e2eJump(rank, cell.max));
        barHost.appendChild(segment);
      });
      row.appendChild(barHost);
      chart.appendChild(row);
    });
    sec.appendChild(chart);
    stage.appendChild(sec);
  }

  /* A scatter plot makes cross-card skew visible without forcing the reader
   * to compare two columns of numbers.  Upper-right = expensive on both;
   * off-diagonal = rank-specific work or imbalance. */
  function renderE2EOperatorScatter(stage, projection) {
    const sec = el('section');
    const ranks = projection.ranks.slice(0, 2);
    if (ranks.length < 2) { renderFuncSummary(stage); return; }
    const ops = projection.operators.map((op) => {
      op.x = op.ranks[ranks[0]] ? op.ranks[ranks[0]].coreUs : 0;
      op.y = op.ranks[ranks[1]] ? op.ranks[ranks[1]].coreUs : 0;
      op.maxCore = Math.max(op.x, op.y);
      op.jumpRank = op.x >= op.y ? ranks[0] : ranks[1];
      return op;
    }).filter((op) => op.maxCore > 0).sort((a, b) => a.maxCore - b.maxCore).slice(-32);
    const max = Math.max.apply(null, ops.map((op) => Math.max(op.x, op.y))) || 1;
    sec.appendChild(sectionHead('算子偏斜散点', ranks[0] + ' × ' + ranks[1] + ' · 右上 = 双卡共同热点，偏轴 = rank 偏斜'));
    sec.appendChild(e2eStageLegend(projection));
    const plot = el('div', 'tc-e2e-scatter');
    plot.appendChild(el('span', 'axis axis-y', ranks[1] + ' core-time'));
    plot.appendChild(el('span', 'axis axis-x', ranks[0] + ' core-time'));
    ops.forEach((op) => {
      const point = el('button', 'tc-e2e-scatter-point');
      point.type = 'button';
      point.dataset.stage = op.stage;
      if (op.pathRanks.length) point.dataset.critical = 'true';
      const scale = (value) => Math.log1p(value) / Math.log1p(max);
      point.style.left = (6 + scale(op.x) * 88).toFixed(2) + '%';
      point.style.bottom = (8 + scale(op.y) * 84).toFixed(2) + '%';
      const size = 8 + scale(op.maxCore) * 14;
      point.style.width = size.toFixed(1) + 'px';
      point.style.height = size.toFixed(1) + 'px';
      point.title = op.name + ' · ' + ranks[0] + ' ' + num(op.x, 1) + ' us · ' + ranks[1] + ' ' + num(op.y, 1) + ' us'
        + (op.pathRanks.length ? ' · 依赖关键路径' : '');
      point.setAttribute('aria-label', point.title);
      point.addEventListener('click', () => e2eJump(op.jumpRank, op.ranks[op.jumpRank].max));
      plot.appendChild(point);
    });
    sec.appendChild(plot);
    stage.appendChild(sec);
  }

  function viewE2E(stage) {
    if (!hasE2E()) { viewE2EAbsent(stage); renderFuncSummary(stage); return; }

    const projection = e2eProjection();
    renderE2ERuntimeOverview(stage, projection);
    renderE2ETraceAtlas(stage, projection);
    renderE2EStageComposition(stage, projection);
    renderE2EOperatorScatter(stage, projection);

    /* --- rank x invocation table --- */
    const rows = [];
    Object.keys(D.e2e).forEach((rank) => {
      Object.keys(D.e2e[rank]).forEach((inv) => {
        const sp = D.e2e[rank][inv];
        const get = (n) => (sp[n] ? sp[n].us : null);
        rows.push({
          rank: rank, inv: +inv,
          dev: get('chip.run.runner_run.device_wall'),
          sched: get('chip.run.runner_run.device_wall.sched'),
          build: get('chip.run.runner_run.device_wall.graph_build'),
          orch: get('chip.run.runner_run.device_wall.orch'),
          host: get('chip.run.runner_run'),
          bind: get('chip.run.bind'),
          traced: TRACE_MATCH[rank] && TRACE_MATCH[rank].inv === +inv,
        });
      });
    });
    const maxDev = Math.max.apply(null, rows.map((r) => r.dev));
    const secTab = el('section');
    secTab.appendChild(sectionHead('调用采样', 'device_wall 为设备时钟',
      el('span', 'tc-readout', '点行 = 选中要带去 L2 / L1 的 rank')));
    secTab.appendChild(table([
      { label: 'Rank', key: 'rank', mono: true },
      { label: 'inv', key: 'inv', num: true },
      {
        label: 'device_wall', num: true,
        cell: (r) => (r.traced ? '<span class="ok">' : '<span>') + num(r.dev, 1) + '</span>',
      },
      { label: '', cell: (r) => bar(r.dev / maxDev, r.dev === maxDev ? 'warn' : 'neutral') },
      { label: 'sched', key: 'sched', num: true, cell: (r) => num(r.sched, 1) },
      { label: 'graph_build', num: true, cell: (r) => (r.build > r.sched ? '<span class="warn">' : '<span>') + num(r.build, 1) + '</span>' },
      { label: 'orch', num: true, cell: (r) => num(r.orch, 2) },
      { label: 'host runner_run', num: true, cell: (r) => num(r.host, 1) },
      { label: 'trace', cell: (r) => (r.traced ? '<span class="ok">已采</span>' : '—') },
    ], rows.map((r) => Object.assign(r, { __selected: r.rank === S.rank && r.traced })), {
      onPick: (r) => { S.rank = r.rank; S.focus = null; render(); },
    }));
    stage.appendChild(secTab);

    /* --- hierarchical span breakdown, both ranks on one shared scale --- */
    const rankKeys = Object.keys(D.ranks);
    const spans = {};
    rankKeys.forEach((r) => { spans[r] = D.e2e[r][TRACE_MATCH[r].inv]; });
    /* one denominator for every bar, so the two columns compare directly */
    const total = Math.max.apply(null, rankKeys.map((r) => spans[r]['chip.run'].us));
    const secBreak = el('section');
    secBreak.appendChild(sectionHead('调用剖分', rankKeys.length + ' rank · 共用刻度 · STRACE host span，device_wall 及子段为设备时钟'));
    const rowsWrap = el('div', 'tc-spanrows');
    const shead = el('div', 'tc-spanrow tc-spanhead');
    shead.appendChild(el('span', 'lbl', 'span'));
    rankKeys.forEach((r) => {
      shead.appendChild(el('span', 'h' + (r === S.rank ? ' is-armed' : ''), r + ' inv=' + TRACE_MATCH[r].inv));
      shead.appendChild(el('span', 'h val' + (r === S.rank ? ' is-armed' : ''), 'us'));
    });
    rowsWrap.appendChild(shead);
    SPAN_TREE.forEach((entry) => {
      const name = entry[0];
      const depth = entry[1];
      if (!rankKeys.some((r) => spans[r][name])) return;
      const row = el('div', 'tc-spanrow');
      row.dataset.depth = depth;
      row.appendChild(el('span', 'lbl', name.replace(/^chip\.run\.?/, '') || 'chip.run'));
      const tone = /graph_build/.test(name) ? 'warn' : /device_wall$/.test(name) ? 'good' : /sched/.test(name) ? 'neutral' : null;
      rankKeys.forEach((r) => {
        const span = spans[r][name];
        if (!span) {
          row.appendChild(el('span', 'tc-bar-empty'));
          row.appendChild(el('span', 'val muted', '—'));
          return;
        }
        row.appendChild(bar(span.us / total, tone));
        row.appendChild(el('span', 'val' + (r === S.rank ? ' is-armed' : ''), num(span.us, 2)));
      });
      rowsWrap.appendChild(row);
    });
    secBreak.appendChild(rowsWrap);
    stage.appendChild(secBreak);

    /* --- reconciliation: host span vs device trace --- */
    const recSec = el('section');
    recSec.appendChild(sectionHead('对账', 'host sched ↔ device trace'));
    const recRows = Object.keys(D.ranks).map((rank) => {
      const m = TRACE_MATCH[rank];
      const sw = D.ranks[rank].swimlane;
      return {
        rank: rank, inv: m.inv, host: m.hostUs, trace: sw.spanUs, diff: m.diff,
        tasks: D.ranks[rank].tasks.length,
        blocks: sw.blocks.reduce((a, b) => a + b.length, 0),
        crit: D.ranks[rank].critical.tags.length,
        aic: D.ranks[rank].occupancy.aicUtil,
        aiv: D.ranks[rank].occupancy.aivUtil,
        __selected: rank === S.rank,
      };
    });
    recSec.appendChild(table([
      { label: 'Rank', key: 'rank', mono: true },
      { label: '匹配 inv', key: 'inv', num: true },
      { label: 'host sched', num: true, cell: (r) => num(r.host, 1) },
      { label: 'trace span', num: true, cell: (r) => num(r.trace, 1) },
      { label: '偏差', num: true, cell: (r) => (r.diff / r.host < 0.02 ? '<span class="ok">' : '<span class="warn">') + num(r.diff, 1) + ' us</span>' },
      { label: '任务', key: 'tasks', num: true },
      { label: '块', key: 'blocks', num: true },
      { label: '依赖关键路径', cell: (r) => r.crit + ' 节点', num: true },
      { label: 'AIC 占用', num: true, cell: (r) => pct(r.aic) },
      { label: 'AIV 占用', num: true, cell: (r) => pct(r.aiv) },
    ], recRows, { onPick: (r) => { S.rank = r.rank; render(); } }));
    stage.appendChild(recSec);
    renderFuncSummary(stage);
  }

  /* The L2 dump has no host STRACE log, so there is no end-to-end layer to
   * show. This is a state, not an error: name the missing artifact, say what
   * it would have answered, and point at the layer that still works. */
  function viewE2EAbsent(stage) {
    const sec = el('section');
    sec.appendChild(sectionHead('端到端', '本 dump 缺少 host STRACE log'));
    sec.appendChild(table([
      { label: '缺失产物', cell: (r) => esc(r[0]), mono: true },
      { label: '本可回答', cell: (r) => esc(r[1]) },
      { label: '状态', cell: () => '<span class="bad">缺失</span>' },
    ], [
      ['dfx_outputs/**/host.*.log', 'chip.run / bind / runner_run / device_wall 的 span 树'],
      ['  └ inv=', '本次录制里程序被调用了几次（迭代次数 n）'],
      ['  └ device_wall', '设备墙钟，调优的主指标与复测基准'],
      ['  └ bind.prebuilt', 'JIT 建图是否命中缓存，第一次调用能不能用'],
      ['distributed_meta.json', '绑定参数的 shape / dtype，case 是否固定'],
    ], {}));
    stage.appendChild(sec);

    const alt = el('section');
    alt.appendChild(sectionHead('仍然可测', '设备侧 trace 完整'));
    const R0 = R();
    alt.appendChild(tiles([
      { k: 'trace span', v: num(R0.swimlane.spanUs, 1), u: 'us（设备钟）' },
      { k: '任务', v: String(R0.tasks.length) },
      { k: '块', v: String(R0.swimlane.blocks.reduce((a, b) => a + b.length, 0)) },
      { k: '依赖关键路径', v: R0.critical.tags.length, u: '节点 · 静态 CPM' },
      { k: 'AIC 占用', v: pct(R0.occupancy.aicUtil), tone: R0.occupancy.aicUtil < 40 ? 'bad' : 'good' },
      { k: 'AIV 占用', v: pct(R0.occupancy.aivUtil), tone: R0.occupancy.aivUtil < 40 ? 'bad' : null },
    ]));
    const jump = el('div', 'tc-actions');
    jump.appendChild(btn('去 L2 调度', { on: () => { S.view = 'l2'; render(); } }));
    jump.appendChild(btn('去 ISA / 布局', { variant: 'ghost', on: () => { S.view = 'isa'; render(); } }));
    alt.appendChild(jump);
    stage.appendChild(alt);
  }

  /* ========================================================= L2 view */
  function laneRows() {
    const all = R().swimlane.lanes;
    if (S.laneFilter === 'aic') return all.filter((l) => l.kind === 'aic');
    if (S.laneFilter === 'aiv') return all.filter((l) => l.kind === 'aiv');
    return all;
  }

  function viewL2(stage) {
    const rank = R();
    const crit = rank.critical;
    const critSet = {};
    crit.tags.forEach((t) => { critSet[t] = 1; });
    /* whichever path the "只看" filter is pointed at */
    const pathSet = {};
    (S.pathOnly === 'cpm' ? crit.tags : rank.cpath.segments.map((sg) => sg.tag))
      .forEach((t) => { pathSet[t] = 1; });
    const subj = subjectTaskSet();      /* tag -> 1-based marker number */
    const subjLane = subjectLaneSet();
    const hasSubjects = Object.keys(subj).length > 0;
    const dim = S.focusEvidence && (hasSubjects || Object.keys(subjLane).length > 0);

    /* --- path ribbon -------------------------------------------------
     * Drawn on the real time axis, so it shows the path that actually
     * tiles that axis: the observed blame walk. The dependency floor does
     * NOT tile it (14 nodes, 3066.8 + 185.8 gap against a 4879.8 makespan),
     * so putting it here used to make the header claim "走完 4879.8 us"
     * about a chain that covers two thirds of it. CPM nodes are ticked. */
    const cp = rank.cpath;
    const cpmOnPath = cp.segments.filter((sg) => sg.onCpm).length;
    const ribSec = el('section');
    ribSec.appendChild(sectionHead('观测路径 · ' + cp.segments.length + ' 节点',
      '计算 ' + us(cp.computeTotal) + ' + stall ' + us(cp.stallTotal) + ' = ' + us(cp.makespan),
      el('span', 'tc-readout', '其中 ' + cpmOnPath + ' 个也在依赖关键路径上（共 '
        + crit.tags.length + ' 个 · ' + us(crit.chainSpan) + '）')));
    const ribHost = el('div', 'tc-canvas-strip');
    const ribCanvas = el('canvas');
    ribHost.appendChild(ribCanvas);
    ribSec.appendChild(ribHost);
    stage.appendChild(ribSec);

    /* --- worker swimlane --- */
    const laneSec = el('section', 'tc-stage-fill');
    /* A 62-way categorical scale has no readable legend. With per-operator
     * colouring the hue is an identity key for telling neighbouring blocks
     * apart, not something to look up — the name comes from hover or the
     * scope ranking. So: state the scale, don't enumerate it. */
    const legend = el('div', 'tc-legend');
    if (S.colorOn && S.colorMode !== 'engine') {
      legend.appendChild(el('span', 'tc-readout',
        rank.scopes.length + ' scope 各一色 · 颜色只用于区分相邻块，名字看悬停或右侧排行'));
    } else if (!S.colorOn) {
      const s0 = el('span');
      const i0 = el('i');
      i0.style.background = cssVar('--surface-4');
      s0.appendChild(i0);
      s0.appendChild(el('span', null, '任务（配色已关）'));
      legend.appendChild(s0);
      const s1 = el('span');
      const i1 = el('i');
      i1.style.background = cssVar('--warning');
      s1.appendChild(i1);
      s1.appendChild(el('span', null, '空转窗口 ' + (rank.idleRuns || []).length + ' 段'));
      legend.appendChild(s1);
      const s2 = el('span');
      const i2 = el('i');
      i2.style.background = cssVar('--danger');
      s2.appendChild(i2);
      s2.appendChild(el('span', null, '依赖关键路径 ' + crit.tags.length + ' 节点'));
      legend.appendChild(s2);
    } else if (S.colorMode === 'engine') {
      [['aic', 'AIC'], ['aiv', 'AIV'], ['mix', 'MIX']].forEach((p) => {
        const s = el('span');
        const i = el('i');
        i.style.background = CMAP.colorForTask({ laneKind: p[0] }, 'engine');
        s.appendChild(i);
        s.appendChild(el('span', null, p[1]));
        legend.appendChild(s);
      });
    }
    laneSec.appendChild(sectionHead('Chip swimlane · ' + rank.swimlane.lanes.length + ' core lane',
      laneRows().length + ' 泳道 · ' + rank.swimlane.blocks.reduce((a, b) => a + b.length, 0) + ' 块', legend));
    const laneHost = el('div', 'tc-canvas-host');
    const laneCanvas = el('canvas', 'tc-lanes');
    laneCanvas.tabIndex = 0;
    laneHost.appendChild(laneCanvas);
    laneSec.appendChild(laneHost);
    stage.appendChild(laneSec);

    /* ---------- rendering ---------- */
    const LBL = 66;
    function drawRibbon() {
      const w = ribHost.clientWidth || 800;
      const evRow = hasSubjects ? 26 : 0;
      const h = 74 + evRow;
      const ctx = fitCanvas(ribCanvas, w, h);
      const plotX = LBL, plotW = Math.max(40, w - LBL - 10);
      drawTimeRuler(ctx, plotX, plotW, 0, S.t0, S.t1, { dense: true });
      ctx.font = '500 11px ' + cssVar('--font-sans');
      ctx.fillStyle = cssVar('--foreground-muted');
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      const sx = (t) => plotX + ((t - S.t0) / (S.t1 - S.t0)) * plotW;

      /* evidence row: where the active finding's subjects sit on this axis */
      if (evRow) {
        ctx.fillStyle = cssVar('--foreground');
        ctx.fillText('证据', 4, 40);
        Object.keys(subj).forEach((tag) => {
          const t = tasksOf[S.rank][tag];
          if (!t) return;
          const x = clamp(sx(t.start), plotX, plotX + plotW);
          const x2 = clamp(sx(t.end), plotX, plotX + plotW);
          if (x2 <= plotX || x >= plotX + plotW) return;
          SW.drawTaskBar(ctx, {
            task: barTask(t, null, 'evidence'),
            x: x, y: 32, width: Math.max(3, x2 - x), height: 16,
            baseColor: CMAP.colorForTask({ colorKey: t.callable, label: t.callable }, 'semantic'),
            isSelected: true,
            isEmphasized: t.tag === S.task,
            fontFamily: cssVar('--font-sans'),
          });
          /* numbered marker matching the evidence chip above */
          const cx = Math.min(plotX + plotW - 7, Math.max(plotX + 7, x + 7));
          ctx.beginPath();
          ctx.arc(cx, 28, 7, 0, Math.PI * 2);
          ctx.fillStyle = cssVar('--background');
          ctx.fill();
          ctx.strokeStyle = cssVar('--foreground');
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.fillStyle = cssVar('--foreground');
          ctx.font = '700 9px ' + cssVar('--font-mono');
          ctx.textAlign = 'center';
          ctx.fillText(String(subj[tag]), cx, 28);
          ctx.textAlign = 'left';
          ctx.font = '500 11px ' + cssVar('--font-sans');
        });
      }

      ctx.fillStyle = cssVar('--foreground-muted');
      ctx.fillText('OBS PATH', 4, 40 + evRow);
      ctx.fillText('STALL', 4, 62 + evRow);
      let cursor = null;
      cp.segments.forEach((node) => {
        const t = tasksOf[S.rank][node.tag];
        if (!t) return;
        const x = sx(t.start), x2 = sx(t.end);
        if (x2 < plotX || x > plotX + plotW) { cursor = t.end; return; }
        ctx.globalAlpha = dim && !subj[t.tag] ? 0.28 : 1;
        SW.drawTaskBar(ctx, {
          task: barTask(t, null, 'critical'),
          x: Math.max(plotX, x), y: 31 + evRow, width: Math.max(2, Math.min(plotX + plotW, x2) - Math.max(plotX, x)), height: 18,
          baseColor: taskColor(t),
          isSelected: !!subj[t.tag] || t.tag === S.task,
          isEmphasized: true,
          fontFamily: cssVar('--font-sans'),
        });
        ctx.globalAlpha = 1;
        /* a tick above the bar marks a node that is ALSO on the dependency
         * floor -- touching one of those lowers the floor, not just the stall */
        if (node.onCpm) {
          ctx.fillStyle = cssVar('--foreground');
          ctx.fillRect(Math.max(plotX, x), 27 + evRow, Math.max(2, Math.min(plotX + plotW, x2) - Math.max(plotX, x)), 2);
        }
        /* gap markers are not task bars: page-local data-viz marks */
        if (cursor !== null && t.start > cursor) {
          const gx = sx(cursor), gx2 = sx(t.start);
          ctx.fillStyle = cssVar('--warning');
          ctx.globalAlpha = 0.5;
          ctx.fillRect(Math.max(plotX, gx), 56 + evRow, Math.max(1, gx2 - gx), 8);
          ctx.globalAlpha = 1;
        } else if (cursor !== null && t.start < cursor) {
          const ox = sx(t.start), ox2 = sx(cursor);
          ctx.fillStyle = cssVar('--success');
          ctx.globalAlpha = 0.4;
          ctx.fillRect(Math.max(plotX, ox), 58 + evRow, Math.max(1, ox2 - ox), 4);
          ctx.globalAlpha = 1;
        }
        cursor = t.end;
      });
    }

    /* Worker rows are deliberately roomier than the compact scheduler overlay:
     * the worker view is where readers compare adjacent cores. The extra gap
     * makes individual bars and lane labels scannable without turning the
     * trace into a solid colour field. */
    const ROW_H = 11, ROW_GAP = 3, ENGINE_GAP = 7;
    const SCHED_ROW_H = 9, SCHED_ROW_GAP = 2;
    let laneLayout = [];
    function drawLanes() {
      const lanes = laneRows();
      const w = laneHost.clientWidth || 800;
      const plotX = LBL, plotW = Math.max(40, w - LBL - 10);
      const overlayRows = S.overlay === 'sched' ? rank.scheduler.lanes.length : 0;
      const readyH = S.overlay === 'ready' ? 46 : 0;
      const OCC_H = 26;
      const schedH = overlayRows ? overlayRows * (SCHED_ROW_H + SCHED_ROW_GAP) + 8 : 0;
      const top = 30 + OCC_H + schedH + readyH;
      const workerRows = [];
      let workerBottom = top;
      lanes.forEach((lane, i) => {
        workerRows.push(workerBottom);
        workerBottom += ROW_H + ROW_GAP;
        if (i < lanes.length - 1 && lane.kind !== lanes[i + 1].kind) workerBottom += ENGINE_GAP;
      });
      const h = workerBottom + 8;
      const ctx = fitCanvas(laneCanvas, w, Math.max(h, laneHost.clientHeight || h));
      const sx = (t) => plotX + ((t - S.t0) / (S.t1 - S.t0)) * plotW;
      drawTimeRuler(ctx, plotX, plotW, 0, S.t0, S.t1, { dense: true });
      ctx.font = '500 10px ' + cssVar('--font-sans');
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      laneLayout = [];

      /* ---- occupancy band ----
       * Two stacked strips (AIC, AIV) whose opacity tracks how many cores were
       * busy in that window. The idle stretches are the point: they get a
       * warning-tinted wash so a low-occupancy window is visible without
       * reading 72 lanes of bars. */
      (function drawOccBand() {
        const wins = rank.occWindows;
        if (!wins || !wins.length) return;
        const ww = rank.occWindowUs;
        const bandY = 30;
        const strip = (OCC_H - 4) / 2;
        ctx.fillStyle = cssVar('--foreground-muted');
        ctx.font = '500 10px ' + cssVar('--font-sans');
        ctx.fillText('AIC', 4, bandY + strip / 2);
        ctx.fillText('AIV', 4, bandY + strip + 2 + strip / 2);
        wins.forEach((win) => {
          const x = sx(win[0]), x2 = sx(win[0] + ww);
          if (x2 < plotX || x > plotX + plotW) return;
          const xa = Math.max(plotX, x);
          const wpx = Math.max(0.8, Math.min(plotX + plotW, x2) - xa);
          [[win[1], bandY, 'aic'], [win[2], bandY + strip + 2, 'aiv']].forEach((cfg) => {
            ctx.fillStyle = CMAP.colorForLaneKind(cfg[2]);
            ctx.globalAlpha = 0.12 + (clamp(cfg[0], 0, 100) / 100) * 0.85;
            ctx.fillRect(xa, cfg[1], wpx, strip);
            ctx.globalAlpha = 1;
          });
        });
        /* highlight the stretches where both engines were under the threshold */
        (rank.idleRuns || []).forEach((r) => {
          const x = sx(r.t0), x2 = sx(r.t1);
          if (x2 < plotX || x > plotX + plotW) return;
          const xa = Math.max(plotX, x);
          const wpx = Math.min(plotX + plotW, x2) - xa;
          if (wpx < 1) return;
          ctx.fillStyle = cssVar('--warning');
          ctx.globalAlpha = 0.14;
          ctx.fillRect(xa, bandY, wpx, h - bandY - 4);
          ctx.globalAlpha = 0.75;
          ctx.strokeStyle = cssVar('--warning');
          ctx.setLineDash([2, 2]);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(Math.round(xa) + 0.5, bandY);
          ctx.lineTo(Math.round(xa) + 0.5, h - 4);
          ctx.moveTo(Math.round(xa + wpx) - 0.5, bandY);
          ctx.lineTo(Math.round(xa + wpx) - 0.5, h - 4);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.globalAlpha = 1;
          if (wpx > 52) {
            ctx.fillStyle = cssVar('--warning');
            ctx.font = '500 10px ' + cssVar('--font-sans');
            ctx.textAlign = 'center';
            ctx.fillText(num(r.us, 0) + ' us 空转', xa + wpx / 2, bandY - 6);
            ctx.textAlign = 'left';
          }
        });
        ctx.font = '500 10px ' + cssVar('--font-sans');
      })();

      /* AICPU scheduler lanes */
      if (overlayRows) {
        rank.scheduler.lanes.forEach((name, i) => {
          const y = 30 + OCC_H + i * (SCHED_ROW_H + SCHED_ROW_GAP);
          ctx.fillStyle = cssVar('--foreground-muted');
          ctx.fillText(name, 4, y + SCHED_ROW_H / 2);
          rank.scheduler.blocks[i].forEach((b) => {
            const x = sx(b[0]), x2 = sx(b[0] + b[1]);
            if (x2 < plotX || x > plotX + plotW) return;
            ctx.fillStyle = CMAP.colorForLaneKind('aicpu');
            ctx.globalAlpha = b[2] === 'complete' ? 0.95 : b[2] === 'dispatch' ? 0.7 : 0.45;
            ctx.fillRect(Math.max(plotX, x), y, Math.max(0.8, Math.min(plotX + plotW, x2) - Math.max(plotX, x)), SCHED_ROW_H);
            ctx.globalAlpha = 1;
          });
        });
      }

      /* ready-but-undispatched strip */
      if (readyH) {
        const y0 = 32 + OCC_H, hh = readyH - 8;
        const peak = Math.max(rank.readyStat.peak.AIC, rank.readyStat.peak.AIV, 1);
        ctx.fillStyle = cssVar('--foreground-muted');
        ctx.fillText('READY', 4, y0 + hh / 2);
        [['AIC', 1, '--danger'], ['AIV', 2, '--warning']].forEach((cfg) => {
          ctx.beginPath();
          ctx.moveTo(plotX, y0 + hh);
          rank.readyQueue.forEach((q) => {
            const x = clamp(sx(q[0]), plotX, plotX + plotW);
            const y = y0 + hh - (q[cfg[1]] / peak) * hh;
            ctx.lineTo(x, y);
          });
          ctx.lineTo(plotX + plotW, y0 + hh);
          ctx.closePath();
          ctx.fillStyle = cssVar(cfg[2]);
          ctx.globalAlpha = 0.28;
          ctx.fill();
          ctx.globalAlpha = 1;
        });
        ctx.fillStyle = cssVar('--foreground-muted');
        ctx.textAlign = 'right';
        ctx.fillText('peak ' + peak, plotX + plotW - 2, y0 + 5);
        ctx.textAlign = 'left';
      }

      /* worker lanes */
      const markers = [];
      lanes.forEach((lane, i) => {
        const y = workerRows[i];
        const li = rank.swimlane.laneNames.indexOf(lane.name);
        laneLayout.push({ y: y, laneIdx: li, name: lane.name });
        const laneIsSubject = !!subjLane[lane.name];
        /* A barely-there band restores the row rhythm in a dense trace while
         * leaving task colour and idle washes as the primary signals. */
        ctx.fillStyle = cssVar('--surface-2');
        ctx.globalAlpha = i % 2 ? 0.32 : 0.16;
        ctx.fillRect(plotX, y - 1, plotW, ROW_H + 2);
        ctx.globalAlpha = 1;
        if (laneIsSubject) {
          ctx.fillStyle = cssVar('--warning');
          ctx.globalAlpha = 0.12;
          ctx.fillRect(plotX, y - 2, plotW, ROW_H + 4);
          ctx.globalAlpha = 1;
        }
        ctx.fillStyle = laneIsSubject ? cssVar('--warning')
          : lane.util > 60 ? cssVar('--foreground-secondary') : cssVar('--foreground-muted');
        ctx.font = (laneIsSubject ? '600' : '500') + ' 10px ' + cssVar('--font-sans');
        ctx.fillText(lane.name, 4, y + ROW_H / 2);
        rank.swimlane.blocks[li].forEach((b) => {
          const t = rank.tasks[b[2]];
          if (S.critOnly && !pathSet[t.tag]) return;
          const x = sx(b[0]), x2 = sx(b[0] + b[1]);
          if (x2 < plotX || x > plotX + plotW) return;
          const xa = Math.max(plotX, x);
          const wBar = Math.max(0.8, Math.min(plotX + plotW, x2) - xa);
          const isSubj = !!subj[t.tag];
          if (isSubj) markers.push({ x: xa, y: y, n: subj[t.tag] });
          ctx.globalAlpha = dim && !isSubj && !laneIsSubject ? 0.16 : 1;
          if (wBar < 2.2) {
            /* below task-bar legibility: draw a density tick, not a fake bar */
            ctx.fillStyle = taskColor(t);
            ctx.fillRect(xa, y, wBar, ROW_H);
            ctx.globalAlpha = 1;
            return;
          }
          SW.drawTaskBar(ctx, {
            task: barTask(t, b, lane.name),
            x: xa, y: y, width: wBar, height: ROW_H, radius: 1,
            baseColor: taskColor(t),
            isSelected: isSubj || t.tag === S.task,
            isRelated: !isSubj && t.tag !== S.task && !!pathSet[t.tag] && !S.critOnly,
            isEmphasized: isSubj,
            fontFamily: cssVar('--font-sans'),
          });
          ctx.globalAlpha = 1;
        });
        /* A wider separator at each engine boundary makes the two core pools
         * legible even when the chart is scrolled. */
        if (i < lanes.length - 1 && lane.kind !== lanes[i + 1].kind) {
          const separatorY = y + ROW_H + (ROW_GAP + ENGINE_GAP) / 2;
          ctx.strokeStyle = cssVar('--border-default');
          ctx.globalAlpha = 0.75;
          ctx.beginPath();
          ctx.moveTo(4, separatorY + 0.5);
          ctx.lineTo(plotX + plotW, separatorY + 0.5);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      });

      /* numbered markers matching the evidence chips, drawn last so nothing covers them */
      const seen = {};
      markers.forEach((m) => {
        if (seen[m.n]) return;
        seen[m.n] = 1;
        const cx = clamp(m.x, plotX + 7, plotX + plotW - 7);
        ctx.beginPath();
        ctx.arc(cx, m.y + ROW_H / 2, 7, 0, Math.PI * 2);
        ctx.fillStyle = cssVar('--background');
        ctx.fill();
        ctx.strokeStyle = cssVar('--foreground');
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.fillStyle = cssVar('--foreground');
        ctx.font = '700 9px ' + cssVar('--font-mono');
        ctx.textAlign = 'center';
        ctx.fillText(String(m.n), cx, m.y + ROW_H / 2);
        ctx.textAlign = 'left';
        ctx.font = '500 10px ' + cssVar('--font-sans');
      });

      /* auto-scroll a chip-selected lane into view */
      if (S.scrollToLane) {
        const row = laneLayout.find((r) => r.name === S.scrollToLane);
        if (row) laneHost.scrollTop = Math.max(0, row.y - laneHost.clientHeight / 2);
        S.scrollToLane = null;
      }
    }

    function hitTest(event) {
      const rect = laneCanvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const plotX = LBL, plotW = Math.max(40, (laneHost.clientWidth || 800) - LBL - 10);
      if (x < plotX) return null;
      const t = S.t0 + ((x - plotX) / plotW) * (S.t1 - S.t0);
      const row = laneLayout.find((r) => y >= r.y - 1 && y <= r.y + ROW_H + 1);
      if (!row) return null;
      const tolerance = ((S.t1 - S.t0) / plotW) * 2;
      const blocks = rank.swimlane.blocks[row.laneIdx];
      for (let i = 0; i < blocks.length; i++) {
        const b = blocks[i];
        if (t >= b[0] - tolerance && t <= b[0] + b[1] + tolerance) {
          const task = rank.tasks[b[2]];
          if (S.critOnly && !pathSet[task.tag]) continue;
          return { task: task, block: b, lane: row.name };
        }
      }
      return null;
    }

    attachTooltip(laneHost, laneCanvas, (event) => {
      const hit = hitTest(event);
      return hit ? barTask(hit.task, hit.block, hit.lane) : null;
    });
    laneCanvas.addEventListener('click', (event) => {
      const hit = hitTest(event);
      if (!hit) return;
      S.task = hit.task.tag;
      S.focus = 'task';
      renderInspector();
      drawLanes();
      drawRibbon();
    });

    /* horizontal pan by drag */
    let drag = null;
    laneCanvas.addEventListener('pointerdown', (event) => {
      if (event.button !== 0 || !event.shiftKey) return;
      drag = { x: event.clientX, t0: S.t0, t1: S.t1 };
      laneHost.classList.add('is-grabbing');
      laneCanvas.setPointerCapture(event.pointerId);
    });
    laneCanvas.addEventListener('pointermove', (event) => {
      if (!drag) return;
      const plotW = Math.max(40, (laneHost.clientWidth || 800) - LBL - 10);
      const dt = ((event.clientX - drag.x) / plotW) * (drag.t1 - drag.t0);
      setWindow(drag.t0 - dt, drag.t1 - dt);
      drawLanes(); drawRibbon(); renderToolbar();
    });
    const endDrag = () => { drag = null; laneHost.classList.remove('is-grabbing'); };
    laneCanvas.addEventListener('pointerup', endDrag);
    laneCanvas.addEventListener('pointercancel', endDrag);

    const redraw = () => { drawRibbon(); drawLanes(); };
    requestAnimationFrame(redraw);
    stage.__redraw = redraw;
    if (stage.__ro) stage.__ro.disconnect();
    stage.__ro = new ResizeObserver(() => redraw());
    stage.__ro.observe(laneHost);
    stage.__ro.observe(ribHost);
  }

  function setWindow(a, b) {
    const span = R().swimlane.spanUs;
    let t0 = a, t1 = b;
    const w = t1 - t0;
    if (w >= span) { S.t0 = 0; S.t1 = span; return; }
    if (t0 < 0) { t0 = 0; t1 = w; }
    if (t1 > span) { t1 = span; t0 = span - w; }
    S.t0 = t0; S.t1 = t1;
  }
  function zoom(factor) {
    const mid = (S.t0 + S.t1) / 2;
    const half = ((S.t1 - S.t0) / 2) * factor;
    setWindow(mid - half, mid + half);
  }

  /* ====================================================== L1 / L0 view */
  const DTYPES = [
    { id: 'int8', label: 'INT8', bytes: 1, mult: 512 },
    { id: 'bf16', label: 'BF16', bytes: 2, mult: 256 },
    { id: 'fp16', label: 'FP16', bytes: 2, mult: 256 },
    { id: 'fp32', label: 'FP32', bytes: 4, mult: 128 },
  ];
  const ACC_DTYPES = [
    { id: 'int32', label: 'INT32', bytes: 4 },
    { id: 'fp32', label: 'FP32', bytes: 4 },
  ];
  const dt = (id) => DTYPES.find((d) => d.id === id) || DTYPES[1];
  const adt = (id) => ACC_DTYPES.find((d) => d.id === id) || ACC_DTYPES[1];

  function defaultTile() {
    /* seeded from the AutoTileMatmulL0 dump: the qr_acc K-loop this run actually emitted
     * (Left INT8[16,64], Right INT8[64,512], Acc INT32[16,512], stage=2) */
    return { m: 16, n: 512, k: 64, ab: 'int8', acc: 'int32', live: 1, depth: 2 };
  }

  function tileBudget() {
    const T = S.tile || (S.tile = defaultTile());
    const ab = dt(T.ab), ac = adt(T.acc);
    const left = T.m * T.k * ab.bytes;
    const right = T.k * T.n * ab.bytes;
    const acc = T.m * T.n * ac.bytes * Math.max(1, T.live);
    const freeLR = D.budgets.Right ? D.budgets.Right.freeB : null;
    const accObserved = Math.max.apply(null, D.l0Tiles.filter((x) => x.mem === 'Acc').map((x) => x.bytes));
    const innermost = T.n * ab.bytes;
    const cacheLine = (D.hints.find((h) => h.cacheLineB) || {}).cacheLineB || 512;
    const need = Math.max(left, right) * T.depth;
    /* The run's own counter-example: qkv_proj_rope.py:375 asks for depth 2 with
     * 32768 B per stage against 65536 B free — exactly 100% — and MemoryReuse
     * still fits only one buffer, because co-resident tiles take part of that
     * space. So "exactly at the limit" is a fail in practice, not a pass. */
    const ratio = freeLR ? need / freeLR : null;
    return {
      T: T, ab: ab, ac: ac, left: left, right: right, acc: acc,
      freeLR: freeLR, accObserved: accObserved,
      innermost: innermost, cacheLine: cacheLine,
      depthNeed: need, depthRatio: ratio,
      depthState: ratio == null ? 'warn' : ratio > 1 ? 'fail' : ratio === 1 ? 'fail' : ratio > 0.8 ? 'warn' : 'pass',
      atLimit: ratio === 1,
      lineOk: innermost >= cacheLine,
      needElems: Math.ceil(cacheLine / ab.bytes),
    };
  }

  function viewL1(stage) {
    const rank = R();
    const t = curTask();
    const role = pathRole(rank, t.tag);

    /* --- identity + measured split --- */
    const idSec = el('section');
    idSec.appendChild(sectionHead(t.callable, t.tag + ' · ' + t.kind.toUpperCase() + ' · task ' + t.id
      + ' · ' + t.kernelCount + ' kernel',
      (function () {
        const ro = el('span', 'tc-readout' + (role.onCpm ? ' is-crit' : ''), pathLabel(role));
        ro.title = pathTitle(role);
        return ro;
      })()));
    const kernelMean = t.kdurSum / t.blockCount;
    idSec.appendChild(tiles([
      { k: 'span', v: num(t.span, 1), u: 'us' },
      { k: '块 / 核', v: t.blockCount + ' / ' + t.coreCount, u: 'block_num ' + t.blockNum },
      { k: '块中位', v: num(t.durMed, 2), u: 'us' },
      { k: '块最长', v: num(t.durMax, 2), u: 'us', tone: t.imbalance > 3 ? 'warn' : null },
      { k: '离散度', v: num(t.imbalance, 2) + 'x', u: 'max / med', tone: t.imbalance > 3 ? 'bad' : t.imbalance > 2 ? 'warn' : 'good' },
      { k: 'kernel 均值', v: num(kernelMean, 2), u: 'us' },
      { k: 'setup 均值', v: num(t.setupMean, 2), u: pct(t.setupShare * 100, 0) + ' of block', tone: t.setupShare > 0.3 ? 'bad' : t.setupShare > 0.1 ? 'warn' : null },
      { k: 'AICPU 视角', v: num(t.svAicpuMean, 1), u: t.svOverhead != null ? '+' + num(t.svOverhead, 1) + ' us hand-off' : '', tone: t.svOverhead > 20 ? 'bad' : t.svOverhead > 5 ? 'warn' : null },
    ]));
    stage.appendChild(idSec);

    /* --- one scope, two kernels: the Cube/Vec pairing, measured --- */
    if (t.kernelCount > 1) {
      const pairSec = el('section');
      pairSec.appendChild(sectionHead('引擎配对',
        t.kernels.map((k) => k.name).join(' + ') + ' · 同一次 Group launch',
        el('span', 'tc-readout', '最长块比 ' + t.pairRatio + 'x')));
      pairSec.appendChild(table([
        { label: 'kernel', cell: (k) => esc(k.name), mono: true },
        { label: 'FuncId', cell: (k) => String(k.funcId), mono: true, num: true },
        { label: '引擎', cell: (k) => (k.engine === 'aic' ? 'AIC (Cube)' : 'AIV (Vec)') },
        { label: '块 / 核', cell: (k) => k.blocks + ' / ' + k.cores, num: true },
        { label: 'core-time', cell: (k) => num(k.coreTime, 1), mono: true, num: true },
        { label: '最长块', cell: (k) => num(k.durMax, 2), mono: true, num: true },
        { label: '均值', cell: (k) => num(k.durMean, 2), mono: true, num: true },
      ], t.kernels, {}));
      pairSec.appendChild(el('p', 'tc-note',
        '两半共用一个 taskId，只有 event-hint 里的 FuncId 能把它们分开 —— 按 taskId 汇总会把 Vec 侧的 '
        + num(t.engines.aiv.coreTime, 0) + ' us 记到 Cube 侧的名下。'
        + '两侧最长块 ' + num(t.engines.aic.durMax, 1) + ' / ' + num(t.engines.aiv.durMax, 1)
        + ' us，而整段 span 只有 ' + num(t.span, 1) + ' us：'
        + (t.pairRatio < 1.3 ? '两半没有错开，是块内串行。' : '慢的一侧决定整块时长。')));
      stage.appendChild(pairSec);
    }

    /* --- three measurements of the same block, side by side --- */
    const splitSec = el('section');
    splitSec.appendChild(sectionHead('一个块的三种口径',
      'kernel · +local_setup · +hand-off (dispatch→finish)'));
    const splitHost = el('div', 'tc-canvas-strip');
    const splitCanvas = el('canvas');
    splitHost.appendChild(splitCanvas);
    splitSec.appendChild(splitHost);
    stage.appendChild(splitSec);

    /* --- per-core block strip + duration distribution --- */
    const distSec = el('section');
    distSec.appendChild(sectionHead('块分布',
      t.blockCount + ' 块 / ' + t.coreCount + ' 核 · ' + num(t.blockCount / t.coreCount, 2) + ' 波'));
    const distHost = el('div', 'tc-canvas-host');
    const distCanvas = el('canvas');
    distHost.appendChild(distCanvas);
    distSec.appendChild(distHost);
    stage.appendChild(distSec);

    /* --- on-chip tile budget --- */
    const calcSec = el('section');
    calcSec.appendChild(sectionHead('片上预算试算',
      'Left / Right = 编译器选的 L0A / L0B staging',
      el('span', 'tc-readout', '上限取自本 run MemoryReuse 报告')));
    calcSec.appendChild(renderCalc());
    stage.appendChild(calcSec);

    /* --- compiler hints, honestly unlinked --- */
    const hintSec = el('section');
    const mods = ['all'].concat(D.tileFiles.map((f) => f.file));
    hintSec.appendChild(sectionHead('编译提示', D.hints.length + ' 条 · 按模块聚合 · '
      + (D.sourceMap ? 'scope→源码已重建，提示仍按 hint 自带的行号' : '无 kernel→源码映射'),
      field('模块', select(mods.map((m) => ({ id: m, label: m === 'all' ? '全部模块' : m })), S.hintModule,
        (v) => { S.hintModule = v; render(); }))));
    const hintRows = D.hints
      .filter((h) => S.hintModule === 'all' || h.file === S.hintModule)
      .map((h, i) => Object.assign({ __i: i }, h));
    hintSec.appendChild(table([
      { label: 'Code', key: 'code', mono: true },
      { label: '位置', mono: true, cell: (h) => esc(h.file + ':' + h.line) },
      { label: '类型', cell: (h) => (h.kind === 'pipeline-depth' ? '流水深度' : '搬运粒度') },
      {
        label: '事实', cell: (h) => (h.kind === 'pipeline-depth'
          ? 'depth ' + h.reqDepth + ' → ' + h.fit + ' @' + h.unit + '（' + kb(h.perStageB) + '/stage，' + kb(h.freeB) + ' free）'
          : esc(h.op) + ' 末维 <span class="' + (h.innermostB < 128 ? 'bad' : 'warn') + '">' + h.innermostB + 'B</span> · tile ' + esc(h.dtype + '[' + h.tileShape + ']') + ' → ' + esc(h.mem)),
      },
      { label: '次数', key: 'occurrences', num: true },
    ], hintRows.slice(0, 80), {
      onPick: (h) => { S.view = 'compiler'; S.compilerTab = h.kind === 'pipeline-depth' ? 'depth' : 'granularity'; S.hintSite = h.file + ':' + h.line; render(); },
    }));
    if (hintRows.length > 80) {
      hintSec.appendChild(el('p', 'tc-note', '前 80 / ' + hintRows.length + ' 条 · 完整列表见 Problems 面板'));
    }
    stage.appendChild(hintSec);

    /* ---------- canvases ---------- */
    function drawSplit() {
      const w = splitHost.clientWidth || 700;
      const h = 82;
      const ctx = fitCanvas(splitCanvas, w, h);
      const rows = [
        { k: 'kernel', v: kernelMean, tone: '--success' },
        { k: '+ setup', v: t.durMean, tone: '--warning' },
        { k: '+ hand-off', v: t.svAicpuMean == null ? t.durMean : t.svAicpuMean, tone: '--danger' },
      ];
      const max = Math.max.apply(null, rows.map((r) => r.v));
      const x0 = 84, plotW = Math.max(40, w - x0 - 110);
      ctx.font = '500 11px ' + cssVar('--font-sans');
      ctx.textBaseline = 'middle';
      rows.forEach((r, i) => {
        const y = 14 + i * 22;
        ctx.textAlign = 'right';
        ctx.fillStyle = cssVar('--foreground-muted');
        ctx.fillText(r.k, x0 - 8, y + 7);
        ctx.fillStyle = cssVar('--surface-3');
        ctx.fillRect(x0, y, plotW, 14);
        ctx.fillStyle = cssVar(r.tone);
        ctx.globalAlpha = 0.85;
        ctx.fillRect(x0, y, Math.max(1, (r.v / max) * plotW), 14);
        ctx.globalAlpha = 1;
        ctx.textAlign = 'left';
        ctx.fillStyle = cssVar('--foreground');
        ctx.font = '500 11px ' + cssVar('--font-mono');
        ctx.fillText(num(r.v, 2) + ' us', x0 + plotW + 8, y + 7);
        ctx.font = '500 11px ' + cssVar('--font-sans');
      });
    }

    function drawDist() {
      const w = distHost.clientWidth || 700;
      /* per-core strip for this task only */
      const lanes = [];
      rank.swimlane.laneNames.forEach((name, li) => {
        const blocks = rank.swimlane.blocks[li].filter((b) => rank.tasks[b[2]].tag === t.tag);
        if (blocks.length) lanes.push({ name: name, blocks: blocks });
      });
      const ROW = 9, GAP = 1;
      const stripH = 22 + lanes.length * (ROW + GAP) + 10;
      const sorted = rank.swimlane.blocks.flat().filter((b) => rank.tasks[b[2]].tag === t.tag)
        .map((b) => b[1]).sort((a, b) => a - b);
      const histH = 96;
      distHost.style.height = Math.min(520, stripH + histH + 4) + 'px';
      const ctx = fitCanvas(distCanvas, w, stripH + histH);
      const x0 = 70, plotW = Math.max(40, w - x0 - 12);
      drawTimeRuler(ctx, x0, plotW, 12, t.start, t.end);
      const sx = (x) => x0 + ((x - t.start) / Math.max(1e-6, t.end - t.start)) * plotW;
      ctx.font = '500 10px ' + cssVar('--font-sans');
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'left';
      lanes.forEach((lane, i) => {
        const y = 22 + i * (ROW + GAP);
        ctx.fillStyle = cssVar('--foreground-muted');
        ctx.fillText(lane.name, 4, y + ROW / 2);
        lane.blocks.forEach((b) => {
          const x = sx(b[0]);
          const wBar = Math.max(1, sx(b[0] + b[1]) - x);
          if (wBar < 2.2) {
            ctx.fillStyle = CMAP.colorForTask({ colorKey: t.callable, label: t.callable }, 'semantic');
            ctx.fillRect(x, y, wBar, ROW);
            return;
          }
          SW.drawTaskBar(ctx, {
            task: barTask(t, b, lane.name),
            x: x, y: y, width: wBar, height: ROW, radius: 1,
            baseColor: CMAP.colorForTask({ colorKey: t.callable, label: t.callable }, 'semantic'),
            isEmphasized: b[1] >= t.durP90,
            fontFamily: cssVar('--font-sans'),
          });
        });
      });

      /* sorted block-duration profile: data-viz, not a task bar */
      const hy = stripH + 16;
      const hh = histH - 34;
      const max = sorted[sorted.length - 1] || 1;
      ctx.fillStyle = cssVar('--foreground-muted');
      ctx.font = '500 11px ' + cssVar('--font-sans');
      ctx.textAlign = 'left';
      ctx.fillText('块时长排序（' + sorted.length + ' 块，' + num(sorted[0], 2) + ' → ' + num(max, 2) + ' us）', 4, hy - 6);
      const bw = plotW / sorted.length;
      sorted.forEach((v, i) => {
        const bh = (v / max) * hh;
        ctx.fillStyle = v >= t.durP90 ? cssVar('--danger') : cssVar('--primary');
        ctx.globalAlpha = v >= t.durP90 ? 0.9 : 0.55;
        ctx.fillRect(x0 + i * bw, hy + hh - bh, Math.max(0.7, bw - 0.4), bh);
        ctx.globalAlpha = 1;
      });
      [['med', t.durMed, '--foreground-secondary'], ['p90', t.durP90, '--warning']].forEach((m) => {
        const y = hy + hh - (m[1] / max) * hh;
        ctx.strokeStyle = cssVar(m[2]);
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x0 + plotW, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = cssVar(m[2]);
        ctx.textAlign = 'right';
        ctx.fillText(m[0] + ' ' + num(m[1], 2), x0 - 4, y);
        ctx.textAlign = 'left';
      });
    }

    const redraw = () => { drawSplit(); drawDist(); };
    requestAnimationFrame(redraw);
    stage.__redraw = redraw;
    if (stage.__ro) stage.__ro.disconnect();
    stage.__ro = new ResizeObserver(() => redraw());
    stage.__ro.observe(distHost);
    stage.__ro.observe(splitHost);
  }

  function renderCalc() {
    const B = tileBudget();
    const wrap = el('div', 'tc-calc');

    const form = el('div', 'tc-calc-form');
    const numRow = (label, key, min, max, step) => {
      const row = el('div', 'tc-calc-row');
      row.appendChild(el('label', null, label));
      const i = el('input');
      i.type = 'number'; i.min = min; i.max = max; i.step = step || 1;
      i.value = S.tile[key];
      i.addEventListener('change', () => {
        S.tile[key] = clamp(parseInt(i.value, 10) || min, min, max);
        render();
      });
      row.appendChild(i);
      return row;
    };
    form.appendChild(numRow('M_tile', 'm', 8, 512, 8));
    form.appendChild(numRow('N_tile', 'n', 16, 1024, 16));
    form.appendChild(numRow('K_tile', 'k', 16, 1024, 16));
    const abRow = el('div', 'tc-calc-row');
    abRow.appendChild(el('label', null, 'A / B dtype'));
    abRow.appendChild(select(DTYPES.map((d) => ({ id: d.id, label: d.label })), S.tile.ab, (v) => { S.tile.ab = v; render(); }));
    form.appendChild(abRow);
    const accRow = el('div', 'tc-calc-row');
    accRow.appendChild(el('label', null, 'Acc dtype'));
    accRow.appendChild(select(ACC_DTYPES.map((d) => ({ id: d.id, label: d.label })), S.tile.acc, (v) => { S.tile.acc = v; render(); }));
    form.appendChild(accRow);
    form.appendChild(numRow('live acc', 'live', 1, 8, 1));
    const dRow = el('div', 'tc-calc-row');
    dRow.appendChild(el('label', null, 'pipeline stage'));
    dRow.appendChild(select([1, 2, 3, 4].map((n) => ({ id: String(n), label: String(n) })), String(S.tile.depth),
      (v) => { S.tile.depth = +v; render(); }));
    form.appendChild(dRow);
    const presets = el('div', 'tc-actions');
    presets.appendChild(btn('回到本 run 实测值', {
      size: 'sm', on: () => { S.tile = defaultTile(); render(); },
    }));
    form.appendChild(presets);
    wrap.appendChild(form);

    const out = el('div', 'tc-calc-out');
    const budget = el('div', 'tc-budget');
    const row = (name, bytes, cap, fx, tone) => {
      const r = el('div', 'tc-budget-row');
      r.appendChild(el('span', 'nm', name));
      r.appendChild(bar(cap ? bytes / cap : 0, tone));
      r.appendChild(el('span', 'fx', kb(bytes) + (cap ? ' / ' + kb(cap) : '')));
      return r;
    };
    budget.appendChild(row('Left (L0A)', B.left, B.freeLR, B.left > B.freeLR ? 'bad' : 'neutral'));
    budget.appendChild(row('Right (L0B)', B.right, B.freeLR, B.right > B.freeLR ? 'bad' : 'neutral'));
    budget.appendChild(row('Acc (L0C)', B.acc, B.accObserved, B.acc > B.accObserved ? 'warn' : 'good'));
    budget.appendChild(row('stage × max(L,R)', B.depthNeed, B.freeLR, B.depthState === 'pass' ? 'good' : B.depthState === 'warn' ? 'warn' : 'bad'));
    out.appendChild(budget);

    const verdict = el('div', 'tc-verdict');
    const vrow = (state, tag, html) => {
      const r = el('div', 'tc-verdict-row');
      r.dataset.state = state;
      r.appendChild(el('span', 'tag', tag));
      const p = el('p');
      p.innerHTML = html;
      r.appendChild(p);
      return r;
    };
    if (B.freeLR == null) {
      verdict.appendChild(vrow('warn', 'depth ?',
        'stage ' + B.T.depth + ' × max(L,R) = <strong>' + kb(B.depthNeed)
        + '</strong>，但本 dump 无 PH-MR-001，Left/Right 可用字节未知 · 无法判定'));
    } else {
      const depthMsg = B.atLimit
        ? ' · 同配置实测回退：' + D.depthSites[0].file + ':' + D.depthSites[0].line
        : B.depthState === 'fail' ? ' · 超出，MemoryReuse 降至 depth 1'
          : B.depthState === 'warn' ? ' · 余量不足以容纳同驻 tile' : '';
      verdict.appendChild(vrow(B.depthState, B.depthState === 'pass' ? 'depth' : 'depth ↓',
        'stage ' + B.T.depth + ' × max(L,R) = <strong>' + kb(B.depthNeed) + '</strong> / '
        + kb(B.freeLR) + ' free = ' + pct(B.depthRatio * 100, 0) + depthMsg));
    }
    verdict.appendChild(vrow(B.lineOk ? 'pass' : 'warn', B.lineOk ? 'cache line' : '末维不足',
      'N × ' + B.ab.label + ' = <strong>' + B.innermost + 'B</strong> / ' + B.cacheLine + 'B'
      + (B.lineOk ? '' : ' · 需 ' + B.ab.mult + ' 元素倍数（≥ ' + B.needElems + ' 个 ' + B.ab.label + '）')));
    verdict.appendChild(vrow(B.acc > B.accObserved ? 'warn' : 'pass', 'Acc',
      'M × N × ' + B.ac.bytes + 'B × ' + B.T.live + ' = <strong>' + kb(B.acc) + '</strong> · 本 run 实测最大 '
      + kb(B.accObserved) + '（dump 未报上限，超出即待验证）'));
    out.appendChild(verdict);

    const obs = D.l0Tiles.slice(0, 8).map((x) => ({
      mem: x.mem, shape: x.dtype + '[' + x.rows + ',' + x.cols + ']',
      bytes: x.bytes, innermost: x.innermostB, n: x.n,
    }));
    out.appendChild(sectionHead('本 run 出现的 L0 tile', D.l0Tiles.length + ' 种 · 点行回填'));
    out.appendChild(table([
      { label: '空间', key: 'mem', mono: true },
      { label: 'tile', key: 'shape', mono: true },
      { label: '字节', num: true, cell: (r) => kb(r.bytes) },
      { label: '末维', num: true, cell: (r) => (r.innermost >= 512 ? '<span class="ok">' : '<span class="warn">') + r.innermost + 'B</span>' },
      { label: '出现', key: 'n', num: true },
    ], obs, {
      onPick: (r) => {
        const m = r.shape.match(/\[(\d+),(\d+)\]/);
        if (!m) return;
        if (r.mem === 'Right') { S.tile.k = +m[1]; S.tile.n = +m[2]; }
        else if (r.mem === 'Left') { S.tile.m = +m[1]; S.tile.k = +m[2]; }
        else { S.tile.m = +m[1]; S.tile.n = +m[2]; }
        render();
      },
    }));
    wrap.appendChild(out);
    return wrap;
  }

  /* ==================================================== compiler view */
  function viewCompiler(stage) {
    if (S.compilerTab === 'passes') {
      const detailByPass = {};
      (D.passEvidence || []).forEach((d) => { detailByPass[d.idx] = d; });
      const selected = D.passes.find((p) => p.idx === S.pass) || D.passes[0];
      const detail = detailByPass[selected.idx] || { add: 0, del: 0, groups: 0, scopes: [], hunks: [], links: [] };
      const changed = D.passes.filter((p) => {
        const d = detailByPass[p.idx];
        return d && (d.add || d.del);
      }).length;
      const sec = el('section', 'tc-pass-workspace');
      sec.appendChild(sectionHead('编译 IR 全流程 · ' + D.case.program,
        changed + ' / ' + Math.max(0, D.passes.length - 1) + ' 个 Pass 改动了 IR'));

      /* The river is a view of the same selected-pass state as the evidence
       * panel below.  Its strata describe IR form, its dots describe actual
       * snapshot changes — no separate, decorative pipeline is introduced. */
      const strata = [
        { id: 's0', label: 'S0 · 前端', form: 'Tensor IR', until: 0 },
        { id: 's1', label: 'S1 · 规范化张量', form: 'SSA / Tensor', until: 9 },
        { id: 's2', label: 'S2 · 层级化', form: 'Structured IR', until: 12 },
        { id: 's3', label: 'S3 · Tile', form: 'Tile IR', until: 21 },
        { id: 's4', label: 'S4 · 双核 Kernel', form: 'AIC / AIV Kernel', until: 28 },
        { id: 's5', label: 'S5 · 物理内存', form: 'MemRef / 物理内存', until: 34 },
        { id: 's6', label: 'S6 · 运行时', form: 'Runtime IR', until: Infinity },
      ];
      const stratumFor = (p) => strata.find((s) => p.idx <= s.until) || strata[strata.length - 1];
      const kindFor = (p, d) => {
        if (!p.idx || !(d.add || d.del)) return 'same';
        if (/MemoryReuse/.test(p.name) && D.depthSites.some((s) => s.fittedDepth < s.maxReqDepth)) return 'bad';
        if (/MemRef|Memory|Addr|Layout/.test(p.name)) return 'memory';
        if (/Runtime|Host|CallDirection|CommDomain/.test(p.name)) return 'runtime';
        if (/Inline|Outline|Unroll|Split|Expand|LowerPipeline/.test(p.name)) return 'struct';
        if (/Tile|Pipeline|Prefetch|Matmul/.test(p.name)) return 'intent';
        return 'touch';
      };
      const groups = [];
      D.passes.forEach((p, i) => {
        const s = stratumFor(p);
        const last = groups[groups.length - 1];
        if (!last || last.id !== s.id) groups.push({ id: s.id, label: s.label, form: s.form, from: i, to: i });
        else last.to = i;
      });
      const river = el('div', 'tc-pass-river');
      const riverHead = el('div', 'tc-pass-river-head');
      riverHead.appendChild(el('span', null, '相邻快照 · ' + D.passes[0].lines + ' → ' + D.passes[D.passes.length - 1].lines + ' 行'));
      riverHead.appendChild(el('span', null, changed + ' 个关键事件 · ' + D.passes.length + ' 个 Pass'));
      river.appendChild(riverHead);
      const riverScroll = el('div', 'tc-pass-river-scroll');
      const riverScene = el('div', 'tc-pass-river-scene');
      riverScene.style.minWidth = Math.max(1040, D.passes.length * 28) + 'px';
      const formLabel = el('span', 'tc-pass-river-label', 'IR 形态');
      riverScene.appendChild(formLabel);
      groups.forEach((group) => {
        const band = el('div', 'tc-pass-form-band');
        band.dataset.stratum = group.id;
        band.style.left = (group.from / D.passes.length * 100) + '%';
        band.style.width = ((group.to - group.from + 1) / D.passes.length * 100) + '%';
        band.textContent = group.form;
        riverScene.appendChild(band);
      });
      const sequenceLabel = el('span', 'tc-pass-river-label tc-pass-river-seq-label', '执行序 →');
      riverScene.appendChild(sequenceLabel);
      const spine = el('i', 'tc-pass-river-spine'); riverScene.appendChild(spine);
      groups.forEach((group) => {
        const box = el('div', 'tc-pass-stage-band');
        box.dataset.stratum = group.id;
        box.style.left = (group.from / D.passes.length * 100) + '%';
        box.style.width = ((group.to - group.from + 1) / D.passes.length * 100) + '%';
        box.appendChild(el('span', null, group.label));
        riverScene.appendChild(box);
      });
      D.passes.forEach((p, i) => {
        const d = detailByPass[p.idx] || {};
        const kind = kindFor(p, d);
        const b = el('button', 'tc-pass-river-node is-' + kind + (p.idx === selected.idx ? ' is-selected' : ''));
        b.type = 'button'; b.dataset.stratum = stratumFor(p).id;
        b.style.left = ((i + 0.5) / D.passes.length * 100) + '%';
        b.title = String(p.idx).padStart(2, '0') + ' · ' + p.name + (d.add || d.del ? ' · +' + d.add + ' / −' + d.del : ' · 未改动 IR');
        b.setAttribute('aria-label', b.title);
        b.appendChild(el('i', 'dot'));
        if (kind !== 'touch' && kind !== 'same') b.appendChild(el('span', 'idx', String(p.idx).padStart(2, '0')));
        b.addEventListener('click', () => { S.pass = p.idx; S.focus = 'pass'; S.passMode = 'overview'; render(); });
        riverScene.appendChild(b);
      });
      riverScroll.appendChild(riverScene); river.appendChild(riverScroll);
      const legend = el('div', 'tc-pass-river-legend');
      [['struct', '结构变换'], ['intent', '意图相关'], ['bad', '意图被破坏'], ['memory', '内存'], ['runtime', '运行时'], ['touch', '普通改动'], ['same', '未改动']]
        .forEach(([kind, label]) => { const item = el('span'); item.appendChild(el('i', 'is-' + kind)); item.appendChild(el('span', null, label)); legend.appendChild(item); });
      river.appendChild(legend);
      sec.appendChild(river);

      const body = el('div', 'tc-pass-workspace-body');

      const work = el('div', 'tc-pass-detail');
      const head = el('div', 'tc-pass-detail-head');
      const title = el('div');
      title.appendChild(el('span', 'eyebrow', selected.idx === 0 ? '流水线输入' : 'Pass #' + selected.idx + ' · 从 ' + detail.from));
      title.appendChild(el('h3', null, selected.name));
      head.appendChild(title);
      const modes = el('div', 'segmented-control segmented-control-muted');
      [['overview', '变化概览'], ['diff', '代码 Diff']].forEach(([id, label]) => {
        modes.appendChild(btn(label, { size: 'sm', selected: S.passMode === id,
          on: () => { S.passMode = id; render(); } }));
      });
      head.appendChild(modes);
      work.appendChild(head);

      const metrics = el('div', 'tc-pass-metrics');
      [
        ['IR 行', selected.lines, selected.delta === 0 ? '与上一快照等长' : (selected.delta > 0 ? '+' : '') + selected.delta],
        ['实测变更', detail.add + ' + / ' + detail.del + ' −', detail.groups + ' 个改写区域'],
        ['受影响作用域', String(detail.scopes.length), detail.scopes.length ? detail.scopes.slice(0, 2).map((s) => s.name).join(' · ') : '无'],
      ].forEach(([k, v, sub]) => {
        const m = el('div', 'tc-pass-metric');
        m.appendChild(el('span', 'k', k)); m.appendChild(el('strong', null, v)); m.appendChild(el('span', 'sub', sub));
        metrics.appendChild(m);
      });
      work.appendChild(metrics);

      if (detail.links && detail.links.length) {
        const links = el('div', 'tc-pass-links');
        links.appendChild(el('span', 'label', '运行内关联'));
        detail.links.forEach((link) => {
          links.appendChild(btn(link.label, { size: 'sm', on: () => {
            if (link.findingId && findingById[link.findingId]) {
              S.finding = link.findingId; S.focus = 'finding'; applyFocus(findingById[link.findingId]);
            } else if (link.view) S.view = link.view;
            render();
          } }));
        });
        work.appendChild(links);
      }

      if (!detail.add && !detail.del) {
        work.appendChild(el('div', 'tc-pass-empty', selected.idx === 0
          ? '前端 IR 是流水线的事实起点；选择后续 Pass 查看相邻快照的改写证据。'
          : '相邻快照逐行一致：这个 Pass 在本次编译输入上是空操作。'));
      } else if (S.passMode === 'overview') {
        const scopes = el('div', 'tc-pass-scopes');
        scopes.appendChild(el('span', 'label', '受影响作用域'));
        detail.scopes.forEach((scope) => {
          const chip = el('span', 'tc-pass-scope');
          chip.appendChild(el('code', null, scope.name));
          chip.appendChild(el('span', null, scope.lines + ' 行变更'));
          scopes.appendChild(chip);
        });
        work.appendChild(scopes);
        const hunkList = el('div', 'tc-pass-hunks');
        detail.hunks.slice(0, 3).forEach((hunk, i) => {
          const card = el('article', 'tc-pass-hunk');
          card.appendChild(el('div', 'h', '改写区域 ' + (i + 1) + ' · ' + hunk.scopes.join(' / ')));
          const pre = el('pre', 'tc-pass-code');
          pre.textContent = hunk.before.map((line) => '− ' + line).join('\n')
            + (hunk.beforeMore ? '\n− … ' + hunk.beforeMore + ' 行' : '')
            + (hunk.before.length && hunk.after.length ? '\n' : '')
            + hunk.after.map((line) => '+ ' + line).join('\n')
            + (hunk.afterMore ? '\n+ … ' + hunk.afterMore + ' 行' : '');
          card.appendChild(pre);
          hunkList.appendChild(card);
        });
        work.appendChild(hunkList);
      } else {
        const diffList = el('div', 'tc-pass-diff-list');
        detail.hunks.forEach((hunk, i) => {
          const card = el('article', 'tc-pass-diff-card');
          card.appendChild(el('div', 'h', '区域 ' + (i + 1) + ' · ' + hunk.scopes.join(' / ')
            + ' · 前 ' + hunk.beforeLine + ' / 后 ' + hunk.afterLine + ' 行'));
          const grid = el('div', 'tc-pass-diff-grid');
          [['删除', hunk.before, hunk.beforeMore, 'before'], ['新增', hunk.after, hunk.afterMore, 'after']].forEach(([label, lines, more, side]) => {
            const sideEl = el('div', 'tc-pass-diff-side'); sideEl.dataset.side = side;
            sideEl.appendChild(el('span', 'label', label));
            const pre = el('pre', 'tc-pass-code');
            pre.textContent = lines.length ? lines.join('\n') + (more ? '\n… ' + more + ' 行' : '') : '—';
            sideEl.appendChild(pre); grid.appendChild(sideEl);
          });
          card.appendChild(grid); diffList.appendChild(card);
        });
        work.appendChild(diffList);
      }
      body.appendChild(work);
      sec.appendChild(body);
      stage.appendChild(sec);
    }

    if (S.compilerTab === 'depth') {
      const sec = el('section');
      if (!D.depthSites.length) {
        /* MemoryReuse never reported a degradation here. That is a different
         * statement from "we found nothing", so spell out what was checked. */
        sec.appendChild(sectionHead('软流水深度回退', '本 run 无 PH-MR-001'));
        sec.appendChild(table([
          { label: '事实', cell: (r) => esc(r[0]), mono: true },
          { label: '含义', cell: (r) => esc(r[1]) },
        ], [
          ['PH-MR-001 × 0', 'MemoryReuse 未报告过任何一次深度回退'],
          ['pl.pipeline × ' + D.pipelineSites.length, '请求的 stage 都放得下，或该 kernel 未进 MemoryReuse'],
          ['Left / Right / Vec 可用字节 缺失', '片上预算试算器无本 run 实测上限可对账'],
        ], {}));
      } else {
        sec.appendChild(sectionHead('软流水深度回退',
          D.depthSites.length + ' 个源码点，' + D.hints.filter((h) => h.code === 'PH-MR-001').length + ' 条 PH-MR-001'));
        sec.appendChild(table([
          { label: '源码点', mono: true, cell: (s) => esc(s.module + ':' + s.line) },
          { label: '空间', cell: (s) => s.units.join(' / '), mono: true },
          { label: '组数', key: 'groupCount', num: true },
          { label: '请求 → 实得', num: true, cell: (s) => s.maxReqDepth + ' → <span class="bad">' + s.fittedDepth + '</span>' },
          { label: '每 stage', num: true, cell: (s) => kb(s.perStageB) },
          { label: '可用', num: true, cell: (s) => kb(s.freeB) },
          { label: '需求 / 可用', cell: (s) => bar((s.perStageB * s.maxReqDepth) / s.freeB, 'bad') },
        ], D.depthSites.map((s) => Object.assign({
          __selected: s.key === S.hintSite, __subject: !!subjectSiteSet()[s.key],
        }, s)), {
          onPick: (s) => { S.hintSite = s.key; S.focus = 'hint'; render(); },
        }));
      }
      stage.appendChild(sec);

      const stageGroups = {};
      D.pipelineSites.forEach((p) => {
        const k = 'stage=' + p.stage;
        (stageGroups[k] = stageGroups[k] || []).push(p);
      });
      const sec2 = el('section');
      sec2.appendChild(sectionHead('IR 里请求的流水',
        D.pipelineSites.length + ' 个 pl.pipeline 站点（AutoTileMatmulL0 之后）· 前端手写 ' + D.dsl.pipeline + ' 处'));
      sec2.appendChild(table([
        { label: 'stage', key: 'k', mono: true },
        { label: '站点数', cell: (r) => r.n, num: true },
        { label: '循环次数（trip）', cell: (r) => esc(r.trips), mono: true },
        { label: '携带值', cell: (r) => r.carriers, num: true },
      ], Object.keys(stageGroups).sort().map((k) => ({
        k: k, n: stageGroups[k].length,
        trips: Array.from(new Set(stageGroups[k].map((p) => p.trip))).slice(0, 8).join(', '),
        carriers: Math.max.apply(null, stageGroups[k].map((p) => p.carriers)),
      })), {}));
      stage.appendChild(sec2);
    }

    if (S.compilerTab === 'granularity') {
      const cacheLine = (D.hints.find((h) => h.cacheLineB) || {}).cacheLineB || 512;
      const sec = el('section');
      sec.appendChild(sectionHead('搬运末维粒度',
        D.hints.filter((h) => h.code === 'PH001').reduce((a, h) => a + h.occurrences, 0) + ' 次命中 · cache line ' + cacheLine + 'B · backend ' + D.case.backend));
      sec.appendChild(table([
        { label: '模块', key: 'module', mono: true },
        { label: '源码点', key: 'siteCount', num: true },
        { label: '命中', key: 'occ', num: true },
        { label: '最小末维', num: true, cell: (f) => '<span class="' + (f.minB < 128 ? 'bad' : 'warn') + '">' + f.minB + 'B</span>' },
        { label: '距一行', cell: (f) => bar(f.minB / cacheLine, 'bad') },
      ], D.tileFiles, { onPick: (f) => { S.hintModule = f.file; S.view = 'l1'; render(); } }));
      stage.appendChild(sec);

      const sec2 = el('section');
      const memFilter = ['all', 'Vec', 'Mat', 'Acc'];
      sec2.appendChild(sectionHead('最差的源码点', '按末维字节升序',
        field('目标空间', select(memFilter.map((m) => ({ id: m, label: m === 'all' ? '全部' : m })), S.__mem || 'all',
          (v) => { S.__mem = v; render(); }))));
      const rows = D.tileSites.filter((s) => {
        const m = S.__mem || 'all';
        return m === 'all' || s.mems[m];
      }).slice(0, 60);
      sec2.appendChild(table([
        { label: '源码点', mono: true, cell: (s) => esc(s.module + ':' + s.line) },
        { label: '末维', num: true, cell: (s) => '<span class="' + (s.minB < 128 ? 'bad' : 'warn') + '">' + s.minB + 'B</span>' },
        { label: '算子', cell: (s) => Object.keys(s.ops).join(', '), mono: true },
        { label: '空间', cell: (s) => Object.keys(s.mems).join(', '), mono: true },
        { label: 'tile', cell: (s) => esc(s.shapes.join(' ')), mono: true },
        { label: '命中', key: 'occ', num: true },
      ], rows.map((s) => Object.assign({
        __selected: s.key === S.hintSite, __subject: !!subjectSiteSet()[s.key],
      }, s)), {
        onPick: (s) => { S.hintSite = s.key; S.focus = 'hint'; render(); },
      }));
      stage.appendChild(sec2);

      const guide = el('section');
      guide.appendChild(sectionHead('末维目标', '凑满 ' + cacheLine + 'B 所需元素倍数'));
      guide.appendChild(table([
        { label: 'dtype', key: 'label', mono: true },
        { label: '每元素', num: true, cell: (d) => d.bytes + 'B' },
        { label: '末维元素倍数', num: true, cell: (d) => d.mult },
        { label: '对应字节', num: true, cell: (d) => kb(d.mult * d.bytes) },
      ], DTYPES, {}));
      stage.appendChild(guide);
    }
  }

  /* ========================================================= ISA view */
  function viewISA(stage) {
    const layoutPass = D.passes.find((p) => p.name === 'ResolveBackendOpLayouts');
    const spacePass = D.passes.find((p) => p.name === 'InferTileMemorySpace');
    const cacheLine = (D.hints.find((h) => h.cacheLineB) || {}).cacheLineB || 512;

    const have = el('section');
    have.appendChild(sectionHead('工具链与约束', 'binary_context · Pass dump · L0 tile'));
    have.appendChild(tiles([
      { k: 'platform', v: D.case.toolchain.platform || D.case.backend },
      { k: 'pto-isa', v: D.case.toolchain.ptoIsaRevision ? D.case.toolchain.ptoIsaRevision.slice(0, 8) : '—', u: 'revision' },
      { k: 'runtime', v: (D.case.toolchain.runtimeName || '—').split('_')[0],
        u: D.case.toolchain.runtimeRevision ? D.case.toolchain.runtimeRevision.slice(0, 8) : (D.case.toolchain.aicpuThreads ? D.case.toolchain.aicpuThreads + ' AICPU 线程' : '') },
      { k: 'cache line', v: cacheLine, u: 'B' },
      { k: 'L0 tile 形状', v: D.l0Tiles.length, u: '种（AutoTileMatmulL0）' },
      { k: 'Left/Right 可用', v: D.budgets.Right ? kb(D.budgets.Right.freeB) : '—',
        u: D.budgets.Right ? 'MemoryReuse 报告' : '无 PH-MR-001' },
    ]));
    stage.appendChild(have);

    const lay = el('section');
    lay.appendChild(sectionHead('布局与内存空间分配', 'ResolveBackendOpLayouts +' + layoutPass.delta + ' 行，InferTileMemorySpace +' + spacePass.delta + ' 行'));
    lay.appendChild(table([
      { label: '空间', key: 'mem', mono: true },
      { label: 'tile 形状', cell: (r) => esc(r.shape), mono: true },
      { label: 'dtype', key: 'dtype', mono: true },
      { label: '字节', num: true, cell: (r) => kb(r.bytes) },
      { label: '末维', num: true, cell: (r) => (r.innermostB >= cacheLine ? '<span class="ok">' : '<span class="warn">') + r.innermostB + 'B</span>' },
      { label: D.budgets.Right ? '占 ' + kb(D.budgets.Right.freeB) : '相对最大',
        cell: (r) => {
          const base = D.budgets.Right ? D.budgets.Right.freeB
            : Math.max.apply(null, D.l0Tiles.map((x) => x.bytes));
          return r.mem === 'Acc' ? '—' : bar(r.bytes / base, r.bytes > base ? 'bad' : 'neutral');
        } },
      { label: '出现', key: 'n', num: true },
    ], D.l0Tiles.map((r) => Object.assign({ shape: '[' + r.rows + ',' + r.cols + ']' }, r)), {}));
    stage.appendChild(lay);

    const missing = el('section');
    const A = D.case.artifacts;

    /* kernel -> source: reconstructed, not read out of the dump */
    const SM = D.sourceMap;
    const srcSec = el('section');
    if (SM) {
      srcSec.appendChild(sectionHead('scope → 源码',
        SM.covered + '/' + SM.total + ' 覆盖 · ' + SM.unique + ' 唯一 · ' + SM.ambiguous + ' 多候选'));
      srcSec.appendChild(table([
        { label: '项', cell: (r) => esc(r[0]), mono: true },
        { label: '值', cell: (r) => esc(r[1]) },
      ], [
        ['来源', '不在 dump 内 —— 由 ' + SM.root + '/ 的源码重建'],
        ['入口', SM.entry + ' · 传递导入 ' + SM.modules.length + ' 个模块'],
        ['依据', 'pl.spmd(..., name_hint="X") 与 OutlineIncoreScopes 外联出的函数同名'],
        ['索引到的 name_hint', String(SM.hintCount)],
        ['唯一定位', SM.unique + ' 个 scope'],
        ['多候选', SM.ambiguous + ' 个（同名 hint 出现在多处，名字消不掉歧义）'],
        ['未匹配', String(SM.missing)],
        ['粒度', '映射的是 scope 不是 kernel —— 混合 scope 拆出的 _aic / _aiv '
          + '两个 kernel 共用同一个 name_hint，因此指向同一处源码'],
        ['不能做的事', '只给出 scope 写在哪里，不把实测块时长归到某一行'],
      ], {}));
    } else {
      srcSec.appendChild(sectionHead('scope → 源码', '源码树不在仓库内'));
      srcSec.appendChild(table([
        { label: '项', cell: (r) => esc(r[0]), mono: true },
        { label: '值', cell: (r) => esc(r[1]) },
        { label: '状态', cell: () => '<span class="bad">缺失</span>' },
      ], [
        ['模型源码', D.case.sourceRoot || '未记录'],
        ['可重建性', '拿到源码树后可按 name_hint 重建，方法同 decode_csa'],
      ], {}));
    }
    stage.appendChild(srcSec);

    /* PTOAS sources, when this dump carries them */
    if (A.ptoas) {
      const src = el('section');
      const units = D.case.ptoasUnits;
      const maxPto = Math.max.apply(null, units.map((u) => u.ptoLines));
      src.appendChild(sectionHead('PTOAS 单元', units.length + ' 个 · .pto → .cpp · '
        + Object.keys(A.kernelDirs).map((k) => k + ' ' + A.kernelDirs[k]).join(' / ')));
      src.appendChild(table([
        { label: '单元', key: 'name', mono: true },
        { label: '.pto 行', key: 'ptoLines', num: true },
        { label: '', cell: (r) => bar(r.ptoLines / maxPto, r.ptoLines === maxPto ? 'warn' : 'neutral') },
        { label: '.cpp 行', num: true, cell: (r) => (r.cppLines == null ? '—' : r.cppLines) },
        { label: '展开比', num: true, cell: (r) => (r.cppLines == null ? '—'
          : num(r.cppLines / r.ptoLines, 2) + 'x') },
      ], units.slice().sort((a, b) => b.ptoLines - a.ptoLines), { tall: true }));
      stage.appendChild(src);
    }

    /* what is still missing — computed, not a fixed list */
    const gaps = [
      [A.ptoas ? null : 'ptoas/*.pto', '每个 kernel 的 PTOAS 源与展开后的 cpp'],
      [Object.keys(A.kernelDirs).length ? null : 'kernels/', '实际编译出的 AIC / AIV 二进制'],
      ['PTOAS TileLib 模板记录', '模板候选与选中原因'],
      ['VPTO scheduler 排布报告', '依赖、延迟、寄存器压力、重物化'],
      ['cycle cost model 预测', '与实测块时长对账'],
      ['PMU counter', 'Cube / Vec / MTE / FIXPIPE，需单独建 PMU-on 基线'],
    ].filter((r) => r[0]);
    missing.appendChild(sectionHead('缺失产物', gaps.length + ' 项'));
    missing.appendChild(table([
      { label: '产物', cell: (r) => esc(r[0]), mono: true },
      { label: '用于', cell: (r) => esc(r[1]) },
      { label: '状态', cell: () => '<span class="bad">缺失</span>' },
    ], gaps, {}));
    stage.appendChild(missing);
  }

  /* ======================================================== inspector */
  /* The rail opens the three questions the page exists to answer and folds
   * the supporting detail. With everything expanded L2 was 6.3 screens of
   * scrolling against 1-2 on every other tab. Nothing is removed: a folded
   * section is one click from its full content, and the fold state is kept
   * per title so it survives the rail's re-render. */
  const FOLD_BY_DEFAULT = { '统计口径': 1, '引擎配对': 1, 'spmd 展开': 1 };

  function inspectorSection(title, kicker) {
    const s = el('section', 'inspector-section');
    const h = el('div', 'inspector-section-head');
    if (!FOLD_BY_DEFAULT[title]) {
      h.appendChild(el('h3', 'inspector-section-title', title));
      if (kicker) h.appendChild(el('span', 'inspector-section-kicker', kicker));
      s.appendChild(h);
      return s;
    }
    if (S.folded[title] === undefined) S.folded[title] = true;
    const open = !S.folded[title];
    s.classList.add('is-foldable');
    if (!open) s.classList.add('is-folded');
    const btn = el('button', 'inspector-section-toggle');
    btn.type = 'button';
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    btn.appendChild(el('span', 'chev', open ? '−' : '+'));
    btn.appendChild(el('h3', 'inspector-section-title', title));
    if (kicker) btn.appendChild(el('span', 'inspector-section-kicker', kicker));
    btn.addEventListener('click', () => {
      S.folded[title] = !S.folded[title];
      renderInspector();
    });
    h.appendChild(btn);
    s.appendChild(h);
    return s;
  }

  /* [label, value] or [label, value, title] -- the third slot is where an
   * explanation goes that used to cost a whole card. */
  function kv(pairs) {
    const d = el('dl', 'tc-kv');
    pairs.forEach((p) => {
      if (p[1] == null) return;
      const dt = el('dt', p[2] ? 'has-q' : null, p[0]);
      const dd = el('dd', null, p[1]);
      if (p[2]) { dt.title = p[2]; dd.title = p[2]; }
      d.appendChild(dt); d.appendChild(dd);
    });
    return d;
  }

  function renderInspector() {
    const host = $('#inspector');
    host.textContent = '';
    const title = $('[data-bind="inspectorTitle"]');
    const meta = $('[data-bind="inspectorMeta"]');

    const focus = S.focus || defaultFocus();
    const crumb = scopeCrumb();
    if (crumb) host.appendChild(crumb);
    if (focus === 'scope') renderScopeInspector(host, title, meta);
    else if (focus === 'finding' && S.finding) renderFindingInspector(host, title, meta);
    else if (focus === 'hint' && S.hintSite) renderHintInspector(host, title, meta);
    else if (focus === 'pass') renderPassInspector(host, title, meta);
    else if (focus === 'run') renderRunInspector(host, title, meta);
    else renderTaskInspector(host, title, meta);

    host.appendChild(renderLedger());
  }

  /* the inspector follows the view unless the user pinned something else.
   * L2 asks "which scope ate the time / when was the machine idle";
   * L1 asks "what happened inside one kernel". Different panel. */
  function defaultFocus() {
    if (S.view === 'e2e' || S.view === 'isa') return 'run';
    if (S.view === 'compiler') return S.compilerTab === 'passes' ? 'pass' : (S.hintSite ? 'hint' : 'run');
    if (S.view === 'l2') return 'scope';
    return 'task';
  }

  /* ------------------------------------------------------ L2 inspector
   * Answers the two questions the swimlane alone cannot: where the core-time
   * went by scope, and which stretches of wall time the machine sat idle.
   * Σdur on its own ranks the fat scopes; pairing it with DAG slack separates
   * "fat" from "fat and pinned to the critical path". */
  const lanesOf = () => R().swimlane.lanes.length;

  /* membership in each of the two paths, for one task */
  function pathRole(rank, tag) {
    const onCpm = rank.critical.tags.indexOf(tag);
    const seg = rank.cpath.segments.filter((sg) => sg.tag === tag)[0];
    const obsIdx = seg ? rank.cpath.segments.indexOf(seg) : -1;
    return {
      onCpm: onCpm >= 0, cpmIdx: onCpm + 1, cpmN: rank.critical.tags.length,
      onObs: !!seg, obsIdx: obsIdx + 1, obsN: rank.cpath.segments.length,
      seg: seg || null,
    };
  }
  function pathLabel(r) {
    if (!r.onCpm && !r.onObs) return '两条路径都不在';
    const bits = [];
    if (r.onCpm) bits.push('依赖关键路径 ' + r.cpmIdx + '/' + r.cpmN);
    if (r.onObs) bits.push('观测路径 ' + r.obsIdx + '/' + r.obsN);
    return '在 ' + bits.join(' · ');
  }
  function pathTitle(r) {
    return '依赖关键路径 = 静态 CPM，依赖决定的延迟下界；动它降下界。' + NL
      + '观测路径 = 从最后完成的任务反向归责走出的链，计算 + stall 精确铺满 makespan；动它去掉 stall。' + NL
      + '两条不是同一个集合，所以「在不在关键路径上」必须说清是哪一条。'
      + (r.seg ? NL + '本节点前的 stall ' + num(r.seg.stall, 2) + ' us（' + r.seg.kind + '）' : '');
  }
  /* scopes are source-level pl.spmd regions; kernels are what launched */
  const scopeCountOf = () => R().scopes.length;

  /* file:line, plus an honest marker when the name matches more than one
   * source site or only matched after a compiler suffix was stripped */
  /* Cube / Vector, and the split when a scope compiled into both */
  function engineChip(sc) {
    if (sc.kind === 'mix') return 'C+V';
    return sc.kind === 'aic' ? 'C' : 'V';
  }
  function engineTitle(sc) {
    const bits = [];
    ['aic', 'aiv'].forEach((en) => {
      const e = sc.engines && sc.engines[en];
      if (!e) return;
      bits.push((en === 'aic' ? 'AIC (Cube)' : 'AIV (Vec)') + ' ' + num(e.coreTime, 0)
        + ' us · ' + e.blocks + ' 块 / ' + e.cores + ' 核 · 最长 ' + num(e.durMax, 1) + ' us');
    });
    if (sc.kernelCount > 1) {
      bits.push('拆成 ' + sc.kernelCount + ' 个 kernel：' + sc.kernels.map((k) => k.name).join('、'));
    }
    return bits.join(String.fromCharCode(10));
  }

  function srcLabel(src) {
    if (!src) return null;
    return src.file + ':' + src.line
      + (src.candidates > 1 ? ' +' + (src.candidates - 1) : '');
  }
  function srcTitle(src) {
    if (!src) return '';
    const bits = ['name_hint="' + src.hint + '"'];
    if (!src.exact) bits.push('（callable 去掉编译器后缀后匹配）');
    if (src.candidates > 1) {
      bits.push(src.candidates + ' 处同名候选：'
        + src.sites.map((x) => x.file + ':' + x.line).join('、'));
    }
    return bits.join('\n');
  }

  /* Restore the window and the panel the drill-down replaced. */
  function scopeBack() {
    const r = S.scopeReturn;
    S.scopeReturn = null;
    if (r) { S.t0 = r.t0; S.t1 = r.t1; S.task = r.task || S.task; }
    S.focus = 'scope';
    S.focusEvidence = false;
    render();
  }

  /* A breadcrumb the drill-down can be undone from. Rendered by whichever
   * panel the drill landed on, so the trail is visible where the user is. */
  function scopeCrumb() {
    const r = S.scopeReturn;
    if (!r || S.view !== 'l2') return null;
    /* a scope drill left the panel behind; an idle drill only moved the window */
    const panelMoved = (S.focus || defaultFocus()) !== 'scope';
    const bar = el('div', 'tc-crumb');
    const back = el('button', 'tc-crumb-back',
      panelMoved ? '← scope 排行' : '← 恢复时间窗');
    back.type = 'button';
    back.title = (panelMoved ? '回到 L2 面板，并恢复 ' : '恢复 ')
      + num(r.t0, 0) + '–' + num(r.t1, 0) + ' us 的时间窗（Esc）';
    back.addEventListener('click', scopeBack);
    bar.appendChild(back);
    bar.appendChild(el('span', 'sep', '/'));
    bar.appendChild(el('span', 'cur',
      r.scope || (num(S.t0, 0) + '–' + num(S.t1, 0) + ' us')));
    return bar;
  }

  function renderScopeInspector(host, title, meta) {
    const rank = R();
    title.textContent = 'L2 · ' + S.rank;
    meta.textContent = rank.scopes.length + ' scope';

    /* --- 1. accounting: two views of the same block, stated once --- */
    const A = rank.accounting;
    const s0 = inspectorSection('统计口径', A.schedCoreTime ? 'Worker / Scheduler' : 'Worker');
    const kernelTotal = rank.scopes.reduce((a, x) => a + x.kernelCount, 0);
    const splitScopes = rank.scopes.filter((x) => x.kernelCount > 1);
    s0.appendChild(kv([
      ['Worker View', num(A.workerCoreTime, 0) + ' us · ' + A.workerBlocks + ' 块'],
      ['  kernel', num(A.workerKernelTime, 0) + ' us'],
      ['  setup', num(A.workerSetupTime, 0) + ' us'],
      ['Scheduler View', A.schedCoreTime
        ? num(A.schedCoreTime, 0) + ' us · ' + A.schedBlocks + ' 块' : '—'],
      ['hand-off 差', A.handoff == null ? '—' : '+' + num(A.handoff, 0) + ' us'],
      ['scope / kernel', rank.scopes.length + ' / ' + kernelTotal,
        splitScopes.length
          ? 'scope = 源码里一个 pl.spmd 区域；kernel = 设备上真正 launch 的函数。' + NL
            + 'ExpandMixedKernel 把 ' + splitScopes.length + ' 个混合 scope 各拆成 AIC + AIV，'
            + '所以多 ' + (kernelTotal - rank.scopes.length) + ' 个：'
            + splitScopes.map((x) => x.name).join('、') + NL
            + '两半共用一次 launch（同一个 taskId），只能靠 FuncId 分开。'
          : 'scope = 源码里一个 pl.spmd 区域；kernel = 设备上真正 launch 的函数。'
            + '本 case 没有混合 scope，两者一一对应。'],
    ]));
    if (A.naiveSum) {
      s0.appendChild(el('div', 'inspector-soft-card is-warning',
        '同一个块在 trace 里出现两次。两边相加得 ' + num(A.naiveSum, 0)
        + ' us —— 这是重复计数，不是总量。下面的 scope 排行只用 Worker View。'));
    }
    /* s0 is built here but appended below: the rail leads with the three
     * questions the L2 page exists to answer, not with its bookkeeping. */

    /* --- 1. where the makespan went, attributed --- */
    host.appendChild(renderCriticalPath(rank));

    /* --- 2. scope ranking: core-time × slack --- */
    const top = rank.scopes.slice(0, 10);
    const maxCore = top[0] ? top[0].coreTime : 1;
    const s1 = inspectorSection('scope 排行', 'Worker core-time · 前 ' + top.length + ' / ' + rank.scopes.length);
    const rows = el('div', 'tc-scoperows');
    const hd = el('div', 'tc-scoperow is-head');
    [['scope', 'l', '外联后的 incore scope，按 Worker core-time 排'],
     ['引擎', 'e', 'C = AIC (Cube)，V = AIV (Vec)，C+V = 混合 scope'],
     ['core-time', 'n', 'Σ 块时长，跨所有核'],
     ['占', 'n', '占本 rank 总 core-time'],
     ['slack', 'n', 'DAG 上这个 scope 最紧的任务能被推迟多久。'
       + '0 = 在依赖关键路径上，动它直接缩短总时长；slack 大 = 它胖但不急，先看并行度。' + NL
       + '不含资源争抢 —— 等核那部分在「关键路径归责」里记作 core-wait。']]
      .forEach((c) => { const x = el('span', c[1], c[0]); x.title = c[2]; hd.appendChild(x); });
    rows.appendChild(hd);
    top.forEach((sc) => {
      const b = el('button', 'tc-scoperow' + (sc.onCrit ? ' is-crit' : ''));
      b.type = 'button';
      b.title = sc.taskCount + ' 任务 / ' + sc.blocks + ' 块 · wall ' + num(sc.wall, 1) + ' us'
        + (sc.onCrit ? ' · 依赖关键路径上 ' + sc.critNodes + ' 个节点' : ' · 不在依赖关键路径上')
        + (sc.src ? '\n' + srcLabel(sc.src) + '\n' + srcTitle(sc.src) : '');
      const nm = el('span', 'l');
      nm.appendChild(el('i', 'bar'));
      nm.lastChild.style.width = ((sc.coreTime / maxCore) * 100).toFixed(1) + '%';
      nm.appendChild(el('span', 'tx', sc.name));
      b.appendChild(nm);
      if (sc.src) {
        const sl = el('span', 'src' + (sc.src.candidates > 1 ? ' is-amb' : ''), srcLabel(sc.src));
        nm.appendChild(sl);
      }
      /* which engine burns this scope's core-time -- C, V, or both */
      const eg = el('span', 'e is-' + sc.kind, engineChip(sc));
      eg.title = engineTitle(sc);
      b.appendChild(eg);
      b.appendChild(el('span', 'n', num(sc.coreTime, 0)));
      b.appendChild(el('span', 'n muted', pct(sc.coreShare, 1)));
      /* zero slack on a fat scope is the actionable combination */
      b.appendChild(el('span', 'n' + (sc.minSlack === 0 ? ' hot' : ''),
        sc.minSlack === 0 ? '0' : num(sc.minSlack, 0)));
      b.addEventListener('click', () => {
        const from = S.scopeReturn || { t0: S.t0, t1: S.t1, task: S.task, focus: S.focus };
        S.scopeReturn = { t0: from.t0, t1: from.t1, scope: sc.name, task: from.task, focus: from.focus };
        S.task = sc.tags[0];
        S.focus = 'task';
        const t = tasksOf[S.rank][sc.tags[0]];
        if (t) { const pad = Math.max(40, t.span * 0.3); setWindow(t.start - pad, t.end + pad); }
        render();
      });
      rows.appendChild(b);
    });
    s1.appendChild(rows);
    host.appendChild(s1);

    /* --- 3. idle windows --- */
    const idle = rank.idleRuns || [];
    const idleUs = idle.reduce((a, r) => a + r.us, 0);
    const s2 = inspectorSection('空转窗口',
      idle.length ? '前 ' + Math.min(5, idle.length) + ' / ' + idle.length + ' 段 · '
        + pct((idleUs / rank.swimlane.spanUs) * 100, 1) : '无');
    if (!idle.length) {
      s2.appendChild(el('div', 'tc-foot',
        '没有 AIC 与 AIV 同时低于 ' + rank.idlePct + '% 的窗口（窗宽 '
        + num(rank.occWindowUs, 1) + ' us）。'));
    } else {
      const ir = el('div', 'tc-scoperows');
      const ih = el('div', 'tc-scoperow is-idle is-head');
      ['窗口', '时长', 'AIC', 'AIV'].forEach((t, i) => ih.appendChild(el('span', i ? 'n' : 'l', t)));
      ir.appendChild(ih);
      idle.slice(0, 5).forEach((r) => {
        const b = el('button', 'tc-scoperow is-idle');
        b.type = 'button';
        b.title = '窗口内实际在跑：' + (r.running.top.map((t) => t.callable + ' ' + num(t.us, 0) + ' us/' + t.blocks + ' 块').join('，') || '无')
          + String.fromCharCode(10) + '核容量占用 ' + pct(r.running.capacityPct, 1);
        b.appendChild(el('span', 'l', num(r.t0, 0) + '–' + num(r.t1, 0) + ' us'));
        b.appendChild(el('span', 'n hot', num(r.us, 0)));
        b.appendChild(el('span', 'n muted', pct(r.aic, 0)));
        b.appendChild(el('span', 'n muted', pct(r.aiv, 0)));
        b.addEventListener('click', () => {
          if (!S.scopeReturn) {
            S.scopeReturn = { t0: S.t0, t1: S.t1, scope: null, task: S.task, focus: S.focus };
          }
          const pad = Math.max(30, r.us * 0.2);
          setWindow(r.t0 - pad, r.t1 + pad);
          redrawStage(); renderToolbar(); renderDock(); renderInspector();
        });
        ir.appendChild(b);
      });
      s2.appendChild(ir);
      const worst = idle[0];
      /* what is actually executing, by block overlap — not by task envelope */
      const hog = worst.spanning.filter((t) => t.blocks <= 2 && t.onCrit)[0];
      /* headline + cause; the block-by-block breakdown moves to hover */
      const ic = el('div', 'inspector-soft-card is-warning');
      ic.appendChild(el('div', 'hd', '最长一段 ' + num(worst.us, 0) + ' us，'
        + lanesOf() + ' 核只用掉 ' + pct(worst.running.capacityPct, 1)));
      ic.appendChild(el('div', 'bd', hog
        ? hog.callable + ' 单块跨越整段（span ' + num(hog.span, 0) + ' us，在依赖关键路径上）——挡住全部核。'
        : (worst.running.top.length ? '窗口内只有零星块在跑。' : '窗口内没有任何块在执行。')));
      ic.title = '占 ' + pct(worst.share, 1) + ' 的 makespan。' + NL
        + (worst.running.top.length
          ? '窗口内在跑：' + worst.running.top.map((t) => t.callable + ' ' + num(t.us, 0) + ' us/' + t.blocks + ' 块').join('、')
          : '窗口内没有任何块在执行。');
      s2.appendChild(ic);
    }
    host.appendChild(s2);

    /* --- supporting detail, folded by default --- */
    host.appendChild(s0);
    host.appendChild(renderEnginePairing(rank));
    host.appendChild(renderSpmdShape(rank));
  }

  /* --------------------------------------------- critical path, attributed
   * Ports simpler_setup.tools.critical_path. The scope ranking answers
   * "what is fat"; this answers "what did the makespan actually consist of",
   * and unlike the structural slack it can name WHY a task waited. */
  const KIND_LABEL = { 'data-wait': '数据', 'core-wait': '核', 'front-gap': '启动' };
  const KIND_FULL = {
    'data-wait': 'data-wait —— 在等上游生产者',
    'core-wait': 'core-wait —— 在等分到的核空出来（资源串行化）',
    'front-gap': 'front-gap —— 第一个任务前的 launch / dispatch 延迟',
  };
  const BOUND_LABEL = {
    dependency: '依赖受限', comm: '通信受限', resource: '资源受限',
    stall: '调度受限', compute: '计算受限',
  };

  function renderCriticalPath(rank) {
    const c = rank.cpath;
    const sec = inspectorSection('路径归责', c.segments.length + ' 节点');

    if (!c.acyclic) {
      sec.appendChild(el('div', 'inspector-soft-card is-warning',
        'happens-before 图有环，静态 CPM 无法计算 —— 下面只有观测路径。'));
    }

    sec.appendChild(kv([
      ['makespan', num(c.makespan, 0) + ' us'],
      ['静态 CPM', num(c.cpm.len, 0) + ' us · ' + pct(c.cpm.share, 1),
        '依赖决定的延迟下界（无限核），' + c.cpm.nodes + ' 个节点。'
        + '接近 makespan = 依赖受限，图本身就是地板。'],
      ['真正计算', num(c.workSpan, 0) + ' us · ' + pct(c.workShare, 1)],
      ['通信等待', c.waitNodes
        ? num(c.waitSpan, 0) + ' us · ' + pct(c.waitShare, 1) + '（' + c.waitNodes + ' 个 *_wait）'
        : '—'],
      ['调度 stall', num(c.stallTotal, 0) + ' us · ' + pct(c.stallShare, 1),
        '等上游 data-wait ' + num(c.stallByKind['data-wait'], 1) + ' us' + NL
        + '等核 core-wait ' + num(c.stallByKind['core-wait'], 1) + ' us' + NL
        + '启动 front-gap ' + num(c.stallByKind['front-gap'], 1) + ' us'],
      ['  数据 / 核 / 启动',
        num(c.stallByKind['data-wait'], 0) + ' / ' + num(c.stallByKind['core-wait'], 0)
        + ' / ' + num(c.stallByKind['front-gap'], 0) + ' us'],
    ]));

    /* the invariant that makes the per-task attribution sound */
    /* The verdict, with the validity gate folded into its own line: if the
     * walk did not tile the makespan the verdict is not usable at all. */
    const bd = el('div', 'inspector-soft-card' + (c.bound === 'compute' && c.tiling.exact ? '' : ' is-warning'));
    bd.appendChild(el('div', 'hd', (BOUND_LABEL[c.bound] || c.bound)
      + (c.tiling.exact ? '' : ' · 归责未闭合')));
    bd.appendChild(el('div', 'bd', c.tiling.exact
      ? c.boundWhy
      : '走查没有铺满 makespan（' + num(c.tiling.sum, 2) + ' vs ' + num(c.tiling.makespan, 2)
        + '），逐节点归责不成立，下面的数字不要引用。'));
    const gate = el('div', 'bd gate');
    gate.textContent = (c.tiling.exact ? '✓ ' : '✗ ')
      + '归责闭合 compute + stall = makespan（差 ' + num(c.tiling.delta, 2) + ' us）';
    gate.title = 'compute + stall = ' + num(c.tiling.sum, 2) + ' us vs makespan '
      + num(c.tiling.makespan, 2) + ' us。' + NL
      + '这条不成立，逐节点归责就不成立 —— 它是整段分析能不能用的前提。';
    bd.appendChild(gate);
    /* the floor check: a dependency-limited floor cannot exceed the wall
     * time it is a floor for. Nothing checked this until it was violated. */
    const cr = rank.critical;
    const floor = el('div', 'bd gate');
    floor.textContent = (cr.floorValid ? '✓ ' : '✗ ')
      + '依赖下界 CPM ≤ makespan（' + num(cr.chainSpan, 0) + ' ≤ ' + num(cr.walltime, 0) + ' us）';
    floor.title = '静态 CPM 是「无限核时依赖能压到多短」，它不可能超过实测总时长。' + NL
      + '这条曾经被违反过：未过滤时间戳的最长链算出 102.2%，把实际并行的两段时长相加了。' + NL
      + '路径上残留重叠 ' + num(cr.overlapOnPath, 2) + ' us'
      + (cr.overlapWithinTol ? '（在边保留容差内）' : '（超出容差，要查）');
    bd.appendChild(floor);
    sec.appendChild(bd);

    /* --- path nodes, worst stall first --- */
    const slow = c.segments.filter((sg) => sg.stall > 1)
      .sort((a, b) => b.stall - a.stall);
    const shown = (slow.length ? slow : c.segments.slice().sort((a, b) => b.compute - a.compute))
      .slice(0, 6);
    const rows = el('div', 'tc-scoperows');
    const hd = el('div', 'tc-scoperow is-cpath is-head');
    [['路径节点', 'l'], ['因', 'e'], ['stall', 'n'], ['span', 'n']]
      .forEach((col) => hd.appendChild(el('span', col[1], col[0])));
    rows.appendChild(hd);
    const maxStall = shown[0] ? Math.max.apply(null, shown.map((x) => x.stall)) || 1 : 1;
    shown.forEach((sg) => {
      const b = el('button', 'tc-scoperow is-cpath' + (sg.onCpm ? ' is-crit' : ''));
      b.type = 'button';
      b.title = sg.tag + ' · ' + sg.callable + NL
        + KIND_FULL[sg.kind] + NL
        + 'stall ' + num(sg.stall, 2) + ' us · 本节点 span ' + num(sg.dur, 2)
        + ' us · 非重叠计入 ' + num(sg.compute, 2) + ' us' + NL
        + (sg.onCpm ? '也在静态 CPM 路径上 —— 动它能降依赖下界'
                    : '只在观测路径上 —— 动它去掉的是 stall，不是依赖下界')
        + (sg.isWait ? NL + '这是 *_wait，它的 span 是等待不是计算' : '');
      const nm = el('span', 'l');
      nm.appendChild(el('i', 'bar'));
      nm.lastChild.style.width = ((sg.stall / maxStall) * 100).toFixed(1) + '%';
      nm.appendChild(el('span', 'tx', (sg.stall > 1 ? '🐌 ' : '') + sg.callable));
      b.appendChild(nm);
      b.appendChild(el('span', 'e is-' + sg.kind.split('-')[0], KIND_LABEL[sg.kind]));
      b.appendChild(el('span', 'n' + (sg.stall > 1 ? ' hot' : ' muted'), num(sg.stall, 1)));
      b.appendChild(el('span', 'n muted' + (sg.isWait ? ' is-wait' : ''), num(sg.dur, 0)));
      b.addEventListener('click', () => {
        const from = S.scopeReturn || { t0: S.t0, t1: S.t1, task: S.task, focus: S.focus };
        S.scopeReturn = { t0: from.t0, t1: from.t1, scope: null, task: from.task, focus: from.focus };
        S.task = sg.tag;
        S.focus = 'task';
        const t = tasksOf[S.rank][sg.tag];
        if (t) { const pad = Math.max(40, t.span * 0.3); setWindow(t.start - pad, t.end + pad); }
        render();
      });
      rows.appendChild(b);
    });
    sec.appendChild(rows);
    /* One line, not three cards: the marks, and the one distinction that
     * changes what a proposal is actually worth. */
    const note = el('div', 'tc-foot');
    note.appendChild(el('span', null,
      (slow.length ? '🐌 stall > 1 us（' + c.slowNodes + ' 个）' : '无 stall > 1 us')
      + ' · 红边框 = 也在静态 CPM 上'));
    if (c.cpm.onlyOnCpm && c.cpm.onlyOnCpm.length) {
      const more = el('span', 'q', '两条路怎么选 ?');
      more.title = '静态 CPM 的 ' + c.cpm.nodes + ' 个节点里 ' + c.cpm.shared
        + ' 个也在观测路径上，另外 ' + c.cpm.onlyOnCpm.length + ' 个观测路径从不经过（'
        + c.cpm.onlyOnCpm.join('、') + '）。' + NL
        + '动只在 CPM 上的节点 → 降依赖下界。' + NL
        + '动只在观测路径上的节点 → 去掉 stall。' + NL
        + '两者不能互换，提建议时要说清在动哪一条。';
      note.appendChild(more);
    }
    const how = el('span', 'q', '怎么算的 ?');
    how.title = '依赖边按实测时间戳过滤：只有 end(前驱) ≤ start(本节点) + ' + num(c.tol, 3)
      + ' us 且 start(前驱) < start(本节点) 才保留（' + c.edgesKept + ' 留 / ' + c.edgesDropped + ' 弃）。' + NL
      + '容差' + (c.tolSource === 'clock' ? '取 2 个时钟 tick。' : '本 case 没记时钟频率，退回时间戳精度的 2 个量子。') + NL
      + 'core-wait 的前驱是同一条泳道上此前被释放的最晚时刻（running max），流水重叠的块也算得对。';
    note.appendChild(how);
    sec.appendChild(note);

    return sec;
  }

  /* ------------------------------------------------------ engine pairing
   * A general trace tool sees 72 identical blocks. It cannot say that 24 of
   * them are the Cube half and 48 the Vector half of ONE source scope, nor
   * that the two ran on paired cores. Split by FuncId and state the ratio. */
  function renderEnginePairing(rank) {
    let aicT = 0, aivT = 0;
    rank.scopes.forEach((sc) => {
      if (sc.engines.aic) aicT += sc.engines.aic.coreTime;
      if (sc.engines.aiv) aivT += sc.engines.aiv.coreTime;
    });
    const tot = Math.max(aicT + aivT, 1e-9);
    const split = rank.scopes.filter((sc) => sc.kernelCount > 1)
      .sort((a, b) => b.coreTime - a.coreTime);
    const sec = inspectorSection('引擎配对', split.length
      ? split.length + ' 个混合 scope · AIC ' + pct((aicT / tot) * 100, 0) + ' / AIV ' + pct((aivT / tot) * 100, 0)
      : 'AIC ' + pct((aicT / tot) * 100, 0) + ' / AIV ' + pct((aivT / tot) * 100, 0));

    sec.appendChild(kv([
      ['AIC (Cube)', num(aicT, 0) + ' us · ' + pct((aicT / tot) * 100, 1)
        + ' · ' + rank.occupancy.aicUtil + '% 占用'],
      ['AIV (Vec)', num(aivT, 0) + ' us · ' + pct((aivT / tot) * 100, 1)
        + ' · ' + rank.occupancy.aivUtil + '% 占用'],
      ['Cube : Vec', aicT <= aivT
        ? '1 : ' + num(aivT / Math.max(aicT, 1e-9), 2)
        : num(aicT / Math.max(aivT, 1e-9), 2) + ' : 1'],
    ]));

    if (!split.length) {
      sec.appendChild(el('div', 'inspector-soft-card',
        '本 case 没有 mixed kernel —— 每个 scope 只编译出一个 kernel，纯 Cube 或纯 Vec。'));
      return sec;
    }

    const rows = el('div', 'tc-scoperows');
    const hd = el('div', 'tc-scoperow is-pair is-head');
    [['kernel', 'l'], ['引擎', 'e'], ['core-time', 'n'], ['块/核', 'n'], ['最长块', 'n']]
      .forEach((c) => hd.appendChild(el('span', c[1], c[0])));
    rows.appendChild(hd);

    split.forEach((sc) => {
      const head = el('div', 'tc-pairhead');
      head.appendChild(el('span', 'nm', sc.name));
      head.appendChild(el('span', 'ct', num(sc.coreTime, 0) + ' us'));
      rows.appendChild(head);
      sc.kernels.forEach((k) => {
        const b = el('button', 'tc-scoperow is-pair is-sub');
        b.type = 'button';
        b.title = k.name + ' · FuncId ' + k.funcId + NL
          + k.blocks + ' 块 / ' + k.cores + ' 核 · ' + num(k.coreTime, 1) + ' us · 占本 scope ' + pct(k.share, 1);
        const nm = el('span', 'l');
        nm.appendChild(el('i', 'bar'));
        nm.lastChild.style.width = k.share.toFixed(1) + '%';
        nm.appendChild(el('span', 'tx', k.name));
        b.appendChild(nm);
        const eg = el('span', 'e is-' + k.engine, k.engine === 'aic' ? 'C' : 'V');
        b.appendChild(eg);
        b.appendChild(el('span', 'n', num(k.coreTime, 0)));
        b.appendChild(el('span', 'n muted', k.blocks + '/' + k.cores));
        b.appendChild(el('span', 'n muted', num(k.durMax, 0)));
        b.addEventListener('click', () => {
          const from = S.scopeReturn || { t0: S.t0, t1: S.t1, task: S.task, focus: S.focus };
          S.scopeReturn = { t0: from.t0, t1: from.t1, scope: sc.name, task: from.task, focus: from.focus };
          S.task = sc.tags[0];
          S.focus = 'task';
          const t = tasksOf[S.rank][sc.tags[0]];
          if (t) { const pad = Math.max(40, t.span * 0.3); setWindow(t.start - pad, t.end + pad); }
          render();
        });
        rows.appendChild(b);
      });
    });
    sec.appendChild(rows);

    const worst = split[0];
    const card = el('div', 'inspector-soft-card' + (worst.pairRatio < 1.3 ? ' is-warning' : ''));
    card.appendChild(el('div', 'hd', worst.name + ' 两侧最长块相差 ' + worst.pairRatio + ' 倍'));
    card.appendChild(el('div', 'bd', worst.pairRatio < 1.3
      ? '几乎相等 = 两半没有错开，Cube 段和 Vec 段在块内串行。解耦成 GM FIFO 才能真正并行。'
      : '差距明显 = 慢的一侧决定整块时长，快的一侧在等。'));
    card.title = num(worst.engines.aic.durMax, 1) + ' / ' + num(worst.engines.aiv.durMax, 1)
      + ' us，整段 span ' + num(worst.wall, 0) + ' us。' + NL
      + '通用 trace 工具只看到 ' + worst.spmd.blocks + ' 个同名块 —— AIC / AIV 的归属只在 '
      + 'event-hint 的 FuncId 里，而两半共用同一个 taskId。';
    sec.appendChild(card);
    return sec;
  }

  /* --------------------------------------------------------- spmd shape
   * pl.spmd(N) fans one scope out to N cores. The trace records blocks and
   * core ids but never the launch shape, so "how wide, how many waves, how
   * evenly" needs the blocks regrouped per scope. */
  function renderSpmdShape(rank) {
    const sc = rank.scopes.slice().sort((a, b) => b.spmd.cores - a.spmd.cores
      || b.coreTime - a.coreTime);
    const wide = sc.filter((x) => x.spmd.cores > 1);
    const single = sc.length - wide.length;
    const waved = sc.filter((x) => x.spmd.waves > 1);
    const sec = inspectorSection('spmd 展开',
      wide.length + ' 个多核 scope · ' + single + ' 个单核');

    sec.appendChild(kv([
      ['最宽展开', sc[0] ? sc[0].spmd.cores + ' 核（' + sc[0].name + '）' : '—'],
      ['多波 scope', waved.length + (waved.length
        ? ' · 最多 ' + num(Math.max.apply(null, waved.map((x) => x.spmd.waves)), 2) + ' 波' : '')],
      ['单核 scope', single + ' 个'],
    ]));

    const rows = el('div', 'tc-scoperows');
    const hd = el('div', 'tc-scoperow is-spmd is-head');
    [['scope', 'l', '外联后的 incore scope'],
     ['核', 'n', '最宽一次 pl.spmd 展开占了几个核'],
     ['块', 'n', '块数'],
     ['波', 'n', '块数 / 核数。1 波 = 一次填满；>1 波 = 同一批核要跑好几轮，'
       + '每轮之间有一次完成回收。'],
     ['离散', 'n', '最长块 / 中位块。>2 = 同一次展开里各块负载不均，'
       + '最慢的那块决定整个 scope 什么时候结束。']]
      .forEach((c) => { const x = el('span', c[1], c[0]); x.title = c[2]; hd.appendChild(x); });
    rows.appendChild(hd);

    /* rank by what makes a fan-out worth looking at: many waves, or uneven */
    const notable = sc.filter((x) => x.spmd.waves > 1 || x.spmd.imbalance > 1.5
      || x.spmd.widths.length > 1);
    const pick = (notable.length ? notable : sc.filter((x) => x.spmd.cores > 1))
      .slice().sort((a, b) =>
        (b.spmd.waves - 1) * b.spmd.imbalance - (a.spmd.waves - 1) * a.spmd.imbalance
        || b.spmd.imbalance - a.spmd.imbalance).slice(0, 10);
    pick.forEach((x) => {
      const b = el('button', 'tc-scoperow is-spmd');
      b.type = 'button';
      b.title = x.name + NL + x.spmd.launches + ' 次 launch · 宽度 ' + x.spmd.widths.join('/')
        + ' 核 · ' + x.spmd.blocks + ' 块 · ' + num(x.spmd.waves, 2) + ' 波' + NL
        + '离散度 ' + x.spmd.imbalance + 'x（最长块 / 中位块）';
      const nm = el('span', 'l');
      nm.appendChild(el('span', 'tx', x.name));
      b.appendChild(nm);
      b.appendChild(el('span', 'n', String(x.spmd.cores)));
      b.appendChild(el('span', 'n muted', String(x.spmd.blocks)));
      b.appendChild(el('span', 'n' + (x.spmd.waves > 1 ? ' hot' : ' muted'), num(x.spmd.waves, x.spmd.waves > 1 ? 1 : 0)));
      b.appendChild(el('span', 'n' + (x.spmd.imbalance > 2 ? ' hot' : ' muted'), num(x.spmd.imbalance, 2)));
      b.addEventListener('click', () => {
        const from = S.scopeReturn || { t0: S.t0, t1: S.t1, task: S.task, focus: S.focus };
        S.scopeReturn = { t0: from.t0, t1: from.t1, scope: x.name, task: from.task, focus: from.focus };
        S.task = x.tags[0];
        S.focus = 'task';
        const t = tasksOf[S.rank][x.tags[0]];
        if (t) { const pad = Math.max(40, t.span * 0.3); setWindow(t.start - pad, t.end + pad); }
        render();
      });
      rows.appendChild(b);
    });
    sec.appendChild(rows);
    if (!notable.length) {
      sec.appendChild(el('div', 'tc-foot',
        '没有多波、不均或变宽的展开；本 case 的形状问题是 ' + single + ' 个 scope 只用 1 个核。'));
    }
    return sec;
  }

  function renderRunInspector(host, title, meta) {
    title.textContent = D.case.program;
    meta.textContent = D.case.toolchain.platform || D.case.backend;

    const s1 = inspectorSection('运行对象', D.case.runDir.slice(0, 16) + '…');
    s1.appendChild(kv([
      ['model', D.case.model],
      ['采集时间', D.case.capturedAt],
      ['ranks', D.case.ranks.join(', ') + ' · ' + D.case.device],
      ['核', D.case.numCores + '（AIC ' + D.case.aicCount + ' / AIV ' + D.case.aivCount + '）'],
      ['kernel / scope', D.case.callables + ' / ' + scopeCountOf()],
      ['绑定参数', D.case.params.length ? String(D.case.params.length) : '未记录'],
      ['pto-isa', D.case.toolchain.ptoIsaRevision ? D.case.toolchain.ptoIsaRevision.slice(0, 12) : '—'],
      ['runtime', D.case.toolchain.runtimeName || '—'],
    ]));
    host.appendChild(s1);

    if (!multiRank() || !hasE2E()) { renderRunInspectorSingle(host); return; }
    const RK = D.case.ranks;
    const s2 = inspectorSection('两卡对比', 'inv=' + TRACE_MATCH[RK[0]].inv + ' / ' + TRACE_MATCH[RK[1]].inv);
    const a = D.ranks[RK[0]], b = D.ranks[RK[1]];
    s2.appendChild(kv([
      ['device_wall', num(D.e2e.rank0[2]['chip.run.runner_run.device_wall'].us, 1) + ' / '
        + num(D.e2e.rank1[2]['chip.run.runner_run.device_wall'].us, 1) + ' us'],
      ['trace span', num(a.swimlane.spanUs, 1) + ' / ' + num(b.swimlane.spanUs, 1) + ' us'],
      ['AIC 占用', pct(a.occupancy.aicUtil) + ' / ' + pct(b.occupancy.aicUtil)],
      ['AIV 占用', pct(a.occupancy.aivUtil) + ' / ' + pct(b.occupancy.aivUtil)],
      ['依赖关键路径', a.critical.tags.length + ' / ' + b.critical.tags.length + ' 节点'],
      ['调度器占用', pct(a.scheduler.perLaneUtil) + ' / ' + pct(b.scheduler.perLaneUtil)],
    ]));
    /* the observation, then what the host clock says causes it */
    s2.appendChild(el('div', 'inspector-soft-card is-warning',
      'rank0 更慢却更闲：+' + num(a.swimlane.spanUs - b.swimlane.spanUs, 0) + ' us span，'
      + '−' + num(b.occupancy.aicUtil - a.occupancy.aicUtil, 1) + ' pt AIC 占用'));
    const K = D.launchSkew;
    if (K) {
      const cause = el('div', 'inspector-soft-card');
      cause.appendChild(el('span', 'hd', 'rank1 晚发 ' + num(K.runnerUs, 1) + ' us'));
      cause.appendChild(el('span', 'bd', 'AIC busy ' + num(K.work.rank0.aic.busy, 0) + ' / '
        + num(K.work.rank1.aic.busy, 0) + ' us（差 '
        + pct(Math.abs(K.work.rank0.aic.busy - K.work.rank1.aic.busy) / K.work.rank1.aic.busy * 100, 2)
        + '）——计算量相同，多出来的 span 是等待'));
      const rows = el('div', 'tc-skewrows');
      const hd = el('div', 'tc-skewrow is-head');
      ['wait', 'rank0 等', '错峰上界', '占比'].forEach((t, i) => hd.appendChild(el('span', i ? 'n' : 'l', t)));
      rows.appendChild(hd);
      K.checks.forEach((c) => {
        const r = el('button', 'tc-skewrow');
        r.type = 'button';
        r.appendChild(el('span', 'l', c.callable.replace(/_wait$/, '')));
        r.appendChild(el('span', 'n', num(c.measured, 1)));
        r.appendChild(el('span', 'n muted', num(c.bound, 1)));
        r.appendChild(el('span', 'n' + (c.fitPct >= 80 ? ' hot' : ''), pct(c.fitPct, 0)));
        r.addEventListener('click', () => {
          S.rank = 'rank0'; S.view = 'l2'; S.task = c.tag0; S.focus = 'task';
          const t = tasksOf.rank0[c.tag0];
          if (t) { const pad = Math.max(40, t.span * 0.35); setWindow(t.start - pad, t.end + pad); }
          render();
        });
        rows.appendChild(r);
      });
      cause.appendChild(rows);
      cause.appendChild(el('span', 'bd', (K.allUnderBound ? '4 个 wait 全部落在上界内' : '有 wait 超出上界')
        + ' · 合计 ' + num(K.measuredSum, 0) + ' / ' + num(K.boundSum, 0) + ' us'));
      s2.appendChild(cause);
    }
    host.appendChild(s2);

    const s3 = inspectorSection('瓶颈链', topChains(3).length + ' 条');
    topChains(3).forEach((f) => {
      s3.appendChild(btn(f.id + ' · ' + f.title, {
        size: 'sm',
        on: () => { S.finding = f.id; S.focus = 'finding'; applyFocus(f); render(); },
      }));
    });
    host.appendChild(s3);
  }

  function renderTaskInspector(host, title, meta) {
    const t = curTask();
    const rank = R();
    title.textContent = t.callable;
    meta.textContent = t.tag + ' · ' + S.rank;

    const s1 = inspectorSection('对象', t.kind.toUpperCase());
    s1.appendChild(kv([
      ['task id', t.id],
      /* a split scope has no single funcId -- name every kernel's */
      ['scope', t.callable + '（funcId ' + (t.kernels || []).map((k) => k.funcId).join(' + ') + '）'],
      ['ring / scope', 'r' + t.ring + ' · ' + t.scope + (t.earlyDispatch ? ' · early_dispatch' : '')],
      ['窗口', num(t.start, 1) + ' → ' + num(t.end, 1) + ' us'],
      ['块 / 核', t.blockCount + ' / ' + t.coreCount + '（block_num ' + t.blockNum + '）'],
      ['块时长', num(t.durMin, 2) + ' / ' + num(t.durMed, 2) + ' / ' + num(t.durP90, 2) + ' / ' + num(t.durMax, 2) + ' us'],
      ['kernel · setup', num(t.kdurSum / t.blockCount, 2) + ' · ' + num(t.setupMean, 2) + ' us'],
      ['AICPU 视角', t.svAicpuMean == null ? null : num(t.svAicpuMean, 1) + ' us（+' + num(t.svOverhead, 1) + '）'],
      ['前驱 / 后继', t.pred.length + ' / ' + t.succ.length],
      (function () { const r = pathRole(rank, t.tag); return ['路径', pathLabel(r), pathTitle(r)]; })(),
      ['kernel', (t.kernels || []).map((k) => k.name).join(' + ') || t.callable],
      ['源码', t.src ? srcLabel(t.src) : (D.sourceMap ? '未匹配' : '源码树不在仓库内')],
    ]));
    /* one scope, two kernels: state the split instead of one merged number */
    if (t.kernelCount > 1 && t.engines.aic && t.engines.aiv) {
      s1.appendChild(el('div', 'inspector-soft-card',
        '这是一个 mixed scope：ExpandMixedKernel 拆成 ' + t.kernelCount
        + ' 个 kernel，共用一次 Group launch。'
        + 'Cube 侧 ' + t.engines.aic.blocks + ' 块 / ' + num(t.engines.aic.coreTime, 0)
        + ' us（最长 ' + num(t.engines.aic.durMax, 1) + '），'
        + 'Vec 侧 ' + t.engines.aiv.blocks + ' 块 / ' + num(t.engines.aiv.coreTime, 0)
        + ' us（最长 ' + num(t.engines.aiv.durMax, 1) + '）。'
        + '上面「块 / 核」是两半合计。'));
    }
    if (t.src && (t.src.candidates > 1 || !t.src.exact)) {
      s1.appendChild(el('div', 'inspector-soft-card',
        (t.src.exact ? '' : 'callable 去掉编译器后缀后按 name_hint="' + t.src.hint + '" 匹配。')
        + (t.src.candidates > 1
          ? t.src.candidates + ' 处同名候选，名字本身消不掉歧义：'
            + t.src.sites.map((x) => x.file + ':' + x.line).join('、')
          : '')));
    }
    host.appendChild(s1);

    if (t.args.length) {
      const s2 = inspectorSection('绑定张量', t.args.length + ' 个');
      const list = el('div', 'tc-evidence');
      t.args.slice(0, 8).forEach((a) => {
        const r = el('div', 'tc-evidence-row');
        r.appendChild(el('span', 'a', 'idx ' + a.idx + ' · ' + a.type));
        r.appendChild(el('span', 'l', a.dtype + ' [' + a.shape.join(', ') + ']'));
        list.appendChild(r);
      });
      s2.appendChild(list);
      if (t.args.length > 8) s2.appendChild(el('p', 'tc-note', '另有 ' + (t.args.length - 8) + ' 个'));
      host.appendChild(s2);
    }

    const s3 = inspectorSection('依赖', 'fanin / fanout hints');
    const chips = el('div', 'tc-chipbar');
    t.pred.concat(t.succ).forEach((tag) => {
      const p = tasksOf[S.rank][tag];
      const b = btn(tag + (p ? ' · ' + p.callable : ''), {
        variant: 'ghost', size: 'sm',
        on: () => { if (p) { S.task = tag; S.focus = 'task'; render(); } },
      });
      if (!p) b.disabled = true;
      chips.appendChild(b);
    });
    s3.appendChild(chips);
    host.appendChild(s3);

    /* a task belongs to a chain if ANY rung marks it, so walking down from
     * L2 to a compiler site still shows the task its chain came from */
    const rungsFor = (f) => (f.chain || []).filter((st) => st.subjects.tasks.indexOf(t.tag) >= 0);
    const rel = D.findings.filter((f) => (f.focus && f.focus.task === t.tag)
      || f.subjects.tasks.indexOf(t.tag) >= 0 || rungsFor(f).length);
    if (rel.length) {
      const s4 = inspectorSection('关联瓶颈', rel.length + ' 条');
      rel.forEach((f) => {
        const at = rungsFor(f).map((st) => LEVEL_LABEL[st.level] || st.level);
        s4.appendChild(btn(f.id + ' · ' + f.title
          + (at.length ? '（' + Array.from(new Set(at)).join(' / ') + '）' : ''), {
          size: 'sm', on: () => { S.finding = f.id; S.chainStep = null; S.focus = 'finding'; applyFocus(f); render(); },
        }));
      });
      host.appendChild(s4);
    }

    const s5 = inspectorSection('所在核', '本任务涉及的泳道');
    const laneNames = {};
    rank.swimlane.laneNames.forEach((name, li) => {
      if (rank.swimlane.blocks[li].some((b) => rank.tasks[b[2]].tag === t.tag)) laneNames[name] = 1;
    });
    const laneStats = rank.swimlane.lanes.filter((l) => laneNames[l.name])
      .sort((a, b) => b.util - a.util).slice(0, 6);
    s5.appendChild(table([
      { label: 'lane', key: 'name', mono: true },
      { label: '占用', num: true, cell: (l) => pct(l.util) },
      { label: '', cell: (l) => bar(l.util / 100, l.util > 70 ? 'warn' : 'neutral') },
      { label: '最大空洞', num: true, cell: (l) => num(l.maxGap, 1) },
    ], laneStats, {}));
    host.appendChild(s5);
  }

  function renderFindingInspector(host, title, meta) {
    const f = findingById[S.finding];
    const chain = f.chain || [];
    title.textContent = f.id + ' · ' + (f.kind === 'hygiene' ? '体检项' : '瓶颈链');
    meta.textContent = f.cost ? f.cost.share + '% of makespan' : '无归因';

    const s1 = inspectorSection(f.title, f.metric);
    if (f.cost) {
      s1.appendChild(el('div', 'inspector-soft-card is-info',
        '代价 ' + us(f.cost.us, 1) + '（makespan 的 ' + f.cost.share + '%）· 口径：' + f.cost.basis));
    } else if (f.unattributed) {
      s1.appendChild(el('div', 'inspector-soft-card is-warning', '不作为瓶颈：' + f.unattributed));
    }
    s1.appendChild(el('p', 'tc-note', f.claim));
    host.appendChild(s1);

    if (chain.length) {
      const s0 = inspectorSection('跨层链条',
        chain.map((st) => LEVEL_LABEL[st.level] || st.level).join(' → '));
      const lad = el('div', 'tc-ladder');
      chain.forEach((st, i) => {
        const key = f.id + ':' + i;
        const r = el('div', 'tc-ladder-step' + (S.chainStep === key ? ' is-selected' : ''));
        r.dataset.role = st.role;
        const hd = el('div', 'hd');
        hd.appendChild(el('span', 'lv', LEVEL_LABEL[st.level] || st.level));
        hd.appendChild(el('span', 'role', (ROLE[st.role] || { label: st.role }).label));
        r.appendChild(hd);
        r.appendChild(el('span', 'ti', st.headline));
        if (st.detail) r.appendChild(el('span', 'dt', st.detail));
        /* only a rung that names something on the stage is clickable; a stop
         * rung has nothing to jump to and must not pretend otherwise */
        if (st.chips.length || st.role !== 'stop') {
          r.tabIndex = 0;
          r.setAttribute('role', 'button');
          r.classList.add('is-linked');
          const go = () => { applyStep(f, st); render(); };
          r.addEventListener('click', go);
          r.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); go(); }
          });
        }
        lad.appendChild(r);
      });
      s0.appendChild(lad);
      if (f.terminus) {
        s0.appendChild(el('div', 'inspector-soft-card'
          + (f.terminus.level === 'compiler' ? ' is-info' : ' is-warning'),
          '链止于 ' + (LEVEL_LABEL[f.terminus.level] || f.terminus.level) + '：' + f.terminus.reason));
      }
      host.appendChild(s0);
    }

    const s2 = inspectorSection('证据', f.chips.length ? f.evidence.length + ' 项 · 已在中间标号' : f.evidence.length + ' 项');
    const list = el('div', 'tc-evidence');
    f.evidence.forEach((e) => {
      /* link an evidence row to the marked objects its locator names, so the
       * inspector text and the numbered markers on the stage are the same thing */
      const keysOf = (c) => {
        const keys = [c.id, c.label];
        if (c.kind === 'site') keys.push(c.id.split(':')[0]);   /* file without line */
        return keys.filter(Boolean);
      };
      const linked = f.chips.filter((c) => keysOf(c)
        .some((k) => e.locator.indexOf(k) >= 0 || e.value.indexOf(k) >= 0));
      const r = el('div', 'tc-evidence-row');
      const a = el('span', 'a', e.artifact);
      if (linked.length) {
        a.appendChild(document.createTextNode(' · '));
        a.appendChild(el('span', 'jump', '标号 ' + linked.map((c) => f.chips.indexOf(c) + 1).join(' / ')));
        r.classList.add('is-linked');
        r.tabIndex = 0;
        r.setAttribute('role', 'button');
        const go = () => gotoChip(linked[0]);
        r.addEventListener('click', go);
        r.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); go(); } });
      }
      r.appendChild(a);
      r.appendChild(el('span', 'l', e.locator));
      r.appendChild(el('span', 'v', e.value));
      list.appendChild(r);
    });
    s2.appendChild(list);
    host.appendChild(s2);

    const s3 = inspectorSection('杠杆与护栏');
    s3.appendChild(el('div', 'inspector-soft-card is-info', '杠杆：' + f.lever));
    s3.appendChild(el('div', 'inspector-soft-card is-warning', '护栏：' + f.guardrail));
    s3.appendChild(el('div', 'inspector-soft-card', '复测：' + f.verify));
    host.appendChild(s3);

    host.appendChild(renderComposer(f));
  }

  function renderHintInspector(host, title, meta) {
    const depth = D.depthSites.find((s) => s.key === S.hintSite);
    const tile = D.tileSites.find((s) => s.key === S.hintSite);
    const site = depth || tile;
    if (!site) { renderTaskInspector(host, title, meta); return; }
    title.textContent = site.file + ':' + site.line;
    meta.textContent = depth ? 'PH-MR-001' : 'PH001';

    const s1 = inspectorSection('源码点', site.module);
    if (depth) {
      s1.appendChild(kv([
        ['pipeline 组', depth.groupCount + ' 组'],
        ['空间', depth.units.join(' / ')],
        ['请求深度', String(depth.maxReqDepth)],
        ['实得深度', String(depth.fittedDepth)],
        ['每 stage', kb(depth.perStageB)],
        ['可用', kb(depth.freeB)],
        ['需求 / 可用', num((depth.perStageB * depth.maxReqDepth) / depth.freeB, 2) + 'x'],
      ]));
      const list = el('div', 'tc-evidence');
      depth.groups.forEach((g) => {
        const r = el('div', 'tc-evidence-row');
        r.appendChild(el('span', 'a', 'group ' + g.group + ' @' + g.unit));
        r.appendChild(el('span', 'l', 'depth ' + g.reqDepth + ' → ' + g.fit));
        r.appendChild(el('span', 'v', kb(g.perStageB) + ' / stage，' + kb(g.freeB) + ' free'
          + (g.ownDepth ? '；单独放得下 depth ' + g.ownDepth : '')));
        list.appendChild(r);
      });
      s1.appendChild(list);
    } else {
      s1.appendChild(kv([
        ['算子', Object.keys(tile.ops).join(', ')],
        ['目标空间', Object.keys(tile.mems).join(', ')],
        ['dtype', Object.keys(tile.dtypes).join(', ')],
        ['tile', tile.shapes.join(' ')],
        ['最小末维', tile.minB + 'B'],
        ['建议', '≥ ' + tile.recB + 'B（cache line ' + tile.cacheLineB + 'B）'],
        ['命中', tile.occ + ' 次 / ' + tile.n + ' 条'],
      ]));
    }
    host.appendChild(s1);

    const s2 = inspectorSection('处理');
    s2.appendChild(el('div', 'inspector-soft-card is-info', depth
      ? '减少同驻 tile，而非调大 stage'
      : '末维凑满一个 cache line'));
    s2.appendChild(el('div', 'inspector-soft-card is-warning', depth
      ? '调大 stage 会再触发一次回退'
      : '加大末维会抬高 L0 / UB 占用，可能触发深度回退'));
    host.appendChild(s2);
  }

  function renderPassInspector(host, title, meta) {
    const p = D.passes.find((x) => x.idx === S.pass) || D.passes[0];
    title.textContent = p.name;
    meta.textContent = '#' + p.idx;
    const s1 = inspectorSection('Pass', p.file);
    s1.appendChild(kv([
      ['IR 行数', String(p.lines)],
      ['Δ 行', (p.delta > 0 ? '+' : '') + p.delta],
      ['pl.pipeline', String(p.counts.pipeline)],
      ['tile.matmul', String(p.counts.matmul)],
      ['pl.spmd', String(p.counts.spmd)],
      ['pl.range', String(p.counts.range)],
      ['Mem.Left / Right', p.counts.left + ' / ' + p.counts.right],
      ['Mem.Acc / Vec', p.counts.acc + ' / ' + p.counts.vec],
    ]));
    host.appendChild(s1);

    const prev = D.passes.find((x) => x.idx === p.idx - 1);
    if (prev) {
      const s2 = inspectorSection('相对上一个 Pass', prev.name);
      const diffs = [
        ['pl.pipeline', p.counts.pipeline - prev.counts.pipeline],
        ['tile.matmul', p.counts.matmul - prev.counts.matmul],
        ['Mem.Left', p.counts.left - prev.counts.left],
        ['Mem.Right', p.counts.right - prev.counts.right],
        ['Mem.Acc', p.counts.acc - prev.counts.acc],
        ['pl.range', p.counts.range - prev.counts.range],
      ].filter((d) => d[1] !== 0);
      if (diffs.length) s2.appendChild(kv(diffs.map((d) => [d[0], (d[1] > 0 ? '+' : '') + d[1]])));
      else s2.appendChild(el('p', 'tc-note', '结构计数无变化 · 行数 ' + (p.delta > 0 ? '+' : '') + p.delta));
      host.appendChild(s2);
    }
  }

  /* ------------------------------------------------------- experiment */
  function renderComposer(f) {
    const open = openExperiment();
    const s = inspectorSection('实验台账', open ? '已有 1 个进行中' : '每轮只验证一个假设');
    if (open && open.findingId !== f.id) {
      s.appendChild(el('div', 'inspector-soft-card is-warning',
        open.id + ' 进行中 · 结论后才能开下一个'));
      s.appendChild(btn('查看 ' + open.id, {
        size: 'sm',
        on: () => { if (open.findingId) { S.finding = open.findingId; S.focus = 'finding'; render(); } },
      }));
      return s;
    }
    if (open && open.findingId === f.id) {
      s.appendChild(renderStepper(open));
      return s;
    }
    const form = el('div', 'tc-form');
    const hyp = el('textarea');
    hyp.rows = 3;
    hyp.value = f.lever;
    const chg = el('input');
    chg.type = 'text';
    chg.placeholder = '例如：hc_post.py:51 把 co-live tile 从 5 组降到 2 组';
    const l1 = el('label');
    l1.appendChild(el('span', null, '假设'));
    l1.appendChild(hyp);
    const l2 = el('label');
    l2.appendChild(el('span', null, '改动'));
    l2.appendChild(chg);
    form.appendChild(l1);
    form.appendChild(l2);
    const acts = el('div', 'tc-actions');
    acts.appendChild(btn('开始实验', {
      variant: 'solid', size: 'sm',
      on: () => {
        ledgerSeq += 1;
        S.ledger.push({
          id: 'E' + ledgerSeq,
          state: 'open',
          findingId: f.id,
          title: f.id + ' · ' + f.title,
          hypothesis: hyp.value.trim() || f.lever,
          change: chg.value.trim() || '（未填写改动位置）',
          correctness: null, perf: null, keep: null,
          verify: f.verify, guardrail: f.guardrail,
        });
        render();
      },
    }));
    form.appendChild(acts);
    s.appendChild(form);
    return s;
  }

  function renderStepper(row) {
    const wrap = el('div', 'tc-form');
    const steps = el('div', 'tc-ledger-steps');
    const step = (k, v, done) => {
      const r = el('div', 'tc-ledger-step');
      r.dataset.done = done ? 'true' : 'false';
      r.appendChild(el('span', 'k', k));
      r.appendChild(el('span', 'v', v));
      return r;
    };
    steps.appendChild(step('假设', row.hypothesis, true));
    steps.appendChild(step('改动', row.change, true));
    steps.appendChild(step('正确性', row.correctness || '待记录', !!row.correctness));
    steps.appendChild(step('性能', row.perf || row.verify, !!row.perf));
    steps.appendChild(step('结论', row.keep || '待决定', !!row.keep));
    wrap.appendChild(steps);

    const acts = el('div', 'tc-actions');
    if (!row.correctness) {
      const inp = el('input');
      inp.type = 'text';
      inp.placeholder = '精度阈值 / 对比基准';
      const lb = el('label');
      lb.appendChild(el('span', null, '正确性'));
      lb.appendChild(inp);
      wrap.appendChild(lb);
      acts.appendChild(btn('记录正确性', {
        size: 'sm', variant: 'solid',
        on: () => { row.correctness = inp.value.trim() || '通过（未填写细节）'; render(); },
      }));
    } else if (!row.perf) {
      const inp = el('input');
      inp.type = 'text';
      inp.placeholder = '复测结果，例如 device_wall 5132.8 → ? us';
      const lb = el('label');
      lb.appendChild(el('span', null, '性能'));
      lb.appendChild(inp);
      wrap.appendChild(lb);
      acts.appendChild(btn('记录性能', {
        size: 'sm', variant: 'solid',
        on: () => { row.perf = inp.value.trim() || '（未填写复测数值）'; render(); },
      }));
    } else {
      const guard = el('div', 'inspector-soft-card is-warning');
      guard.textContent = row.guardrail;
      wrap.appendChild(guard);
      acts.appendChild(btn('保留', { size: 'sm', variant: 'solid', on: () => { row.keep = '保留'; row.state = 'kept'; render(); } }));
      acts.appendChild(btn('回退', { size: 'sm', on: () => { row.keep = '回退'; row.state = 'reverted'; render(); } }));
    }
    acts.appendChild(btn('放弃这轮', {
      size: 'sm', variant: 'ghost',
      on: () => { S.ledger = S.ledger.filter((r) => r !== row); render(); },
    }));
    wrap.appendChild(acts);
    return wrap;
  }

  function renderLedger() {
    const s = inspectorSection('台账', S.ledger.length + ' 条');
    const list = el('div', 'tc-ledger');
    S.ledger.slice().reverse().forEach((row) => {
      const item = el('div', 'tc-ledger-item');
      item.dataset.state = row.state;
      const hd = el('div', 'hd');
      hd.appendChild(el('span', 'id', row.id));
      hd.appendChild(el('span', 'st', row.state));
      item.appendChild(hd);
      item.appendChild(el('span', 'ti', row.title));
      const steps = el('div', 'tc-ledger-steps');
      [['假设', row.hypothesis], ['改动', row.change], ['正确性', row.correctness],
        ['性能', row.perf], ['结论', row.keep]].forEach((p) => {
        const r = el('div', 'tc-ledger-step');
        r.dataset.done = p[1] ? 'true' : 'false';
        r.appendChild(el('span', 'k', p[0]));
        r.appendChild(el('span', 'v', p[1] || '—'));
        steps.appendChild(r);
      });
      item.appendChild(steps);
      if (row.findingId) {
        item.appendChild(btn('回到 ' + row.findingId, {
          variant: 'ghost', size: 'sm',
          on: () => { S.finding = row.findingId; S.focus = 'finding'; applyFocus(findingById[row.findingId]); render(); },
        }));
      }
      list.appendChild(item);
    });
    s.appendChild(list);
    return s;
  }

  /* ======================================================= bottom dock */
  function renderDock() {
    const body = $('#dockBody');
    body.textContent = '';
    /* the ready-queue tooltip lives outside #dockBody, so clear it by hand */
    const staleTip = document.querySelector('[data-tc-tip="readyq"]');
    if (staleTip) staleTip.remove();
    const rank = R();
    $('[data-bind="dockMeta"]').textContent = S.rank + ' · 与上方时间轴同窗口 '
      + num(S.t0, 0) + '–' + num(S.t1, 0) + ' us';
    const modeHost = $('#dockMode');
    modeHost.textContent = '';
    modeHost.appendChild(group('segmented-control segmented-control-muted', [
      { id: 'sched', label: 'AICPU 调度' },
      { id: 'ready', label: 'Ready queue' },
      { id: 'lanes', label: '核占用' },
    ], S.dockMode, (v) => { S.dockMode = v; renderDock(); }));

    if (S.dockMode === 'lanes') {
      const rows = rank.swimlane.lanes.slice().sort((a, b) => b.util - a.util);
      body.appendChild(table([
        { label: 'lane', key: 'name', mono: true },
        { label: '类型', key: 'kind', mono: true },
        { label: '块', key: 'blocks', num: true },
        { label: '占用', num: true, cell: (l) => pct(l.util) },
        { label: '', cell: (l) => bar(l.util / 100, l.util > 70 ? 'warn' : l.util < 30 ? 'bad' : 'neutral') },
        { label: '空洞数', key: 'nGap', num: true },
        { label: '最大空洞', num: true, cell: (l) => num(l.maxGap, 1) },
        { label: '首块 → 末块', cell: (l) => num(l.first, 0) + ' → ' + num(l.last, 0), num: true },
      ], rows, {}));
      return;
    }

    if (S.dockMode === 'sched') {
      const phases = Object.keys(rank.scheduler.phases)
        .map((k) => Object.assign({ phase: k }, rank.scheduler.phases[k]))
        .sort((a, b) => b.us - a.us);
      const maxUs = phases[0].us;
      const head = el('div', 'tc-tiles');
      head.style.flex = '0 0 auto';
      const t = tiles([
        { k: '调度线程', v: rank.scheduler.lanes.length },
        { k: '合计 busy', v: num(rank.scheduler.busy, 0), u: 'us' },
        { k: '单线程平均占用', v: pct(rank.scheduler.perLaneUtil), tone: rank.scheduler.perLaneUtil > 40 ? 'warn' : null },
        { k: 'orchestrator submit', v: rank.orchestrator.count, u: num(rank.orchestrator.busy, 1) + ' us' },
        { k: 'hb_violation', v: rank.hbViolations.length, u: '对', tone: rank.hbViolations.length ? 'warn' : 'good' },
      ]);
      body.appendChild(t);
      body.appendChild(table([
        { label: 'phase', key: 'phase', mono: true },
        { label: '段数', key: 'n', num: true },
        { label: '总时长', num: true, cell: (p) => num(p.us, 1) },
        { label: '', cell: (p) => bar(p.us / maxUs, p.phase === 'complete' ? 'warn' : 'neutral') },
        { label: '处理任务', key: 'tasks', num: true },
        { label: 'us / 任务', num: true, cell: (p) => (p.usPerTask == null ? '—' : num(p.usPerTask, 3)) },
      ], phases, {}));
      return;
    }

    /* ready queue
     * shared_ready_queue is a Chrome-trace counter (ph "C"): each sample says
     * how many tasks are dependency-satisfied but not yet dispatched, split by
     * engine. A counter holds its value until the next sample, so it is drawn
     * as a step — the same semantics build-data.cjs integrates busyTime with. */
    const SERIES = [
      { key: 'AIC', idx: 1, token: '--danger' },
      { key: 'AIV', idx: 2, token: '--warning' },
      { key: 'MIX', idx: 3, token: '--accent' },
    ];
    const legend = el('div', 'tc-legend');
    SERIES.forEach((sr) => {
      const item = el('span');
      const swatch = el('i');
      swatch.style.background = cssVar(sr.token);
      item.appendChild(swatch);
      item.appendChild(el('span', null, sr.key + ' ready · peak ' + rank.readyStat.peak[sr.key]));
      legend.appendChild(item);
    });
    legend.appendChild(el('span', 'tc-readout', rank.readyQueue.length + ' 个采样点 · 悬停读数'));
    body.appendChild(legend);

    const host = el('div', 'tc-canvas-strip');
    host.style.flex = '0 0 auto';
    const canvas = el('canvas');
    host.appendChild(canvas);
    body.appendChild(host);
    body.appendChild(tiles([
      { k: 'AIC ready>0', v: pct(rank.readyStat.busyShare.AIC), u: num(rank.readyStat.busyTime.AIC, 0) + ' us', tone: 'warn' },
      { k: 'AIV ready>0', v: pct(rank.readyStat.busyShare.AIV), u: num(rank.readyStat.busyTime.AIV, 0) + ' us' },
      { k: 'MIX ready>0', v: pct(rank.readyStat.busyShare.MIX), u: num(rank.readyStat.busyTime.MIX, 0) + ' us' },
      { k: '峰值', v: rank.readyStat.peak.AIC + ' / ' + rank.readyStat.peak.AIV + ' / ' + rank.readyStat.peak.MIX, u: 'AIC / AIV / MIX' },
      { k: 'AIC 核占用', v: pct(rank.occupancy.aicUtil), tone: rank.occupancy.aicUtil < 40 ? 'bad' : null },
      { k: 'AIV 核占用', v: pct(rank.occupancy.aivUtil) },
    ]));

    const q = rank.readyQueue;
    /* last sample whose timestamp is <= t */
    const sampleAt = (t) => {
      let lo = 0, hi = q.length - 1, hit = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (q[mid][0] <= t) { hit = mid; lo = mid + 1; } else { hi = mid - 1; }
      }
      return hit;
    };

    let hoverX = null;   /* pointer position, canvas css px */
    let geom = null;     /* written by draw(), read by the pointer handler */

    const draw = () => {
      const w = host.clientWidth || 700;
      const h = 92;
      const ctx = fitCanvas(canvas, w, h);
      const x0 = 46, plotW = Math.max(40, w - x0 - 12), y0 = 18, hh = h - 40;
      geom = { x0: x0, plotW: plotW, y0: y0, hh: hh };
      drawTimeRuler(ctx, x0, plotW, 10, S.t0, S.t1);
      const peak = Math.max(rank.readyStat.peak.AIC, rank.readyStat.peak.AIV, 1);
      const sx = (t) => x0 + ((t - S.t0) / (S.t1 - S.t0)) * plotW;
      const sy = (v) => y0 + hh - (v / peak) * hh;
      SERIES.forEach((sr) => {
        ctx.beginPath();
        ctx.moveTo(x0, y0 + hh);
        let prevY = y0 + hh;
        q.forEach((sample) => {
          const x = clamp(sx(sample[0]), x0, x0 + plotW);
          const y = sy(sample[sr.idx]);
          ctx.lineTo(x, prevY);   /* hold the previous value up to this sample */
          ctx.lineTo(x, y);       /* then step */
          prevY = y;
        });
        ctx.lineTo(x0 + plotW, prevY);
        ctx.lineTo(x0 + plotW, y0 + hh);
        ctx.closePath();
        ctx.fillStyle = cssVar(sr.token);
        ctx.globalAlpha = 0.3;
        ctx.fill();
        ctx.globalAlpha = 0.9;
        ctx.strokeStyle = cssVar(sr.token);
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.globalAlpha = 1;
      });
      ctx.font = '500 11px ' + cssVar('--font-sans');
      ctx.fillStyle = cssVar('--foreground-muted');
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(peak), x0 - 6, y0 + 4);
      ctx.fillText('0', x0 - 6, y0 + hh);

      /* crosshair: the read head the tooltip is reporting */
      if (hoverX != null) {
        const hx = clamp(hoverX, x0, x0 + plotW);
        const i = sampleAt(S.t0 + ((hx - x0) / plotW) * (S.t1 - S.t0));
        ctx.save();
        ctx.strokeStyle = cssVar('--foreground-secondary');
        ctx.globalAlpha = 0.55;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(Math.round(hx) + 0.5, y0 - 6);
        ctx.lineTo(Math.round(hx) + 0.5, y0 + hh);
        ctx.stroke();
        ctx.restore();
        if (i >= 0) {
          SERIES.forEach((sr) => {
            ctx.beginPath();
            ctx.arc(hx, sy(q[i][sr.idx]), 2.5, 0, Math.PI * 2);
            ctx.fillStyle = cssVar(sr.token);
            ctx.fill();
          });
        }
      }
    };

    /* the shared pattern owns the tooltip chrome; only the rows are ours */
    const tipRow = (k, v, cls) => '<div class="pto-swimlane-task-tooltip__row">'
      + '<span class="pto-swimlane-task-tooltip__key">' + esc(k) + '</span>'
      + '<span class="pto-swimlane-task-tooltip__value' + (cls ? ' ' + cls : '') + '">'
      + esc(v) + '</span></div>';

    /* the tooltip lives on the frame layer: .tc-canvas-strip clips to its
     * rounded corners, and the strip is only 92px tall. */
    const tipLayer = document.querySelector('.tc-frame');
    const tip = SW.createTooltip();
    tip.dataset.tcTip = 'readyq';
    tipLayer.appendChild(tip);

    canvas.addEventListener('pointermove', (event) => {
      if (!geom) return;
      hoverX = event.clientX - canvas.getBoundingClientRect().left;
      draw();
      const hx = clamp(hoverX, geom.x0, geom.x0 + geom.plotW);
      const t = S.t0 + ((hx - geom.x0) / geom.plotW) * (S.t1 - S.t0);
      const i = sampleAt(t);
      if (i < 0) { SW.hideTooltip(tip); return; }
      const held = (i + 1 < q.length ? q[i + 1][0] : q[q.length - 1][0]) - q[i][0];
      const total = q[i][1] + q[i][2] + q[i][3];
      const html = '<div class="pto-swimlane-task-tooltip__title">t = ' + num(t, 1) + ' us</div>'
        + tipRow('采样', num(q[i][0], 2) + ' us · #' + (i + 1) + '/' + q.length)
        + tipRow('保持', num(held, 2) + ' us')
        + tipRow('AIC ready', String(q[i][1]), q[i][1] > 0 ? 'is-warn' : '')
        + tipRow('AIV ready', String(q[i][2]))
        + tipRow('MIX ready', String(q[i][3]))
        + tipRow('待派发合计', String(total));
      SW.showTooltip(tip, { counterReadout: true }, event,
        { bounds: tipLayer, target: canvas, getTooltipHtml: () => html });
    });
    canvas.addEventListener('pointerleave', () => {
      hoverX = null;
      draw();
      SW.hideTooltip(tip);
    });

    requestAnimationFrame(draw);
    if (body.__ro) body.__ro.disconnect();
    body.__ro = new ResizeObserver(draw);
    body.__ro.observe(host);
  }

  /* -------------------------------------------------------- terminal */
  const TERM_TABS = [
    { id: 'problems', label: 'Problems' },
    { id: 'output', label: 'Output' },
    { id: 'artifacts', label: 'Artifacts' },
  ];

  function renderTerminal() {
    const tabs = $('#terminalTabs');
    tabs.textContent = '';
    TERM_TABS.forEach((t) => {
      const b = el('span', 'pto-ide-frame__terminal-tab' + (t.id === S.termTab ? ' is-selected' : ''),
        t.label + (t.id === 'problems' ? ' (' + D.hints.length + ')' : ''));
      b.tabIndex = 0;
      b.addEventListener('click', () => { S.termTab = t.id; renderTerminal(); });
      tabs.appendChild(b);
    });

    const body = $('#terminalBody');
    body.textContent = '';

    if (S.termTab === 'problems') {
      const list = el('div', 'tc-term-list');
      D.hints.forEach((h) => {
        const row = el('button', 'tc-term-row');
        row.type = 'button';
        row.dataset.sev = h.kind === 'pipeline-depth' ? 'warn' : 'info';
        row.appendChild(el('span', 'sev', h.code));
        row.appendChild(el('span', 'loc', h.file + ':' + h.line));
        row.appendChild(el('span', 'msg', h.kind === 'pipeline-depth'
          ? 'MemoryReuse: depth ' + h.reqDepth + ' → ' + h.fit + ' @' + h.unit + ' group ' + h.group
            + ' (' + h.perStageB + ' B/stage, ' + h.freeB + ' B free)'
          : 'TileInnermostDimGranularity: ' + h.op + ' innermost ' + h.innermostB + 'B, tile '
            + h.dtype + '[' + h.tileShape + '] → ' + h.mem + ', recommended ≥ ' + h.recB + 'B'));
        row.addEventListener('click', () => {
          S.view = 'compiler';
          S.compilerTab = h.kind === 'pipeline-depth' ? 'depth' : 'granularity';
          S.hintSite = h.file + ':' + h.line;
          S.focus = 'hint';
          render();
        });
        list.appendChild(row);
      });
      body.appendChild(list);
      return;
    }

    if (S.termTab === 'output') {
      const lines = [];
      Object.keys(D.e2e || {}).forEach((rank) => {
        Object.keys(D.e2e[rank]).forEach((inv) => {
          lines.push('# ' + rank + ' inv=' + inv);
          Object.keys(D.e2e[rank][inv]).forEach((name) => {
            const sp = D.e2e[rank][inv][name];
            lines.push('  [' + sp.clk.padEnd(6) + '] ' + name.padEnd(46) + ' dur=' + sp.us.toFixed(2) + ' us');
          });
        });
      });
      const pre = el('pre', 'tc-term-static', lines.join('\n'));
      body.appendChild(pre);
      return;
    }

    /* Artifact inventory for the active case: present rows carry what they
     * answer, absent rows say so instead of being dropped silently. */
    const A = D.case.artifacts;
    const rankDir = (r) => 'dfx_outputs/' + (multiRank() ? r + '/d0/' : '');
    const rows = [
      ['distributed_meta.json', A.distributedMeta
        ? D.case.params.length + ' 个绑定参数，schema ' + D.case.metaSchema : '缺失'],
    ].concat(D.case.ranks.map((r) => [
      rankDir(r) + 'merged_swimlane_*.json',
      'Worker + Scheduler view，' + D.ranks[r].tasks.length + ' 任务 / '
        + D.ranks[r].swimlane.blocks.reduce((a, b) => a + b.length, 0) + ' 块',
    ])).concat([
      [rankDir(D.defaultRank) + 'deps.json', 'scope / 绑定张量'
        + (D.ranks[D.defaultRank].tasks[0].blockNum != null ? ' / block_num / early_dispatch' : '')],
      [rankDir(D.defaultRank) + 'name_map.json', D.case.callables + ' 个 kernel 名（level '
        + D.case.level + '）—— 对应 ' + scopeCountOf() + ' 个 scope，混合 scope 各拆两个'],
      [rankDir(D.defaultRank) + 'host.*.log', A.hostSpans
        ? 'STRACE host span（bind / runner_run / device_wall / sched）' : '缺失 —— 无 E2E 层'],
      ['report/perf_hints.log', D.hints.length + ' 条 perf hint（'
        + Array.from(new Set(D.hints.map((h) => h.code))).sort().join(' / ') + '）'],
      ['passes_dump/', D.passes.length + ' 个 IR dump，' + D.passes[0].lines
        + ' → ' + D.passes[D.passes.length - 1].lines + ' 行'],
      ['chip_swimlane_records.json', A.chipSwimlaneRecords
        ? '核清单与时钟频率' : '缺失 —— 核数由 trace 线程名推得'],
      ['binary_context.json', A.binaryContext
        ? 'platform ' + D.case.toolchain.platform + ' · pto-isa '
          + D.case.toolchain.ptoIsaRevision.slice(0, 12) + ' · runtime ' + D.case.toolchain.runtimeName
        : '缺失'],
      ['ptoas/', A.ptoas ? A.ptoas + ' 个单元（.pto + .cpp）' : '缺失'],
      ['kernels/', Object.keys(A.kernelDirs).length
        ? Object.keys(A.kernelDirs).map((k) => k + ' ' + A.kernelDirs[k]).join(' · ') : '缺失'],
      ['orchestration/', A.orchestration.length ? A.orchestration.join(' · ') : '缺失'],
      ['kernel_config.py', A.kernelConfig
        ? 'runtime ' + (D.case.toolchain.runtimeName || '—')
          + (D.case.toolchain.aicpuThreads ? ' · ' + D.case.toolchain.aicpuThreads + ' AICPU 线程' : '')
        : '缺失'],
    ]);
    body.appendChild(table([
      { label: '产物', cell: (r) => esc(r[0]), mono: true },
      { label: '内容', cell: (r) => esc(r[1]) },
    ], rows, {}));
  }

  /* One device, or no host spans: there is no cross-rank comparison to make,
   * so the inspector reports this run on its own terms. */
  function renderRunInspectorSingle(host) {
    const R0 = R();
    const s2 = inspectorSection('本次运行', D.case.ranks.length + ' 个执行单元');
    s2.appendChild(kv([
      ['trace span', num(R0.swimlane.spanUs, 1) + ' us'],
      ['device_wall', hasE2E()
        ? num(D.e2e[S.rank][TRACE_MATCH[S.rank].inv]['chip.run.runner_run.device_wall'].us, 1) + ' us'
        : '无 host log'],
      ['AIC 占用', pct(R0.occupancy.aicUtil)],
      ['AIV 占用', pct(R0.occupancy.aivUtil)],
      ['依赖关键路径', R0.critical.tags.length + ' 节点（静态 CPM）'],
      ['调度器占用', pct(R0.scheduler.perLaneUtil)],
    ]));
    if (!hasE2E()) {
      s2.appendChild(el('div', 'inspector-soft-card is-warning',
        '无 host STRACE log：迭代次数、device_wall、bind 缓存命中都不可得，'
        + '基线只能锁在 trace span ' + num(R0.swimlane.spanUs, 1) + ' us 上'));
    }
    host.appendChild(s2);

    const s3 = inspectorSection('瓶颈链', topChains(3).length + ' 条 · 体检项见左栏');
    topChains(3).forEach((f) => {
      s3.appendChild(btn(f.id + ' · ' + f.title, {
        variant: 'ghost', size: 'sm',
        on: () => { S.finding = f.id; S.focus = 'finding'; applyFocus(f); render(); },
      }));
    });
    host.appendChild(s3);
    host.appendChild(renderLedger());
  }

  /* ========================================================= chrome */
  function renderTabs() {
    const host = $('#levelTabs');
    host.textContent = '';
    LEVELS.forEach((l) => {
      const b = el('button', 'tab-control-item' + (l.id === S.view ? ' is-selected' : ''), l.label);
      b.type = 'button';
      b.title = l.hint;
      b.setAttribute('aria-pressed', l.id === S.view ? 'true' : 'false');
      b.addEventListener('click', () => { S.view = l.id; S.focus = null; render(); });
      host.appendChild(b);
    });
  }

  function renderToolbar() {
    const host = $('#viewToolbar');
    host.textContent = '';
    /* sub-view switch sits where the view's own tab would: on the left */
    if (S.view === 'compiler') {
      host.appendChild(group('segmented-control segmented-control-muted', [
        { id: 'passes', label: 'Pass 轨迹' },
        { id: 'depth', label: '流水深度' },
        { id: 'granularity', label: '搬运粒度' },
      ], S.compilerTab, (v) => { S.compilerTab = v; render(); }));
    }

    const right = el('div', 'tc-toolbar-right');

    if (S.view === 'compiler') {
      right.appendChild(el('span', 'tc-readout', D.passes.length + ' Pass dump · ' + D.hints.length + ' perf hint'));
    }

    if (S.view === 'l2') {
      right.appendChild(field('泳道', select([
        { id: 'all', label: '全部 ' + R().swimlane.lanes.length },
        { id: 'aic', label: 'AIC ' + D.case.aicCount },
        { id: 'aiv', label: 'AIV ' + D.case.aivCount },
      ], S.laneFilter, (v) => { S.laneFilter = v; render(); })));
      const colorField = el('div', 'tc-field');
      colorField.appendChild(el('span', null, '着色'));
      colorField.appendChild(btn(S.colorOn ? '开' : '关', {
        size: 'sm', selected: S.colorOn,
        title: S.colorOn ? '关闭算子配色，让占用率与空转段更突出' : '恢复算子配色',
        on: () => { S.colorOn = !S.colorOn; render(); },
      }));
      if (S.colorOn) {
        colorField.appendChild(select([
          { id: 'semantic', label: '按算子' },
          { id: 'engine', label: '按引擎' },
        ], S.colorMode, (v) => { S.colorMode = v; render(); }));
      }
      right.appendChild(colorField);
      right.appendChild(field('叠加', select([
        { id: 'sched', label: 'AICPU 调度' },
        { id: 'ready', label: 'Ready queue' },
        { id: 'none', label: '无' },
      ], S.overlay, (v) => { S.overlay = v; render(); })));
      right.appendChild(field('只看', select([
        { id: 'off', label: '全部任务' },
        { id: 'obs', label: '观测路径' },
        { id: 'cpm', label: '依赖关键路径' },
      ], S.pathOnly, (v) => { S.pathOnly = v; S.critOnly = v !== 'off'; render(); })));
      const zoomGroup = el('div', 'toolbar-control');
      zoomGroup.appendChild(btn('−', { variant: 'ghost', size: 'icon', title: '缩小', on: () => { zoom(2); redrawStage(); renderToolbar(); renderDock(); } }));
      zoomGroup.appendChild(btn('Fit', { variant: 'ghost', size: 'sm', on: () => { S.t0 = 0; S.t1 = R().swimlane.spanUs; redrawStage(); renderToolbar(); renderDock(); } }));
      zoomGroup.appendChild(btn('+', { variant: 'ghost', size: 'icon', title: '放大', on: () => { zoom(0.5); redrawStage(); renderToolbar(); renderDock(); } }));
      right.appendChild(zoomGroup);
      right.appendChild(el('span', 'tc-readout', num(S.t0, 0) + '–' + num(S.t1, 0) + ' us · shift+拖动平移'));
    }

    if (S.view === 'l1') {
      const ordered = R().tasks.slice().sort((a, b) => b.span - a.span);
      right.appendChild(field('kernel', select(ordered.map((t) => ({
        id: t.tag, label: t.callable + ' · ' + t.tag + '（' + num(t.span, 0) + ' us）',
      })), S.task, (v) => { S.task = v; S.focus = 'task'; render(); })));
    }

    if ((S.view === 'l2' || S.view === 'l1') && multiRank()) {
      right.appendChild(field('rank', select(Object.keys(D.ranks).map((r) => ({ id: r, label: r })), S.rank,
        (v) => {
          S.rank = v;
          S.t0 = 0; S.t1 = R().swimlane.spanUs;
          if (!tasksOf[S.rank][S.task]) S.task = R().tasks[0].tag;
          render();
        })));
    }

    host.appendChild(right);
    host.hidden = !host.childNodes.length || (host.childNodes.length === 1 && !right.childNodes.length);
  }

  function renderExplorer() {
    const tree = $('#runTree');
    tree.textContent = '';
    const row = (depth, label, metaText, opts) => {
      const o = opts || {};
      const b = el('button', 'tc-tree-row' + (o.selected ? ' is-selected' : ''));
      b.type = 'button';
      b.dataset.depth = depth;
      b.appendChild(el('span', 'n', label));
      if (metaText) b.appendChild(el('span', 'm', metaText));
      if (o.on) b.addEventListener('click', o.on);
      else b.disabled = true;
      tree.appendChild(b);
      return b;
    };
    row(0, D.case.program, D.case.backend);
    Object.keys(D.ranks).forEach((rank) => {
      const m = TRACE_MATCH[rank];
      row(1, rank + ' / ' + D.case.device, us(D.ranks[rank].swimlane.spanUs, 0), {
        selected: rank === S.rank,
        on: () => {
          S.rank = rank;
          S.t0 = 0; S.t1 = R().swimlane.spanUs;
          if (!tasksOf[S.rank][S.task]) S.task = R().tasks[0].tag;
          render();
        },
      });
      if (D.e2e && D.e2e[rank]) {
        Object.keys(D.e2e[rank]).forEach((inv) => {
          row(2, 'inv=' + inv + (m && m.inv === +inv ? ' · traced' : ''),
            us(D.e2e[rank][inv]['chip.run.runner_run.device_wall'].us, 0), {
              on: () => { S.rank = rank; S.view = 'e2e'; render(); },
            });
        });
      } else {
        row(2, 'host.*.log', '缺失');
      }
    });
    row(0, 'artifacts', D.passes.length + ' passes');
    row(1, 'report/perf_hints.log', D.hints.length, {
      on: () => { S.view = 'compiler'; S.compilerTab = 'depth'; render(); },
    });
    row(1, 'passes_dump/', D.passes.length, {
      on: () => { S.view = 'compiler'; S.compilerTab = 'passes'; render(); },
    });
    row(1, 'binary_context.json', D.case.toolchain.platform, {
      on: () => { S.view = 'isa'; render(); },
    });

    $('[data-bind="explorerMeta"]').textContent = D.case.runDir.slice(0, 18) + '…';

    /* findings queue */
    const filterHost = $('#findingFilter');
    filterHost.textContent = '';
    /* a chain spans several layers, so it counts under every layer it visits;
     * filtering by "编译器" must not hide the chain that landed there */
    const counts = { all: D.findings.length };
    D.findings.forEach((f) => {
      (f.levels || [f.level]).forEach((lv) => { counts[lv] = (counts[lv] || 0) + 1; });
    });
    [{ id: 'all', label: '全部' }].concat(LEVELS.filter((l) => counts[l.id]).map((l) => ({ id: l.id, label: l.label })))
      .forEach((o) => {
        filterHost.appendChild(btn(o.label + ' ' + (counts[o.id] || 0), {
          size: 'sm', selected: S.findingLevel === o.id,
          on: () => { S.findingLevel = o.id; render(); },
        }));
      });

    const list = $('#findingList');
    list.textContent = '';
    const shown = D.findings.filter((f) => S.findingLevel === 'all'
      || (f.levels || [f.level]).indexOf(S.findingLevel) >= 0);
    const logged = {};
    S.ledger.forEach((r) => { if (r.findingId) logged[r.findingId] = 1; });

    const findingRow = (f) => {
      const b = el('button', 'tc-finding'
        + (f.kind === 'hygiene' ? ' is-hygiene' : '')
        + (f.id === S.finding && S.focus === 'finding' ? ' is-selected' : '')
        + (logged[f.id] ? ' is-logged' : ''));
      b.type = 'button';
      b.dataset.sev = f.severity;
      const hd = el('div', 'hd');
      hd.appendChild(el('span', 'id', f.id));
      /* a chain advertises the layers it crosses; that path IS its identity */
      hd.appendChild(el('span', 'lv', (f.levels || [f.level])
        .map((lv) => LEVEL_LABEL[lv] || lv).join(' → ')));
      b.appendChild(hd);
      b.appendChild(el('span', 'ti', f.title));
      const mt = el('div', 'mt');
      mt.appendChild(el('span', 'm', f.metric));
      mt.appendChild(el('span', f.cost ? 'cost' : 'cost is-none',
        f.cost ? f.cost.share + '%' : '无归因'));
      b.appendChild(mt);
      b.addEventListener('click', () => {
        S.finding = f.id;
        S.chainStep = null;
        S.focus = 'finding';
        applyFocus(f);
        render();
      });
      return b;
    };

    /* Two groups, never interleaved: a chain carries a makespan attribution,
     * a hygiene item carries the reason it does not. Mixing them is how a
     * hint count ends up looking as urgent as a 39% finding. */
    const group = (label, kicker, rows) => {
      if (!rows.length) return;
      const h = el('div', 'tc-finding-group');
      h.appendChild(el('span', 'gl', label));
      h.appendChild(el('span', 'gk', kicker));
      list.appendChild(h);
      rows.forEach((f) => list.appendChild(findingRow(f)));
    };
    const chains = shown.filter((f) => f.kind !== 'hygiene');
    const hyg = shown.filter((f) => f.kind === 'hygiene');
    group('瓶颈链', chains.length
      ? '合计 ' + num(chains.reduce((n, f) => n + (f.cost ? f.cost.us : 0), 0), 0)
        + ' us · 与 makespan 有时间重叠，不可相加'
      : '本层无', chains);
    group('体检项', hyg.length ? '无 makespan 归因' : '本层无', hyg);
    $('[data-bind="findingCount"]').textContent = shown.length + ' / ' + D.findings.length;
  }

  /* Activating a finding puts the stage where its evidence lives and turns on
   * evidence focus, so the reader never has to guess which parts of the screen
   * the inspector is talking about. */
  function applyFocus(f) {
    if (!f) return;
    applySubjects(f.subjects || {}, f.focus && f.focus.pass);
  }

  /* A chain step is focusable in its own right: the reader walks the ladder
   * and the stage follows one layer at a time, instead of the whole chain
   * always dumping them on its first layer. */
  function applyStep(f, st) {
    if (!f || !st) return;
    S.chainStep = f.id + ':' + (f.chain || []).indexOf(st);
    applySubjects(st.subjects || {}, st.level === 'compiler' ? (f.rootPass || 'MemoryReuse') : null);
  }

  function applySubjects(s, passName) {
    S.view = s.view || S.view;
    if (s.tab) S.compilerTab = s.tab;
    if (s.overlay) S.overlay = s.overlay;
    if (s.tasks && s.tasks.length && tasksOf[S.rank][s.tasks[0]]) S.task = s.tasks[0];
    if (s.sites && s.sites.length) S.hintSite = s.sites[0];
    if (s.lanes && s.lanes.length) S.laneFilter = s.lanes[0].indexOf('AIC') === 0 ? 'aic' : 'aiv';
    if (passName) {
      const p = D.passes.find((x) => x.name === passName);
      if (p) S.pass = p.idx;
    }
    S.critOnly = false;
    S.pathOnly = 'off';
    S.focusEvidence = true;
    if (S.view === 'l2') { S.t0 = 0; S.t1 = R().swimlane.spanUs; }
  }

  function renderStatus() {
    const host = $('#statusStrip');
    host.textContent = '';
    const rank = R();
    const open = openExperiment();
    const items = [
      ['case', D.case.program],
      ['rank', S.rank + (TRACE_MATCH[S.rank] ? ' inv=' + TRACE_MATCH[S.rank].inv : ' · 无 host log')],
      ['span', us(rank.swimlane.spanUs, 1)],
      ['tasks', String(rank.tasks.length)],
      ['crit', rank.critical.tags.length + ' CPM / ' + rank.cpath.segments.length + ' 观测'],
      ['AIC / AIV', pct(rank.occupancy.aicUtil, 0) + ' / ' + pct(rank.occupancy.aivUtil, 0)],
      ['sched', pct(rank.scheduler.perLaneUtil, 0)],
      ['hints', String(D.hints.length)],
    ];
    items.forEach((it) => {
      const s = el('span', 'tc-status-item');
      s.appendChild(el('span', 'k', it[0]));
      s.appendChild(el('span', 'v', it[1]));
      host.appendChild(s);
    });
    const pmu = el('span', 'tc-status-item');
    pmu.appendChild(el('span', 'k', 'pmu'));
    const pv = el('span', 'v warn', 'off');
    pmu.appendChild(pv);
    host.appendChild(pmu);
    const exp = el('span', 'tc-status-item');
    exp.appendChild(el('span', 'k', '实验'));
    exp.appendChild(el('span', 'v' + (open ? ' warn' : ' ok'), open ? open.id + ' 进行中' : '无进行中'));
    host.appendChild(exp);
  }

  function renderFingerprint() {
    const host = $('#fingerprint');
    host.textContent = '';
    const head = el('header', 'panel-shell-header');
    head.appendChild(el('h2', 'panel-shell-title', 'Case fingerprint'));
    head.appendChild(el('span', 'panel-shell-meta', D.case.runDir));
    const close = el('button', 'panel-shell-close', '×');
    close.type = 'button';
    close.setAttribute('aria-label', '关闭');
    close.addEventListener('click', () => toggleFingerprint(false));
    head.appendChild(close);
    host.appendChild(head);
    const body = el('div', 'panel-shell-body');
    const dl = el('dl', 'tc-fp-grid');
    const add = (k, v) => { dl.appendChild(el('dt', null, k)); dl.appendChild(el('dd', null, v)); };
    add('program', D.case.program);
    add('model', D.case.model);
    add('platform / backend', D.case.toolchain.platform + ' / ' + D.case.backend);
    add('pto-isa', D.case.toolchain.ptoIsaRevision);
    add('runtime', D.case.toolchain.runtimeName + ' @ ' + D.case.toolchain.runtimeRevision);
    add('ranks / device', D.case.ranks.join(', ') + ' / ' + D.case.device);
    add('cores', D.case.numCores + '（AIC ' + D.case.aicCount + ' + AIV ' + D.case.aivCount + '，每核 ' + D.case.threadsPerCore + ' thread）');
    add('trace clock', (D.case.clockHz / 1e6) + ' MHz');
    add('kernel / scope', D.case.callables + ' 个 kernel 名 / ' + scopeCountOf()
      + ' 个 scope（IR incore scope ' + D.case.incoreScopes.length + '）');
    add('captured', D.case.capturedAt);
    add('source root', D.case.sourceRoot);
    body.appendChild(dl);
    body.appendChild(sectionHead('绑定参数', '前 12 / ' + D.case.params.length));
    body.appendChild(table([
      { label: 'name', cell: (p) => esc(p.name.replace(/__ssa_v0$/, '')), mono: true },
      { label: 'dir', key: 'dir' },
      { label: 'dtype', key: 'dtype', mono: true },
      { label: 'shape', cell: (p) => '[' + p.shape.join(', ') + ']', mono: true },
    ], D.case.params.slice(0, 12), {}));
    host.appendChild(body);
  }

  function toggleFingerprint(force) {
    const host = $('#fingerprint');
    const chip = document.querySelector('[data-act="toggle-fingerprint"]');
    const open = force == null ? host.hidden : force;
    host.hidden = !open;
    chip.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  /* ---------------------------------------------------------- search */
  function buildSearchIndex() {
    const idx = [];
    Object.keys(D.ranks).forEach((rank) => {
      D.ranks[rank].tasks.forEach((t) => {
        idx.push({
          kind: 'task', text: t.callable + ' ' + t.tag, name: t.callable,
          value: t.tag + ' · ' + num(t.span, 0) + ' us · ' + rank,
          go: () => { S.rank = rank; S.view = 'l1'; S.task = t.tag; S.focus = 'task'; },
        });
      });
    });
    D.findings.forEach((f) => {
      idx.push({
        kind: 'finding',
        text: [f.id, f.title, f.axis, f.kind].concat(f.levels || [])
          .concat((f.chain || []).map((st) => st.headline)).join(' '),
        name: f.id + ' ' + f.title,
        value: f.cost ? f.cost.share + '% · ' + f.metric : '无归因 · ' + f.metric,
        go: () => { S.finding = f.id; S.chainStep = null; S.focus = 'finding'; applyFocus(f); },
      });
    });
    D.passes.forEach((p) => {
      idx.push({
        kind: 'pass', text: p.name, name: p.name, value: '#' + p.idx + ' · ' + p.lines + ' 行',
        go: () => { S.view = 'compiler'; S.compilerTab = 'passes'; S.pass = p.idx; S.focus = 'pass'; },
      });
    });
    D.depthSites.forEach((s) => {
      idx.push({
        kind: 'hint', text: 'PH-MR-001 ' + s.module + ':' + s.line, name: s.file + ':' + s.line,
        value: 'depth ' + s.maxReqDepth + ' → ' + s.fittedDepth,
        go: () => { S.view = 'compiler'; S.compilerTab = 'depth'; S.hintSite = s.key; S.focus = 'hint'; },
      });
    });
    D.tileSites.forEach((s) => {
      idx.push({
        kind: 'hint', text: 'PH001 ' + s.module + ':' + s.line, name: s.file + ':' + s.line,
        value: '末维 ' + s.minB + 'B',
        go: () => { S.view = 'compiler'; S.compilerTab = 'granularity'; S.hintSite = s.key; S.focus = 'hint'; },
      });
    });
    return idx;
  }
  const SEARCH = buildSearchIndex();

  function runSearch(q) {
    const box = $('#searchResults');
    const input = $('#searchInput');
    box.textContent = '';
    const query = q.trim().toLowerCase();
    if (!query) { box.hidden = true; input.setAttribute('aria-expanded', 'false'); return; }
    const hits = SEARCH.filter((h) => h.text.toLowerCase().indexOf(query) >= 0).slice(0, 24);
    if (!hits.length) {
      box.appendChild(el('div', 'tc-search-empty', '没有匹配的 kernel、任务、提示或 Pass'));
    } else {
      hits.forEach((h) => {
        const b = el('button', 'tc-search-hit');
        b.type = 'button';
        b.appendChild(el('span', 'k', h.kind));
        b.appendChild(el('span', 'n', h.name));
        b.appendChild(el('span', 'v', h.value));
        b.addEventListener('click', () => {
          h.go();
          input.value = '';
          box.hidden = true;
          input.setAttribute('aria-expanded', 'false');
          render();
        });
        box.appendChild(b);
      });
    }
    box.hidden = false;
    input.setAttribute('aria-expanded', 'true');
  }

  /* ============================================================ render */
  function redrawStage() {
    const stage = $('#stage');
    if (stage.__redraw) stage.__redraw();
  }

  function render() {
    const stage = $('#stage');
    if (stage.__ro) { stage.__ro.disconnect(); stage.__ro = null; }
    stage.__redraw = null;
    stage.textContent = '';

    renderTabs();
    renderToolbar();
    renderExplorer();
    findingBar(stage);

    if (S.view === 'e2e') viewE2E(stage);
    else if (S.view === 'l2') viewL2(stage);
    else if (S.view === 'l1') viewL1(stage);
    else if (S.view === 'compiler') viewCompiler(stage);
    else viewISA(stage);

    renderInspector();
    renderDock();
    renderTerminal();
    renderStatus();
    $('[data-bind="caseChip"]').textContent = D.case.program + ' · ' + S.rank;
  }

  /* ------------------------------------------------------------- boot */
  /* Switching case swaps the whole dataset. The two dumps carry different
   * artifacts, so every layer re-derives what it can and says what it cannot. */
  function switchCase(id) {
    if (id === D.case.id) { toggleCaseMenu(false); return; }
    loadCase(id);
    S.tile = defaultTile();
    toggleCaseMenu(false);
    renderFingerprint();
    render();
  }

  function toggleCaseMenu(force) {
    const menu = $('#caseMenu');
    const chip = document.querySelector('[data-act="toggle-fingerprint"]');
    const open = force === undefined ? menu.hidden : force;
    menu.hidden = !open;
    chip.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) toggleFingerprint(false);
  }

  function renderCaseMenu() {
    const menu = $('#caseMenu');
    menu.textContent = '';
    CASES.forEach((c) => {
      const run = RUNS[c.id];
      const b = el('button', 'tc-case-item' + (c.id === D.case.id ? ' is-selected' : ''));
      b.type = 'button';
      const hd = el('div', 'hd');
      hd.appendChild(el('span', 'nm', c.label));
      hd.appendChild(el('span', 'sub', c.sub));
      b.appendChild(hd);
      /* say up front which layers this dump can answer */
      const layers = el('div', 'ly');
      [
        ['E2E', !!run.e2e],
        ['L2', true],
        ['L1/L0', true],
        ['编译器', run.passes.length > 0],
        ['ISA', run.case.artifacts.ptoas > 0],
      ].forEach((pair) => {
        layers.appendChild(el('span', pair[1] ? 'on' : 'off', pair[0]));
      });
      b.appendChild(layers);
      b.appendChild(el('div', 'mt', run.ranks[run.defaultRank].tasks.length + ' 任务 · '
        + (run.chainCount != null ? run.chainCount + ' 条瓶颈链 · ' + run.hygieneCount + ' 条体检项'
          : run.findings.length + ' 条瓶颈')
        + ' · ' + run.hints.length + ' 条提示'));
      b.addEventListener('click', () => switchCase(c.id));
      menu.appendChild(b);
    });
    const fp = el('button', 'tc-case-item is-action', 'Case fingerprint …');
    fp.type = 'button';
    fp.addEventListener('click', () => { toggleCaseMenu(false); toggleFingerprint(true); });
    menu.appendChild(fp);
  }

  function boot() {
    if (window.PtoIdeFrame) window.PtoIdeFrame.initAll();
    if (EMBED_VIEW) document.body.classList.add('tc-embed-view');
    loadCase(initialCase);
    S.tile = defaultTile();
    S.view = EMBED_VIEW || 'e2e';
    S.focus = null;
    renderCaseMenu();
    renderFingerprint();
    render();

    document.querySelector('[data-act="toggle-fingerprint"]').addEventListener('click', () => toggleCaseMenu());
    document.querySelector('[data-act="theme"]').addEventListener('click', () => {
      const root = document.documentElement;
      root.dataset.theme = root.dataset.theme === 'light' ? 'dark' : 'light';
      render();
    });
    document.querySelector('[data-act="focus-search"]').addEventListener('click', () => $('#searchInput').focus());
    document.querySelector('[data-act="show-ledger"]').addEventListener('click', () => {
      const btnEl = document.querySelector('[data-ide-toggle="inspector"]');
      if (btnEl && btnEl.getAttribute('aria-expanded') === 'false') btnEl.click();
      $('#inspector').scrollTop = $('#inspector').scrollHeight;
    });
    document.querySelector('[data-act="open-terminal"]').addEventListener('click', () => {
      const btnEl = document.querySelector('[data-ide-toggle="terminal"]');
      if (btnEl && btnEl.getAttribute('aria-expanded') === 'false') btnEl.click();
    });

    const input = $('#searchInput');
    input.addEventListener('input', () => runSearch(input.value));
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { input.value = ''; runSearch(''); input.blur(); }
      if (e.key === 'Enter') {
        const first = $('#searchResults').querySelector('.tc-search-hit');
        if (first) first.click();
      }
    });
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.tc-search')) { $('#searchResults').hidden = true; }
      if (!e.target.closest('#fingerprint') && !e.target.closest('[data-act="toggle-fingerprint"]')) {
        toggleFingerprint(false);
      }
      if (!e.target.closest('#caseMenu') && !e.target.closest('[data-act="toggle-fingerprint"]')) {
        toggleCaseMenu(false);
      }
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && S.scopeReturn && S.view === 'l2'
        && document.activeElement !== input && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault(); scopeBack(); return;
      }
      if (e.key === '/' && document.activeElement !== input) { e.preventDefault(); input.focus(); }
      if (e.key >= '1' && e.key <= '5' && !e.metaKey && !e.ctrlKey
        && document.activeElement !== input && document.activeElement.tagName !== 'INPUT'
        && document.activeElement.tagName !== 'TEXTAREA') {
        S.view = LEVELS[+e.key - 1].id;
        render();
      }
    });
    window.addEventListener('resize', () => { redrawStage(); });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot, { once: true });
  else boot();
})();
