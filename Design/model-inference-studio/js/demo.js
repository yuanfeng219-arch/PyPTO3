/* PyPTO Model Inference Studio — 交互逻辑 */
(function () {
  'use strict';

  const D = window.MIS_DATA;
  const COV = D.COVERAGE_META;
  const SEM = D.SEM;
  const PREC = D.PRECISION_ORDER;
  const PRIO_COLOR = { P0: '#FF2D7A', P1: '#FF9D00', P2: '#FFE600' };

  const STEPS = [
    { id: 'import', label: '导入模型工件', view: 'architecture' },
    { id: 'detect', label: '识别模型结构', view: 'architecture' },
    { id: 'coverage', label: '生成覆盖报告', view: 'coverage' },
    { id: 'precision', label: '精度与量化画像', view: 'precision' },
    { id: 'recipe', label: '组合 Recipe 并生成骨架', view: 'recipe' }
  ];

  const state = {
    model: D.models['qwen3-moe'],
    hardware: 'a3',
    phase: 'decode',
    step: 0,
    imported: false,
    tab: 'architecture',
    selectedNode: null,
    selectedRecipe: 'r-moe',
    selectedPattern: 'p-moe',
    covFilter: null,
    view: { x: 0, y: 0, k: 1 }
  };

  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  let toastTimer;
  function toast(msg) {
    const el = $('#toast');
    el.textContent = msg;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 2400);
  }

  function status(phase, text, kind) {
    $('#statusPhase').textContent = phase;
    $('#statusText').textContent = text;
    const dot = $('#statusDot');
    dot.className = 'mis-status-dot' + (kind ? ' is-' + kind : '');
  }

  function covOf(nodeId) {
    return state.model.coverage.find((c) => c.node === nodeId) || null;
  }
  function nodeOf(id) {
    return state.model.graph.nodes.find((n) => n.id === id) || null;
  }
  function covTotals() {
    const t = {};
    Object.keys(COV).forEach((k) => { t[k] = 0; });
    state.model.coverage.forEach((c) => { t[c.cov] += 1; });
    return t;
  }
  function fallbackCost() {
    return state.model.coverage
      .filter((c) => c.cov === 'fallback')
      .reduce((a, c) => a + c.share, 0);
  }
  function blockingCount() {
    return state.model.coverage.filter((c) => c.cov === 'unsupported').length;
  }

  /* ============ Explorer ============ */
  function renderArtifacts() {
    const list = $('#artifactList');
    list.innerHTML = state.model.artifacts.map((a) => {
      const cls = state.imported ? (a.status === 'warn' ? 'is-warn' : 'is-loaded') : '';
      const icon = state.imported ? (a.status === 'warn' ? '!' : '✓') : '○';
      return `<li class="${cls}" data-artifact="${a.id}" title="${esc(a.detail)}">
        <i>${icon}</i><b>${esc(a.name)}</b><small>${esc(a.size)}</small></li>`;
    }).join('');
    $('#artifactCount').textContent = (state.imported ? state.model.artifacts.length : 0) + ' / ' + state.model.artifacts.length;
    $('#explorerMeta').textContent = state.imported ? state.model.family : '未导入';
  }

  function renderSteps() {
    $('#stepList').innerHTML = STEPS.map((s, i) => {
      const done = i < state.step;
      const cur = i === state.step;
      const locked = i > state.step;
      const cls = ['mis-step', done ? 'is-done' : '', cur ? 'is-selected' : '', locked ? 'is-locked' : ''].filter(Boolean).join(' ');
      return `<li><button class="${cls}" type="button" data-step="${i}" ${locked ? 'disabled' : ''}>
        <span class="mis-step-num">${done ? '✓' : i + 1}</span><span>${esc(s.label)}</span></button></li>`;
    }).join('');
    $('#stepState').textContent = state.step + ' / ' + STEPS.length;
  }

  function renderStructures() {
    const box = $('#structureTags');
    if (!state.imported || state.step < 2) {
      box.innerHTML = '<span class="mis-empty-hint">导入后由 Architecture Detector 识别</span>';
      $('#structureCount').textContent = '—';
      return;
    }
    box.innerHTML = state.model.structures.map((s) => {
      const sel = state.selectedNode && (s.id === state.selectedNode);
      return `<button class="mis-struct-tag ${sel ? 'is-selected' : ''}" type="button" data-struct="${s.id}"
        title="${esc(s.value)}" style="color:${SEM[s.sem] || 'var(--comp-tag-fg)'}">
        <i></i>${esc(s.label)}</button>`;
    }).join('');
    $('#structureCount').textContent = state.model.structures.length + ' 项';
  }

  function renderExplorer() {
    renderArtifacts();
    renderSteps();
    renderStructures();
  }

  /* ============ Architecture graph (model-graphviz 视觉契约) ============ */
  const NS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs) {
    const el = document.createElementNS(NS, tag);
    if (attrs) Object.keys(attrs).forEach((k) => el.setAttribute(k, attrs[k]));
    return el;
  }

  function relatedIds(id) {
    const set = new Set([id]);
    state.model.graph.edges.forEach(([a, b]) => {
      if (a === id) set.add(b);
      if (b === id) set.add(a);
    });
    return set;
  }

  function buildGraph() {
    const g = state.model.graph;
    const svg = svgEl('svg', {
      viewBox: `0 0 ${g.width} ${g.height}`,
      preserveAspectRatio: 'xMidYMin meet',
      role: 'img',
      'aria-label': '模型架构图'
    });
    const root = svgEl('g', { id: 'graphRoot' });
    svg.appendChild(root);

    // clusters first (behind)
    g.clusters.forEach((c) => {
      const cg = svgEl('g', { class: 'mis-cluster' });
      cg.appendChild(svgEl('rect', { x: c.x, y: c.y, width: c.w, height: c.h, rx: 16, ry: 16 }));
      const t = svgEl('text', { x: c.x + 12, y: c.y + 18 });
      t.textContent = c.label;
      cg.appendChild(t);
      root.appendChild(cg);
    });

    // edges
    const edgeLayer = svgEl('g', { id: 'edgeLayer' });
    root.appendChild(edgeLayer);
    g.edges.forEach(([a, b]) => {
      const na = nodeOf(a), nb = nodeOf(b);
      if (!na || !nb) return;
      const x1 = na.x + na.w / 2, y1 = na.y + na.h;
      const x2 = nb.x + nb.w / 2, y2 = nb.y;
      const my = (y1 + y2) / 2;
      const path = svgEl('path', {
        class: 'mis-edge',
        'data-from': a,
        'data-to': b,
        d: `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`
      });
      edgeLayer.appendChild(path);
    });

    // nodes
    const nodeLayer = svgEl('g', { id: 'nodeLayer' });
    root.appendChild(nodeLayer);
    g.nodes.forEach((n) => {
      const pill = n.kind === 'module' || n.kind === 'op';
      const r = pill ? n.h / 2 : 6;
      const accent = SEM[n.sem] || 'var(--foreground-muted)';
      const ng = svgEl('g', { class: 'mis-node', 'data-node': n.id, tabindex: '0', role: 'button',
        'aria-label': `${n.label} ${n.op}` });

      ng.appendChild(svgEl('rect', { class: 'mis-node-body', x: n.x, y: n.y, width: n.w, height: n.h, rx: r, ry: r }));
      // 语义色左侧强调条
      ng.appendChild(svgEl('rect', { class: 'mis-node-accent', x: n.x, y: n.y + n.h / 2 - 7, width: 3, height: 14, rx: 1.5, fill: accent }));

      const label = svgEl('text', { class: 'mis-node-label', x: n.x + 12, y: n.y + (n.op ? n.h / 2 - 1 : n.h / 2 + 4) });
      label.textContent = n.label;
      ng.appendChild(label);

      if (n.op) {
        const ty = svgEl('text', { class: 'mis-node-type', x: n.x + 12, y: n.y + n.h / 2 + 10 });
        ty.textContent = n.op;
        ng.appendChild(ty);
      }

      // 覆盖状态环
      const c = covOf(n.id);
      if (c) {
        ng.appendChild(svgEl('circle', {
          class: 'mis-cov-ring', cx: n.x + n.w - 12, cy: n.y + n.h / 2, r: 4.5, stroke: COV[c.cov].color
        }));
        if (c.p) {
          ng.appendChild(svgEl('circle', {
            class: 'mis-prio-dot', cx: n.x + n.w - 12, cy: n.y + n.h / 2, r: 2.4, fill: PRIO_COLOR[c.p]
          }));
        }
      }
      nodeLayer.appendChild(ng);
    });

    return svg;
  }

  function applyGraphHighlight() {
    const sel = state.selectedNode;
    const rel = sel ? relatedIds(sel) : null;
    const filter = state.covFilter;
    $$('.mis-node').forEach((el) => {
      const id = el.dataset.node;
      const c = covOf(id);
      let dim = false;
      if (filter && (!c || c.cov !== filter)) dim = true;
      if (sel && !rel.has(id)) dim = true;
      el.classList.toggle('is-dim', dim);
      el.classList.toggle('is-selected', id === sel);
    });
    $$('.mis-edge').forEach((el) => {
      const on = sel && (el.dataset.from === sel || el.dataset.to === sel);
      el.classList.toggle('is-related', !!on);
      el.style.opacity = sel && !on ? '0.16' : '';
    });
  }

  function applyTransform() {
    const root = $('#graphRoot');
    if (!root) return;
    const v = state.view;
    root.setAttribute('transform', `translate(${v.x},${v.y}) scale(${v.k})`);
  }

  function wireGraphStage(stage) {
    let drag = null;
    stage.addEventListener('pointerdown', (e) => {
      if (e.target.closest('.mis-node')) return;
      drag = { x: e.clientX, y: e.clientY, ox: state.view.x, oy: state.view.y };
      stage.classList.add('is-panning');
      stage.setPointerCapture(e.pointerId);
    });
    stage.addEventListener('pointermove', (e) => {
      if (!drag) return;
      state.view.x = drag.ox + (e.clientX - drag.x);
      state.view.y = drag.oy + (e.clientY - drag.y);
      applyTransform();
    });
    stage.addEventListener('pointerup', (e) => {
      drag = null;
      stage.classList.remove('is-panning');
      try { stage.releasePointerCapture(e.pointerId); } catch (_) {}
    });
    stage.addEventListener('wheel', (e) => {
      e.preventDefault();
      const k = Math.min(2.4, Math.max(0.4, state.view.k * (e.deltaY < 0 ? 1.1 : 0.9)));
      state.view.k = k;
      applyTransform();
    }, { passive: false });
    stage.addEventListener('click', (e) => {
      const n = e.target.closest('.mis-node');
      if (n) selectNode(n.dataset.node, 'graph');
      else selectNode(null, 'graph');
    });
    stage.addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      const n = e.target.closest('.mis-node');
      if (n) { e.preventDefault(); selectNode(n.dataset.node, 'graph'); }
    });
  }

  /* ============ Views ============ */
  function covLegendHTML() {
    return Object.keys(COV).map((k) => {
      const m = COV[k];
      const dim = state.covFilter && state.covFilter !== k;
      return `<button type="button" data-cov-filter="${k}" class="${dim ? 'is-dim' : ''}"
        style="color:${m.color}" title="${esc(m.desc)}"><i></i>${esc(m.short)}</button>`;
    }).join('');
  }

  function viewArchitecture() {
    const m = state.model;
    return `<section class="mis-view is-active" data-view="architecture">
      <div class="mis-graph-toolbar">
        <div class="segmented-control segmented-control-muted" role="group" aria-label="图缩放">
          <button class="btn btn-sm" type="button" data-zoom="out" aria-label="缩小">−</button>
          <button class="btn btn-sm" type="button" data-zoom="fit">适应</button>
          <button class="btn btn-sm" type="button" data-zoom="in" aria-label="放大">+</button>
        </div>
        <span class="mis-state-text">${esc(m.name)} · ${m.layers} 层 · ${esc(m.params)}</span>
        <div class="mis-graph-legend">${covLegendHTML()}</div>
      </div>
      <div class="mis-graph-stage" id="graphStage"></div>
    </section>`;
  }

  function viewCoverage() {
    const t = covTotals();
    const total = state.model.coverage.length;
    const bar = Object.keys(COV).map((k) => t[k] ? `<span style="width:${(t[k] / total * 100).toFixed(2)}%;background:${COV[k].color}"></span>` : '').join('');
    const sum = Object.keys(COV).map((k) => `<button type="button" data-cov-filter="${k}"
      class="${state.covFilter === k ? 'is-active' : ''}" style="color:${COV[k].color}">
      <i></i>${esc(COV[k].label)} <b>${t[k]}</b></button>`).join('');

    const rows = state.model.coverage.map((c) => {
      const n = nodeOf(c.node);
      const m = COV[c.cov];
      const hidden = state.covFilter && state.covFilter !== c.cov;
      return `<tr data-node="${c.node}" class="${hidden ? 'is-hidden' : ''} ${state.selectedNode === c.node ? 'is-selected' : ''}">
        <td class="mono"><b>${esc(n ? n.label : c.node)}</b></td>
        <td class="mono">${esc(c.op)}</td>
        <td><span class="mis-cov-pill" style="color:${m.color}"><i></i>${esc(m.short)}</span></td>
        <td class="mono">${esc(c.impl)}</td>
        <td class="num mono">${c.latency ? c.latency.toFixed(2) : '—'}</td>
        <td><div class="mis-share-bar"><span style="width:${Math.min(100, c.share * 4).toFixed(1)}%;background:${m.color}"></span></div></td>
        <td>${c.p ? `<span class="mis-prio" style="background:${PRIO_COLOR[c.p]}">${c.p}</span>` : ''}</td>
      </tr>`;
    }).join('');

    const fb = fallbackCost();
    const blocking = blockingCount();
    return `<section class="mis-view" data-view="coverage">
      <div class="mis-view-heading">
        <div><span class="mis-eyebrow">OPERATOR COVERAGE</span>
          <h1>逐模型节点的实现来源与回落代价</h1>
          <p>覆盖状态按 PyPTO 原生 / 融合 / 库调用 / 回落 / 不支持五档给出，回落项直接标注端到端耗时占比与优先级。</p></div>
        <div class="mis-heading-aside">
          <span class="mis-state-text ${blocking ? 'danger' : 'success'}">${blocking ? blocking + ' 项阻塞整网' : '整网可闭环'}</span>
          <span class="mis-state-text">${esc(state.model.name)} · ${state.phase === 'decode' ? 'Decode' : 'Prefill'}</span>
        </div>
      </div>
      <div class="mis-metrics">
        <div><span>节点总数</span><strong>${total}</strong><small>traced nodes</small></div>
        <div><span>原生 + 融合</span><strong class="success">${((t.native + t.fused) / total * 100).toFixed(0)}%</strong><small>${t.native + t.fused} 个节点</small></div>
        <div><span>回落耗时占比</span><strong class="warning">${fb.toFixed(1)}%</strong><small>${t.fallback} 个 fallback</small></div>
        <div><span>不支持</span><strong class="${blocking ? 'danger' : ''}">${t.unsupported}</strong><small>${blocking ? '需平台或降级方案' : '无'}</small></div>
      </div>
      <div class="mis-cov-track">${bar}</div>
      <div class="mis-cov-summary">${sum}</div>
      <div class="mis-table-wrap" style="margin-top:var(--space-4)">
        <table class="mis-table">
          <thead><tr><th>模型节点</th><th>算子</th><th>覆盖状态</th><th>实现来源</th><th>耗时 ms</th><th>占比</th><th>优先级</th></tr></thead>
          <tbody id="covBody">${rows}</tbody>
        </table>
      </div>
    </section>`;
  }

  const PREC_GLYPH = { ok: '✓', calib: 'C', no: '✕', na: '–' };

  function viewPrecision() {
    const p = state.model.precision;
    const head = PREC.map((x) => `<th>${x}</th>`).join('');
    const rows = p.modules.map((m) => {
      const cells = PREC.map((x) => {
        const s = m.support[x] || 'na';
        return `<td><span class="mis-prec-cell ${s}" title="${esc(x)} · ${s}">${PREC_GLYPH[s]}</span></td>`;
      }).join('');
      const sel = state.selectedNode === m.id;
      return `<tr data-node="${m.id}" class="${sel ? 'is-selected' : ''}">
        <td>${esc(m.name)}</td>${cells}
        <td><span class="mis-prec-current">${esc(m.current)}</span></td></tr>`;
    }).join('');

    const bounds = p.boundaries.map((b) => `<div class="mis-boundary ${b.level}">
      <b>${esc(b.from)} → ${esc(b.to)}</b>
      <span>${esc(b.at)}</span>
      <p>${esc(b.reason)}</p></div>`).join('');

    const q = p.quant;
    const metrics = q.metrics.map((m) => `<div class="mis-quant-row">
      <span>${esc(m.name)}</span><code>${esc(m.before)}</code><code>→ ${esc(m.after)}</code>
      <b class="${m.good ? 'good' : 'bad'}">${esc(m.delta)}</b></div>`).join('');
    const risks = q.risks.map((r) => `<li>${esc(r)}</li>`).join('');

    return `<section class="mis-view" data-view="precision">
      <div class="mis-view-heading">
        <div><span class="mis-eyebrow">PRECISION &amp; QUANTIZATION</span>
          <h1>支持精度、混合精度边界与校准要求</h1>
          <p>每个模块给出可用精度档位，标注需校准（C）与禁用（✕）的组合，并显式列出跨模块的精度转换边界。</p></div>
        <div class="mis-heading-aside">
          <span class="mis-state-text">基线 ${esc(p.baseline)}</span>
          <span class="mis-state-text ${q.calib.status === 'ready' ? 'success' : 'warning'}">校准集${q.calib.status === 'ready' ? '就绪' : '待准备'}</span>
        </div>
      </div>

      <div class="mis-grid-2">
        <section class="panel-shell panel-shell-quiet">
          <header class="panel-shell-header"><div><h2 class="panel-shell-title">精度支持矩阵</h2>
            <span class="panel-shell-meta">✓ 直接支持 · C 需校准 · ✕ 禁用</span></div></header>
          <div class="panel-shell-body">
            <table class="mis-prec-matrix">
              <thead><tr><th>模块</th>${head}<th>当前</th></tr></thead>
              <tbody id="precBody">${rows}</tbody>
            </table>
          </div>
        </section>

        <section class="panel-shell panel-shell-quiet">
          <header class="panel-shell-header"><div><h2 class="panel-shell-title">混合精度边界</h2>
            <span class="panel-shell-meta">${p.boundaries.length} 处转换点</span></div></header>
          <div class="panel-shell-body"><div class="mis-boundary-list">${bounds}</div></div>
        </section>
      </div>

      <div class="mis-grid-2">
        <section class="panel-shell panel-shell-quiet">
          <header class="panel-shell-header"><div><h2 class="panel-shell-title">量化参数</h2>
            <span class="panel-shell-meta">${esc(q.scheme)}</span></div></header>
          <div class="panel-shell-body">
            <div class="mis-kv" style="margin-bottom:var(--space-3)">
              <div><span>方案</span><b class="strong">${esc(q.scheme)}</b></div>
              <div><span>校准集</span><b>${esc(q.calib.dataset)}</b></div>
              <div><span>样本数</span><b>${q.calib.samples} × seqlen ${q.calib.seqlen}</b></div>
            </div>
            <div class="mis-quant-metrics">${metrics}</div>
          </div>
        </section>

        <section class="panel-shell panel-shell-quiet">
          <header class="panel-shell-header"><div><h2 class="panel-shell-title">校准风险</h2>
            <span class="panel-shell-meta">需在 Recipe 中显式处理</span></div></header>
          <div class="panel-shell-body"><ul class="mis-risk-list">${risks}</ul></div>
        </section>
      </div>
    </section>`;
  }

  function viewRecipe() {
    const cards = state.model.recipes.map((r) => {
      const on = r.selected;
      const sel = state.selectedRecipe === r.id;
      return `<button class="mis-recipe-card ${sel ? 'is-selected' : ''}" type="button" data-recipe="${r.id}">
        <div class="mis-recipe-head">
          <i style="background:${SEM[r.sem] || 'var(--foreground-muted)'}"></i>
          <b>${esc(r.name)}</b>
          <span class="mis-recipe-toggle" data-recipe-toggle="${r.id}">${on ? '已启用' : '未启用'}</span>
        </div>
        <p>${esc(r.summary)}</p>
        <div class="mis-recipe-baseline"><strong>${esc(r.baseline.value)}</strong><small>${esc(r.baseline.metric)} · ${esc(r.baseline.at)}</small></div>
        <div class="mis-recipe-meta">
          <div><span>适用模型</span><b>${esc(r.applies.models)}</b></div>
          <div><span>形状</span><b>${esc(r.applies.shapes)}</b></div>
          <div><span>平台</span><b>${esc(r.applies.platform)}</b></div>
          <div><span>精度阈值</span><b>${esc(r.precision.threshold)}</b></div>
        </div>
      </button>`;
    }).join('');

    const enabled = state.model.recipes.filter((r) => r.selected).length;
    const covered = new Set();
    state.model.recipes.filter((r) => r.selected).forEach((r) => r.nodes.forEach((n) => covered.add(n)));

    return `<section class="mis-view" data-view="recipe">
      <div class="mis-view-heading">
        <div><span class="mis-eyebrow">MODEL RECIPE</span>
          <h1>可组合的推理 Recipe</h1>
          <p>按 prefill / decode / paged attention / RMSNorm+RoPE / MoE expert / lm_head 组合，每个 Recipe 携带参考实现、精度阈值、性能基线、已知限制与调优点。</p></div>
        <div class="mis-heading-aside">
          <span class="mis-state-text success">已启用 ${enabled} / ${state.model.recipes.length}</span>
          <span class="mis-state-text">覆盖 ${covered.size} 个节点</span>
        </div>
      </div>
      <div class="mis-recipe-grid">${cards}</div>
    </section>`;
  }

  function viewSkeleton() {
    const pats = state.model.patterns;
    const active = pats.find((p) => p.id === state.selectedPattern) || pats[0];

    const list = pats.map((p) => {
      const r = state.model.recipes.find((x) => x.id === p.recipe);
      return `<button class="mis-pattern-row ${p.id === state.selectedPattern ? 'is-selected' : ''}" type="button" data-pattern="${p.id}">
        <div><b>${esc(p.name)}</b><small>${esc(p.source)}</small></div>
        <small>${p.matched.length} 节点命中</small>
        <span class="mis-cov-pill" style="color:${r ? SEM[r.sem] : 'var(--foreground-muted)'}"><i></i>${(p.confidence * 100).toFixed(0)}%</span>
      </button>`;
    }).join('');

    const lines = active.code.map((ln) => {
      const n = ln.map ? nodeOf(ln.map) : null;
      const c = ln.map ? covOf(ln.map) : null;
      const sel = ln.map && ln.map === state.selectedNode;
      const accent = c ? COV[c.cov].color : 'transparent';
      const title = n ? `映射到模型节点 ${n.label} (${n.op})` : '';
      return `<div class="mis-code-line kind-${ln.kind} ${sel ? 'is-selected' : ''}"
        ${ln.map ? `data-map="${ln.map}"` : ''} title="${esc(title)}">
        <em>${ln.l}</em><i style="background:${ln.map ? accent : 'transparent'}"></i><code>${esc(ln.t || ' ')}</code></div>`;
    }).join('');

    const gaps = active.gaps.map((g) => `<li>${esc(g)}</li>`).join('');

    return `<section class="mis-view" data-view="skeleton">
      <div class="mis-view-heading">
        <div><span class="mis-eyebrow">SUBGRAPH PATTERN MATCHING</span>
          <h1>生成可编辑实现骨架，保留节点到源码的映射</h1>
          <p>子图匹配命中后生成 PyPTO 实现骨架；每一行都保留到模型节点的双向映射，点击代码行即可定位节点，点击节点即可定位代码行。</p></div>
        <div class="mis-heading-aside">
          <span class="mis-state-text">${pats.length} 个 pattern 命中</span>
          <span class="mis-state-text warning">${active.gaps.length} 处待补齐</span>
        </div>
      </div>

      <div class="mis-pattern-list">${list}</div>

      <div class="mis-skeleton-head">
        <span class="mis-eyebrow">${esc(active.target)}</span>
        <span class="mis-map-note">左侧色条 = 覆盖状态 · 点击行联动节点</span>
      </div>
      <pre class="mis-code" id="codeBlock">${lines}</pre>

      <section class="panel-shell panel-shell-quiet" style="margin-top:var(--space-4)">
        <header class="panel-shell-header"><div><h2 class="panel-shell-title">待补齐项</h2>
          <span class="panel-shell-meta">来自覆盖报告的 fallback / unsupported</span></div>
          <span class="mis-state-text">源：${esc(active.source)}</span></header>
        <div class="panel-shell-body"><ul class="mis-risk-list">${gaps}</ul></div>
      </section>
    </section>`;
  }

  /* ============ Inspector ============ */
  function inspectorEmpty() {
    return `<div class="mis-insp-empty">
      <span>${state.imported ? '选择模型节点、覆盖行或代码行查看证据' : '导入模型后可查看逐节点证据'}</span>
    </div>`;
  }

  function inspectorForNode(id) {
    const n = nodeOf(id);
    const c = covOf(id);
    const struct = state.model.structures.find((s) => s.id === id);
    const prec = state.model.precision.modules.find((m) => m.id === id);
    const recipes = state.model.recipes.filter((r) => r.nodes.indexOf(id) >= 0);
    const pats = state.model.patterns.filter((p) => p.matched.indexOf(id) >= 0 || p.code.some((l) => l.map === id));
    if (!n && !struct) return inspectorEmpty();

    const title = n ? n.label : struct.label;
    const sub = n ? `${n.op} · ${n.kind}` : struct.value;
    let html = `<div class="mis-insp-section"><div class="mis-insp-title">
      <b>${esc(title)}</b><span>${esc(sub)}</span></div>`;
    if (c) {
      const m = COV[c.cov];
      html += `<div style="margin-top:var(--space-2)"><span class="mis-cov-pill" style="color:${m.color}"><i></i>${esc(m.label)}</span></div>`;
    }
    html += `</div>`;

    if (n && (n.shape || n.dtype)) {
      html += `<div class="mis-insp-section"><h3>TENSOR</h3><div class="mis-kv">
        ${n.shape ? `<div><span>shape</span><b class="strong">${esc(n.shape)}</b></div>` : ''}
        ${n.dtype ? `<div><span>dtype</span><b>${esc(n.dtype)}</b></div>` : ''}
        <div><span>语义</span><b>${esc(n.sem)}</b></div></div></div>`;
    }

    if (struct) {
      html += `<div class="mis-insp-section"><h3>DETECTOR 依据</h3><div class="mis-kv">
        <div><span>置信度</span><b class="strong">${struct.confidence === 'high' ? '高' : '中'}</b></div>
        <div><span>参数</span><b>${esc(struct.value)}</b></div></div>
        <p style="margin:var(--space-2) 0 0;color:var(--foreground-secondary);font-size:var(--font-size-body-sm)">${esc(struct.evidence)}</p></div>`;
    }

    if (c) {
      const m = COV[c.cov];
      const sev = c.p || (c.cov === 'unsupported' ? 'P0' : 'INFO');
      const ruleId = 'MIS-' + c.cov.slice(0, 3).toUpperCase() + '-' + String(state.model.coverage.indexOf(c) + 1).padStart(3, '0');
      html += `<div class="mis-insp-section"><h3>诊断</h3><div class="mis-diag">
        <div class="mis-diag-head">
          ${c.p ? `<span class="mis-prio" style="background:${PRIO_COLOR[c.p]}">${c.p}</span>` : `<span class="mis-cov-pill" style="color:${m.color}"><i></i>${sev}</span>`}
          <code>${ruleId}</code></div>
        <div class="mis-diag-row"><span>结论</span><p>${esc(m.label)}：${esc(m.desc)}</p></div>
        <div class="mis-diag-row"><span>位置</span><p>${esc(c.node)} · ${esc(c.op)}</p></div>
        <div class="mis-diag-row"><span>影响</span><p>${c.share ? c.share.toFixed(1) + '% 端到端耗时（' + c.latency.toFixed(2) + ' ms）' : (c.cov === 'unsupported' ? '阻塞整网推理' : '无独立开销')}</p></div>
        <div class="mis-diag-row"><span>依据</span><p>${esc(c.impl)}</p></div>
        ${c.note ? `<div class="mis-diag-row"><span>建议</span><p>${esc(c.note)}</p></div>` : ''}
      </div></div>`;
    }

    if (prec) {
      const cells = PREC.map((x) => {
        const s = prec.support[x] || 'na';
        return `<span class="mis-prec-cell ${s}" title="${x}">${PREC_GLYPH[s]}</span>`;
      }).join(' ');
      html += `<div class="mis-insp-section"><h3>精度画像</h3>
        <div style="display:flex;gap:var(--space-2);align-items:center">${cells}</div>
        <div class="mis-kv" style="margin-top:var(--space-2)">
          <div><span>当前</span><b class="strong">${esc(prec.current)}</b></div></div>
        <p style="margin:var(--space-2) 0 0;color:var(--foreground-secondary);font-size:var(--font-size-body-sm)">${esc(prec.note)}</p></div>`;
    }

    if (recipes.length) {
      html += `<div class="mis-insp-section"><h3>关联 RECIPE</h3><div class="mis-structure-tags">` +
        recipes.map((r) => `<button class="mis-struct-tag ${r.selected ? 'is-selected' : ''}" type="button"
          data-goto-recipe="${r.id}" style="color:${SEM[r.sem]}"><i></i>${esc(r.name)}</button>`).join('') +
        `</div></div>`;
    }

    if (pats.length) {
      html += `<div class="mis-insp-section"><h3>源码映射</h3>` + pats.map((p) => {
        const line = p.code.find((l) => l.map === id);
        return `<button class="mis-source-link" type="button" data-goto-code="${p.id}" data-goto-node="${id}">
          <b>${esc(p.source)}</b>
          <span>→ ${esc(p.target)}${line ? ' : ' + line.l : ''}</span></button>`;
      }).join('') + `</div>`;
    }

    return html;
  }

  function renderInspector() {
    const box = $('#inspector');
    if (state.selectedNode) {
      box.innerHTML = inspectorForNode(state.selectedNode);
      $('#inspectorId').textContent = state.selectedNode;
      return;
    }
    if (state.tab === 'recipe' && state.selectedRecipe) {
      box.innerHTML = inspectorForRecipe(state.selectedRecipe);
      $('#inspectorId').textContent = state.selectedRecipe;
      return;
    }
    box.innerHTML = inspectorEmpty();
    $('#inspectorId').textContent = '—';
  }

  function inspectorForRecipe(id) {
    const r = state.model.recipes.find((x) => x.id === id);
    if (!r) return inspectorEmpty();
    const li = (a) => a.map((x) => `<li>${esc(x)}</li>`).join('');
    return `<div class="mis-insp-section"><div class="mis-insp-title">
        <b>${esc(r.name)}</b><span>${esc(r.stage)} · ${r.selected ? '已启用' : '未启用'}</span></div>
        <p style="margin:var(--space-2) 0 0;color:var(--foreground-secondary);font-size:var(--font-size-body-sm)">${esc(r.summary)}</p></div>
      <div class="mis-insp-section"><h3>适用范围</h3><div class="mis-kv">
        <div><span>模型</span><b>${esc(r.applies.models)}</b></div>
        <div><span>形状</span><b>${esc(r.applies.shapes)}</b></div>
        <div><span>平台</span><b>${esc(r.applies.platform)}</b></div></div></div>
      <div class="mis-insp-section"><h3>精度阈值</h3><div class="mis-kv">
        <div><span>阈值</span><b class="strong">${esc(r.precision.threshold)}</b></div>
        <div><span>累加</span><b>${esc(r.precision.accum)}</b></div></div></div>
      <div class="mis-insp-section"><h3>性能基线</h3><div class="mis-kv">
        <div><span>${esc(r.baseline.metric)}</span><b class="strong">${esc(r.baseline.value)}</b></div>
        <div><span>条件</span><b>${esc(r.baseline.at)}</b></div></div></div>
      <div class="mis-insp-section"><h3>参考实现</h3>${r.refs.map((f) =>
        `<button class="mis-source-link" type="button"><b>${esc(f)}</b><span>参考实现</span></button>`).join('')}</div>
      <div class="mis-insp-section"><h3>已知限制</h3><ul class="mis-risk-list">${li(r.limits)}</ul></div>
      <div class="mis-insp-section"><h3>调优点</h3><ul class="mis-risk-list">${li(r.tuning)}</ul></div>
      <div class="mis-insp-section"><h3>覆盖节点</h3><div class="mis-structure-tags">${
        r.nodes.map((nid) => { const n = nodeOf(nid); const c = covOf(nid);
          return `<button class="mis-struct-tag" type="button" data-struct="${nid}"
            style="color:${c ? COV[c.cov].color : 'var(--comp-tag-fg)'}"><i></i>${esc(n ? n.label : nid)}</button>`; }).join('')
      }</div></div>`;
  }

  /* ============ Render orchestration ============ */
  const TAB_META = {
    architecture: { title: '模型结构识别', builder: viewArchitecture },
    coverage: { title: '算子覆盖报告', builder: viewCoverage },
    precision: { title: '精度与量化画像', builder: viewPrecision },
    recipe: { title: 'Model Recipe 组合', builder: viewRecipe },
    skeleton: { title: '实现骨架与源码映射', builder: viewSkeleton }
  };

  function paneActionsHTML() {
    if (!state.imported) return '';
    if (state.tab === 'coverage') {
      return `<button class="btn btn-sm" type="button" id="exportCoverage">导出覆盖报告</button>
        <button class="btn btn-solid btn-sm" type="button" id="gotoRecipe">按报告选择 Recipe</button>`;
    }
    if (state.tab === 'recipe') {
      return `<button class="btn btn-sm" type="button" id="resetRecipes">恢复推荐组合</button>
        <button class="btn btn-solid btn-sm" type="button" id="genSkeleton">生成实现骨架</button>`;
    }
    if (state.tab === 'skeleton') {
      return `<button class="btn btn-sm" type="button" id="copyCode">复制骨架</button>
        <button class="btn btn-solid btn-sm" type="button" id="openWorkspace">写入工作区</button>`;
    }
    if (state.tab === 'precision') {
      return `<button class="btn btn-sm" type="button" id="exportPrecision">导出精度画像</button>`;
    }
    return `<button class="btn btn-sm" type="button" id="focusFallback">定位回落节点</button>`;
  }

  function renderMain() {
    const meta = TAB_META[state.tab];
    $('#mainBody').innerHTML = meta.builder();
    $('#viewTitle').textContent = meta.title;
    $('#viewMeta').textContent = state.imported
      ? `${state.model.name} · ${state.hardware.toUpperCase()} · ${state.phase}`
      : '未导入模型';
    $('#paneActions').innerHTML = paneActionsHTML();

    // activate current view section
    const sec = $(`.mis-view[data-view="${state.tab}"]`);
    if (sec) sec.classList.add('is-active');

    if (state.tab === 'architecture') {
      const stage = $('#graphStage');
      stage.appendChild(buildGraph());
      const hint = document.createElement('div');
      hint.className = 'mis-graph-hint';
      hint.textContent = '拖拽平移 · 滚轮缩放 · 点击节点查看证据';
      stage.appendChild(hint);
      wireGraphStage(stage);
      applyTransform();
      applyGraphHighlight();
    }
    renderInspector();
  }

  function syncTabs() {
    $$('.tab-control-item').forEach((b) => {
      const on = b.dataset.tab === state.tab;
      b.classList.toggle('is-selected', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
  }

  function setTab(tab) {
    if (!TAB_META[tab]) return;
    state.tab = tab;
    syncTabs();
    renderMain();
  }

  function selectNode(id, origin) {
    state.selectedNode = id;
    if (state.tab === 'architecture') {
      applyGraphHighlight();
    } else {
      // update row/line selection in-place without full rebuild
      $$('[data-node]').forEach((el) => el.classList.toggle('is-selected', el.dataset.node === id));
      $$('.mis-code-line').forEach((el) => el.classList.toggle('is-selected', el.dataset.map === id));
    }
    if (origin !== 'explorer') renderStructures();
    renderInspector();
  }

  function runImport() {
    if (state.imported) { toast('模型已导入，可点击「重置 Demo」重新开始'); return; }
    status('解析中', '正在读取 config / 权重索引 / tokenizer …', 'busy');
    $('#importBtn').disabled = true;
    let i = 0;
    const tick = () => {
      i += 1;
      if (i <= 2) {
        state.imported = i >= 1;
        state.step = i;
        renderExplorer();
        status(i === 1 ? '工件已加载' : '结构已识别',
          i === 1 ? '5 个工件解析完成，检测到 trust_remote_code 自定义代码' : `Architecture Detector 命中 ${state.model.structures.length} 类关键结构`,
          i === 1 ? 'busy' : 'ready');
        renderMain();
        setTimeout(tick, 520);
        return;
      }
      state.step = 2;
      $('#importBtn').disabled = false;
      $('#importBtn').textContent = '重新解析模型';
      toast('模型解析完成，已识别 ' + state.model.structures.length + ' 类关键结构');
      status('已就绪', `${state.model.name} 结构识别完成，可查看算子覆盖报告`, 'ready');
      renderExplorer();
    };
    setTimeout(tick, 420);
  }

  function advanceTo(stepIdx) {
    state.step = Math.max(state.step, stepIdx);
    renderSteps();
  }

  function resetDemo() {
    state.imported = false;
    state.step = 0;
    state.selectedNode = null;
    state.covFilter = null;
    state.tab = 'architecture';
    state.view = { x: 0, y: 0, k: 1 };
    state.model.recipes.forEach((r) => { r.selected = r.id !== 'r-lmhead'; });
    $('#importBtn').textContent = '导入并解析模型';
    $('#importBtn').disabled = false;
    renderExplorer();
    syncTabs();
    renderMain();
    status('等待导入', '选择模型仓库后点击「导入并解析模型」', null);
    toast('已重置');
  }

  /* ============ Events ============ */
  document.addEventListener('click', (e) => {
    const t = e.target;

    const tab = t.closest('.tab-control-item');
    if (tab) { setTab(tab.dataset.tab); return; }

    const rail = t.closest('[data-open-tab]');
    if (rail) { setTab(rail.dataset.openTab); return; }

    if (t.closest('#importBtn')) { runImport(); return; }
    if (t.closest('#resetDemo')) { resetDemo(); return; }
    if (t.closest('#commandTrigger')) { toast('命令面板在完整产品中打开：节点 / 算子 / Recipe 统一搜索'); return; }

    const step = t.closest('[data-step]');
    if (step && !step.disabled) {
      const i = Number(step.dataset.step);
      if (i === 0 && !state.imported) { runImport(); return; }
      state.step = Math.max(state.step, i);
      renderSteps();
      setTab(STEPS[i].view);
      return;
    }

    const filter = t.closest('[data-cov-filter]');
    if (filter) {
      const k = filter.dataset.covFilter;
      state.covFilter = state.covFilter === k ? null : k;
      state.selectedNode = null;
      renderMain();
      toast(state.covFilter ? '仅显示 ' + COV[k].label : '已清除筛选');
      return;
    }

    const zoom = t.closest('[data-zoom]');
    if (zoom) {
      const mode = zoom.dataset.zoom;
      if (mode === 'fit') state.view = { x: 0, y: 0, k: 1 };
      else state.view.k = Math.min(2.4, Math.max(0.4, state.view.k * (mode === 'in' ? 1.2 : 0.83)));
      applyTransform();
      return;
    }

    const toggle = t.closest('[data-recipe-toggle]');
    if (toggle) {
      e.stopPropagation();
      const r = state.model.recipes.find((x) => x.id === toggle.dataset.recipeToggle);
      if (r) {
        r.selected = !r.selected;
        state.selectedRecipe = r.id;
        renderMain();
        toast(r.name + (r.selected ? ' 已启用' : ' 已停用'));
      }
      return;
    }

    const card = t.closest('[data-recipe]');
    if (card) {
      state.selectedRecipe = card.dataset.recipe;
      state.selectedNode = null;
      $$('.mis-recipe-card').forEach((el) => el.classList.toggle('is-selected', el.dataset.recipe === state.selectedRecipe));
      renderInspector();
      return;
    }

    const gotoRecipeTag = t.closest('[data-goto-recipe]');
    if (gotoRecipeTag) {
      state.selectedRecipe = gotoRecipeTag.dataset.gotoRecipe;
      state.selectedNode = null;
      setTab('recipe');
      return;
    }

    const pat = t.closest('[data-pattern]');
    if (pat) { state.selectedPattern = pat.dataset.pattern; renderMain(); return; }

    const gotoCode = t.closest('[data-goto-code]');
    if (gotoCode) {
      state.selectedPattern = gotoCode.dataset.gotoCode;
      const nid = gotoCode.dataset.gotoNode;
      advanceTo(4);
      setTab('skeleton');
      selectNode(nid, 'inspector');
      const line = $(`.mis-code-line[data-map="${nid}"]`);
      if (line) line.scrollIntoView({ block: 'center', behavior: 'smooth' });
      return;
    }

    const line = t.closest('.mis-code-line[data-map]');
    if (line) { selectNode(line.dataset.map, 'code'); return; }

    const struct = t.closest('[data-struct]');
    if (struct) {
      const id = struct.dataset.struct;
      selectNode(state.selectedNode === id ? null : id, 'explorer');
      return;
    }

    const row = t.closest('tr[data-node]');
    if (row) { selectNode(row.dataset.node === state.selectedNode ? null : row.dataset.node, 'table'); return; }

    if (t.closest('#gotoRecipe')) { advanceTo(3); setTab('recipe'); toast('已按覆盖报告预选 Recipe 组合'); return; }
    if (t.closest('#genSkeleton')) {
      advanceTo(5);
      setTab('skeleton');
      toast('已按启用的 Recipe 生成实现骨架');
      status('骨架已生成', '子图 pattern 匹配完成，节点到源码映射已保留', 'ready');
      return;
    }
    if (t.closest('#resetRecipes')) {
      state.model.recipes.forEach((r) => { r.selected = r.id !== 'r-lmhead'; });
      renderMain();
      toast('已恢复推荐组合');
      return;
    }
    if (t.closest('#copyCode')) {
      const p = state.model.patterns.find((x) => x.id === state.selectedPattern);
      const text = p ? p.code.map((l) => l.t).join('\n') : '';
      if (navigator.clipboard) navigator.clipboard.writeText(text).catch(() => {});
      toast('骨架已复制到剪贴板');
      return;
    }
    if (t.closest('#openWorkspace')) {
      const p = state.model.patterns.find((x) => x.id === state.selectedPattern);
      toast('已写入 ' + (p ? p.target : 'workspace'));
      return;
    }
    if (t.closest('#exportCoverage')) { toast('覆盖报告已导出为 coverage_report.json'); return; }
    if (t.closest('#exportPrecision')) { toast('精度画像已导出为 precision_profile.json'); return; }
    if (t.closest('#focusFallback')) {
      state.covFilter = 'fallback';
      setTab('coverage');
      toast('已筛选出 fallback 节点');
      return;
    }
  });

  document.addEventListener('change', (e) => {
    const id = e.target.id;
    if (id === 'hardwareSelect') {
      state.hardware = e.target.value;
      const map = { a3: 'A3 / 910C · CANN 9.0.0', a2: 'A2 / 910B · CANN 8.2.RC1', a5: 'A5 / 950B · CANN 9.1.0' };
      $('#runtimeLabel').textContent = map[state.hardware];
      renderMain();
      toast('目标硬件切换为 ' + state.hardware.toUpperCase());
    } else if (id === 'phaseSelect') {
      state.phase = e.target.value;
      renderMain();
      toast('推理阶段切换为 ' + (state.phase === 'decode' ? 'Decode' : 'Prefill'));
    } else if (id === 'modelSelect') {
      if (e.target.value !== 'qwen3-moe') {
        toast('Demo 仅内置 Qwen3-30B-A3B 完整数据，已回退');
        e.target.value = 'qwen3-moe';
      }
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.selectedNode) { selectNode(null, 'key'); }
  });

  /* ============ Boot ============ */
  state.model.recipes.forEach((r) => { r.selected = r.id !== 'r-lmhead'; });
  renderExplorer();
  syncTabs();
  renderMain();
  status('等待导入', '选择模型仓库后点击「导入并解析模型」', null);

  window.__MIS = { state, setTab, selectNode, renderMain };
})();
