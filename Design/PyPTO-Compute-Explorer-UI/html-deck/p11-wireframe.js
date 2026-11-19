/* P11 variant B — wireframe Venn, rebuilt from the design reference
   (“算子理解8.png”): thin light strokes that disappear where a ring passes
   behind another, translucent lens overlaps, chip badges at the intersections,
   oversized watermark words.
   Variant A (p11-native.js) keeps its own stage; this module only builds a
   sibling pane and fronts the deck-facing story API. */
(() => {
  const NS = 'http://www.w3.org/2000/svg';
  const root = document.querySelector('[data-p11]');
  if (!root || !root.ptoP11Story) return;
  const paneB = root.querySelector('[data-p11-pane="b"]');
  if (!paneB) return;
  const paneA = root.querySelector('[data-p11-pane="a"]');
  const storyA = root.ptoP11Story;

  /* ---------- design tokens (sampled from the reference) ---------- */
  const VIEW = { w: 1366, h: 623 };
  const FG = '#F8FAFC';
  const FG_SUB = '#9CA3AF';
  const FG_WEAK = '#7C8798';
  const WATERMARK = 'rgba(255,255,255,0.11)';
  const LEADER = 'rgba(148,163,184,0.55)';
  const LENS = {
    computeData: 'rgba(74,222,128,0.055)',
    computeTime: 'rgba(99,132,255,0.050)',
    dataTime: 'rgba(168,85,247,0.065)',
    core: 'rgba(255,255,255,0.19)',
  };
  const CHIP = { compute: '#9333EA', data: '#16A34A', time: '#2563EB' };
  const ACCENT = { graph: '#9BA8FF', assignment: '#6EE7A0', pipeline: '#FBBF24' };

  /* ---------- Venn geometry ----------
     Measured on the reference at native size (1920 px wide): ring radius 244,
     centres 336 apart horizontally and 306 apart vertically. The stage is the
     content box of the slide, so the composition is fitted to its height:
     623 / 598 → R 189, half-offset 130, row gap 237. Everything else is placed
     with the same 0.7746 factor, which keeps the reference proportions. */
  const R = 189;
  const CX = 683;
  const CY_TOP = 193;
  const CY_LOW = 430;
  const OFF = 130;

  /* A ring is drawn in three tiers, exactly like the reference: solid where it
     is in front, a whisper where it slides under one other ring, and a slightly
     stronger edge where it forms the triple-intersection shield. */
  /* The reference draws two different kinds of line: the ring contours sit at
     30% white, while every lens contour is a near-solid white hairline. Where a
     lens contour passes under the third ring it drops to ~19% (the shield edge). */
  const ARC_TIERS = [
    { opacity: 0.34, width: 2.0 },   // exposed ring contour
    { opacity: 0.92, width: 2.15 },  // lens contour (inside one other ring)
    { opacity: 0.19, width: 1.8 },   // lens contour under the third ring
  ];
  const SEGMENTS = 240;

  const TIERS = [
    { code: '01', name: 'Device', desc: '设备范围 · 跨核调度与显存拓扑' },
    { code: '02', name: 'AI Core', desc: '核内执行 · Cube/Vector 流水并行' },
    { code: '03', name: 'Iteration', desc: '单次迭代 · Tile 分块与指令节拍' },
  ];

  /* ---------- tiny svg factory ---------- */
  const el = (tag, attrs = {}, parent) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) node.setAttribute(k, v);
    parent?.appendChild(node);
    return node;
  };
  const group = (attrs, parent) => el('g', attrs, parent);
  const text = (parent, attrs, content) => {
    const node = el('text', attrs, parent);
    node.textContent = content;
    return node;
  };

  /* ---------- stage ---------- */
  const svg = el('svg', {
    id: 'p11b-stage-svg',
    viewBox: `0 0 ${VIEW.w} ${VIEW.h}`,
    preserveAspectRatio: 'xMidYMid meet',
    role: 'img',
    'aria-label': '三种计算视角的线框表达及其执行层级',
  }, paneB);

  const defs = el('defs', {}, svg);

  const ICONS = [
    ['graph', '<rect x="1" y="1" width="48" height="48" rx="4"/><path d="M10 15h10m-5-5v10m13-5h10M11 29l7 7m0-7-7 7m17-3h10m-5-5v.2m0 9.6v.2"/>'],
    ['assignment', '<rect x="1" y="1" width="28" height="14" rx="2"/><rect x="35" y="1" width="14" height="14" rx="2"/><rect x="1" y="21" width="11" height="28" rx="2"/><rect x="18" y="21" width="11" height="28" rx="2"/><rect x="35" y="21" width="14" height="11" rx="2"/><rect x="35" y="38" width="14" height="11" rx="2"/>'],
    ['pipeline', '<path d="M3 5h9m10 0h16a10 10 0 0 1 0 20H10a10 10 0 0 0 0 20h16m10 0h9"/><circle cx="17" cy="5" r="5"/><circle cx="31" cy="45" r="5"/>'],
  ];
  for (const [name, body] of ICONS) {
    const symbol = el('symbol', {
      id: `p11b-icon-${name}`, viewBox: '0 0 50 50', overflow: 'visible',
      fill: 'none', 'stroke-width': '3.2', 'stroke-linecap': 'round', 'stroke-linejoin': 'round',
    }, defs);
    symbol.innerHTML = body;
  }

  const lensBlur = el('filter', { id: 'p11b-lens-blur', x: '-14%', y: '-14%', width: '128%', height: '128%' }, defs);
  el('feGaussianBlur', { stdDeviation: '2.5' }, lensBlur);
  const coreBlur = el('filter', { id: 'p11b-core-blur', x: '-20%', y: '-20%', width: '140%', height: '140%' }, defs);
  el('feGaussianBlur', { stdDeviation: '0.4' }, coreBlur);
  const glow = el('filter', { id: 'p11b-glow', x: '-70%', y: '-70%', width: '240%', height: '240%' }, defs);
  el('feGaussianBlur', { stdDeviation: '2.2', result: 'blur' }, glow);
  const merge = el('feMerge', {}, glow);
  el('feMergeNode', { in: 'blur' }, merge);
  el('feMergeNode', { in: 'SourceGraphic' }, merge);

  const clip = {
    graph: el('clipPath', { id: 'p11b-clip-graph' }, defs),
    assign: el('clipPath', { id: 'p11b-clip-assign' }, defs),
    pipe: el('clipPath', { id: 'p11b-clip-pipe' }, defs),
  };
  el('circle', { cx: CX, cy: CY_TOP, r: R }, clip.graph);
  el('circle', { cx: CX - OFF, cy: CY_LOW, r: R }, clip.assign);
  el('circle', { cx: CX + OFF, cy: CY_LOW, r: R }, clip.pipe);

  /* ---------- layers (back → front) ---------- */
  const layerAxis = group({ id: 'p11b-axis-layer', opacity: 0 }, svg);
  const layerArrows = group({ id: 'p11b-arrows-layer', opacity: 0 }, svg);
  const layerFills = group({ id: 'p11b-lens-fills', opacity: 0 }, svg);
  const layerInst = group({ id: 'p11b-instances-layer' }, svg);
  const layerLensLabels = group({ id: 'p11b-lens-labels', opacity: 0 }, svg);
  const layerCallouts = group({ id: 'p11b-callouts-layer', opacity: 0 }, svg);

  /* ---------- rings split by how deeply they are buried ---------- */
  function ringRuns(geo, others) {
    const xs = new Float64Array(SEGMENTS);
    const ys = new Float64Array(SEGMENTS);
    const dep = new Uint8Array(SEGMENTS);
    for (let i = 0; i < SEGMENTS; i++) {
      const t = (i / SEGMENTS) * Math.PI * 2;
      const x = geo.cx + geo.rx * Math.cos(t);
      const y = geo.cy + geo.ry * Math.sin(t);
      xs[i] = x;
      ys[i] = y;
      let d = 0;
      for (const o of others) {
        if (!o.rx || !o.ry) continue;
        const ux = (x - o.cx) / o.rx;
        const uy = (y - o.cy) / o.ry;
        if (ux * ux + uy * uy < 1) d++;
      }
      dep[i] = d > 1 ? 2 : d;
    }
    let seam = 0;
    for (let i = 1; i < SEGMENTS; i++) if (dep[i] !== dep[i - 1]) { seam = i; break; }
    const out = ['', '', ''];
    let depth = dep[seam];
    let run = `M${xs[seam].toFixed(1)} ${ys[seam].toFixed(1)}`;
    for (let k = 1; k <= SEGMENTS; k++) {
      const i = (seam + k) % SEGMENTS;
      if (dep[i] === depth) {
        run += `L${xs[i].toFixed(1)} ${ys[i].toFixed(1)}`;
      } else {
        out[depth] += run;
        depth = dep[i];
        run = `M${xs[i].toFixed(1)} ${ys[i].toFixed(1)}`;
      }
    }
    out[depth] += run;
    return out;
  }

  /* ---------- islands: three rings + labels ---------- */
  function buildLabel(parent, icon, title, subtitle) {
    const detail = group({ class: 'detail' }, parent);
    /* the icon tucks closer to the word than in the first draft: ACT 01 centres
       the text on the ring, so a wide label ("Assignment") has to keep its icon
       clear of the circle's outline */
    el('use', { href: `#p11b-icon-${icon}`, x: -43, y: -16.65, width: 34, height: 33.9, stroke: FG }, detail);
    /* the two text lines live in their own group: ACT 01 centres that group on
       the ring, so the icon may hang left exactly like the reference does */
    const textG = group({ class: 'label-text' }, detail);
    text(textG, { x: 0, y: 0, fill: FG, 'font-size': 25.5, 'font-weight': '700', 'letter-spacing': '-0.4' }, title);
    text(textG, { x: 0, y: 21, fill: FG_SUB, 'font-size': 12, 'font-weight': '400' }, subtitle);

    const compact = group({ class: 'compact', opacity: 0 }, parent);
    el('use', { href: `#p11b-icon-${icon}`, x: -30, y: -11, width: 18, height: 18, stroke: ACCENT[icon] }, compact);
    text(compact, { x: -6, y: 3, fill: ACCENT[icon], 'font-size': 15, 'font-weight': '700' }, title);
    return { detail, textG, compact };
  }

  const islands = [];
  function buildIslands() {
    for (let i = 0; i < 3; i++) {
      const g = group({ id: `p11b-island-${i}` }, layerInst);
      const ringGroup = group({}, g);
      const rings = ['graph', 'assign', 'pipe'].map(() => {
        const paths = ARC_TIERS.map((tier) => el('path', {
          fill: 'none', stroke: '#FFFFFF',
          'stroke-opacity': tier.opacity, 'stroke-width': tier.width,
          'stroke-linecap': 'round',
        }, ringGroup));
        return { paths };
      });
      const labelG = group({}, g);
      const labelA = group({}, g);
      const labelP = group({}, g);
      const parts = {
        graph: buildLabel(labelG, 'graph', 'Graph', '计算依赖关系网络化'),
        assign: buildLabel(labelA, 'assignment', 'Assignment', '数据与任务分配空间化'),
        pipe: buildLabel(labelP, 'pipeline', 'Pipeline', '计算组织过程可视化'),
      };
      const wm = {
        graph: text(g, { 'text-anchor': 'middle', fill: WATERMARK, 'font-size': 40, 'font-weight': '700', 'letter-spacing': '-0.8' }, 'What/Why'),
        assign: text(g, { 'text-anchor': 'middle', fill: WATERMARK, 'font-size': 40, 'font-weight': '700', 'letter-spacing': '-0.8' }, 'Who/Where'),
        pipe: text(g, { 'text-anchor': 'middle', fill: WATERMARK, 'font-size': 40, 'font-weight': '700', 'letter-spacing': '-0.8' }, 'How/When'),
      };
      const core = group({ class: 'core' }, g);
      const coreHalo = el('circle', { fill: 'rgba(255,255,255,0.05)' }, core);
      const coreRing = el('circle', { fill: 'rgba(14,15,18,0.92)', stroke: 'rgba(255,255,255,0.45)', 'stroke-width': 1.4 }, core);
      const coreText = group({}, core);
      text(coreText, { x: 0, y: -3, 'text-anchor': 'middle', fill: '#FFFFFF', 'font-size': 13, 'font-weight': '700' }, 'Paged');
      text(coreText, { x: 0, y: 14, 'text-anchor': 'middle', fill: '#E5E7EB', 'font-size': 13, 'font-weight': '600' }, 'Attention');
      const reticle = group({ opacity: 0 }, g);
      el('circle', { cx: 0, cy: 0, r: 15, fill: 'none', stroke: 'rgba(255,255,255,0.35)', 'stroke-width': 1 }, reticle);
      el('polygon', { points: '0,-5 5,0 0,5 -5,0', fill: FG }, reticle);

      g.addEventListener('click', () => { if (current === 3) selectTier(i); });

      islands.push({ g, rings, labelG, labelA, labelP, wm, parts, core, coreHalo, coreRing, coreText, reticle, state: null });
    }
  }

  /* ---------- label metrics ----------
     Each label is left-aligned from its anchor, so the anchor is the left edge
     of the block, not its middle. ACT 01 centres the whole block on its ring —
     icon plus both text lines — because centring only the two text lines leaves
     the icon parked against the left edge and the composition reads as shifted.
     Only the browser can tell us the real box, so measure it. */
  const labelMetrics = {
    graph: { x: 32.9, y: 0 },
    assign: { x: 48.3, y: 0 },
    pipe: { x: 32.5, y: 0 },
  };
  let metricsReady = false;

  function measureMetrics() {
    if (metricsReady || !islands.length) return metricsReady;
    /* getBBox needs a laid-out pane: until P11 is actually on screen the
       fallback numbers above stand in, and they are within a unit of the
       measured ones, so nothing jumps when the real values land */
    if (paneB.hidden || !paneB.getClientRects().length) return false;
    const probe = {};
    for (const key of ['graph', 'assign', 'pipe']) {
      const box = islands[0].parts[key].detail?.getBBox();
      if (box && box.width > 1) probe[key] = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
    }
    if (Object.keys(probe).length !== 3) return false;
    Object.assign(labelMetrics, probe);
    metricsReady = true;
    return true;
  }

  /* ---------- lens fills + chips (scene 1) ---------- */
  function buildLenses() {
    const centers = {
      graph: { x: CX, y: CY_TOP },
      assign: { x: CX - OFF, y: CY_LOW },
      pipe: { x: CX + OFF, y: CY_LOW },
    };
    /* blur inside the clip, so the haze never leaks past the lens edges */
    const pair = (outer, inner, fill) => {
      const holder = group({ 'clip-path': `url(#p11b-clip-${outer})` }, layerFills);
      const soft = group({ filter: 'url(#p11b-lens-blur)' }, holder);
      el('circle', { cx: centers[inner].x, cy: centers[inner].y, r: R, fill }, soft);
    };
    pair('graph', 'assign', LENS.computeData);
    pair('graph', 'pipe', LENS.computeTime);
    pair('assign', 'pipe', LENS.dataTime);

    const tripleGraph = group({ 'clip-path': 'url(#p11b-clip-graph)' }, layerFills);
    const tripleAssign = group({ 'clip-path': 'url(#p11b-clip-assign)' }, tripleGraph);
    const tripleSoft = group({ filter: 'url(#p11b-core-blur)' }, tripleAssign);
    el('circle', { cx: centers.pipe.x, cy: centers.pipe.y, r: R, fill: LENS.core }, tripleSoft);
  }

  function buildChips() {
    const W = 40.3;
    const H = 27.9;
    const HALF_GAP = 10.45;
    const chip = (cx, cy, left, leftFill, right, rightFill) => {
      const g = group({ transform: `translate(${cx},${cy})` }, layerLensLabels);
      el('rect', { x: -HALF_GAP - W, y: -H / 2, width: W, height: H, rx: 5, fill: leftFill }, g);
      text(g, { x: -HALF_GAP - W / 2, y: 5, 'text-anchor': 'middle', fill: '#FFFFFF', 'font-size': 15, 'font-weight': '600' }, left);
      text(g, { x: 0, y: 4.5, 'text-anchor': 'middle', fill: 'rgba(255,255,255,0.5)', 'font-size': 12, 'font-weight': '600' }, '×');
      el('rect', { x: HALF_GAP, y: -H / 2, width: W, height: H, rx: 5, fill: rightFill }, g);
      text(g, { x: HALF_GAP + W / 2, y: 5, 'text-anchor': 'middle', fill: '#FFFFFF', 'font-size': 15, 'font-weight': '600' }, right);
    };
    chip(582.7, 285.2, '计算', CHIP.compute, '数据', CHIP.data);
    chip(785.6, 285.2, '计算', CHIP.compute, '时间', CHIP.time);
    chip(684.2, 457.9, '数据', CHIP.data, '时间', CHIP.time);

    text(layerLensLabels, { x: CX, y: 342, 'text-anchor': 'middle', fill: '#FFFFFF', 'font-size': 18.5, 'font-weight': '700' }, 'Paged');
    text(layerLensLabels, { x: CX, y: 369, 'text-anchor': 'middle', fill: '#E2E8F0', 'font-size': 18.5, 'font-weight': '600' }, 'Attention');
  }

  /* ---------- scene 0 arrows ---------- */
  function buildArrows() {
    const marker = el('marker', {
      id: 'p11b-arrow', viewBox: '0 0 10 10', refX: 7, refY: 5,
      markerWidth: 5.5, markerHeight: 5.5, orient: 'auto-start-reverse',
    }, defs);
    el('path', { d: 'M0 0 L10 5 L0 10 z', fill: 'rgba(255,255,255,0.4)' }, marker);

    const core = { x: 683, y: 330 };
    const targets = [{ x: 683, y: 158 }, { x: 473, y: 422 }, { x: 893, y: 422 }];
    for (const target of targets) {
      const dx = target.x - core.x;
      const dy = target.y - core.y;
      const len = Math.hypot(dx, dy);
      const ux = dx / len;
      const uy = dy / len;
      el('line', {
        x1: core.x + ux * 42, y1: core.y + uy * 42,
        x2: target.x - ux * 132, y2: target.y - uy * 132,
        stroke: 'rgba(255,255,255,0.28)', 'stroke-width': 1.3,
        'stroke-dasharray': '4 5', 'marker-end': 'url(#p11b-arrow)',
      }, layerArrows);
    }
  }

  /* ---------- scene 3 axis + callouts ---------- */
  function buildAxis() {
    el('line', {
      x1: 545, y1: 42, x2: 545, y2: 548,
      stroke: 'rgba(255,255,255,0.16)', 'stroke-width': 1, 'stroke-dasharray': '3 6',
    }, layerAxis);
    el('path', { d: 'M545 554 l-4 -7 h8 z', fill: 'rgba(255,255,255,0.3)' }, layerAxis);
    text(layerAxis, {
      x: 545, y: 28, 'text-anchor': 'middle', fill: 'rgba(148,163,184,0.75)',
      'font-size': 9.5, 'font-weight': '600', 'letter-spacing': '2.4',
    }, 'EXECUTION HIERARCHY DRILL-DOWN');
  }

  function buildCallouts() {
    const rows = [
      { y: 110, edge: 854, knee: 100 },
      { y: 280, edge: 792, knee: 270 },
      { y: 450, edge: 740, knee: 440 },
    ];
    rows.forEach((row, i) => {
      const g = group({ id: `p11b-callout-${i}`, class: 'layer-hit-area' }, layerCallouts);
      el('polyline', {
        points: `${row.edge},${row.y} ${row.edge + 34},${row.y} ${row.edge + 56},${row.knee} 872,${row.knee}`,
        fill: 'none', stroke: LEADER, 'stroke-width': 1.2,
      }, g);
      el('circle', { cx: row.edge, cy: row.y, r: 2.6, fill: 'rgba(255,255,255,0.55)' }, g);
      el('circle', { cx: 872, cy: row.knee, r: 3.2, fill: FG, filter: 'url(#p11b-glow)' }, g);
      text(g, {
        x: 890, y: row.knee - 6, fill: 'rgba(148,163,184,0.9)', 'font-size': 12, 'font-weight': '600',
        'font-family': 'ui-monospace, SFMono-Regular, Menlo, monospace',
      }, TIERS[i].code);
      text(g, { x: 916, y: row.knee - 4, fill: FG, 'font-size': 24, 'font-weight': '700', 'letter-spacing': '-0.4' }, TIERS[i].name);
      text(g, { x: 916, y: row.knee + 20, fill: FG_WEAK, 'font-size': 12.5, 'font-weight': '400' }, TIERS[i].desc);
      g.addEventListener('click', () => selectTier(i));
    });
  }

  /* ---------- per-scene targets ---------- */
  const baseIsland = (x, y, scale, opacity) => ({
    x, y, scale, opacity,
    gCx: 0, gCy: 0, gRx: 100, gRy: 100,
    aCx: 0, aCy: 0, aRx: 100, aRy: 100,
    pCx: 0, pCy: 0, pRx: 100, pRy: 100,
    lgX: 0, lgY: 0, laX: 0, laY: 0, lpX: 0, lpY: 0,
    wgX: 0, wgY: 0, waX: 0, waY: 0, wpX: 0, wpY: 0,
    detailAlpha: 0, compactAlpha: 1, wmAlpha: 0,
    coreR: 12, coreAlpha: 0, reticleAlpha: 1,
    uniform: 1,
  });

  const ROW = CY_LOW - CY_TOP;
  /* watermark sits this far below its label (unchanged from the original
     hand-placed ACT 01, now expressed relative to the centred text) */
  const GHOST_OFFSET = 58;

  /* The two spare islands ride along invisibly. They must *start* as an exact
     copy of the visible composition, otherwise they fade in from three
     concentric rings and the squash looks like three unrelated diagrams. */
  const ghostOf = (state) => ({
    ...state,
    opacity: 0,
    detailAlpha: 0, compactAlpha: 0, wmAlpha: 0, coreAlpha: 0, reticleAlpha: 0,
  });

  function sceneStates(idx) {
    /* ACT 01 — three isolated rings pointing at the operator */
    if (idx === 0) {
      /* anchor = left edge of the whole label block (icon + two text lines),
         so centring the measured block on the ring keeps both margins equal */
      const m = labelMetrics;
      const gCx = 0, gCy = -172;
      const aCx = -212, aCy = 92;
      const pCx = 212, pCy = 92;
      const lgX = gCx - m.graph.x, lgY = gCy - m.graph.y;
      const laX = aCx - m.assign.x, laY = aCy - m.assign.y;
      const lpX = pCx - m.pipe.x, lpY = pCy - m.pipe.y;
      const main = {
        ...baseIsland(CX, 330, 1, 1),
        gCy, gRx: 126, gRy: 126,
        aCx, aCy, aRx: 126, aRy: 126,
        pCx, pCy, pRx: 126, pRy: 126,
        lgX, lgY, laX, laY, lpX, lpY,
        wgX: gCx, wgY: lgY + GHOST_OFFSET,
        waX: aCx, waY: laY + GHOST_OFFSET,
        wpX: pCx, wpY: lpY + GHOST_OFFSET,
        detailAlpha: 1, compactAlpha: 0, wmAlpha: 1,
        coreR: 36, coreAlpha: 1, reticleAlpha: 0, uniform: 1,
      };
      return [main, ghostOf(main), ghostOf(main)];
    }

    /* ACT 02 — the reference Venn */
    if (idx === 1) {
      const main = {
        ...baseIsland(CX, CY_TOP, 1, 1),
        gRx: R, gRy: R,
        aCx: -OFF, aCy: ROW, aRx: R, aRy: R,
        pCx: OFF, pCy: ROW, pRx: R, pRy: R,
        lgX: -27.9, lgY: -72.0,
        laX: -239.3, laY: 224.7,
        lpX: 175.1, lpY: 224.7,
        wgX: 0.8, wgY: -7.2,
        waX: -207.2, waY: 293.2,
        wpX: 189.4, wpY: 293.2,
        detailAlpha: 1, compactAlpha: 0, wmAlpha: 1,
        coreR: 0, coreAlpha: 0, reticleAlpha: 0, uniform: 0,
      };
      return [main, ghostOf(main), ghostOf(main)];
    }

    /* ACT 03 — three compressed slices side by side */
    if (idx === 2) {
      return [344, 683, 1022].map((x) => ({
        ...baseIsland(x, 311, 0.74, 1),
        gCy: -30, aCy: 26, pCy: 26,
        gRx: 180, gRy: 46, aRx: 180, aRy: 46, pRx: 180, pRy: 46,
        aCx: -104, pCx: 104,
        lgX: 0, lgY: -30, laX: -158, laY: 26, lpX: 126, lpY: 26,
      }));
    }

    /* ACT 04 — vertical drill-down stack */
    const ys = [140, 310, 480];
    const scales = [1, 0.8, 0.63];
    return ys.map((y, i) => {
      const dim = activeTier !== -1 && activeTier !== i;
      return {
        ...baseIsland(545, y, scales[i], dim ? 0.35 : 1),
        gCy: -30, aCy: 24, pCy: 24,
        gRx: 205, gRy: 50, aRx: 205, aRy: 50, pRx: 205, pRy: 50,
        aCx: -112, pCx: 112,
        lgX: 0, lgY: -30, laX: -170, laY: 24, lpX: 132, lpY: 24,
      };
    });
  }

  /* ---------- state application ---------- */
  const f2 = (v) => Number(v).toFixed(2);
  function applyState(island, s) {
    island.g.setAttribute('transform', `translate(${f2(s.x)}, ${f2(s.y)}) scale(${Number(s.scale).toFixed(4)})`);
    island.g.setAttribute('opacity', Number(s.opacity).toFixed(3));

    const geo = [
      { cx: s.gCx, cy: s.gCy, rx: s.gRx, ry: s.gRy },
      { cx: s.aCx, cy: s.aCy, rx: s.aRx, ry: s.aRy },
      { cx: s.pCx, cy: s.pCy, rx: s.pRx, ry: s.pRy },
    ];
    /* the assign×pipeline lens carries the accent outline; every other arc
       simply disappears where it slides behind a ring */
    geo.forEach((g, i) => {
      const others = geo.filter((_, j) => j !== i);
      const runs = ringRuns(g, others);
      const u = Math.max(0, Math.min(1, s.uniform ?? 1));
      island.rings[i].paths.forEach((p, tier) => {
        p.setAttribute('d', runs[tier]);
        /* flat tiers keep one weight everywhere; the Venn fades each tier back */
        p.setAttribute('stroke-opacity', lerp(ARC_TIERS[tier].opacity, 0.34, u).toFixed(3));
        p.setAttribute('stroke-width', lerp(ARC_TIERS[tier].width, 2.0, u).toFixed(2));
      });
    });

    island.labelG.setAttribute('transform', `translate(${f2(s.lgX)}, ${f2(s.lgY)})`);
    island.labelA.setAttribute('transform', `translate(${f2(s.laX)}, ${f2(s.laY)})`);
    island.labelP.setAttribute('transform', `translate(${f2(s.lpX)}, ${f2(s.lpY)})`);

    island.g.querySelectorAll('.detail').forEach((node) => node.setAttribute('opacity', Number(s.detailAlpha).toFixed(3)));
    island.g.querySelectorAll('.compact').forEach((node) => node.setAttribute('opacity', Number(s.compactAlpha).toFixed(3)));

    island.wm.graph.setAttribute('transform', `translate(${f2(s.wgX)}, ${f2(s.wgY)})`);
    island.wm.assign.setAttribute('transform', `translate(${f2(s.waX)}, ${f2(s.waY)})`);
    island.wm.pipe.setAttribute('transform', `translate(${f2(s.wpX)}, ${f2(s.wpY)})`);
    for (const node of Object.values(island.wm)) node.setAttribute('opacity', Number(s.wmAlpha).toFixed(3));

    island.coreHalo.setAttribute('r', f2(s.coreR * 1.5));
    island.coreHalo.setAttribute('opacity', Number(s.coreAlpha * 0.9).toFixed(3));
    island.coreRing.setAttribute('r', f2(s.coreR));
    island.coreRing.setAttribute('opacity', Number(s.coreAlpha).toFixed(3));
    island.coreText.setAttribute('opacity', Number(s.coreAlpha).toFixed(3));
    island.reticle.setAttribute('opacity', Number(s.reticleAlpha).toFixed(3));
  }

  /* ---------- animation ---------- */
  const DUR = 880;
  /* The Venn and the drill-down never touch directly: the reference squashes the
     rings into three flat copies first, then drops those copies into the stack.
     Each leg gets its own beat so the deformation stays visible instead of
     snapping to the flattened pose. */
  const LEG_FLATTEN = 470;
  const HOLD_FLAT = 190;   /* the flat pose is a beat of its own, not a blur */
  const LEG_SETTLE = 660;
  let raf = null;
  let current = 0;
  let activeTier = -1;
  let generation = 0;

  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const lerp = (a, b, t) => a + (b - a) * t;

  const layerPlan = (target) => ([
    { node: layerArrows, on: target === 0, delay: 0 },
    { node: layerFills, on: target === 1, delay: 0.28 },
    { node: layerLensLabels, on: target === 1, delay: 0.4 },
    { node: layerAxis, on: target === 3, delay: 0.16 },
    { node: layerCallouts, on: target === 3, delay: 0.34 },
  ]);

  /* Type never cross-fades through the squash: the outgoing labels are gone
     before the flattened pose arrives, and the compact labels only appear once
     those slices are formed. Keeps the morph from showing two label systems. */
  const TYPE_KEYS = new Set(['detailAlpha', 'compactAlpha', 'wmAlpha', 'coreAlpha', 'reticleAlpha', 'opacity']);
  const TYPE_OUT = [0.40, 0.58];   /* labels stay crisp, then leave fast — no long ghost */
  const TYPE_IN_START = 0.66;
  const GHOST_IN_START = 0.38;
  const typeRamp = (key, from, to, t) => {
    if (from > to) {
      const [a, b] = TYPE_OUT;
      return Math.max(0, Math.min(1, (t - a) / (b - a)));
    }
    if (to > from) {
      /* the spare copies come back a beat before their labels do */
      const start = key === 'opacity' ? GHOST_IN_START : TYPE_IN_START;
      return Math.max(0, (t - start) / (1 - start));
    }
    return t;
  };

  function render(target, options = {}) {
    const opts = typeof options === 'boolean' ? { immediate: options } : options;
    const immediate = !!opts.immediate;
    const duration = opts.duration ?? DUR;
    const onComplete = opts.onComplete ?? null;
    const swap = !!opts.swap;
    const layerSpan = opts.layerSpan ?? 1;
    const gen = ++generation;

    /* the first render that happens while P11 is on screen locks in the real
       text widths, so ACT 01 centres on the actual ink, not a guess */
    measureMetrics();
    const states = sceneStates(target);
    const plan = layerPlan(target);

    if (raf) { cancelAnimationFrame(raf); raf = null; }

    if (immediate || !islands[0].state) {
      islands.forEach((island, i) => {
        island.state = { ...states[i] };
        applyState(island, island.state);
      });
      plan.forEach((item) => item.node.setAttribute('opacity', item.on ? '1' : '0'));
      onComplete?.();
      return;
    }

    const from = islands.map((island) => ({ ...island.state }));
    const fromOpacity = plan.map((item) => Number(item.node.getAttribute('opacity') ?? 0));
    const start = performance.now();

    const tick = (now) => {
      if (gen !== generation) return;
      const t = Math.min(1, (now - start) / duration);
      const eased = ease(t);
      islands.forEach((island, i) => {
        const next = {};
        for (const key of Object.keys(states[i])) {
          const kt = swap && TYPE_KEYS.has(key)
            ? ease(typeRamp(key, from[i][key], states[i][key], t))
            : eased;
          next[key] = lerp(from[i][key], states[i][key], kt);
        }
        island.state = next;
        applyState(island, next);
      });
      const lt = Math.min(1, t / layerSpan);
      plan.forEach((item, i) => {
        const to = item.on ? 1 : 0;
        const local = item.on ? Math.max(0, (lt - item.delay) / (1 - item.delay)) : lt;
        item.node.setAttribute('opacity', lerp(fromOpacity[i], to, ease(Math.min(1, local))).toFixed(3));
      });
      if (t < 1) {
        raf = requestAnimationFrame(tick);
      } else {
        raf = null;
        onComplete?.();
      }
    };
    raf = requestAnimationFrame(tick);
  }

  function goToScene(idx, immediate = false) {
    if (!Number.isInteger(idx) || idx < 0 || idx > 3) return false;
    if (idx === 2) idx = 3;
    if (idx !== 3) activeTier = -1;
    const source = current;
    current = idx;
    const viaFlatten = (source === 1 && idx === 3) || (source === 3 && idx === 1);

    if (viaFlatten && !immediate) {
      /* leg 1 — squeeze the Venn into the flat three-slice pose (labels hand
         over at the midpoint, lens tints are gone by then) */
      render(2, {
        duration: LEG_FLATTEN,
        swap: true,
        layerSpan: source === 1 ? 0.45 : 1,
        onComplete: () => {
          if (current !== idx) return;
          /* hold the flattened pose, then — leg 2 — collapse the slices into the
             stack, or unfold them back */
          const token = generation;
          setTimeout(() => {
            if (generation !== token || current !== idx) return;
            render(idx, { duration: LEG_SETTLE, swap: source === 3 });
          }, HOLD_FLAT);
        },
      });
    } else if (viaFlatten) {
      render(idx, { immediate: true });
    } else {
      render(idx, { immediate: immediate || source === idx });
    }
    return true;
  }

  function selectTier(tier) {
    if (current !== 3) return;
    activeTier = activeTier === tier ? -1 : tier;
    for (let i = 0; i < 3; i++) {
      const callout = layerCallouts.querySelector(`#p11b-callout-${i}`);
      if (callout) callout.style.opacity = activeTier === -1 || activeTier === tier ? '1' : '0.35';
    }
    render(3);
  }

  /* ---------- build ---------- */
  buildIslands();
  buildLenses();
  buildChips();
  buildArrows();
  buildAxis();
  buildCallouts();
  render(0, true);

  const storyB = Object.freeze({
    goToStep(index, { animate = true } = {}) {
      if (!Number.isInteger(index) || index < 0 || index > 3) return false;
      goToScene(index === 2 ? 3 : index, !animate);
      return true;
    },
    next() { return current >= 3 ? false : goToScene(current + 1); },
    previous() { return current <= 0 ? false : goToScene(current === 3 ? 1 : current - 1); },
    setActive(active) { if (!active) render(current, true); },
    get step() { return current; },
  });

  /* ---------- variant switch ---------- */
  const VARIANT_KEY = 'pto.p11.variant';
  const storedVariant = () => {
    try {
      return localStorage.getItem(VARIANT_KEY) === 'b' ? 'b' : 'a';
    } catch (err) {
      return 'a';
    }
  };
  const rememberVariant = (next) => {
    try {
      localStorage.setItem(VARIANT_KEY, next);
    } catch (err) {
      /* private mode / blocked storage: keep the in-session choice */
    }
  };

  let variant = storedVariant();
  let controlButton = null;
  root.dataset.p11Variant = variant;
  if (paneA) paneA.hidden = variant !== 'a';
  paneB.hidden = variant !== 'b';
  if (variant === 'b') {
    storyA?.setActive?.(false);
    storyB.goToStep(0, { animate: false });
  }

  const activeStory = () => (variant === 'a' ? storyA : storyB);

  function syncButton() {
    if (!controlButton) return;
    const isB = variant === 'b';
    controlButton.textContent = isB ? '视觉 B' : '视觉 A';
    controlButton.setAttribute('aria-pressed', isB ? 'true' : 'false');
    controlButton.title = isB ? '切换 P11 的视觉版本（当前：线框版 B）' : '切换 P11 的视觉版本（当前：原版 A）';
  }

  function setVariant(next) {
    if (next !== 'a' && next !== 'b') return variant;
    if (next === variant) return variant;
    const step = activeStory().step ?? 0;
    activeStory().setActive(false);
    variant = next;
    root.dataset.p11Variant = variant;
    if (paneA) paneA.hidden = variant !== 'a';
    paneB.hidden = variant !== 'b';
    activeStory().setActive(true);
    activeStory().goToStep(step, { animate: false });
    rememberVariant(variant);
    syncButton();
    return variant;
  }

  Object.defineProperty(root, 'ptoP11Story', {
    configurable: true,
    value: Object.freeze({
      goToStep(step, opts) { return activeStory().goToStep(step, opts); },
      next() { return activeStory().next(); },
      previous() { return activeStory().previous(); },
      setActive(active) { activeStory().setActive(active); },
      get step() { return activeStory().step; },
    }),
  });

  window.PtoP11Variant = Object.freeze({
    get value() { return variant; },
    set: setVariant,
    toggle() { return setVariant(variant === 'a' ? 'b' : 'a'); },
    attach(button) { controlButton = button ?? null; syncButton(); },
    refresh: syncButton,
  });
  syncButton();
})();
