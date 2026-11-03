/* Phase0 uses g_idx = KV * active_batch + Batch; this is not Attention Task order. */
(function (global) {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const mix = (a, b, p) => a + (b - a) * p;
  const clamp = p => Math.max(0, Math.min(1, p));
  function createModel({ blocks, kvHeads, qPerKv, activeBatch, ropeCores, itemsPerCore = Math.ceil(activeBatch * kvHeads / ropeCores), lanesPerBlock }) {
    const total = activeBatch * kvHeads;
    const lanes = Array.from({ length: blocks * lanesPerBlock }, (_, ropeCore) => {
      const block = Math.floor(ropeCore / lanesPerBlock), lane = ropeCore % lanesPerBlock;
      const groups = ropeCore >= ropeCores ? [] : Array.from({ length: itemsPerCore }, (_, it) => {
        const gIndex = ropeCore + it * ropeCores, kvHead = Math.floor(gIndex / activeBatch), batch = gIndex % activeBatch;
        return { id: `g:${gIndex}`, gIndex, it, iteration: it, batch, kvHead, ropeCore, block, lane,
          attentionTaskId: batch * kvHeads + kvHead, qStart: kvHead * qPerKv, qEnd: (kvHead + 1) * qPerKv - 1 };
      }).filter(group => group.gIndex < total);
      return { ropeCore, block, lane, groups };
    });
    const items = lanes.flatMap(lane => lane.groups).sort((a, b) => a.gIndex - b.gIndex);
    return { blocks, kvHeads, activeBatch, ropeCores, qPerKv, total, lanesPerBlock, itemsPerCore, lanes, items,
      activeLanes: lanes.filter(lane => lane.groups.length),
      groupsForBlock: block => lanes.filter(lane => lane.block === block).flatMap(lane => lane.groups),
      ownerOf: (batch, kvHead) => lanes[(kvHead * activeBatch + batch) % ropeCores],
      itemFor: (batch, kvHead) => items[kvHead * activeBatch + batch],
    };
  }
  function createLayout(model) {
    const size = { width: 1250, height: 860, itemWidth: 114, itemHeight: 18, headerHeight: 20,
      padding: 8, labelWidth: 38, labelGap: 8, columnGap: 6, laneGap: 7, pairGap: 12,
      groupGap: 24, top: 54, syncGap: 18, summaryHeight: 18, syncBarGap: 8, syncBarHeight: 26 };
    const { width, height, itemWidth, itemHeight } = size;
    const columnPitch = 1048 / model.kvHeads;
    const columnOrder = [...Array.from({ length: model.kvHeads }, (_, kv) => kv).filter(kv => kv % 2 === 0),
      ...Array.from({ length: model.kvHeads }, (_, kv) => kv).filter(kv => kv % 2 === 1)];
    const sourceX = index => 179 + index * columnPitch + (columnPitch - itemWidth) / 2;
    const halfColumns = Math.ceil(model.kvHeads / 2);
    const itemPitch = itemWidth + size.columnGap;
    const groupWidth = size.padding * 2 + size.labelWidth + size.labelGap + halfColumns * itemWidth + (halfColumns - 1) * size.columnGap;
    const groupLeft = (width - groupWidth * 2 - size.groupGap) / 2;
    const targetX = index => groupLeft + (index >= halfColumns ? groupWidth + size.groupGap : 0)
      + size.padding + size.labelWidth + size.labelGap + index % halfColumns * itemPitch;
    const blockHeight = size.headerHeight + size.padding * 2 + itemHeight * 2 + size.laneGap;
    const pairPitch = blockHeight + size.pairGap;
    const firstItemY = size.top + size.headerHeight + size.padding;
    const sourceRowPitch = (pairPitch * Math.ceil(model.activeBatch / 2) - size.pairGap - itemHeight) / Math.max(1, model.activeBatch - 1);
    const sources = [], targets = [];
    model.items.forEach(item => {
      sources[item.gIndex] = { x: sourceX(item.kvHead), y: firstItemY + item.batch * sourceRowPitch, w: itemWidth, h: itemHeight };
      targets[item.gIndex] = { x: targetX(columnOrder.indexOf(item.kvHead)),
        y: firstItemY + Math.floor(item.batch / 2) * pairPitch + item.batch % 2 * (itemHeight + size.laneGap), w: itemWidth, h: itemHeight };
    });
    const blockRects = Array.from({ length: model.blocks }, (_, block) => {
      const items = model.groupsForBlock(block);
      if (!items.length) return null;
      const rects = items.map(item => targets[item.gIndex]);
      const minX = Math.min(...rects.map(r => r.x)), maxX = Math.max(...rects.map(r => r.x + r.w));
      const minY = Math.min(...rects.map(r => r.y)), maxY = Math.max(...rects.map(r => r.y + r.h));
      const x = minX - size.labelWidth - size.labelGap - size.padding;
      return { x, y: minY - size.headerHeight - size.padding, w: maxX + size.padding - x,
        h: maxY - minY + size.headerHeight + size.padding * 2 };
    });
    const laneRects = model.lanes.map(lane => {
      if (!lane.groups.length) return null;
      const block = blockRects[lane.block], rects = lane.groups.map(item => targets[item.gIndex]);
      const y = Math.min(...rects.map(r => r.y)), bottom = Math.max(...rects.map(r => r.y + r.h));
      return { x: block.x + size.padding, y: y - 1, w: block.w - size.padding * 2, h: bottom - y + 2 };
    });
    const gridBottom = Math.max(...blockRects.filter(Boolean).map(r => r.y + r.h));
    const syncSummaryY = gridBottom + size.syncGap + size.summaryHeight / 2;
    const syncBarY = gridBottom + size.syncGap + size.summaryHeight + size.syncBarGap;
    return { width, height, sources, targets, blockRects, laneRects, columnOrder, columnPitch, pairPitch,
      size, sourceRowPitch, syncSummaryY, syncBarY };
  }
  const make = (tag, attrs, parent, text) => {
    const el = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    if (text != null) el.textContent = text;
    parent.append(el); return el;
  };
  function mount(host, { model, seqLens = [], onSelect, onCamera }) {
    const layout = createLayout(model);
    host.classList.add('viz2-p0-morph');
    host.innerHTML = `<div class="viz2-am-head"><div><h2 data-p0-title-state></h2><p data-p0-formula></p></div><div class="viz2-am-controls"><button class="btn btn-ghost btn-sm" data-p0-play>暂停</button><button class="btn btn-ghost btn-sm" data-p0-replay>重播</button><div class="segmented-control segmented-control-muted"><button class="btn btn-sm" data-p0-state="generation">Work Generation</button><button class="btn btn-sm" data-p0-state="assignment">Block Assignment</button></div><label><input type="checkbox" data-p0-loop checked> 循环播放</label></div></div>
      <div class="viz2-p0-viewport viz2-am-viewport" tabindex="0" role="group" aria-label="Phase0 工作项生成与 Block 分配画布"><svg class="viz2-p0-svg" aria-label="Phase0 Batch × KV Head 与 AIV Lane 分配"><g data-p0-scene><g data-p0-columns></g><g data-p0-generation></g><g data-p0-board></g><g data-p0-items></g></g></svg><div class="kf-pa-map__tooltip" role="tooltip" hidden></div></div>
      <footer class="viz2-am-footer"><p data-p0-selection aria-live="polite">${model.total} 个工作项 · 每项 ${model.qPerKv}Q + K + V</p><span>Q/K Norm → RoPE → KV Cache Append · 静态分配，不表示执行时序</span><div class="kf-pa-map__tools"><button class="btn btn-ghost btn-sm" type="button" data-p0-zoom="0.8" aria-label="缩小 Assignment">−</button><output data-p0-scale>100%</output><button class="btn btn-ghost btn-sm" type="button" data-p0-zoom="1.25" aria-label="放大 Assignment">＋</button><button class="btn btn-ghost btn-sm" type="button" data-p0-fit>适应</button></div></footer>
      <aside class="pto-ide-frame__pane viz2-p0-inspector" aria-label="Phase0 分配详情" hidden><header class="pto-ide-frame__pane-header"><b data-p0-title></b><button class="btn btn-ghost btn-sm" data-p0-clear aria-label="关闭 Phase0 详情">×</button></header><div class="pto-ide-frame__pane-body" data-p0-detail></div></aside>`;
    const viewport = host.querySelector('.viz2-p0-viewport'), scene = host.querySelector('[data-p0-scene]'), tooltip = host.querySelector('[role="tooltip"]');
    const columns = host.querySelector('[data-p0-columns]'), generation = host.querySelector('[data-p0-generation]'), board = host.querySelector('[data-p0-board]'), itemLayer = host.querySelector('[data-p0-items]');
    const columnNodes = Array.from({ length: model.kvHeads }, (_, kv) => make('text', { x: layout.sources[model.itemFor(0, kv).gIndex].x + layout.sources[0].w / 2, y: 30, class: 'viz2-p0-axis', 'text-anchor': 'middle' }, columns, `KV${kv}`));
    make('text', { x: 100, y: 30, class: 'viz2-p0-axis' }, generation, 'BATCH');
    for (let b = 0; b < model.activeBatch; b++) make('text', { x: 100, y: layout.sources[b].y + layout.size.itemHeight / 2, class: 'viz2-p0-batch' }, generation, `B${b}`);
    const blockNodes = layout.blockRects.map((r, block) => {
      if (!r) return null;
      const group = make('g', { class: 'viz2-p0-block', 'data-p0-block': block }, board);
      make('rect', { x: r.x, y: r.y, width: r.w, height: r.h, rx: 2, class: 'viz2-p0-frame' }, group);
      make('line', { x1: r.x + layout.size.padding, y1: r.y + layout.size.headerHeight,
        x2: r.x + r.w - layout.size.padding, y2: r.y + layout.size.headerHeight, class: 'viz2-p0-header-rule' }, group);
      const pick = make('g', { 'data-p0-pick-block': block, role: 'button', tabindex: '-1', 'aria-label': `Phase0 Block ${block}，${model.groupsForBlock(block).length} 项工作，参与同步` }, group);
      make('rect', { x: r.x, y: r.y, width: r.w, height: layout.size.headerHeight, fill: 'transparent' }, pick);
      make('text', { x: r.x + layout.size.padding, y: r.y + layout.size.headerHeight / 2, 'dominant-baseline': 'central', class: 'viz2-p0-block-label' }, pick, `BLK ${String(block).padStart(2, '0')}`);
      return group;
    });
    const laneNodes = model.lanes.map((lane, index) => {
      const r = layout.laneRects[index];
      const group = make('g', { class: 'viz2-p0-lane', 'data-p0-lane': lane.ropeCore, role: 'button', tabindex: '-1', 'aria-label': `Block ${lane.block} AIV${lane.lane}，rope_core ${lane.ropeCore}，${lane.groups.length} 项工作` }, blockNodes[lane.block] || board);
      if (r) {
        make('rect', { x: r.x, y: r.y, width: r.w, height: r.h, rx: 2, class: 'viz2-p0-lane-frame' }, group);
        make('text', { x: r.x + layout.size.labelWidth / 2, y: r.y + r.h / 2,
          'text-anchor': 'middle', 'dominant-baseline': 'central', class: 'viz2-p0-lane-label' }, group, `AIV${lane.lane}`);
      }
      return group;
    });
    const sync = make('g', { 'data-p0-sync': '' }, board);
    const idle = model.blocks - blockNodes.filter(Boolean).length;
    make('text', { x: 624, y: layout.syncSummaryY, 'text-anchor': 'middle', 'dominant-baseline': 'central', class: 'viz2-p0-idle-label' }, sync, `${idle} Blocks · Sync Only`);
    make('rect', { x: 24, y: layout.syncBarY, width: 1200, height: layout.size.syncBarHeight, rx: 2, class: 'viz2-p0-sync-frame' }, sync);
    make('text', { x: 624, y: layout.syncBarY + layout.size.syncBarHeight / 2, 'text-anchor': 'middle', 'dominant-baseline': 'central', class: 'viz2-p0-sync-label' }, sync, `SYNCALL · ${model.blocks} Blocks 参与 · cacheinvalid → fence → SYNCALL(MIX) → cacheinvalid`);
    const itemNodes = model.items.map((item, id) => {
      const group = make('g', { class: 'viz2-p0-item', 'data-p0-item': item.gIndex, role: 'button', tabindex: id === 0 ? '0' : '-1', 'aria-pressed': 'false', 'aria-label': `G${item.gIndex}，Batch ${item.batch}，KV${item.kvHead}，Block ${item.block}，AIV${item.lane}` }, itemLayer);
      const cell = make('rect', { class: 'viz2-p0-item-cell', rx: 3 }, group);
      const symbols = make('g', { class: 'viz2-p0-item-symbols', 'aria-hidden': 'true' }, group);
      const q = Array.from({ length: model.qPerKv }, () => make('circle', { class: 'viz2-p0-q' }, symbols));
      const label = make('text', { class: 'viz2-p0-item-id', 'text-anchor': 'middle', 'dominant-baseline': 'central' }, group, `G${item.gIndex}`);
      const r = layout.sources[id], pitch = 9;
      cell.setAttribute('width', r.w); cell.setAttribute('height', r.h);
      label.setAttribute('x', r.w - 20); label.setAttribute('y', r.h / 2);
      q.forEach((dot, index) => { dot.setAttribute('cx', 5 + (index + .5) * pitch); dot.setAttribute('cy', r.h / 2); dot.setAttribute('r', 2.5); });
      return { group, cell, symbols, q, label };
    });
    let current = null, camera = null, active = false, inspectorOpen = false, hovered = null;
    let drag = null, suppressClick = false, lastSize = null;
    const mapSelection = selection => {
      if (selection?.kind !== 'task') return selection;
      const item = model.itemFor(selection.batch, selection.kvHead);
      return item ? { ...selection, block: item.block, lane: item.lane, ropeCore: item.ropeCore, gIndex: item.gIndex } : null;
    };
    const fitted = (w = viewport.clientWidth, h = viewport.clientHeight) => {
      const scale = Math.max(.05, Math.min((w - 24) / layout.width, (h - 16) / layout.height));
      return { scale, x: (w - layout.width * scale) / 2, y: (h - layout.height * scale) / 2 };
    };
    function paintCamera() {
      const c = camera || fitted();
      host.querySelector('[data-p0-scale]').textContent = `${Math.round(c.scale / fitted().scale * 100)}%`;
      scene.setAttribute('transform', `translate(${c.x} ${c.y}) scale(${c.scale})`);
      scene.style.setProperty('--p0-label-size', `${Math.max(12, 11 / c.scale)}px`);
    }
    function draw(p) {
      host.dataset.p0Progress = String(p);
      generation.style.opacity = String(1 - p);
      const alpha = clamp((p - .55) / .4);
      board.style.opacity = String(alpha); board.style.pointerEvents = alpha > .8 ? '' : 'none';
      [...laneNodes.filter((_, index) => model.lanes[index].groups.length), ...host.querySelectorAll('[data-p0-pick-block]')].forEach(n => n.setAttribute('tabindex', alpha > .8 ? '0' : '-1'));
      columnNodes.forEach((node, kv) => node.setAttribute('transform', `translate(${mix(layout.sources[model.itemFor(0, kv).gIndex].x, layout.targets[model.itemFor(0, kv).gIndex].x, p) - layout.sources[model.itemFor(0, kv).gIndex].x} 0)`));
      itemNodes.forEach((n, id) => {
        const a = layout.sources[id], b = layout.targets[id];
        n.group.setAttribute('transform', `translate(${mix(a.x, b.x, p)} ${mix(a.y, b.y, p)})`);
      });
      host.querySelector('[data-p0-title-state]').textContent = p < .5 ? `Work Generation · ${model.total} Items` : `Block Assignment · ${blockNodes.filter(Boolean).length} Active Blocks`;
      host.querySelector('[data-p0-formula]').textContent = p < .5 ? `g_idx = KV Head × ${model.activeBatch} + Batch · 每项处理当前 token 的 ${model.qPerKv}Q + K + V` : `rope_core = g_idx % ${model.ropeCores} · Block = ⌊rope_core / ${model.lanesPerBlock}⌋ · 逻辑分组，非物理排列`;
      host.querySelectorAll('[data-p0-state]').forEach(button => { const selected = button.dataset.p0State === (p < .5 ? 'generation' : 'assignment'); button.setAttribute('aria-pressed', String(selected)); button.classList.toggle('is-selected', selected); });
    }
    const player = global.PtoVisualization2MorphPlayer.create({ draw, pauseOnReturn: true, onChange: s => { host.querySelector('[data-p0-play]').textContent = s.playing ? '暂停' : '播放'; } });
    const pause = () => player.pause();
    function itemDescription(item) {
      const len = seqLens[item.batch];
      return `G${item.gIndex} · B${item.batch} / KV${item.kvHead} · Q${item.qStart}–Q${item.qEnd} + K/V · BLK ${String(item.block).padStart(2, '0')} / AIV${item.lane} · rope_core ${item.ropeCore} · it ${item.iteration}${len == null ? '' : ` · seq_lens ${len} · position ${len - 1}`} · Attention Task ${item.attentionTaskId}`;
    }
    function selectItem(gIndex) {
      const item = model.items[gIndex]; if (!item) return;
      pause(); inspectorOpen = true;
      onSelect({ kind: 'task', task: item.attentionTaskId, block: item.attentionTaskId % model.blocks, batch: item.batch, kvHead: item.kvHead });
    }
    function paintSelection() {
      const selected = mapSelection(current);
      blockNodes.forEach((node, block) => node?.classList.toggle('is-selected', block === selected?.block));
      laneNodes.forEach((node, index) => { const lane = model.lanes[index], related = lane.block === selected?.block && (selected.lane == null || lane.lane === selected.lane); node.classList.toggle('is-selected', related); node.setAttribute('aria-pressed', String(related)); });
      itemNodes.forEach(({ group }, id) => {
        const item = model.items[id], related = item.block === selected?.block && (selected.lane == null || item.lane === selected.lane), exact = item.gIndex === selected?.gIndex;
        group.classList.toggle('is-assigned', related); group.classList.toggle('is-selected', exact); group.classList.toggle('is-hovered', item.gIndex === hovered);
        group.setAttribute('aria-pressed', String(exact)); group.setAttribute('tabindex', id === (selected?.gIndex ?? 0) ? '0' : '-1');
      });
      host.querySelector('[data-p0-selection]').textContent = selected?.gIndex != null ? itemDescription(model.items[selected.gIndex]) : selected ? `Block ${selected.block}${selected.lane == null ? '' : ` / AIV${selected.lane}`} · ${model.groupsForBlock(selected.block).filter(item => selected.lane == null || item.lane === selected.lane).map(item => `G${item.gIndex}`).join(', ') || '无预处理工作，仍参与 SYNCALL'}` : `${model.total} 个工作项 · 每项 ${model.qPerKv}Q + K + V`;
    }
    function detail() {
      const selection = mapSelection(current), aside = host.querySelector('aside');
      aside.hidden = !active || !selection || !inspectorOpen;
      if (!selection) return;
      const item = selection.gIndex == null ? null : model.items[selection.gIndex];
      const lanes = model.lanes.filter(lane => lane.block === selection.block && (selection.lane == null || lane.lane === selection.lane));
      host.querySelector('[data-p0-title]').textContent = item ? `G${item.gIndex} · B${item.batch} / KV${item.kvHead}` : `BLK ${String(selection.block).padStart(2, '0')}${selection.lane == null ? '' : ` / AIV${selection.lane}`}`;
      const len = item ? seqLens[item.batch] : null;
      host.querySelector('[data-p0-detail]').innerHTML = `${item ? `<section class="viz2-p0-detail-section"><h3>工作项映射</h3><p>Block ${item.block} / AIV${item.lane}<br>rope_core ${item.ropeCore} · iteration ${item.iteration}<br>Q${item.qStart}–Q${item.qEnd} + K + V${len == null ? '' : `<br>seq_lens ${len} · position ${len - 1}`}</p><p>对应 Attention Task ${item.attentionTaskId}<br>Attention Block ${item.attentionTaskId % model.blocks}</p></section>` : ''}
        ${lanes.map(lane => `<section class="viz2-p0-detail-section"><h3>AIV${lane.lane} <small>rope_core ${lane.ropeCore}</small></h3>${lane.groups.length ? `<table class="viz2-p0-table"><thead><tr><th>工作项</th><th>Batch</th><th>KV</th></tr></thead><tbody>${lane.groups.map(g => `<tr><td><button class="btn btn-ghost btn-sm" data-p0-pick-item="${g.gIndex}">G${g.gIndex}</button></td><td>${g.batch}</td><td>${g.kvHead}</td></tr>`).join('')}</tbody></table>` : `<p>无有效预处理工作。${lane.ropeCore >= model.ropeCores ? `rope_core ≥ ${model.ropeCores}，跳过工作循环。` : '该 Lane 没有有效工作项。'}</p>`}</section>`).join('')}
        <section class="viz2-p0-detail-section"><h3>当前 token 的预处理</h3><p>Q/K Norm → RoPE → KV Cache Append。写回旋转后的 Q，并追加当前 token 的 K/V。</p><p>seq_lens 决定 RoPE position，不表示本阶段扫描历史序列。</p></section><section class="viz2-p0-detail-section"><h3>同步后进入 Attention</h3><p>全部 ${model.blocks} Blocks 参与 SYNCALL。</p><code>cacheinvalid → fence<br>→ SYNCALL(MIX)<br>→ cacheinvalid</code></section>`;
    }
    function update(selection, nextCamera, { selectionChanged = true, active: nextActive = true } = {}) {
      const changed = JSON.stringify(current) !== JSON.stringify(selection);
      current = selection; camera = nextCamera; active = nextActive;
      if (changed && selectionChanged) inspectorOpen = !!selection;
      player.setActive(active); paintSelection(); paintCamera();
      if (selectionChanged || changed || !active) detail();
    }
    function fit() { if (viewport.clientWidth && viewport.clientHeight) onCamera(fitted()); }
    host.addEventListener('click', event => {
      if (suppressClick) { suppressClick = false; return; }
      if (event.target.closest('[data-p0-play]')) { player.toggle(); return; }
      if (event.target.closest('[data-p0-replay]')) { player.replay(); return; }
      const state = event.target.closest('[data-p0-state]'); if (state) { player.goTo(state.dataset.p0State === 'assignment' ? 1 : 0); return; }
      if (event.target.closest('[data-p0-fit]')) { fit(); return; }
      const zoomButton = event.target.closest('[data-p0-zoom]');
      if (zoomButton) { const c = camera || fitted(), scale = Math.max(.08, Math.min(6, c.scale * Number(zoomButton.dataset.p0Zoom))), ratio = scale / c.scale, x = viewport.clientWidth / 2, y = viewport.clientHeight / 2; onCamera({ scale, x: x - (x - c.x) * ratio, y: y - (y - c.y) * ratio }); return; }
      if (event.target.closest('[data-p0-clear]')) { pause(); inspectorOpen = false; detail(); return; }
      const item = event.target.closest('[data-p0-item],[data-p0-pick-item]');
      if (item) { selectItem(Number(item.dataset.p0Item ?? item.dataset.p0PickItem)); return; }
      const lane = event.target.closest('[data-p0-lane]'), block = event.target.closest('[data-p0-pick-block]');
      if (lane || block) { pause(); inspectorOpen = true; if (lane) { const l = model.lanes[Number(lane.dataset.p0Lane)]; onSelect({ kind: 'lane', block: l.block, lane: l.lane, ropeCore: l.ropeCore, task: null }); } else onSelect({ kind: 'block', block: Number(block.dataset.p0PickBlock), task: null }); }
    });
    host.querySelector('[data-p0-loop]').addEventListener('change', event => player.setLoop(event.target.checked));
    viewport.addEventListener('pointerover', event => { if (event.target.closest('[data-p0-item],[data-p0-lane],[data-p0-pick-block]')) pause(); });
    viewport.addEventListener('keydown', event => {
      pause();
      const target = event.target.closest('[data-p0-item],[data-p0-lane],[data-p0-pick-block]');
      if (['Enter', ' '].includes(event.key) && target) { event.preventDefault(); target.dispatchEvent(new MouseEvent('click', { bubbles: true })); return; }
      if (event.key === 'Escape') { onSelect(null); viewport.focus(); return; }
      const delta = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -model.kvHeads, ArrowDown: model.kvHeads }[event.key];
      if (delta != null) { event.preventDefault(); const id = Math.max(0, Math.min(model.total - 1, (current?.task ?? 0) + delta)); const batch = Math.floor(id / model.kvHeads), kv = id % model.kvHeads; selectItem(kv * model.activeBatch + batch); itemNodes[kv * model.activeBatch + batch].group.focus(); }
    });
    viewport.addEventListener('dblclick', event => { const item = event.target.closest('[data-p0-item]'); if (item) selectItem(Number(item.dataset.p0Item)); else fit(); });
    viewport.addEventListener('wheel', event => {
      event.preventDefault(); pause(); const r = viewport.getBoundingClientRect(), c = camera || fitted(), unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? r.height : 1;
      const factor = Math.exp(Math.max(-.22, Math.min(.22, -event.deltaY * unit * .003))), scale = Math.max(.08, Math.min(6, c.scale * factor)), ratio = scale / c.scale;
      const x = event.clientX - r.left, y = event.clientY - r.top; onCamera({ scale, x: x - (x - c.x) * ratio, y: y - (y - c.y) * ratio });
    }, { passive: false });
    viewport.addEventListener('pointerdown', event => { pause(); if (event.button === 0) { suppressClick = false; drag = { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, id: event.pointerId, moved: false }; } });
    viewport.addEventListener('pointermove', event => {
      if (drag) {
        if (!drag.moved && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) <= 4) return;
        drag.moved = true; viewport.setPointerCapture(drag.id); tooltip.hidden = true;
        const c = camera || fitted(); onCamera({ ...c, x: c.x + event.clientX - drag.x, y: c.y + event.clientY - drag.y }); drag.x = event.clientX; drag.y = event.clientY; return;
      }
      const item = event.target.closest('[data-p0-item]'), lane = event.target.closest('[data-p0-lane]'), block = event.target.closest('[data-p0-pick-block]');
      hovered = item ? Number(item.dataset.p0Item) : null; paintSelection(); tooltip.hidden = !item && !lane && !block;
      if (!tooltip.hidden) {
        tooltip.textContent = item ? itemDescription(model.items[hovered]) : lane ? `rope_core ${lane.dataset.p0Lane} · ${model.lanes[Number(lane.dataset.p0Lane)].groups.map(g => `G${g.gIndex}`).join(', ') || '无预处理工作 · 仍参与同步'}` : `Block ${block.dataset.p0PickBlock} · ${model.groupsForBlock(Number(block.dataset.p0PickBlock)).length} 项工作 · 参与 SYNCALL`;
        const r = viewport.getBoundingClientRect(); tooltip.style.left = `${Math.max(8, Math.min(event.clientX - r.left + 12, r.width - tooltip.offsetWidth - 8))}px`; tooltip.style.top = `${Math.max(8, Math.min(event.clientY - r.top + 18, r.height - tooltip.offsetHeight - 8))}px`;
      }
    });
    viewport.addEventListener('pointerleave', () => { tooltip.hidden = true; hovered = null; paintSelection(); });
    const endDrag = () => { if (!drag) return; suppressClick = drag.moved; drag = null; };
    ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(type => viewport.addEventListener(type, endDrag));
    const observer = new ResizeObserver(() => {
      const w = viewport.clientWidth, h = viewport.clientHeight; if (!w || !h || !active) return;
      if (!camera) onCamera(fitted());
      else if (lastSize && (w !== lastSize.w || h !== lastSize.h)) { const ratio = fitted().scale / fitted(lastSize.w, lastSize.h).scale; onCamera({ scale: camera.scale * ratio, x: w / 2 + (camera.x - lastSize.w / 2) * ratio, y: h / 2 + (camera.y - lastSize.h / 2) * ratio }); }
      lastSize = { w, h };
    });
    observer.observe(viewport); paintCamera();
    return { update, hideTooltip: () => { tooltip.hidden = true; }, model, pause, settle: player.settle, get progress() { return player.progress; }, destroy() { player.destroy(); observer.disconnect(); } };
  }
  global.PtoVisualization2Phase0 = Object.freeze({ createModel, createLayout, mount });
})(window);
