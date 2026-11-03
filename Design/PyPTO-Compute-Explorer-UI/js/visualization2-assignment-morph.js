/* Persistent Task identities in a shared KV-column space. Morph changes Y only. */
(function (global) {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const clamp = value => Math.max(0, Math.min(1, value));
  function partitionWidth(width, count, gap = 1.5) {
    const cellWidth = (width - gap * (count - 1)) / count;
    return Array.from({ length: count }, (_, index) => ({ x: index * (cellWidth + gap), width: cellWidth }));
  }
  function sequenceSegments(seqLen, width, count, segmentTokens = 512, gap = 1.5) {
    return partitionWidth(width, count, gap).map((segment, index) => {
      const fraction = clamp((seqLen - index * segmentTokens) / segmentTokens);
      return { ...segment, fraction, fillWidth: segment.width * fraction };
    });
  }
  const svg = (tag, attrs, parent) => {
    const node = document.createElementNS(NS, tag);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, value);
    parent.append(node);
    return node;
  };

  function createLayout({ batch, kvHeads, blocks }) {
    const width = 1250, height = 600, top = 58, bottom = 580;
    const columnPitch = 1048 / kvHeads, frameWidth = columnPitch - 10;
    const columns = Array.from({ length: kvHeads }, (_, kv) => ({ kv, x: 172 + kv * columnPitch, width: frameWidth }));
    const assignments = Array.from({ length: blocks }, () => []);
    for (let id = 0; id < batch * kvHeads; id++) assignments[id % blocks].push(id);
    // Whole KV cycles produce one Block per KV column in each Batch-residue group.
    // For other configurations, each Block gets a full-width band, while Task X
    // still stays in its KV column and modulo ownership remains exact.
    const aligned = blocks % kvHeads === 0;
    const groupCount = aligned ? blocks / kvHeads : blocks;
    const capacities = Array.from({ length: groupCount }, (_, index) => aligned
      ? Math.max(1, Math.ceil((batch - index) / groupCount))
      : Math.max(1, ...columns.map(({ kv }) => assignments[index].filter(id => id % kvHeads === kv).length)));
    const groupGap = 12, header = 24, padding = 7;
    const pitch = Math.min(25, (bottom - top - groupGap * (groupCount - 1) - (header + padding) * groupCount) / capacities.reduce((sum, n) => sum + n, 0));
    const taskHeight = Math.min(22, (bottom - top - 16) / batch - 5, pitch - 3);
    let y = top;
    const groups = capacities.map((capacity, index) => {
      const group = { index, y, height: header + capacity * pitch + padding, capacity,
        label: aligned ? `B%${groupCount}=${index}` : `Task%${blocks}=${index}` };
      y += group.height + groupGap;
      return group;
    });
    const sources = [], targets = [];
    for (let id = 0; id < batch * kvHeads; id++) {
      const batchId = Math.floor(id / kvHeads), kv = id % kvHeads, block = id % blocks;
      const group = groups[aligned ? Math.floor(block / kvHeads) : block];
      const index = aligned ? Math.floor(batchId / groupCount) : assignments[block].filter(task => task % kvHeads === kv).indexOf(id);
      const geometry = { x: columns[kv].x + 7, w: frameWidth - 14, h: taskHeight };
      sources.push({ ...geometry, y: top + 8 + batchId * ((bottom - top - 16) / batch) });
      targets.push({ ...geometry, y: group.y + header + index * pitch });
    }
    const blockRects = assignments.map((tasks, block) => {
      const group = groups[aligned ? Math.floor(block / kvHeads) : block];
      return { x: aligned ? columns[block % kvHeads].x : columns[0].x, y: group.y,
        w: aligned ? frameWidth : columns.at(-1).x + frameWidth - columns[0].x, h: group.height };
    });
    return { width, height, columns, groups, sources, targets, assignments, blockRects, aligned };
  }

  function mount(host, { config, seqLens, onPause, onEnter }) {
    const { batch, kvHeads, blocks, qPerKv } = config;
    const count = batch * kvHeads, layout = createLayout(config);
    host.innerHTML = `<div class="viz2-am-head"><div><h2 data-am-title>Task Generation · ${count} Tasks</h2><p data-am-explain>${batch} Batches × ${kvHeads} KV Heads → ${count} Tasks</p></div><div class="viz2-am-controls"><button type="button" class="btn btn-ghost btn-sm" data-am-play>暂停</button><button type="button" class="btn btn-ghost btn-sm" data-am-replay>重播</button><div class="segmented-control segmented-control-muted"><button type="button" class="btn btn-sm" data-am-state="generation" aria-pressed="true">任务矩阵</button><button type="button" class="btn btn-sm" data-am-state="assignment" aria-pressed="false">Block 分组</button></div></div></div><div class="kf-pa-map__viewport viz2-am-viewport" tabindex="0" role="group" aria-label="固定 KV 列的 Task 重排画布"><svg class="kf-pa-map__svg viz2-am-svg" viewBox="0 0 ${layout.width} ${layout.height}" preserveAspectRatio="xMidYMid meet" aria-label="Batch 行与 KV Head 列的任务关系"><g class="kf-pa-map__scene"><g data-am-columns></g><g data-am-generation></g><g data-am-blocks></g><g data-am-group-labels></g><g data-am-tasks></g><g data-am-block-hits></g></g></svg><div class="kf-pa-map__tooltip" role="tooltip" hidden></div></div><footer class="viz2-am-footer"><p data-viz2-selection aria-live="polite">每个 Task 对应 ${qPerKv} 个 Q Head</p><span>双击 Task 查看内部 · 静态映射，不表示执行时序</span><button type="button" class="btn btn-ghost btn-sm" data-am-enter hidden>查看内部</button><div class="kf-pa-map__tools"><button class="btn btn-ghost btn-sm" type="button" data-am-zoom="0.8" aria-label="缩小 Assignment">−</button><output data-am-scale>100%</output><button class="btn btn-ghost btn-sm" type="button" data-am-zoom="1.25" aria-label="放大 Assignment">＋</button><button class="btn btn-ghost btn-sm" type="button" data-am-fit>适应</button></div></footer>`;
    const viewport = host.querySelector('.viz2-am-viewport'), scene = host.querySelector('.kf-pa-map__scene');
    const generation = host.querySelector('[data-am-generation]'), blockLayer = host.querySelector('[data-am-blocks]');
    const groupLabels = host.querySelector('[data-am-group-labels]'), taskLayer = host.querySelector('[data-am-tasks]');
    const columnLayer = host.querySelector('[data-am-columns]'), hitLayer = host.querySelector('[data-am-block-hits]');
    layout.columns.forEach(({ kv, x, width }) => {
      svg('text', { x: x + width / 2, y: 22, class: 'viz2-am-kv', 'data-am-kv': kv }, columnLayer).textContent = `KV${kv}`;
      svg('text', { x: x + width / 2, y: 39, class: 'ai-q-axis', 'text-anchor': 'middle' }, columnLayer).textContent = `Q${kv*qPerKv}–${(kv+1)*qPerKv-1}`;
    });
    svg('line', { x1: layout.columns[0].x, x2: layout.columns.at(-1).x + layout.columns.at(-1).width, y1: 44, y2: 44, class: 'viz2-am-column-rule' }, columnLayer);
    svg('text', { x: 8, y: 30, class: 'viz2-am-axis' }, generation).textContent = 'BATCH';
    svg('text', { x: 80, y: 30, class: 'viz2-am-axis' }, generation).textContent = 'seq_lens';
    const sourceConstants=global.PTO_PYPA_SEMANTIC_GRAPH.constants;
    const pageTokens=sourceConstants.BLOCK_SIZE.value,pagesPerStack=sourceConstants.STACK_PAGES.value;
    const stackTokens=pageTokens*pagesPerStack,glyphs=global.PtoTaskObjectGlyphs;
    const segments = Math.max(1, Math.ceil(Math.max(...seqLens) / stackTokens));
    const gridSide = 9, gridGap = 2;
    const gridY = (layout.sources[0].h - gridSide) / 2;
    // Shared square size and pitch for Q Heads and Stacks.
    function drawSequence(parent, len, { x, y, width, height }) {
      return sequenceSegments(len, width, segments, stackTokens, 1.4).map((segment,index) => {
        const track=svg('rect', {x:x+segment.x,y,width:segment.width,height,class:'kf-pa-map__seq-track'},parent);
        const fill=svg('rect', {x:x+segment.x,y,width:segment.fillWidth,height,class:'kf-pa-map__seq-fill'},parent);
        if(segment.fraction>0)parent.insertAdjacentHTML('beforeend',glyphs.stack({index,tokens:Math.min(stackTokens,Math.max(0,len-index*stackTokens)),pageTokens,pagesPerStack,x:x+index*(gridSide+gridGap),y:gridY,w:gridSide,h:gridSide}));
        return {track,fill};
      });
    }
    const batchSequences = [];
    for (let b = 0; b < batch; b++) {
      const row = svg('g', { 'data-am-batch': b }, generation), y = layout.sources[b * kvHeads].y, len = seqLens[b] || 0;
      svg('text', { x: 8, y: y + layout.sources[0].h / 2, class: 'viz2-am-batch' }, row).textContent = `B${b}`;
      svg('text', { x: 75, y: y + layout.sources[0].h / 2, class: 'viz2-am-seq' }, row).textContent = len;
      // Keep the sequence snapshot contract used by Anatomy; metadata now follows Batch rows.
      const seq = svg('g', { class: 'kf-pa-map__sequence', 'aria-label': `seq_lens ${len}` }, row);
      batchSequences.push({ node: seq, parts: drawSequence(seq, len, { x: 24, y: layout.sources[0].h - 8, width: layout.sources[0].w - 30, height: 7 }) });
    }
    layout.groups.forEach(group => {
      svg('text', { x: layout.columns[0].x - 14, y: group.y + group.height / 2, 'text-anchor': 'end', class: 'viz2-am-group-label', 'data-am-group': group.index }, groupLabels).textContent = group.label;
    });
    const blockNodes = layout.blockRects.map((r, block) => {
      const group = svg('g', { 'data-pa-block': block, class: 'viz2-am-block', role: 'button', tabindex: '-1', 'aria-label': `SPMD Block ${block}: Tasks ${layout.assignments[block].join(', ')}` }, blockLayer);
      svg('rect', { x: r.x, y: r.y, width: r.w, height: r.h, rx: 2, class: 'viz2-am-block-frame' }, group);
      svg('text', { x: r.x + 7, y: r.y + 15, class: 'viz2-am-block-label' }, group).textContent = `Block ${String(block).padStart(2, '0')} · ${layout.assignments[block].length} Task`;
      return group;
    });
    const qEncodings = [], seqEncodings = [];
    const nodes = layout.sources.map((r, id) => {
      const group = svg('g', { 'data-pa-task': id, 'data-am-kv-column': id % kvHeads, class: 'kf-pa-map__task viz2-am-task', role: 'button', tabindex: id === 0 ? '0' : '-1', 'aria-label': `Task ${id}, Batch ${Math.floor(id / kvHeads)}, KV Head ${id % kvHeads}, Q Heads ${(id % kvHeads) * qPerKv}–${(id % kvHeads + 1) * qPerKv - 1}, Block ${id % blocks}` }, taskLayer);
      svg('rect', { class: 'kf-pa-map__cell', width: r.w, height: r.h, rx: 3 }, group);
      if (global.PtoAIInfraPresentation?.taskGlyph) group.insertAdjacentHTML('beforeend', global.PtoAIInfraPresentation.taskGlyph(id, r.w, r.h));
      const inner = { x: 24, y: 2, width: r.w - 30, height: Math.max(1, r.h - 4) };
      const qLayer = svg('g', { class: 'viz2-am-q-encoding', 'data-am-q-encoding': '', 'aria-hidden': 'true' }, group);
      const qDetails = svg('g', { class: 'ai-q-detail' }, qLayer);
      for (let q = 0; q < qPerKv; q++) qDetails.insertAdjacentHTML('beforeend', glyphs.qHead({ id: (id % kvHeads) * qPerKv + q, x: 24 + (r.w - 30 - (qPerKv*gridSide+(qPerKv-1)*gridGap))/2 + q*(gridSide+gridGap), y: gridY, w: gridSide, h: gridSide }));
      const len = seqLens[Math.floor(id / kvHeads)] || 0;
      const seqLayer = svg('g', { class: 'viz2-am-seq-encoding', 'data-am-task-sequence': '', 'data-am-seq-len': len, 'aria-hidden': 'true' }, group);
      if (id % kvHeads === 0) {
        // The original Batch strip is the KV0 encoding, not an overlay clone.
        // Keeping data-pa-column on its wrapper preserves Anatomy snapshots.
        seqLayer.setAttribute('data-pa-column', Math.floor(id / kvHeads));
        seqLayer.append(batchSequences[Math.floor(id / kvHeads)].node);
      } else drawSequence(seqLayer, len, { ...inner, y: r.h - 8, height: 7 });
      qEncodings.push(qLayer); seqEncodings.push(seqLayer);
      return group;
    });
    const hits = layout.blockRects.map((r, block) => svg('rect', { x: r.x, y: r.y, width: r.w, height: 22, fill: 'transparent', class: 'viz2-am-block-hit', 'data-am-block-hit': block, role: 'button', tabindex: '-1', 'aria-label': `选择 Block ${block}` }, hitLayer));
    let progress = 0, selection = null;
    const play = host.querySelector('[data-am-play]');
    function describe(p) {
      const assignment = p >= .5;
      if (!selection) host.querySelector('[data-viz2-selection]').textContent = assignment
        ? '一个蓝色正方形 = 一个 Stack · 每个 Stack 最多 512 tokens'
        : `一个蓝色正方形 = 一个 Q Head · 每 Task 固定 ${qPerKv} 个 · Batch 侧方格 = Stack`;
      host.querySelector('[data-am-title]').textContent = assignment ? `执行归属 · ${blocks} SPMD Blocks` : `任务矩阵 · ${count} Tasks`;
      host.querySelector('[data-am-explain]').textContent = assignment ? `同一批 Task 按 T % ${blocks} 分组 → 执行 Block；不是执行顺序` : `${batch} Batches × ${kvHeads} KV Heads → ${count} Tasks · 每 Task ${qPerKv} 个 Q Head`;
      host.querySelectorAll('[data-am-state]').forEach(button => {
        const selected = button.dataset.amState === (assignment ? 'assignment' : 'generation');
        button.setAttribute('aria-pressed', String(selected)); button.classList.toggle('is-selected', selected);
      });
    }
    function draw(p) {
      progress = clamp(p);
      // Keep the Block matrix centered without reserving the Batch metadata gutter.
      const offset = -(layout.columns[0].x + (layout.columns.at(-1).x + layout.columns.at(-1).width - layout.columns[0].x) / 2 - layout.width / 2) * progress;
      [columnLayer,generation,blockLayer,groupLabels,taskLayer,hitLayer].forEach(layer=>layer.setAttribute('transform',`translate(${offset} 0)`));
      host.dataset.amProgress = String(progress);
      generation.style.opacity = String(clamp(1 - progress / .2));
      const alpha = clamp((progress - .82) / .18);
      blockLayer.style.opacity = groupLabels.style.opacity = String(alpha);
      blockLayer.style.pointerEvents = alpha > .8 ? '' : 'none';
      hitLayer.style.pointerEvents = alpha > .8 ? '' : 'none';
      hits.forEach(hit => hit.setAttribute('tabindex', alpha > .8 ? '0' : '-1'));
      nodes.forEach((node, id) => {
        const from = layout.sources[id], to = layout.targets[id];
        const y = from.y + (to.y - from.y) * progress;
        // Tasks remain in fixed KV columns. Batch strips separately travel into KV0.
        node.setAttribute('transform', `translate(${from.x} ${y})`);
        qEncodings[id].style.opacity = String(1-progress);
        if (id % kvHeads === 0) {
          const b = Math.floor(id / kvHeads), width = from.w - 30, height = 7;
          const x = (87-from.x-24)*(1-progress);
          const localY = 0;
          seqEncodings[id].style.opacity = '1';
          seqEncodings[id].setAttribute('transform', `translate(${x} ${localY})`);
          sequenceSegments(seqLens[b] || 0, width, segments, stackTokens, 1.4).forEach((segment, i) => {
            const { track, fill } = batchSequences[b].parts[i];
            for (const rect of [track, fill]) {
              rect.setAttribute('x', 24+segment.x); rect.setAttribute('y',from.h-8); rect.setAttribute('height', height);
            }
            track.setAttribute('width', segment.width); fill.setAttribute('width', segment.fillWidth);
          });
        } else seqEncodings[id].style.opacity = String(progress);
      });
      describe(progress);
    }
    const player = global.PtoVisualization2MorphPlayer.create({once: true, draw, onChange: state => { play.textContent = state.playing ? '暂停' : '播放'; play.setAttribute('aria-label', play.textContent); play.dataset.playing = String(state.playing); }});
    function pause() { player.pause(); onPause?.(); }
    const manual = next => player.goTo(next);
    play.addEventListener('click', player.toggle);
    host.querySelector('[data-am-replay]').addEventListener('click', player.replay);
    host.querySelectorAll('[data-am-state]').forEach(button => button.addEventListener('click', () => manual(button.dataset.amState === 'assignment' ? 1 : 0)));
    viewport.addEventListener('pointerover', event => { if (event.target.closest('[data-pa-task],[data-pa-block],[data-am-block-hit]')) pause(); });
    let zoom = 1, panX = 0, panY = 0, drag = null, suppressClick = false;
    const camera = () => { host.dataset.aiDetail=String(zoom>=2.25); scene.setAttribute('transform', `translate(${panX} ${panY}) translate(625 300) scale(${zoom}) translate(-625 -300)`); host.querySelector('[data-am-scale]').textContent = `${Math.round(zoom * 100)}%`; };
    function fit() { zoom = 1; panX = panY = 0; camera(); }
    host.querySelector('[data-am-fit]').addEventListener('click', fit);
    host.querySelectorAll('[data-am-zoom]').forEach(button => button.addEventListener('click', () => { zoom = Math.max(.8, Math.min(3, zoom * Number(button.dataset.amZoom))); camera(); }));
    viewport.addEventListener('wheel', event => {
      event.preventDefault(); pause();
      const next = Math.max(.8, Math.min(3, zoom * Math.exp(-event.deltaY * .001)));
      const task = event.target.closest('[data-pa-task]');
      if (zoom < 2.8 && next >= 2.8 && task && onEnter) { onEnter(Number(task.dataset.paTask)); return; }
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(host.querySelector('.viz2-am-svg').getScreenCTM().inverse());
      const ratio = next / zoom;
      panX = point.x - 625 - ratio * (point.x - 625 - panX);
      panY = point.y - 300 - ratio * (point.y - 300 - panY);
      zoom = next; camera();
    }, { passive: false });
    viewport.addEventListener('pointerdown', event => { pause(); if (event.button === 0 && !event.target.closest('[data-pa-task],[data-am-block-hit]')) drag = { x: event.clientX, y: event.clientY, id: event.pointerId, moved: false }; });
    viewport.addEventListener('pointermove', event => {
      if (!drag) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) {
        drag.moved = true; viewport.setPointerCapture(drag.id);
        const bounds = viewport.getBoundingClientRect(), unit = Math.min(bounds.width / layout.width, bounds.height / layout.height);
        panX += dx / unit; panY += dy / unit; camera(); drag.x = event.clientX; drag.y = event.clientY;
      }
    });
    viewport.addEventListener('pointerup', () => { suppressClick = !!drag?.moved; drag = null; });
    viewport.addEventListener('pointercancel', () => { drag = null; });
    viewport.addEventListener('click', event => { if (suppressClick) { event.stopPropagation(); suppressClick = false; } }, true);
    viewport.addEventListener('keydown', pause);
    new ResizeObserver(() => {
      const r = viewport.getBoundingClientRect(), unit = Math.min(r.width / layout.width, r.height / layout.height);
      if (unit > 0) scene.style.setProperty('--am-label-size', `${Math.max(12, 11 / unit)}px`);
    }).observe(viewport);
    function update(next, visible) {
      selection = next;
      nodes.forEach((node, id) => {
        node.classList.toggle('is-selected', id === selection?.task);
        node.classList.toggle('is-assigned', selection?.block != null && id % blocks === selection.block);
        node.setAttribute('aria-pressed', String(id === selection?.task));
      });
      blockNodes.forEach((node, id) => { node.classList.toggle('is-selected', id === selection?.block); node.setAttribute('aria-pressed', String(id === selection?.block)); });
      host.querySelector('[data-am-enter]').hidden = selection?.kind !== 'task';
      player.setActive(visible);
    }
    draw(0);
    return { update, pause, manual, fit, settle: player.settle, get progress() { return progress; }, get viewport() { return viewport; }, get scene() { return scene; }, get tooltip() { return host.querySelector('[role="tooltip"]'); } };
  }
  global.PtoVisualization2AssignmentMorph = Object.freeze({ mount, createLayout, partitionWidth, sequenceSegments });
})(window);
