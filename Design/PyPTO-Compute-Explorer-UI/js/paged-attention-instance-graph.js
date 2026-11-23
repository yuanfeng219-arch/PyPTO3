/* Task-local unrolled computation graph (Step 1): logical dependencies with AIV lanes merged.
 * Unified visual grammar:
 *   - Tensors (including loop-carried States m/l_sum/o/Rescale): solid card + left semantic role bar.
 *   - Operations: rounded pill/capsule + op icon + unit badge (AIC / AIV ×2).
 *   - Scores column: explicit score_transfer GM Tensor container holding 4 column slices -> 1 QK_READY edge. */
(function (global) {
  'use strict';

  const stages = [
    'Task Inputs', 'QK Matmul', 'Scores · score_transfer', 'Scale & Mask', 'Online Softmax',
    'Probability', 'V Cache', 'PV Matmul', 'Partial Context', 'Context Accumulate',
    'Output Normalize', 'Task Output',
  ];
  const W = 190, H = 48, DX = 248, DY = 62, STATE_H = 46;
  const xAt = column => 28 + column * DX;
  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);

  const PRESET_SEQ_LEN = 1537; // 4 Stacks (3 full + 1 tail) · 13 Pages · demonstrates Slot 0 reuse (S0 -> S3)

  function roleFor(parentId, type) {
    if (['operation', 'composite'].includes(type)) return 'operation';
    if (type === 'state' || ['scores', 'partial'].includes(parentId)) return 'aggregate';
    return 'activation';
  }

  function opIconKey(parentId) {
    if (['qk', 'pv'].includes(parentId)) return 'matmul';
    if (['softmax', 'scale-mask'].includes(parentId)) return 'softmax';
    if (['context-acc', 'output-norm'].includes(parentId)) return 'reduce';
    return null;
  }

  function unitBadge(parentId, type) {
    if (!['operation', 'composite'].includes(type)) return '';
    if (['qk', 'pv'].includes(parentId)) return 'AIC';
    return 'AIV ×2';
  }

  function iconMarkup(parentId) {
    const key = opIconKey(parentId);
    const entry = key && global.PtoOpIcons?.find(item => item.id === key);
    if (!entry) return '';
    return `<span class="viz2-inst-op-icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="${entry.path}"/></svg></span>`;
  }

  function expand(task, pageInstances) {
    const structure = new Map(task.graph.nodes.map(node => [node.id, node]));
    const pipeline = global.PtoPagedAttentionIntro.model.pipelineFor(task.seqLen);
    const dependency = global.PtoPagedAttentionStackDependencyModel.build(pipeline, task.graph.synchronizations);
    const stageById = new Map(dependency.nodes.map(node => [node.id, node]));
    const qkPages = pageInstances(task, 'qk');
    const pvPages = pageInstances(task, 'pv');
    const nodes = new Map(), edges = [], bands = [];
    const taskKey = `task-${task.id}`;
    const qBusX = xAt(0) + W + 24;
    const transferSlots = task.slots?.length || 3;
    const qHeads = task.q?.validShape?.[0] || 5;
    const qBase = task.kvHead * qHeads;
    const qRange = `Q${qBase}–Q${qBase + qHeads - 1}`;

    function add(parentId, key, column, y, title, meta = {}) {
      const parent = structure.get(parentId);
      const id = `${taskKey}/${parentId}/${key}`;
      const type = parent.type;
      const height = meta.height || (type === 'state' ? STATE_H : H);
      const node = {
        id, parentId, title: title || parent.name, column, x: xAt(column), y, height,
        type, role: roleFor(parentId, type), unit: unitBadge(parentId, type),
        isState: type === 'state',
        source: parent.source, upstream: [], downstream: [], ...meta,
      };
      nodes.set(id, node);
      return node;
    }

    function link(from, to, label, kind = 'data', extra = {}) {
      from.downstream.push({ id: to.id, label, kind });
      to.upstream.push({ id: from.id, label, kind });
      edges.push({ from, to, label, kind, ...extra });
    }

    // Header row: Task-wide shared input & initial loop-carried Tensor states (unified Tensor card style)
    const initY = 48;
    const q = add('q', 'input', 0, initY, 'Q Tile', { note: `[${qHeads}, 128] BF16 · ${qRange}` });
    let state = {
      l: add('l', 'init', 3, initY, 'Initial l_sum', { note: '0 · [5, 1] FP32 · Vec', stateKey: 'l' }),
      m: add('m', 'init', 4, initY, 'Initial m', { note: '−3e38 · [5, 1] FP32 · Vec', stateKey: 'm' }),
      o: add('o', 'init', 9, initY, 'Initial o', { note: '0 · [5, 128] FP32 · Vec', stateKey: 'o' }),
    };

    const qkTargets = [];
    let y = initY + H + 26;

    for (const stack of task.stacks) {
      const index = stack.index;
      const key = `s${index}`;
      const produce = stageById.get(`${key}:softmax`);
      const consume = stageById.get(`${key}:accumulate`);
      const fifo = dependency.edges.find(edge => edge.id === `state:rescale:${key}`);
      const pageCount = stack.pages.length;
      const maxPages = task.pagesPerStack || 4;
      const isTail = pageCount < maxPages;
      const reusedFrom = index >= transferSlots ? index - transferSlots : null;
      const top = y;
      const firstPageY = top + 42;
      const rowAt = row => firstPageY + row * DY;
      const midY = pageCount === 1 ? rowAt(0) : Math.round((rowAt(0) + rowAt(pageCount - 1)) / 2);
      const stateY = rowAt(pageCount) + 18;
      const probBusY = rowAt(pageCount - 1) + H + 10;
      const stackMeta = { stack: index, tick: produce.tick, slot: produce.slot };

      // Single score_transfer GM Tensor container for this Stack (holds up to 4 column slices)
      const scoreTop = firstPageY - 22;
      const scoreBottom = rowAt(pageCount - 1) + H + 8;
      const scoreContainer = add('scores', key, 2, scoreTop, 'Scores · score_transfer', {
        ...stackMeta,
        height: scoreBottom - scoreTop,
        isTransferContainer: true,
        transferVar: 'score_transfer',
        note: `GM · Slot ${produce.slot} · [${qHeads}, 512] FP32`,
        slices: stack.pages.map((pageIndex, offset) => ({
          page: pageIndex,
          offset,
          tokens: qkPages[index].pages[offset].tokens,
          colRange: `${offset * 128}:${(offset + 1) * 128}`,
          y: rowAt(offset),
        })),
        maskedCols: isTail ? `${pageCount * 128}:512 (fillpad)` : null,
      });

      const values = [];
      stack.pages.forEach((pageIndex, offset) => {
        const page = qkPages[index].pages[offset];
        const pageKey = `${key}-p${pageIndex}`;
        const tokenLabel = `${page.tokens} tok${page.tokens < task.pageTokens ? ' · 尾页' : ''}`;
        const meta = { ...stackMeta, page: pageIndex, iteration: offset, note: `P${pageIndex} · ${tokenLabel}` };
        const k = add('k-cache', pageKey, 0, rowAt(offset), 'K Cache Page', { ...meta, note: `P${pageIndex} · [128,128] · ${tokenLabel}` });
        const qk = add('qk', pageKey, 1, rowAt(offset), 'QK Matmul', {
          ...meta, api: page.api, symbol: page.sourceSymbol, stage: 'qk',
        });
        link(q, qk, 'Q Tile (Shared)', 'q-bus', { busX: qBusX });
        link(k, qk, 'K Cache Page', 'data', { portOffsetY: 8 });
        // Connects directly into this page's column slice row inside score_transfer container
        link(qk, scoreContainer, `score_transfer[:, ${offset * 128}:${(offset + 1) * 128}]`, 'data', {
          targetY: rowAt(offset) + H / 2,
        });
        qkTargets.push(qk);
        values.push(add('v-cache', pageKey, 6, rowAt(offset), 'V Cache Page', { ...meta, tick: consume.tick, note: `P${pageIndex} · [128,128] · ${tokenLabel}` }));
      });

      // AIV Softmax path (2 AIV Lanes merged: [2, 512] + [3, 512] = [5, 512])
      const aivMeta = { ...stackMeta, note: `AIV0[2,512] + AIV1[3,512]` };
      const mask = add('scale-mask', key, 3, midY, 'Scale & Mask', { ...aivMeta, stage: 'scale-mask' });
      const softmax = add('softmax', key, 4, midY, 'Online Softmax', { ...aivMeta, stage: 'softmax' });
      const probability = add('probability', key, 5, midY, 'Probability', {
        ...stackMeta, transferVar: 'probability_transfer', note: `probability_transfer · Slot ${produce.slot} · [${qHeads},512]`,
      });

      const nextL = add('l', key, 3, stateY, 'Updated l_sum', { ...stackMeta, stateKey: 'l', note: `l[${index}] · [5, 1] FP32 · Vec` });
      const nextM = add('m', key, 4, stateY, 'Updated m', { ...stackMeta, stateKey: 'm', note: `m[${index}] · [5, 1] FP32 · Vec` });
      const rescale = add('rescale', key, 5, stateY, 'Delayed Rescale', {
        ...stackMeta, stateKey: 'rescale', note: `T${produce.tick}→T${consume.tick} · pending0 · Vec`, versions: fifo.versions,
      });

      // Single clean QK_READY edge from stitched score_transfer container into Scale & Mask
      link(scoreContainer, mask, 'score_transfer · QK_READY · [5, 512]', 'data', {
        sourceY: midY + H / 2,
      });
      link(mask, softmax, 'Scaled & masked scores [5, 512]');
      link(state.m, softmax, 'Previous m', 'state', { track: 'm' });
      link(state.l, softmax, 'Previous l_sum', 'state', { track: 'l' });
      link(softmax, probability, 'BF16 probability [5, 512]');
      link(softmax, nextM, 'next_m', 'state', { track: 'm' });
      link(softmax, nextL, 'next_l', 'state', { track: 'l' });
      link(softmax, rescale, 'rescale_after → pending1 → pending0', 'state', { track: 'rescale' });

      // AIC PV serial accumulation chain
      let accumulator = null;
      stack.pages.forEach((pageIndex, offset) => {
        const page = pvPages[index].pages[offset];
        const pv = add('pv', `${key}-p${pageIndex}`, 7, rowAt(offset),
          offset ? 'PV Matmul Acc' : 'PV Matmul', {
            ...stackMeta, tick: consume.tick, page: pageIndex, iteration: offset, stage: 'pv',
            note: `P${pageIndex} · ${page.tokens} tok · ${offset ? 'matmul_acc' : 'matmul'}`,
            api: page.api, symbol: page.sourceSymbol,
          });
        link(values[offset], pv, 'V Cache Page', 'data', { portOffsetY: 8 });
        link(probability, pv, `probability_transfer[:, ${offset * 128}:${(offset + 1) * 128}] · SOFTMAX_READY`, 'prob-bus', {
          busY: probBusY, spineX: xAt(7) - 18,
        });
        if (accumulator) link(accumulator, pv, 'FP32 page accumulator', 'accum');
        accumulator = pv;
      });

      const partial = add('partial', key, 8, midY, 'Partial Context', {
        ...stackMeta, tick: consume.tick, transferVar: 'pv_transfer', note: `pv_transfer · Slot ${produce.slot} · [${qHeads},128]`,
      });
      link(accumulator, partial, 'Final PV accumulator [5, 128]');

      // AIV Context Accumulate (2 AIV Lanes merged: [2, 128] + [3, 128])
      const acc = add('context-acc', key, 9, midY, 'Context Accumulate', {
        ...stackMeta, tick: consume.tick, stage: 'accumulate', note: 'AIV0[2,128] + AIV1[3,128]',
      });
      const nextO = add('o', key, 9, stateY, 'Updated o', {
        ...stackMeta, tick: consume.tick, stateKey: 'o', note: `o[${index}] · [5, 128] FP32 · Vec`,
      });
      link(partial, acc, 'pv_transfer · PV_READY · [5, 128]');
      link(rescale, acc, 'Delayed pending0', 'state', { track: 'rescale-delay', busY: stateY + STATE_H / 2 });
      link(state.o, acc, 'Previous o', 'state', { track: 'o' });
      link(acc, nextO, 'o_next', 'state', { track: 'o' });

      state = { m: nextM, l: nextL, o: nextO };
      const bandBottom = stateY + STATE_H + 16;
      bands.push({
        index, top, height: bandBottom - top, slot: produce.slot, reusedFrom, isTail,
        pages: stack.pages, validTokens: produce.validTokens, produceTick: produce.tick, consumeTick: consume.tick,
      });
      y = bandBottom + 16;
    }

    // Finalize row: Output Normalize & BF16 Output (merged AIV ×2)
    const norm = add('output-norm', 'task', 10, y + 6, 'Output Normalize', {
      tick: task.tickCount, note: 'AIV0[2,128] + AIV1[3,128] → BF16',
    });
    const output = add('output', 'task', 11, y + 6, 'BF16 Output', {
      tick: task.tickCount, note: `out2d · [${qHeads}, 128] BF16 · ${qRange}`,
    });
    link(state.l, norm, 'Final l_sum', 'state', { track: 'final-l' });
    link(state.o, norm, 'Final o', 'state', { track: 'final-o' });
    link(norm, output, 'BF16 · Task output');

    const operations = [...nodes.values()].filter(node => ['operation', 'composite'].includes(node.type)).length;
    return {
      nodes, edges, bands, operations, qBus: { from: q, targets: qkTargets, busX: qBusX },
      width: xAt(stages.length - 1) + W + 40, height: y + H + 36,
    };
  }

  function edgePath(edge) {
    const { from, to, kind } = edge;
    const fH = from.height || H, tH = to.height || H;
    const sy = edge.sourceY != null ? edge.sourceY : from.y + fH / 2;
    const ty = edge.targetY != null ? edge.targetY : to.y + tH / 2;

    if (kind === 'q-bus') {
      const bx = edge.busX;
      const entryY = to.y + tH / 2 - 9;
      return `M${bx} ${from.y + fH / 2} V${entryY} H${to.x}`;
    }
    if (kind === 'accum') {
      const cx = from.x + W / 2;
      return `M${cx} ${from.y + fH} V${to.y}`;
    }
    if (kind === 'prob-bus') {
      const sx = from.x + W, entryY = to.y + tH / 2 - 8;
      return `M${sx} ${sy} H${sx + 16} V${edge.busY} H${edge.spineX} V${entryY} H${to.x}`;
    }
    if (edge.track === 'rescale-delay') {
      const sx = from.x + W, entryY = to.y + tH / 2 + 11, ch = to.x - 18;
      return `M${sx} ${sy} H${ch} V${entryY} H${to.x}`;
    }
    if (edge.track === 'final-l' || edge.track === 'final-o') {
      const sx = from.x + W, ch = to.x - 22, entryY = edge.track === 'final-l' ? to.y + tH / 2 - 8 : to.y + tH / 2 + 8;
      return `M${sx} ${sy} H${sx + 14} V${entryY} H${ch} H${to.x}`;
    }
    if (kind === 'state') {
      if (from.column === to.column) {
        const ch = from.x + W + 14;
        return `M${from.x + W} ${sy} H${ch} V${ty} H${to.x + W}`;
      }
      if (from.column < to.column) {
        const sx = from.x + W, ch = to.x - 20;
        if (edge.track === 'l') {
          const chL = from.x + W + 14;
          return `M${sx} ${sy} H${chL} V${ty + 10} H${to.x}`;
        }
        return `M${sx} ${sy} H${ch} V${ty} H${to.x}`;
      }
      const sx = from.x, tx = to.x + W;
      const ch = sx - 16;
      return `M${sx} ${sy + 10} H${ch} V${ty} H${tx}`;
    }
    const sx = from.x + W, tx = to.x;
    const entryY = ty + (edge.portOffsetY || 0);
    const ch = tx - (DX - W) / 2;
    return `M${sx} ${sy} H${ch} V${entryY} H${tx}`;
  }

  function mount(host, { pageInstances, scenario, source, onFocus, onNavigate, onClose, parentId }) {
    host.className = 'viz2-instance-graph';
    host.innerHTML = `<div class="viz2-instance-toolbar">
      <div class="viz2-instance-toolbar__left">
        <span data-instance-summary></span>
        <button type="button" class="btn btn-ghost btn-sm viz2-preset-btn" data-instance-preset="${PRESET_SEQ_LEN}" title="切换至 4 Stacks · 13 Pages (1537 tokens)，完整观察 Slot 0 复用与尾页分支">4 Stacks / 13 Pages · Slot 0 复用</button>
        <button type="button" class="btn btn-ghost btn-sm" data-instance-reset hidden>恢复原 Task</button>
      </div>
      <div class="viz2-instance-tools">
        <button type="button" class="btn btn-ghost btn-sm" data-instance-zoom="out" aria-label="缩小实例图">−</button>
        <output data-instance-scale aria-label="缩放比例"></output>
        <button type="button" class="btn btn-ghost btn-sm" data-instance-zoom="in" aria-label="放大实例图">+</button>
        <button type="button" class="btn btn-ghost btn-sm" data-instance-fit>Fit to View</button>
      </div>
    </div>
    <div class="viz2-instance-viewport" role="group" tabindex="0" aria-label="静态循环展开计算图，滚轮缩放、拖拽或方向键平移">
      <div class="viz2-instance-scene"></div>
    </div>
    <aside class="pto-ide-frame__pane viz2-graph-inspector" aria-label="计算实例详情" hidden>
      <header class="pto-ide-frame__pane-header"><b data-instance-title></b>
      <button type="button" class="btn btn-ghost btn-sm" data-instance-close aria-label="关闭实例详情">×</button></header>
      <div class="pto-ide-frame__pane-body" data-viz2-graph-detail></div>
    </aside>`;

    const viewport = host.querySelector('.viz2-instance-viewport');
    const scene = host.querySelector('.viz2-instance-scene');
    const summary = host.querySelector('[data-instance-summary]');
    const presetBtn = host.querySelector('[data-instance-preset]');
    const resetBtn = host.querySelector('[data-instance-reset]');
    const scaleReadout = host.querySelector('[data-instance-scale]');
    const inspector = host.querySelector('aside');
    const details = inspector.querySelector('[data-viz2-graph-detail]');
    let task, graph, camera, drag, suppressBlankClick = false, detailKey, fitMode = true, hoverId = null, lastFocus = null, lastOpen = false;
    let nodeElements = new Map();

    function apply(next) {
      camera = next;
      scaleReadout.textContent = `${Math.round(camera.scale * 100)}%`;
      scene.style.transform = `translate(${camera.x}px,${camera.y}px) scale(${camera.scale})`;
    }
    function fit() {
      if (!graph || !viewport.clientWidth || !viewport.clientHeight) return;
      fitMode = true;
      const scale = Math.min(1, (viewport.clientWidth - 28) / graph.width, (viewport.clientHeight - 28) / graph.height);
      apply({ scale: Math.max(scale, 0.05), x: 14, y: 14 });
    }
    function zoom(factor, x, y) {
      if (!camera) return;
      fitMode = false;
      const { scale, x: left, y: top } = camera;
      const next = Math.min(2, Math.max(0.05, scale * factor));
      apply({ scale: next, x: x - (x - left) * next / scale, y: y - (y - top) * next / scale });
    }

    function rebuild() {
      graph = expand(task, pageInstances);
      scene.style.width = `${graph.width}px`;
      scene.style.height = `${graph.height}px`;
      summary.textContent = `循环展开 · ${task.stackCount} Stacks · ${task.pageCount} Pages · ${graph.operations} 计算实例（AIV ×2 合并）`;
      const isPreset = task.stackCount === 4 && task.pageCount === 13;
      presetBtn.classList.toggle('is-selected', isPreset);
      presetBtn.setAttribute('aria-pressed', String(isPreset));
      resetBtn.hidden = !scenario?.get()?.isCustom;

      const qTrunk = graph.qBus.targets.length
        ? (() => {
            const q = graph.qBus.from, last = graph.qBus.targets[graph.qBus.targets.length - 1];
            const sy = q.y + (q.height || H) / 2, ey = last.y + (last.height || H) / 2 - 9, bx = graph.qBus.busX;
            const dots = graph.qBus.targets.map(t => `<circle cx="${bx}" cy="${t.y + (t.height || H) / 2 - 9}" r="2.6" class="viz2-inst-bus-dot"/>`).join('');
            return `<path class="viz2-inst-q-trunk" d="M${q.x + W} ${sy} H${bx} V${ey}"/>${dots}`;
          })()
        : '';

      const bandHtml = graph.bands.map(band => {
        const pageRange = band.pages.length === 1 ? `P${band.pages[0]}` : `P${band.pages[0]}–P${band.pages[band.pages.length - 1]}`;
        const reuseBadge = band.reusedFrom != null
          ? `<span class="viz2-inst-badge viz2-inst-badge--reuse" style="--slot-color:var(--intra-slot-${band.slot % 3})">复用 S${band.reusedFrom}</span>`
          : '';
        const tailBadge = band.isTail
          ? `<span class="viz2-inst-badge viz2-inst-badge--tail">尾页 · tail=${band.pages.length}</span>`
          : '';
        return `<div class="viz2-instance-band" data-stack-band="${band.index}" style="top:${band.top}px;height:${band.height}px">
          <div class="viz2-instance-band__header">
            <strong>Stack ${band.index}</strong>
            <span>${pageRange} · ${band.pages.length} Page${band.pages.length === 1 ? '' : 's'} · ${band.validTokens} tokens</span>
            <span class="viz2-inst-badge viz2-inst-badge--slot" style="--slot-color:var(--intra-slot-${band.slot % 3})">Slot ${band.slot}</span>
            ${reuseBadge}${tailBadge}
            <span class="viz2-instance-band__ticks">源码循环 T${band.produceTick} → T${band.consumeTick}</span>
          </div>
        </div>`;
      }).join('');

      const nodesHtml = [...graph.nodes.values()].map(node => {
        // Explicit score_transfer GM container showing stitched column slices
        if (node.isTransferContainer) {
          const sliceRows = node.slices.map(s => `<span class="viz2-inst-slice" style="top:${s.y - node.y}px;height:${H}px"><b>P${s.page}</b><small>[:, ${s.colRange}] · ${s.tokens} tok</small></span>`).join('');
          const maskedNote = node.maskedCols ? `<span class="viz2-inst-slice-mask">未填列 ${node.maskedCols}</span>` : '';
          return `<button type="button" class="viz2-instance-node viz2-instance-node--transfer" data-instance-id="${esc(node.id)}"
            data-kind="tensor" data-role="aggregate" data-parent="${esc(node.parentId)}"
            style="left:${node.x}px;top:${node.y}px;width:${W}px;height:${node.height}px" aria-pressed="false">
            <span class="viz2-inst-transfer-head"><b>Scores</b><span class="viz2-inst-transfer-var">score_transfer · Slot ${node.slot}</span></span>
            ${sliceRows}${maskedNote}
          </button>`;
        }

        const unitHtml = node.unit ? `<span class="viz2-inst-unit" data-unit="${node.unit === 'AIC' ? 'aic' : 'aiv'}">${node.unit}</span>` : '';
        const stateTag = node.isState ? `<span class="viz2-inst-state-tag" title="跨 Stack 循环携带 Tensor (pl.yield_)">↻ State</span>` : '';
        return `<button type="button" class="viz2-instance-node" data-instance-id="${esc(node.id)}"
          data-kind="${esc(node.type)}" data-role="${esc(node.role)}" data-parent="${esc(node.parentId)}"
          style="left:${node.x}px;top:${node.y}px;width:${W}px;height:${node.height}px" aria-pressed="false">
          <span class="viz2-inst-head">${iconMarkup(node.parentId)}<b>${esc(node.title)}</b>${unitHtml}${stateTag}</span>
          <small>${esc(node.note)}</small>
        </button>`;
      }).join('');

      scene.innerHTML = stages.map((stage, index) => `<b class="viz2-instance-rank" style="left:${xAt(index)}px">${stage}</b>`).join('')
        + bandHtml
        + `<svg class="viz2-instance-edges" width="${graph.width}" height="${graph.height}" aria-hidden="true">
          <defs>
            <marker id="viz2-instance-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="context-stroke"/></marker>
          </defs>
          ${qTrunk}
          ${graph.edges.map(edge => `<path data-kind="${edge.kind}" data-track="${edge.track || ''}" d="${edgePath(edge)}" marker-end="url(#viz2-instance-arrow)"/>`).join('')}
        </svg>`
        + nodesHtml;

      nodeElements = new Map([...scene.querySelectorAll('[data-instance-id]')].map(el => [el.dataset.instanceId, el]));
      const paths = [...scene.querySelectorAll('.viz2-instance-edges > path[data-kind]')];
      graph.edges.forEach((edge, index) => { edge.element = paths[index]; });
    }

    function resolveMatches(focus) {
      if (!graph || !focus) return { exact: null, matches: [] };
      const exact = focus.executionInstanceId ? graph.nodes.get(focus.executionInstanceId) : null;
      if (exact) return { exact, matches: [exact] };
      const all = [...graph.nodes.values()];
      const parent = parentId(focus)?.split('/')[0];
      if (parent && Number.isInteger(focus.instancePage)) {
        const hit = all.find(n => n.parentId === parent && n.page === focus.instancePage);
        if (hit) return { exact: hit, matches: [hit] };
      }
      if (parent && Number.isInteger(focus.stackIndex)) {
        const hits = all.filter(n => n.parentId === parent && n.stack === focus.stackIndex);
        if (hits.length === 1) return { exact: hits[0], matches: hits };
        if (hits.length > 1) return { exact: null, matches: hits };
      }
      if (parent) {
        const hits = all.filter(n => n.parentId === parent);
        if (hits.length === 1) return { exact: hits[0], matches: hits };
        if (hits.length > 1) return { exact: null, matches: hits };
      }
      if (focus.kind === 'stack' && Number.isInteger(focus.index)) {
        return { exact: null, matches: all.filter(n => n.stack === focus.index) };
      }
      if (focus.kind === 'page' && Number.isInteger(focus.index)) {
        return { exact: null, matches: all.filter(n => n.page === focus.index) };
      }
      return { exact: null, matches: [] };
    }

    function paintHighlights(focus) {
      if (!graph) return;
      const { matches } = resolveMatches(focus);
      const selected = new Set(matches.map(n => n.id));
      const activeId = hoverId || (matches.length === 1 ? matches[0].id : null);
      const activeNode = activeId ? graph.nodes.get(activeId) : null;
      const chainStateKey = activeNode?.stateKey || (['softmax', 'context-acc'].includes(activeNode?.parentId) ? (activeNode.parentId === 'softmax' ? 'm' : 'o') : null);

      const relatedIds = new Set(selected);
      if (activeId) relatedIds.add(activeId);

      graph.edges.forEach(edge => {
        const direct = relatedIds.has(edge.from.id) || relatedIds.has(edge.to.id);
        const stateChain = !!chainStateKey && edge.kind === 'state' && (edge.track === chainStateKey || (chainStateKey === 'm' && edge.track === 'l'));
        const on = direct || stateChain;
        edge.element?.classList.toggle('is-related', on);
        edge.element?.classList.toggle('is-state-chain', stateChain && !direct);
        if (direct) {
          relatedIds.add(edge.from.id);
          relatedIds.add(edge.to.id);
        }
      });

      const hasFocus = selected.size > 0 || !!hoverId;
      scene.classList.toggle('has-focus', hasFocus);
      const qTrunk = scene.querySelector('.viz2-inst-q-trunk');
      if (qTrunk) qTrunk.classList.toggle('is-related', relatedIds.has(graph.qBus.from.id));

      nodeElements.forEach((el, id) => {
        el.setAttribute('aria-pressed', String(selected.has(id)));
        el.classList.toggle('is-related', hasFocus && relatedIds.has(id));
      });
    }

    function select(id) {
      const node = graph.nodes.get(id);
      if (!node) return;
      const stageMap = { qk: 'qk', 'scale-mask': 'scale-mask', softmax: 'softmax', pv: 'pv', 'context-acc': 'accumulate' };
      const stage = stageMap[node.parentId];
      onFocus({
        kind: stage ? 'stage' : node.parentId === 'output' || node.parentId === 'output-norm' ? 'output' : 'graph',
        node: node.parentId,
        ...(stage ? { stage } : {}),
        executionInstanceId: node.id,
        stackIndex: node.stack,
        instancePage: node.page,
        index: node.stack ?? 0,
      });
    }

    function renderDetails(focus, open) {
      lastFocus = focus;
      lastOpen = open;
      paintHighlights(focus);
      const { exact, matches } = resolveMatches(focus);
      inspector.hidden = !open || !matches.length;
      if (inspector.hidden) return;
      const parent = exact?.parentId || parentId(focus)?.split('/')[0];
      const key = exact?.id || `${parent}:${focus?.stackIndex ?? ''}:${focus?.index ?? ''}`;
      if (key === detailKey) return;
      detailKey = key;
      const structure = task.graph.nodes.find(node => node.id === parent);
      inspector.querySelector('[data-instance-title]').textContent = exact?.title || structure?.name || `Stack ${focus?.index ?? 0}`;
      const target = (node, label) => `<li><button type="button" class="btn btn-ghost btn-sm" data-instance-target="${esc(node.id)}">${esc(node.title)} · ${esc(node.note)}</button>${label ? `<small>${esc(label)}</small>` : ''}</li>`;
      const list = (title, items) => `<section><h3>${title}</h3>${items.length ? `<ul>${items.join('')}</ul>` : '<p>—</p>'}</section>`;
      const navButton = onNavigate ? `<div class="viz2-inst-actions"><button type="button" class="btn btn-sm" data-instance-to-pipeline>在 Pipeline · Instance 中定位 ↗</button></div>` : '';

      if (!exact) {
        details.innerHTML = navButton + list(`${matches.length} 个展开实例`, matches.map(node => target(node)));
        return;
      }

      const line = exact.source?.ranges?.[0]?.start || null;
      const snippet = line && source
        ? source.split('\n').slice(Math.max(0, line - 2), line + 10).map((l, i) => `${Math.max(1, line - 1) + i}  ${l}`).join('\n')
        : '';
      const rows = [
        ['结构节点', structure?.name],
        ['对象类型', exact.role === 'operation' ? '计算操作 (OP)' : exact.isState ? '循环携带 Tensor (Vec · pl.yield_)' : exact.transferVar ? `GM 中转 Tensor (${exact.transferVar})` : 'Tensor'],
        ['执行单元', exact.unit === 'AIC' ? 'AIC (Cube / MTE2 / FIX)' : exact.unit === 'AIV ×2' ? 'AIV 0 (2/5 行) + AIV 1 (3/5 行) · 合并展示' : null],
        ['Stack / Slot', exact.stack != null ? `Stack ${exact.stack} · Slot ${exact.slot}${exact.stack >= 3 ? ` (复用 S${exact.stack - 3})` : ''}` : null],
        ['Page 迭代', exact.page != null ? `Page ${exact.page} (Stack 内第 ${exact.iteration + 1} 页)` : exact.slices ? `${exact.slices.length} 个列分片写入同一 Slot` : null],
        ['源码循环 Tick', exact.tick != null ? `Tick ${exact.tick}` : null],
        ['Cache 索引', ['k-cache', 'v-cache'].includes(exact.parentId) ? `block_table[${task.batch}, ${exact.page}]` : null],
        ['源码操作', [exact.api, exact.symbol].filter(Boolean).join(' · ')],
      ];
      details.innerHTML = navButton
        + `<section><dl>${rows.filter(([, value]) => value != null && value !== '').map(([label, value]) =>
          `<dt>${esc(label)}</dt><dd>${esc(value)}</dd>`).join('')}</dl></section>`
        + list('输入依赖', exact.upstream.map(({ id, label }) => target(graph.nodes.get(id), label)))
        + list('输出流向', exact.downstream.map(({ id, label }) => target(graph.nodes.get(id), label)))
        + (exact.versions ? list('Rescale FIFO 状态演化', exact.versions.map(version => `<li>${esc(version)}</li>`)) : '')
        + (snippet ? `<details class="viz2-inspector-details"><summary>PyPTO 源码 · L${line}</summary><pre data-compute-code>${esc(snippet)}</pre></details>` : '');
    }

    host.addEventListener('click', event => {
      const button = event.target.closest('button');
      if (!button) {
        if (suppressBlankClick) { suppressBlankClick = false; return; }
        if (event.target.closest('.viz2-instance-viewport') && lastFocus) {
          detailKey = null;
          onFocus(null);
        }
        return;
      }
      if (button.dataset.instancePreset) { scenario?.preset?.(Number(button.dataset.instancePreset)); return; }
      if (button.hasAttribute('data-instance-reset')) { scenario?.reset?.(); return; }
      if (button.hasAttribute('data-instance-to-pipeline')) { onNavigate?.(); return; }
      if (button.hasAttribute('data-instance-fit')) fit();
      if (button.hasAttribute('data-instance-close')) onClose();
      if (button.dataset.instanceZoom) zoom(button.dataset.instanceZoom === 'in' ? 1.28 : 1 / 1.28,
        viewport.clientWidth / 2, viewport.clientHeight / 2);
      if (button.dataset.instanceId) select(button.dataset.instanceId);
      if (button.dataset.instanceTarget) {
        const node = graph.nodes.get(button.dataset.instanceTarget);
        if (!node) return;
        const scale = Math.max(camera?.scale ?? 0, 0.72);
        fitMode = false;
        apply({
          scale,
          x: viewport.clientWidth / 2 - (node.x + W / 2) * scale,
          y: viewport.clientHeight / 2 - (node.y + (node.height || H) / 2) * scale,
        });
        select(node.id);
      }
    });

    scene.addEventListener('pointerover', event => {
      const btn = event.target.closest('[data-instance-id]');
      const next = btn?.dataset.instanceId || null;
      if (next !== hoverId) { hoverId = next; paintHighlights(lastFocus); }
    });
    scene.addEventListener('pointerleave', () => {
      if (hoverId !== null) { hoverId = null; paintHighlights(lastFocus); }
    });

    viewport.addEventListener('wheel', event => {
      event.preventDefault();
      const box = viewport.getBoundingClientRect();
      zoom(Math.exp(-event.deltaY * 0.002), event.clientX - box.left, event.clientY - box.top);
    }, { passive: false });
    viewport.addEventListener('pointerdown', event => {
      if (event.button !== 0 || event.target.closest('button') || !camera) return;
      suppressBlankClick = false;
      drag = { id: event.pointerId, startX: event.clientX, startY: event.clientY, x: event.clientX - camera.x, y: event.clientY - camera.y, moved: false };
      viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener('pointermove', event => {
      if (drag?.id !== event.pointerId) return;
      if (Math.abs(event.clientX - drag.startX) + Math.abs(event.clientY - drag.startY) > 4) drag.moved = true;
      if (!drag.moved) return;
      fitMode = false;
      apply({ ...camera, x: event.clientX - drag.x, y: event.clientY - drag.y });
    });
    viewport.addEventListener('pointerup', () => { if (drag?.moved) suppressBlankClick = true; drag = null; });
    viewport.addEventListener('lostpointercapture', () => { if (drag?.moved) suppressBlankClick = true; drag = null; });
    viewport.addEventListener('keydown', event => {
      if (event.key === 'Escape' && lastFocus) { detailKey = null; onFocus(null); return; }
      if (event.target !== viewport || !camera) return;
      const step = { ArrowLeft: [48, 0], ArrowRight: [-48, 0], ArrowUp: [0, 48], ArrowDown: [0, -48] }[event.key];
      if (!step) return;
      event.preventDefault();
      fitMode = false;
      apply({ ...camera, x: camera.x + step[0], y: camera.y + step[1] });
    });

    function constrainWidth() {
      if (host.hidden) return;
      const available = document.documentElement.clientWidth - host.getBoundingClientRect().left - 16;
      host.style.maxWidth = `${Math.max(120, available)}px`;
    }
    global.addEventListener('resize', constrainWidth);
    const observer = new ResizeObserver(() => { if (!host.hidden && fitMode) fit(); });
    observer.observe(viewport);

    return {
      update(nextTask, focus, open, visible) {
        const entering = host.hidden && visible;
        host.hidden = !visible;
        if (!visible) return;
        constrainWidth();
        if (task !== nextTask) {
          task = nextTask;
          detailKey = null;
          hoverId = null;
          rebuild();
          fit();
        } else if (entering) {
          fit();
        }
        renderDetails(focus, open);
      },
      getTransitionRect: () => viewport.getBoundingClientRect(),
    };
  }

  global.PtoPagedAttentionInstanceGraph = Object.freeze({ expand, mount, PRESET_SEQ_LEN });
})(window);
