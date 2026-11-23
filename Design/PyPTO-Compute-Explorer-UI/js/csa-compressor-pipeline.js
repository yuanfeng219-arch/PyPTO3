/* Compressor Pipeline Explorer:
 *   - Macro (Inter-core): Runtime Trace (72-core real DFX swimlane) <-> Lane Map (SPMD Lane × Iteration matrix)
 *   - Micro (Block Pipeline): Structure Pipeline (logical step graph + Task Boundary) <-> Instance Pipeline (hardware causal swimlane)
 * Mirrors qk_pv Pipeline's exact navigation model and visual grammar. */
(function (global) {
  'use strict';

  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);

  const OP_ICONS = {
    x: 'cache', wkv: 'cache', wgate: 'cache', load: 'cache', extract: 'reshape',
    kv: 'matmul', score: 'matmul', kvOut: 'cache', scoreOut: 'cache',
    positions: 'cache', projection: 'cache', ring: 'cache', boundary: 'gate',
    zero: 'cache', seed: 'softmax', window: 'cache', carry: 'reduce',
    normalize: 'compress', pooled: 'cache', slots: 'cache', guard: 'gate',
    ape: 'add', weights: 'cache', sum: 'reduce', inv: 'norm', nope: 'norm',
    rope: 'rope', normalized: 'cache', cache: 'cache', mix: 'matmul',
    write: 'cache', round: 'reshape', scale: 'reduce', quant: 'compress',
    scaleOut: 'cache', downstream: 'concat',
    'load-x': 'cache', 'load-w': 'cache', matmul: 'matmul', partial: 'cache',
    atomic: 'add', qr: 'cache', i32: 'cache', 'seed-region': 'cache',
    'qr-seeded': 'cache', gamma: 'cache', reduce: 'reduce', rms: 'norm',
    'quant-scale': 'reduce', 'qr-scale': 'cache', 'qr-int8': 'cache',
    'q-i32': 'cache', 'weight-scale': 'cache', dequant: 'reshape',
    'head-rms': 'norm', rotate: 'rope', q: 'attention', 'cos-sin': 'cache',
    indices: 'reshape', interleave: 'rope', params: 'cache',
  };

  function laneCountFor(kernelInfo) {
    return kernelInfo.coreType === 'AIC' ? 24 : 48;
  }

  function blockTotalFor(model, kernelInfo, tokens) {
    const resolvedTokens = tokens ?? model.cases?.[0]?.tokens;
    return model.blockCount ? model.blockCount(kernelInfo, resolvedTokens) : kernelInfo.blockNum;
  }

  function lanePosition(kernelInfo, block) {
    const lanes = laneCountFor(kernelInfo);
    return { lane: block % lanes, iter: Math.floor(block / lanes), laneCount: lanes };
  }

  function laneBlocksFor(model, kernelInfo, lane, tokens) {
    const lanes = laneCountFor(kernelInfo);
    const total = blockTotalFor(model, kernelInfo, tokens);
    const list = [];
    for (let b = lane; b < total; b += lanes) list.push(b);
    return list;
  }

  function blockCategory(kernelInfo, block, model, localView) {
    const shape = model.blockShape(kernelInfo, block, localView);
    if (kernelInfo.type === 'projection') {
      return shape.col < kernelInfo.head
        ? { cls: 'is-original', key: 'front', label: 'Front Half [0, head)' }
        : { cls: 'is-compressed', key: 'back', label: 'Back Half [head, 2×head)' };
    }
    if (kernelInfo.type === 'qr') {
      return shape.split === 0
        ? { cls: 'is-original', key: 'split0', label: 'Split 0 · K [0, 2048)' }
        : { cls: 'is-compressed', key: 'split1', label: 'Split 1 · K [2048, 4096)' };
    }
    const { iter } = lanePosition(kernelInfo, block);
    if (kernelInfo.type === 'qmat') {
      return iter === 0
        ? { cls: 'is-original', key: 'wave0', label: 'Wave 0 (N 0–23)' }
        : iter === 1
          ? { cls: 'is-compressed', key: 'wave1', label: 'Wave 1 (N 24–47)' }
          : { cls: 'is-wave2', key: 'wave2', label: 'Wave 2 尾波 (N 48–63)' };
    }
    if (kernelInfo.type === 'pool' || kernelInfo.type === 'commit' || kernelInfo.type === 'hadamard' || kernelInfo.type === 'kvrope') {
      return iter === 0
        ? { cls: 'is-compressed', key: 'wave0', label: 'Wave 0' }
        : { cls: 'is-wave1', key: 'wave1', label: 'Wave 1 尾波' };
    }
    return { cls: 'is-original', key: 'active', label: 'Active Block' };
  }

  function buildStructureLayout(kernelInfo, block, model, localView) {
    const graph = model.operationModel(kernelInfo, false, localView);
    const shape = model.blockShape(kernelInfo, block, localView);
    const laneOrder = ['Input', 'Control', 'Load', 'Split 0 · Load', 'Split 0 · Cube', 'Split 0 · Write', 'Split 1 · Load', 'Split 1 · Cube', 'Split 1 · Write', 'MTE2', 'MTE1', 'Gather', 'Vector', 'Cube', 'M', 'FIX', 'Write'];
    const rowTitleMap = {
      Input: 'Inputs',
      Control: 'Control / Guard',
      Load: 'MTE2 · Load',
      'Split 0 · Load': 'Split 0 · Load',
      'Split 0 · Cube': 'Split 0 · Cube',
      'Split 0 · Write': 'Split 0 · Write',
      'Split 1 · Load': 'Split 1 · Load',
      'Split 1 · Cube': 'Split 1 · Cube',
      'Split 1 · Write': 'Split 1 · Write',
      MTE2: 'MTE2 · Load',
      MTE1: 'MTE1 · Extract',
      Gather: 'Gather / Ring',
      Vector: 'Vector Compute',
      Cube: 'Cube · Matmul',
      M: 'Cube · Matmul',
      FIX: 'FIX · Store',
      Write: 'Output / Write',
    };
    const seenLanes = new Set(graph.nodes.map(n => n.lane));
    const usedLanes = [
      ...laneOrder.filter(l => seenLanes.has(l)),
      ...[...seenLanes].filter(l => !laneOrder.includes(l)),
    ];
    const colOf = n => (n.layoutColumn != null ? n.layoutColumn : n.rank);
    const colStep = 216;
    const x0 = 162;
    const y0 = 86;

    const laneY = new Map();
    let curY = y0;
    for (const lane of usedLanes) {
      const cols = new Map();
      for (const n of graph.nodes.filter(item => item.lane === lane)) {
        const c = colOf(n);
        const list = cols.get(c) || [];
        list.push(n);
        cols.set(c, list);
      }
      const maxStack = Math.max(1, ...[...cols.values()].map(arr => arr.length));
      laneY.set(lane, { y: curY, stack: maxStack, height: maxStack * 78 + 22 });
      curY += maxStack * 78 + 28;
    }

    const slotCounter = new Map();
    const nodes = graph.nodes.map(n => {
      const c = colOf(n);
      const key = `${n.lane}:${c}`;
      const slot = slotCounter.get(key) || 0;
      slotCounter.set(key, slot + 1);
      const band = laneY.get(n.lane);
      const x = x0 + c * colStep;
      const y = band.y + 8 + slot * 76;
      return {
        ...n,
        x,
        y,
        w: 168,
        h: 56,
        stepIndex: c,
        iconKey: OP_ICONS[n.op] || (n.role === 'activation' ? 'cache' : 'linear'),
      };
    });

    const maxRank = Math.max(0, ...nodes.map(n => n.stepIndex));
    const boundaryX = x0 + (maxRank + 1) * colStep - 12;

    // Downstream Task Boundary nodes grounded in stageEdges
    const downstreamNames = [...new Set(
      model.stageEdges.filter(e => e.from === kernelInfo.name).map(e => e.to)
    )];
    const outputNodes = nodes.filter(n => n.role === 'aggregate');
    const attachOutputs = outputNodes.length ? outputNodes : nodes.filter(n => n.stepIndex === maxRank);
    const downstreamNodes = [];
    const edges = graph.edges.map(e => ({ ...e }));

    if (downstreamNames.length) {
      downstreamNames.forEach((dsName, idx) => {
        const dsKernel = model.kernels.get(dsName);
        const dsId = `downstream:${dsName}`;
        const yBase = attachOutputs.length
          ? Math.min(...attachOutputs.map(n => n.y)) + idx * 80
          : y0 + idx * 80;
        const dsW = Math.max(204, Math.ceil(dsName.length * 7.8 + 48));
        downstreamNodes.push({
          id: dsId,
          op: 'downstream',
          title: dsName,
          detail: dsKernel ? `${dsKernel.coreType} · ${blockTotalFor(model, dsKernel, localView?.tokens)} Blocks` : 'Downstream Kernel',
          lane: 'Downstream',
          role: 'downstream',
          downstreamKernel: dsName,
          lines: dsKernel ? [dsKernel.first, dsKernel.last] : [kernelInfo.first, kernelInfo.last],
          file: dsKernel ? dsKernel.file : kernelInfo.file,
          x: boundaryX + 26,
          y: yBase,
          w: dsW,
          h: 56,
          stepIndex: maxRank + 1,
          iconKey: 'concat',
        });
        for (const out of attachOutputs) {
          edges.push({ from: out.id, to: dsId, type: 'boundary' });
        }
      });
    } else {
      const stageOutLabel = kernelInfo.type === 'post'
        ? 'Q BF16 → Sparse Attention'
        : kernelInfo.type === 'kvrope'
          ? 'KV RoPE Params → KV Proj'
          : kernelInfo.type === 'commit'
            ? 'Persistent State Ring'
            : kernelInfo.branch === 'main'
              ? 'Main Compressed KV Cache'
              : 'Indexer INT8 KV Cache';
      const stageOutDetail = kernelInfo.type === 'post'
        ? '供下游 qk_pv Attention 消费'
        : kernelInfo.type === 'kvrope'
          ? '供下游 KV Proj / Compressor'
          : 'Stage Output · 下游消费';
      const dsId = 'downstream:stage-out';
      const yBase = attachOutputs.length ? attachOutputs[0].y : y0;
      const dsW = Math.max(204, Math.ceil(stageOutLabel.length * 7.8 + 48));
      downstreamNodes.push({
        id: dsId,
        op: 'downstream',
        title: stageOutLabel,
        detail: stageOutDetail,
        lane: 'Downstream',
        role: 'downstream',
        downstreamKernel: null,
        lines: [kernelInfo.first, kernelInfo.last],
        file: kernelInfo.file,
        x: boundaryX + 26,
        y: yBase,
        w: dsW,
        h: 56,
        stepIndex: maxRank + 1,
        iconKey: 'cache',
      });
      for (const out of attachOutputs) {
        edges.push({ from: out.id, to: dsId, type: 'boundary' });
      }
    }

    const allNodes = [...nodes, ...downstreamNodes];
    const maxRight = Math.max(...allNodes.map(n => n.x + n.w));
    const width = Math.max(960, maxRight + 48);
    const height = Math.max(480, curY + 34, ...allNodes.map(n => n.y + n.h + 40));

    return {
      nodes: allNodes,
      edges,
      usedLanes,
      rowTitleMap,
      laneY,
      maxRank,
      colStep,
      x0,
      boundaryX,
      width,
      height,
      shape,
    };
  }

  function create(host, options) {
    const model = options.model || global.PtoCsaCompressorModel;
    const { stage, measured: data, kernels: kernelByName, ticks } = model;
    const stageTitle = model.title || 'Compressor';
    const layoutBranches = model.branches || { main: 'Main', indexer: 'Indexer' };
    const caseOptions = model.cases || [];

    let state = {
      rank: options.rank ?? 0,
      tokens: caseOptions[0]?.tokens,
      tileIndex: 0,
      branch: 'all',
      kernel: options.kernel || model.defaultKernel || 'kv_score_proj',
      filter: 'all', // Runtime Trace kernel/branch filter
      coreId: null,
      sliceId: null,
      level: 'lane', // 'lane' | 'block'
      overviewMode: 'runtime', // 'runtime' | 'lane'
      intraScale: 'instances', // 'structure' | 'instances'
      lane: 0,
      block: 0,
      lastBlock: null,
      selectedLane: null,
      mapGroupFilter: '', // M tile / N Group / Request index
      mapPartFilter: '', // 'front'|'back'|'split0'|'split1'|'wave0'|'wave1'|'wave2'
      structSelection: null, // node id in structure view
      rtZoom: 1,
    };

    let mapCamera = null;
    let mapDrag = null;
    let structCamera = null;
    let structDrag = null;
    let suppressBlankClick = false;
    let currentStructLayout = null;
    let instanceCtrl = null;
    let rtDrag = null;

    const currentRankData = () => data.ranks[String(state.rank)] || data.ranks['0'];
    const currentKernelInfo = () => kernelByName.get(state.kernel) || [...kernelByName.values()][0];
    const localView = () => ({ block: state.block, tokens: state.tokens, tileIndex: state.tileIndex });
    const branchLabelOf = id => layoutBranches[id] || id;
    const blockCountOf = info => blockTotalFor(model, info, state.tokens);

    function setKernel(name) {
      if (!kernelByName.has(name)) return;
      if (state.kernel === name) return;
      state.kernel = name;
      const info = currentKernelInfo();
      if (state.branch !== 'all' && info.branch !== state.branch) state.branch = 'all';
      state.block = 0;
      state.lane = 0;
      state.tileIndex = 0;
      state.lastBlock = null;
      state.selectedLane = null;
      state.mapGroupFilter = '';
      state.mapPartFilter = '';
      state.structSelection = null;
      mapCamera = null;
      structCamera = null;
      options.onKernelChange?.(name, 0);
    }

    function openBlock(blockId, targetScale) {
      const info = currentKernelInfo();
      const total = blockCountOf(info);
      const clamped = Math.max(0, Math.min(total - 1, Number(blockId) || 0));
      const pos = lanePosition(info, clamped);
      state.block = clamped;
      state.lane = pos.lane;
      state.lastBlock = clamped;
      state.level = 'block';
      const sh = model.blockShape(info, clamped, { block: clamped, tokens: state.tokens, tileIndex: state.tileIndex });
      if (sh.tiles && state.tileIndex >= sh.tiles.length) state.tileIndex = 0;
      if (targetScale) state.intraScale = targetScale;
      state.structSelection = null;
      structCamera = null;
      options.onKernelChange?.(info.name, clamped);
      render();
    }

    function openLaneMap(kernelName, laneIdx = null) {
      if (kernelName && kernelByName.has(kernelName)) setKernel(kernelName);
      state.level = 'lane';
      state.overviewMode = 'lane';
      state.selectedLane = laneIdx;
      if (laneIdx != null) state.lane = laneIdx;
      render();
    }

    function openRuntimeTrace(kernelFilter = 'all') {
      state.level = 'lane';
      state.overviewMode = 'runtime';
      state.filter = kernelFilter;
      state.sliceId = null;
      state.coreId = null;
      render();
    }

    function kernelRankStats(name) {
      const k = kernelByName.get(name);
      return k?.ranks?.[String(state.rank)] || k?.ranks?.['0'] || null;
    }

    // ---------- Header / Context Bar (Macro: Runtime Trace <-> Lane Map) ----------
    function macroContextBar() {
      const isRt = state.overviewMode === 'runtime';
      const info = currentKernelInfo();
      const lanes = laneCountFor(info);
      const totalBlocks = blockCountOf(info);
      const maxIter = Math.max(1, Math.ceil(totalBlocks / lanes));
      const branchKernels = stage.kernels.filter(k => state.branch === 'all' || kernelByName.get(k.name).branch === state.branch);

      const modeSwitch = `<div class="segmented-control segmented-control-muted csa-qp-mode-switch" role="group" aria-label="${esc(stageTitle)} 宏观视角">
        <button type="button" class="btn btn-sm${isRt ? ' is-selected' : ''}" data-cmp-overview="runtime" aria-pressed="${isRt}">Runtime Trace</button>
        <button type="button" class="btn btn-sm${!isRt ? ' is-selected' : ''}" data-cmp-overview="lane" aria-pressed="${!isRt}">Lane Map</button>
      </div>`;

      const branchSummary = Object.values(layoutBranches).join(' + ');
      const meta = isRt
        ? `<span class="csa-qp-context-meta">${esc(branchSummary)} · ${stage.kernels.length} Kernels · ${Object.values(currentRankData().kernels).flat().length} 切片</span>`
        : `<span class="csa-qp-context-meta">${esc(info.name)} · ${lanes} ${info.coreType} Lanes × ${maxIter} Iter (${totalBlocks} Blocks)</span>`;

      const rankOpts = [0, 1].map(r => `<option value="${r}"${state.rank === r ? ' selected' : ''}>${r}</option>`).join('');
      const branchOpts = [['all', '全部'], ...Object.entries(layoutBranches)].map(([id, label]) =>
        `<option value="${id}"${state.branch === id ? ' selected' : ''}>${esc(label)}</option>`
      ).join('');
      const casePicker = caseOptions.length && !isRt
        ? `<label>Q 行数 <select data-cmp-case>${caseOptions.map(c =>
            `<option value="${c.tokens}"${c.tokens === state.tokens ? ' selected' : ''}>${esc(c.label)}</option>`
          ).join('')}</select></label>`
        : '';

      let extraTools = '';
      if (isRt) {
        const filterOpts = [
          `<option value="all"${state.filter === 'all' ? ' selected' : ''}>全部 Kernel</option>`,
          ...Object.entries(layoutBranches).map(([id, label]) =>
            `<option value="${id}"${state.filter === id ? ' selected' : ''}>${esc(label)}</option>`
          ),
          ...stage.kernels.map(k => `<option value="${esc(k.name)}"${state.filter === k.name ? ' selected' : ''}>${esc(k.name)}</option>`),
        ].join('');
        extraTools = `<label>显示 <select data-cmp-rt-filter>${filterOpts}</select></label>`
          + (state.coreId != null ? `<button type="button" class="btn btn-ghost btn-sm" data-cmp-rt-all-cores>显示全部核</button>` : '');
      } else {
        const kernelOpts = branchKernels.map(k => {
          const cnt = blockCountOf(kernelByName.get(k.name));
          return `<option value="${esc(k.name)}"${k.name === state.kernel ? ' selected' : ''}>${esc(k.name)} (${cnt})</option>`;
        }).join('');

        let subFilter = '';
        if (info.type === 'projection') {
          const mCount = totalBlocks / model.blockShape(info, 0, localView()).nTiles;
          const mOpts = ticks(mCount).map(m =>
            `<option value="${m}"${state.mapGroupFilter === String(m) ? ' selected' : ''}>M ${model.range(m * model.rows, model.rows)}</option>`
          ).join('');
          subFilter = `<label>M Tile <select data-cmp-map-group><option value="">全部</option>${mOpts}</select></label>
            <label>分区 <select data-cmp-map-part>
              <option value="">全部</option>
              <option value="front"${state.mapPartFilter === 'front' ? ' selected' : ''}>Front Half</option>
              <option value="back"${state.mapPartFilter === 'back' ? ' selected' : ''}>Back Half</option>
            </select></label>`;
        } else if (info.type === 'qr') {
          const gCount = totalBlocks / 2;
          const gOpts = ticks(gCount).map(g =>
            `<option value="${g}"${state.mapGroupFilter === String(g) ? ' selected' : ''}>N Group ${g}</option>`
          ).join('');
          subFilter = `<label>N Group <select data-cmp-map-group><option value="">全部</option>${gOpts}</select></label>
            <label>Split <select data-cmp-map-part>
              <option value="">全部</option>
              <option value="split0"${state.mapPartFilter === 'split0' ? ' selected' : ''}>Split 0</option>
              <option value="split1"${state.mapPartFilter === 'split1' ? ' selected' : ''}>Split 1</option>
            </select></label>`;
        } else if (info.type === 'qmat') {
          const nOpts = ticks(totalBlocks).map(n =>
            `<option value="${n}"${state.mapGroupFilter === String(n) ? ' selected' : ''}>N Group ${n} (H${n})</option>`
          ).join('');
          subFilter = `<label>N Group <select data-cmp-map-group><option value="">全部</option>${nOpts}</select></label>
            <label>波次 <select data-cmp-map-part>
              <option value="">全部</option>
              <option value="wave0"${state.mapPartFilter === 'wave0' ? ' selected' : ''}>Wave 0 (0–23)</option>
              <option value="wave1"${state.mapPartFilter === 'wave1' ? ' selected' : ''}>Wave 1 (24–47)</option>
              <option value="wave2"${state.mapPartFilter === 'wave2' ? ' selected' : ''}>Wave 2 尾波 (48–63)</option>
            </select></label>`;
        } else if (info.type === 'post') {
          const hOpts = ticks(totalBlocks).map(h =>
            `<option value="${h}"${state.mapGroupFilter === String(h) ? ' selected' : ''}>Head Group ${h} (H${h * 4}–${h * 4 + 3})</option>`
          ).join('');
          subFilter = `<label>Head Group <select data-cmp-map-group><option value="">全部</option>${hOpts}</select></label>`;
        } else if (info.type === 'quant' || info.type === 'qrope' || info.type === 'kvrope') {
          const tOpts = ticks(totalBlocks).map(i =>
            `<option value="${i}"${state.mapGroupFilter === String(i) ? ' selected' : ''}>Token Tile ${i}</option>`
          ).join('');
          subFilter = `<label>Token Tile <select data-cmp-map-group><option value="">全部</option>${tOpts}</select></label>`
            + (info.type === 'kvrope'
              ? `<label>波次 <select data-cmp-map-part>
                  <option value="">全部</option>
                  <option value="wave0"${state.mapPartFilter === 'wave0' ? ' selected' : ''}>Wave 0 (0–47)</option>
                  <option value="wave1"${state.mapPartFilter === 'wave1' ? ' selected' : ''}>Wave 1 尾波 (48–63)</option>
                </select></label>`
              : '');
        } else if (info.type === 'seed') {
          subFilter = '';
        } else if (model.requestScoped?.(info)) {
          const rOpts = ticks(totalBlocks).map(r =>
            `<option value="${r}"${state.mapGroupFilter === String(r) ? ' selected' : ''}>Request ${r}</option>`
          ).join('');
          subFilter = `<label>Request <select data-cmp-map-group><option value="">全部</option>${rOpts}</select></label>
            <label>波次 <select data-cmp-map-part>
              <option value="">全部</option>
              <option value="wave0"${state.mapPartFilter === 'wave0' ? ' selected' : ''}>Wave 0 (0–47)</option>
              <option value="wave1"${state.mapPartFilter === 'wave1' ? ' selected' : ''}>Wave 1 尾波 (48–63)</option>
            </select></label>`;
        } else {
          const mOpts = ticks(totalBlocks).map(m =>
            `<option value="${m}"${state.mapGroupFilter === String(m) ? ' selected' : ''}>M ${model.range(m * model.rows, model.rows)}</option>`
          ).join('');
          subFilter = `<label>M Tile <select data-cmp-map-group><option value="">全部</option>${mOpts}</select></label>`;
        }
        extraTools = `<label>分支 <select data-cmp-branch>${branchOpts}</select></label>
          <label>Kernel <select data-cmp-kernel>${kernelOpts}</select></label>${casePicker}${subFilter}`;
      }

      return `<div class="csa-qp-context">
        <b class="csa-qp-context-title">${esc(stageTitle)}</b>
        ${modeSwitch}
        ${meta}
        <div class="csa-qp-context-tools">
          <span class="csa-qp-context-sep" aria-hidden="true">|</span>
          <label>Rank <select data-qp-rank data-cmp-rank>${rankOpts}</select></label>
          ${extraTools}
          <button type="button" class="btn btn-sm" data-cmp-open-causal>单 Block 因果泳道 →</button>
        </div>
      </div>`;
    }

    // ---------- Macro View 1A: Runtime Trace ----------
    function runtimeTraceView() {
      const rankData = currentRankData();
      const slices = Object.entries(rankData.kernels).flatMap(([name, entries]) => entries.map(e => [...e, name]));
      const start = Math.min(...slices.map(s => s[2]));
      const end = Math.max(...slices.map(s => s[2] + s[3]));
      const visible = slices.filter(([, core, , , , name]) =>
        (state.filter === 'all' || state.filter === name || state.filter === kernelByName.get(name)?.branch) &&
        (state.coreId == null || state.coreId === core)
      );
      const toX = time => 74 + (time - start) / Math.max(0.01, end - start) * 1000;

      // Build inspector HTML if slice or core is selected
      let inspector = '';
      if (state.sliceId != null) {
        const owner = Object.keys(rankData.kernels).find(name => rankData.kernels[name].some(s => s[0] === state.sliceId));
        const rec = owner ? rankData.kernels[owner].find(s => s[0] === state.sliceId) : null;
        if (owner && rec) {
          const kInfo = kernelByName.get(owner);
          const st = kernelRankStats(owner);
          const [id, core, time, dur, kdur] = rec;
          inspector = `<aside class="csa-qp-inspector" aria-label="实测切片详情">
            <header><div><b>${esc(owner)}</b><small>事件 #${id} · ${core < 24 ? 'AIC' : 'AIV'} ${core}</small></div>
              <button type="button" class="btn btn-ghost btn-sm" data-cmp-rt-close aria-label="关闭详情">×</button></header>
            <section><h3>切片实测</h3>
              <dl>
                <dt>分支 / 核型</dt><dd>${esc(branchLabelOf(kInfo.branch))} · ${kInfo.coreType}</dd>
                <dt>物理核</dt><dd>${core < 24 ? 'AIC' : 'AIV'} ${core}</dd>
                <dt>起点</dt><dd>${time} µs</dd>
                <dt>切片耗时</dt><dd>${dur} µs</dd>
                <dt>Kernel 耗时</dt><dd>${kdur} µs</dd>
                <dt>Kernel 规模</dt><dd>${blockCountOf(kInfo)} 逻辑 Blocks · ${st ? `中位 ${st.medDurUs} µs` : ''}</dd>
              </dl>
              <p>DFX 切片记录不含逻辑 Block ID；可切换至 Lane Map 查看逻辑 Block 分布，或直接进入单 Block 因果泳道。</p>
            </section>
            <div class="csa-qp-nav-group">
              <button type="button" class="btn btn-sm csa-rt-cta-primary" data-cmp-drill-block="${esc(owner)}">进入单 Block 因果泳道 →</button>
              <button type="button" class="btn btn-ghost btn-sm" data-cmp-drill-lane="${esc(owner)}">查看 ${esc(owner)} Lane Map →</button>
              <button type="button" class="btn btn-ghost btn-sm" data-cmp-source="${esc(kInfo.file)}:${kInfo.first}:${kInfo.last}">${esc(kInfo.file)} · L${kInfo.first}–${kInfo.last} ↗</button>
              <button type="button" class="btn btn-ghost btn-sm" data-cmp-tab="assignment:${esc(owner)}">在 Assignment 中查看</button>
              <button type="button" class="btn btn-ghost btn-sm" data-cmp-tab="graph:${esc(owner)}">在 Graph 中查看</button>
            </div>
          </aside>`;
        }
      } else if (state.coreId != null) {
        const core = state.coreId;
        const hits = visible.filter(s => s[1] === core);
        const byKernel = new Map();
        for (const s of hits) byKernel.set(s[5], (byKernel.get(s[5]) || 0) + 1);
        const rowsHtml = [...byKernel.entries()].map(([name, cnt]) =>
          `<button type="button" class="btn btn-ghost btn-sm" data-cmp-drill-lane="${esc(name)}">${esc(name)} · ${cnt} 切片 →</button>`
        ).join('');
        inspector = `<aside class="csa-qp-inspector" aria-label="物理核详情">
          <header><div><b>${core < 24 ? 'AIC' : 'AIV'} ${core}</b><small>Rank ${state.rank} · ${hits.length} 条实测切片</small></div>
            <button type="button" class="btn btn-ghost btn-sm" data-cmp-rt-close aria-label="关闭详情">×</button></header>
          <section><h3>该物理核执行的 Kernel</h3><div class="csa-qp-relation">${rowsHtml || '当前筛选无切片'}</div></section>
          <div class="csa-qp-nav-group">
            <button type="button" class="btn btn-ghost btn-sm" data-cmp-rt-all-cores>显示所有物理核</button>
          </div>
        </aside>`;
      }

      const activeKernelFilter = kernelByName.has(state.filter) ? state.filter : null;
      const quickBar = `<div class="csa-cmp-rt-strip">
        <span class="csa-cmp-rt-summary">显示 <b>${visible.length}</b> / ${slices.length} 条切片 · Rank ${state.rank} · 横轴合并记录时间（µs）</span>
        <div class="csa-cmp-rt-legend">
          <span><i class="is-aic"></i>AIC (0–23)</span>
          <span><i class="is-aiv"></i>AIV (24–71)</span>
          ${activeKernelFilter ? `<button type="button" class="btn btn-ghost btn-sm" data-cmp-drill-lane="${esc(activeKernelFilter)}">查看 ${esc(activeKernelFilter)} Lane Map →</button>` : ''}
        </div>
      </div>`;

      return `<div class="csa-cmp-rt-body">
        <div class="csa-cmp-rt-main">
          ${quickBar}
          <div class="csa-cmp-rt-viewport" tabindex="0" role="region" aria-label="实测物理 AIC 与 AIV 泳道">
            <div class="csa-compressor-zoom-content" style="--compressor-zoom:${state.rtZoom}">
              <svg class="csa-kvsp-timeline" width="1100" height="914" viewBox="0 0 1100 914" role="group" aria-label="物理 AIC 与 AIV 泳道">
                ${ticks(6).map(i => {
                  const x = 74 + 200 * i;
                  return `<text x="${x}" y="16">${(start + (end - start) * i / 5).toFixed(1)}</text><path d="M${x} 24V902"/>`;
                }).join('')}
                ${ticks(72).map(i => `<g class="csa-compressor-core-label" role="button" tabindex="0" data-cmp-rt-core="${i}" aria-label="物理核 ${i}">
                  <rect x="6" y="${29 + i * 12}" width="60" height="11" rx="2"/><text x="12" y="${39 + i * 12}">${i < 24 ? 'AIC' : 'AIV'} ${i}</text></g><path d="M72 ${42 + i * 12}H1074"/>`).join('')}
                ${visible.map(([id, core, time, duration, , name]) =>
                  `<rect role="button" tabindex="0" data-cmp-rt-slice="${id}" data-unit="${core < 24 ? 'aic' : 'aiv'}" data-highlight="${state.sliceId == null || state.kernel === name}" aria-label="事件 ${id}，${name}，Core ${core}，${time} 微秒" aria-pressed="${state.sliceId === id}" x="${toX(time)}" y="${29 + core * 12}" width="${Math.max(1.5, toX(time + duration) - toX(time))}" height="10" rx="2"><title>#${id} · ${name} · Core ${core} · ${duration} µs</title></rect>`
                ).join('')}
              </svg>
            </div>
          </div>
          <div class="csa-qp-map-footer">
            <span>${esc(rankData.source)} · 点击切片查看详情并下钻 Lane Map / 单 Block 因果泳道</span>
            <div>
              <button type="button" class="btn btn-ghost btn-sm" data-cmp-rt-zoom="out">−</button>
              <output class="csa-cmp-rt-zoom-out">${Math.round(state.rtZoom * 100)}%</output>
              <button type="button" class="btn btn-ghost btn-sm" data-cmp-rt-zoom="in">＋</button>
              <button type="button" class="btn btn-ghost btn-sm" data-cmp-rt-zoom="fit">适应</button>
            </div>
          </div>
        </div>
        ${inspector}
      </div>`;
    }

    // ---------- Macro View 1B: Lane Map (Lane Pipeline) ----------
    function laneMapView() {
      const info = currentKernelInfo();
      const laneCount = laneCountFor(info);
      const totalBlocks = blockCountOf(info);
      const maxIter = Math.max(1, Math.ceil(totalBlocks / laneCount));
      const cell = maxIter <= 4 ? 46 : maxIter <= 12 ? 30 : 21;
      const gap = 4;
      const row = laneCount > 24 ? 19 : 25;
      const cellH = laneCount > 24 ? 14 : 18;
      const left = 84;
      const top = 42;
      const summaryX = left + Math.max(maxIter, 6) * (cell + gap) + 14;
      const width = summaryX + 190;
      const height = top + laneCount * row + 28;
      const view = localView();

      const tickStep = maxIter <= 10 ? 1 : 5;
      const colTicks = ticks(maxIter).map(i =>
        (i % tickStep === 0 || i === maxIter - 1)
          ? `<text x="${left + i * (cell + gap) + cell / 2}" y="26" text-anchor="middle">${i}</text>`
          : ''
      ).join('');

      const rowsHtml = ticks(laneCount).map(laneIdx => {
        const laneBlocks = laneBlocksFor(model, info, laneIdx, state.tokens);
        let cA = 0, cB = 0, cC = 0;
        const cells = laneBlocks.map((blkId, iterIdx) => {
          const shape = model.blockShape(info, blkId, view);
          const cat = blockCategory(info, blkId, model, view);
          if (cat.key === 'front' || cat.key === 'split0' || cat.key === 'wave0' || cat.key === 'active') cA++;
          else if (cat.key === 'wave2') cC++;
          else cB++;

          let related = true;
          if (state.selectedLane != null && state.selectedLane !== laneIdx) related = false;
          if (state.mapGroupFilter !== '') {
            const gVal = Number(state.mapGroupFilter);
            const matchGroup = shape.group != null ? shape.group === gVal : blkId === gVal;
            if (!matchGroup) related = false;
          }
          if (state.mapPartFilter !== '' && cat.key !== state.mapPartFilter) related = false;

          const note = `${info.name} · Block ${blkId} · ${shape.text} · Lane ${String(laneIdx).padStart(2, '0')} / Iter ${iterIdx} · ${cat.label}`;
          return `<rect class="csa-qp-map-cell ${cat.cls}${related ? '' : ' is-dimmed'}${blkId === state.lastBlock ? ' is-last' : ''}" x="${left + iterIdx * (cell + gap)}" y="${top + laneIdx * row}" width="${cell}" height="${cellH}" rx="2" data-cmp-block="${blkId}" role="button" tabindex="0" aria-label="${esc(note)}"><title>${esc(note)}</title></rect>`;
        }).join('');

        const y = top + laneIdx * row;
        const textY = y + Math.round(cellH * 0.78);
        const countLabel = !laneBlocks.length
          ? '0 · Idle'
          : info.type === 'projection'
            ? `${cA} Front · ${cB} Back`
            : info.type === 'qr'
              ? `${cA ? `S0` : ''}${cB ? `S1` : ''} · N${Math.floor(laneBlocks[0] / 2)}`
              : info.type === 'qmat'
                ? `${cA} W0${cB ? ` · ${cB} W1` : ''}${cC ? ` · ${cC} W2` : ''}`
                : (info.type === 'pool' || info.type === 'commit' || info.type === 'hadamard' || info.type === 'kvrope')
                  ? `${cA} W0${cB ? ` · ${cB} W1` : ''}`
                  : info.type === 'post'
                    ? `HG ${laneBlocks[0]} · 4H`
                    : info.type === 'seed'
                      ? '1 Core Group'
                      : `${cA} Tile`;
        const dimmedRow = state.selectedLane != null && state.selectedLane !== laneIdx;

        return `<g class="csa-qp-map-row${dimmedRow ? ' is-dimmed' : ''}">
          <text class="csa-qp-map-lane" x="8" y="${textY}" data-cmp-lane="${laneIdx}" role="button" tabindex="0">${info.coreType === 'AIC' ? 'AIC' : 'AIV'} L${String(laneIdx).padStart(2, '0')}</text>
          ${cells}
          <text class="csa-qp-map-count" x="${summaryX}" y="${textY}">${countLabel}</text>
        </g>`;
      }).join('');

      const st = kernelRankStats(info.name);
      const selBlocks = state.selectedLane != null ? laneBlocksFor(model, info, state.selectedLane, state.tokens) : null;
      const summary = selBlocks
        ? `Lane ${String(state.selectedLane).padStart(2, '0')} · ${selBlocks.length} Blocks · block_idx = ${state.selectedLane} + iter × ${laneCount}${st ? ` · Rank ${state.rank} 实测中位 ${st.medDurUs} µs` : ''}`
        : `${info.name}（${totalBlocks} Blocks）按 block_idx % ${laneCount} 映射至 ${laneCount} 条逻辑 ${info.coreType} Lanes；点击任意 Block 进入 Block Pipeline`;

      const legend = info.type === 'projection'
        ? `<span class="csa-qp-map-legend"><i class="is-original"></i>Front Half [0, ${info.head}) (Slots 0–3) <i class="is-compressed"></i>Back Half [${info.head}, ${info.head * 2}) (Slots 4–7)</span>`
        : info.type === 'qr'
          ? `<span class="csa-qp-map-legend"><i class="is-original"></i>Split 0 · K [0, 2048) <i class="is-compressed"></i>Split 1 · K [2048, 4096) (${totalBlocks}/${laneCount} AIC Lanes 活跃)</span>`
          : info.type === 'qmat'
            ? `<span class="csa-qp-map-legend"><i class="is-original"></i>Wave 0 (0–23) <i class="is-compressed"></i>Wave 1 (24–47) <i class="is-wave2"></i>Wave 2 尾波 (48–63)</span>`
            : (info.type === 'pool' || info.type === 'commit' || info.type === 'hadamard' || info.type === 'kvrope')
              ? `<span class="csa-qp-map-legend"><i class="is-compressed"></i>Wave 0 (主波次) <i class="is-wave1"></i>Wave 1 (尾波 · 产生不均衡尾巴)</span>`
              : `<span class="csa-qp-map-legend"><i class="is-original"></i>Active Block (${totalBlocks}/${laneCount} Lanes 活跃，其余 Idle)</span>`;

      return `<div class="csa-qp-map">
        <div class="csa-qp-map-viewport" tabindex="0" role="group" aria-label="${esc(stageTitle)} Lane Map，拖拽平移，点击 Block 进入 Block Pipeline">
          <svg class="csa-qp-map-svg" viewBox="0 0 ${width} ${height}" data-map-width="${width}" data-map-height="${height}" aria-label="${esc(stageTitle)} SPMD Lane 与 Block 迭代矩阵">
            <text class="csa-qp-map-axis" x="${left}" y="14">Iteration →</text>
            ${colTicks}
            ${rowsHtml}
          </svg>
        </div>
        <div class="csa-qp-map-footer">
          <span>${esc(summary)}</span>
          ${legend}
          <div>
            <button type="button" class="btn btn-ghost btn-sm" data-cmp-map-zoom="0.8">−</button>
            <button type="button" class="btn btn-ghost btn-sm" data-cmp-map-fit>适应</button>
            <button type="button" class="btn btn-ghost btn-sm" data-cmp-map-zoom="1.25">＋</button>
          </div>
        </div>
      </div>`;
    }

    // ---------- Micro View: Block Pipeline (Structure <-> Instance) ----------
    function structureSvgHtml(layout, info) {
      const visual = global.document?.body.classList.contains('csa-visual') && global.PtoAIInfraPresentation?.operationGlyph;
      const selectedId = state.structSelection;
      const byId = new Map(layout.nodes.map(n => [n.id, n]));

      const activeEdges = selectedId
        ? layout.edges.filter(e => e.from === selectedId || e.to === selectedId)
        : [];
      const relatedNodes = new Set();
      if (selectedId) {
        relatedNodes.add(selectedId);
        for (const e of activeEdges) {
          relatedNodes.add(e.from);
          relatedNodes.add(e.to);
        }
      }

      const ticksHtml = Array.from({ length: layout.maxRank + 1 }, (_, i) => {
        const x = layout.x0 + i * layout.colStep;
        return `<g class="csa-qp-tick">
          <rect x="${x - 10}" y="46" width="${layout.colStep - 12}" height="${layout.height - 74}" rx="4"/>
          <text class="csa-qp-tick-label" x="${x + 84}" y="63">${i}</text>
        </g>`;
      }).join('');

      const rowsHtml = layout.usedLanes.map(lane => {
        const band = layout.laneY.get(lane);
        return `<text class="csa-qp-row-label" x="14" y="${band.y + 30}">${esc(layout.rowTitleMap[lane] || lane)}</text>`;
      }).join('');

      const boundaryHtml = `<path class="csa-qp-task-boundary" d="M${layout.boundaryX} 46 V${layout.height - 26}"/>
        <text class="csa-qp-boundary-label" x="${layout.boundaryX + 12}" y="66">Task Boundary</text>
        <text class="csa-qp-boundary-note" x="${layout.boundaryX + 12}" y="82">Downstream Task</text>`;

      const shortMeta = (text, limit = 20) => {
        const s = String(text || '').trim();
        return s.length > limit ? s.slice(0, limit - 1) + '…' : s;
      };

      const edgesHtml = layout.edges.map(e => {
        const from = byId.get(e.from);
        const to = byId.get(e.to);
        if (!from || !to) return '';
        const x1 = from.x + from.w;
        const y1 = from.y + (visual ? 17 : from.h / 2);
        const x2 = to.x;
        const y2 = to.y + (visual ? 17 : to.h / 2);
        const isState = e.type === 'state';
        const isRelated = activeEdges.includes(e);
        const isDimmed = !!selectedId && !isRelated;
        let d;
        if (from.x === to.x) {
          d = `M${x1} ${y1} C${x1 + 26} ${y1} ${x1 + 26} ${y2} ${x1} ${y2}`;
        } else {
          const mid = Math.round((x1 + x2) / 2);
          d = `M${x1} ${y1} H${mid} V${y2} H${x2}`;
        }
        const cls = `${isState ? 'csa-qp-timeline-reuse' : 'csa-qp-timeline-data'}${isRelated ? ' is-related' : ''}${isDimmed ? ' is-dimmed' : ''}`;
        return `<path class="${cls}" data-cmp-from="${esc(e.from)}" data-cmp-to="${esc(e.to)}" d="${d}"${isState ? '' : ' marker-end="url(#cmp-struct-arrow)"'}/>`;
      }).join('');

      const nodesHtml = layout.nodes.map(n => {
        const isSel = selectedId === n.id;
        const isRel = !!selectedId && relatedNodes.has(n.id) && !isSel;
        const isDim = !!selectedId && !relatedNodes.has(n.id);
        const cls = `csa-qp-node${isSel ? ' is-selected' : ''}${isRel ? ' is-related' : ''}${isDim ? ' is-dimmed' : ''}`;
        const slotIdx = n.role === 'downstream' ? 2 : n.role === 'aggregate' ? 1 : 0;
        const slotColor = n.role === 'downstream' ? '#f2c45f' : n.role === 'aggregate' ? 'var(--success)' : 'var(--primary)';
        const meta = shortMeta(n.detail || n.lane, n.w > 190 ? 26 : 19);
        const glyphHtml = visual
          ? global.PtoAIInfraPresentation.operationGlyph('cmp:' + n.id, n.title, n.x, n.y, n.w, n.h, slotIdx, { iconKey: n.iconKey, metadata: esc(meta) })
              .replace(`x="${n.w / 2}" y="48" text-anchor="middle"`, 'x="18" y="48" text-anchor="start"')
          : '';
        const inner = visual
          ? `<rect class="csa-qp-hit" x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}"/>${glyphHtml}`
          : `<rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="8"/>
             <text class="csa-qp-title" x="${n.x + 12}" y="${n.y + 24}">${esc(n.title)}</text>
             <text class="csa-qp-sub" x="${n.x + 12}" y="${n.y + 44}">${esc(meta)}</text>`;
        return `<g class="${cls}" style="--slot-color:${slotColor}" data-cmp-struct-node="${esc(n.id)}" role="button" tabindex="0" aria-label="${esc(n.title)}"><title>${esc([n.title, n.detail, n.condition].filter(Boolean).join(' · '))}</title>${inner}</g>`;
      }).join('');

      return `<svg class="csa-qp-svg" viewBox="0 0 ${layout.width} ${layout.height}" data-selected-operation="${esc(selectedId || '')}" aria-label="Compressor Block Logical Structure Pipeline">
        <defs><marker id="cmp-struct-arrow" viewBox="0 0 8 8" markerUnits="userSpaceOnUse" markerWidth="5" markerHeight="5" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z"/></marker></defs>
        <text class="csa-qp-section" x="${layout.x0}" y="28">Logical Step →</text>
        ${ticksHtml}
        ${rowsHtml}
        ${boundaryHtml}
        ${edgesHtml}
        ${nodesHtml}
      </svg>`;
    }

    function structureInspectorHtml(layout, info) {
      if (!state.structSelection) return '';
      const node = layout.nodes.find(n => n.id === state.structSelection);
      if (!node) return '';
      const byId = new Map(layout.nodes.map(n => [n.id, n]));
      const upstream = layout.edges.filter(e => e.to === node.id).map(e => byId.get(e.from)).filter(Boolean);
      const downstream = layout.edges.filter(e => e.from === node.id).map(e => byId.get(e.to)).filter(Boolean);
      const renderLinks = (title, items) => items.length
        ? `<div class="csa-qp-relation"><b>${title}</b>${items.map(peer =>
            `<button type="button" class="btn btn-ghost btn-sm" data-cmp-struct-jump="${esc(peer.id)}">${esc(peer.title)} ↗</button>`
          ).join('')}</div>`
        : '';
      const file = node.file || info.file;
      const lines = node.lines || [info.first, info.last];
      const irOps = (data.ir[info.name] || []).filter(([, , py]) => py >= lines[0] && py <= lines[1]).slice(0, 6);
      const irHtml = irOps.length
        ? `<section><h3>PTO IR 指令（节选）</h3><ul class="csa-cmp-ir-mini">${irOps.map(([op, l, py]) => `<li><code>${esc(op)}</code><span>IR L${l} · Py L${py}</span></li>`).join('')}</ul></section>`
        : '';

      return `<header>
        <div><b>${esc(node.title)}</b><small>${esc(info.name)} · Block ${state.block} · Step ${node.stepIndex}</small></div>
        <button type="button" class="btn btn-ghost btn-sm" data-cmp-struct-close aria-label="关闭详情">×</button>
      </header>
      <section><h3>Execution</h3><p>${esc(node.detail || node.lane)}${node.condition ? ` · 条件：${esc(node.condition)}` : ''}</p></section>
      <section><h3>Dependencies</h3>
        ${renderLinks('Upstream', upstream)}
        ${renderLinks('Downstream', downstream)}
        ${!upstream.length && !downstream.length ? '<p>Block 边界</p>' : ''}
      </section>
      ${irHtml}
      <section><h3>Source Evidence</h3>
        <button type="button" class="btn btn-ghost btn-sm" data-cmp-source="${esc(file)}:${lines[0]}:${lines[1]}">${esc(file)} · L${lines[0]}–${lines[1]} ↗</button>
        <p>计算依赖来自源码静态分析；逻辑 Step 为展示序，硬件调度请切换至 Instance 因果泳道。</p>
      </section>
      <div class="csa-qp-nav-group">
        ${node.downstreamKernel ? `<button type="button" class="btn btn-sm" data-cmp-goto-kernel="${esc(node.downstreamKernel)}">进入下游 ${esc(node.downstreamKernel)} →</button>` : ''}
        <button type="button" class="btn btn-ghost btn-sm" data-cmp-scale="instances">切换到 Instance 硬件泳道 →</button>
        <button type="button" class="btn btn-ghost btn-sm" data-cmp-tab="assignment:${esc(info.name)}">在 Assignment 中查看</button>
        <button type="button" class="btn btn-ghost btn-sm" data-cmp-tab="graph:${esc(info.name)}">在 Graph 中查看</button>
      </div>`;
    }

    function blockView() {
      const info = currentKernelInfo();
      const view = localView();
      const totalBlocks = blockCountOf(info);
      const shape = model.blockShape(info, state.block, view);
      const pos = lanePosition(info, state.block);
      const laneBlocks = laneBlocksFor(model, info, pos.lane, state.tokens);
      const laneIdxPos = Math.max(0, laneBlocks.indexOf(state.block));
      const st = kernelRankStats(info.name);
      const showInstance = state.intraScale === 'instances' && !!global.PtoCsaCompressorPipelineInstance;

      const scaleToggle = `<div class="segmented-control segmented-control-muted viz2-pipeline-scale csa-scale-bar" role="group" aria-label="Pipeline 观察尺度">
        <button type="button" class="btn btn-sm" data-cmp-scale="runtime" aria-pressed="false">Runtime Trace</button>
        <button type="button" class="btn btn-sm${state.intraScale === 'structure' ? ' is-selected' : ''}" data-cmp-scale="structure" aria-pressed="${state.intraScale === 'structure'}">Structure</button>
        <button type="button" class="btn btn-sm${state.intraScale === 'instances' ? ' is-selected' : ''}" data-cmp-scale="instances" aria-pressed="${state.intraScale === 'instances'}">Instance</button>
      </div>`;

      const kernelOpts = stage.kernels.map(k =>
        `<option value="${esc(k.name)}"${k.name === info.name ? ' selected' : ''}>${esc(k.name)}</option>`
      ).join('');
      const blockOpts = laneBlocks.map((b, i) =>
        `<option value="${b}"${b === state.block ? ' selected' : ''}>Lane ${String(pos.lane).padStart(2, '0')} · Block ${b} (Iter ${i})</option>`
      ).join('');
      const casePicker = caseOptions.length
        ? `<label class="csa-cmp-inline-pick">Q 行数 <select data-cmp-case>${caseOptions.map(c =>
            `<option value="${c.tokens}"${c.tokens === state.tokens ? ' selected' : ''}>${esc(c.label)}</option>`
          ).join('')}</select></label>`
        : '';
      const tilePicker = shape.tiles && shape.tiles.length > 1
        ? `<label class="csa-cmp-inline-pick">M Tile <select data-cmp-tile>${shape.tiles.map((m, i) =>
            `<option value="${i}"${i === state.tileIndex ? ' selected' : ''}>${esc(m.path)} ${model.range(m.start, m.valid)}</option>`
          ).join('')}</select></label>`
        : '';

      const stepper = `<div class="csa-qp-item-stepper">
        <button type="button" class="btn btn-ghost btn-sm" data-cmp-prev aria-label="同 Lane 上一个 Block" ${laneIdxPos <= 0 ? 'disabled' : ''}>←</button>
        <output>${laneIdxPos + 1} / ${laneBlocks.length}</output>
        <button type="button" class="btn btn-ghost btn-sm" data-cmp-next aria-label="同 Lane 下一个 Block" ${laneIdxPos >= laneBlocks.length - 1 ? 'disabled' : ''}>→</button>
      </div>`;

      currentStructLayout = showInstance ? null : buildStructureLayout(info, state.block, model, view);

      const depSummary = (() => {
        if (!currentStructLayout || !state.structSelection) return '';
        const node = currentStructLayout.nodes.find(n => n.id === state.structSelection);
        if (!node) return '';
        const up = currentStructLayout.edges.filter(e => e.to === node.id).map(e => currentStructLayout.nodes.find(x => x.id === e.from)?.title).filter(Boolean);
        const down = currentStructLayout.edges.filter(e => e.from === node.id).map(e => currentStructLayout.nodes.find(x => x.id === e.to)?.title).filter(Boolean);
        return `Block ${state.block} · ${node.title} | ↑ ${up.join(', ') || '输入'} | ↓ ${down.join(', ') || '输出'}`;
      })();

      return `<div class="csa-qp-item-top">
        <div class="csa-qp-item-top-left">
          ${scaleToggle}
          <b>${esc(info.name)} · Block ${state.block}</b>
          <span class="csa-qp-item-meta" title="${esc(shape.text)}">${esc(branchLabelOf(info.branch))} · ${info.coreType} · Lane ${String(pos.lane).padStart(2, '0')} / Iter ${pos.iter}</span>
        </div>
        <div class="csa-qp-item-top-right">
          <label class="csa-cmp-inline-pick">Kernel <select data-cmp-kernel>${kernelOpts}</select></label>
          ${casePicker}
          ${tilePicker}
          <label class="csa-cmp-inline-pick"><select data-cmp-select-block aria-label="当前 Lane 的 Block 列表">${blockOpts}</select></label>
          ${stepper}
          <button type="button" class="btn btn-ghost btn-sm" data-cmp-back-lane>← 返回 Lane Map</button>
          <details class="csa-qp-item-details">
            <summary>Block 详情</summary>
            <div class="csa-qp-item-details-body">
              <dl>
                <dt>Rank</dt><dd>${state.rank}</dd>
                <dt>分支</dt><dd>${esc(branchLabelOf(info.branch))}</dd>
                <dt>Kernel</dt><dd>${esc(info.name)}</dd>
                <dt>核型</dt><dd>${esc(info.coreType)}</dd>
                <dt>Block ID</dt><dd>${state.block} / ${totalBlocks}</dd>
                <dt>逻辑归属</dt><dd>${esc(shape.text)}</dd>
                <dt>Lane / Iter</dt><dd>Lane ${String(pos.lane).padStart(2, '0')} · Iter ${pos.iter}</dd>
                <dt>实测中位耗时</dt><dd>${st ? `${st.medDurUs} µs` : '—'}</dd>
                <dt>实测耗时区间</dt><dd>${st ? `${st.minDurUs}–${st.maxDurUs} µs` : '—'}</dd>
              </dl>
              <button type="button" class="btn btn-ghost btn-sm" data-cmp-tab="assignment:${esc(info.name)}">在 Assignment 中查看 ↗</button>
              <button type="button" class="btn btn-ghost btn-sm" data-cmp-tab="graph:${esc(info.name)}">在 Graph 中查看 ↗</button>
              <button type="button" class="btn btn-ghost btn-sm" data-cmp-source="${esc(info.file)}:${info.first}:${info.last}">查看源码 · L${info.first}–${info.last} ↗</button>
            </div>
          </details>
        </div>
      </div>
      <div class="csa-qp-stage"${showInstance ? ' hidden' : ''}>
        <div class="csa-qp-stage-inner" tabindex="0" role="group" aria-label="Structure Pipeline 画布，点击节点查看依赖，拖拽平移，滚轮缩放">
          ${currentStructLayout ? structureSvgHtml(currentStructLayout, info) : ''}
        </div>
        <aside class="csa-qp-inspector" aria-label="计算节点详情"${state.structSelection ? '' : ' hidden'}>
          ${currentStructLayout ? structureInspectorHtml(currentStructLayout, info) : ''}
        </aside>
      </div>
      <div class="csa-qp-instance-host"${showInstance ? '' : ' hidden'}></div>
      <div class="csa-qp-bottom"${showInstance ? ' hidden' : ''}>
        <span>实线：数据依赖 · 虚线：状态递归 · 竖线：Task Boundary <small data-cmp-dep-summary aria-live="polite">${esc(depSummary)}</small></span>
        <span>Source-derived logical order · 硬件流水请切换 Instance 视角</span>
        <div class="csa-qp-footer-tools">
          <button type="button" class="btn btn-ghost btn-sm" data-cmp-prev aria-label="上一个 Block" ${laneIdxPos <= 0 ? 'disabled' : ''}>←</button>
          <output>${laneIdxPos + 1} / ${laneBlocks.length}</output>
          <button type="button" class="btn btn-ghost btn-sm" data-cmp-next aria-label="下一个 Block" ${laneIdxPos >= laneBlocks.length - 1 ? 'disabled' : ''}>→</button>
          <button type="button" class="btn btn-ghost btn-sm" data-cmp-struct-zoom="0.8">−</button>
          <output data-cmp-struct-scale>100%</output>
          <button type="button" class="btn btn-ghost btn-sm" data-cmp-struct-zoom="1.25">＋</button>
          <button type="button" class="btn btn-ghost btn-sm" data-cmp-struct-fit>适应</button>
        </div>
      </div>`;
    }

    // ---------- Camera helpers for Lane Map & Structure Pipeline ----------
    function applyMapCamera() {
      const svg = host.querySelector('.csa-qp-map-svg');
      if (!svg) return;
      const fullW = Number(svg.dataset.mapWidth);
      const fullH = Number(svg.dataset.mapHeight);
      if (!mapCamera) mapCamera = { x: 0, y: 0, width: fullW, height: fullH };
      mapCamera.width = Math.min(fullW, mapCamera.width);
      mapCamera.height = Math.min(fullH, mapCamera.height);
      mapCamera.x = Math.max(0, Math.min(fullW - mapCamera.width, mapCamera.x));
      mapCamera.y = Math.max(0, Math.min(fullH - mapCamera.height, mapCamera.y));
      svg.setAttribute('viewBox', [mapCamera.x, mapCamera.y, mapCamera.width, mapCamera.height].join(' '));
    }

    function zoomMap(factor) {
      const svg = host.querySelector('.csa-qp-map-svg');
      if (!svg) return;
      applyMapCamera();
      const fullW = Number(svg.dataset.mapWidth);
      const fullH = Number(svg.dataset.mapHeight);
      const scale = Math.max(1, Math.min(6, fullW / mapCamera.width * factor));
      const cx = mapCamera.x + mapCamera.width / 2;
      const cy = mapCamera.y + mapCamera.height / 2;
      mapCamera.width = fullW / scale;
      mapCamera.height = fullH / scale;
      mapCamera.x = cx - mapCamera.width / 2;
      mapCamera.y = cy - mapCamera.height / 2;
      applyMapCamera();
    }

    function applyStructCamera() {
      const svg = host.querySelector('.csa-qp-svg');
      const view = host.querySelector('.csa-qp-stage-inner');
      if (!svg || !view || !structCamera) return;
      svg.setAttribute('viewBox', [structCamera.x, structCamera.y, structCamera.width, structCamera.height].join(' '));
      const out = host.querySelector('[data-cmp-struct-scale]');
      if (out && view.clientWidth) out.textContent = Math.round(view.clientWidth / structCamera.width * 100) + '%';
    }

    function fitStructCamera() {
      const view = host.querySelector('.csa-qp-stage-inner');
      if (!view || !currentStructLayout || view.clientWidth < 1 || view.clientHeight < 1) return;
      const pad = 28;
      const bw = currentStructLayout.width;
      const bh = currentStructLayout.height;
      const scale = Math.min((view.clientWidth - pad * 2) / bw, (view.clientHeight - pad * 2) / bh);
      if (!(scale > 0)) return;
      structCamera = {
        x: (bw - view.clientWidth / scale) / 2,
        y: (bh - view.clientHeight / scale) / 2,
        width: view.clientWidth / scale,
        height: view.clientHeight / scale,
        mode: 'fit',
      };
      applyStructCamera();
    }

    function zoomStructCamera(factor, clientX, clientY) {
      const view = host.querySelector('.csa-qp-stage-inner');
      if (!view || !structCamera || !currentStructLayout) return;
      const rect = view.getBoundingClientRect();
      const px = clientX == null ? 0.5 : Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      const py = clientY == null ? 0.5 : Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
      const curScale = rect.width / structCamera.width;
      const nextScale = Math.max(0.25, Math.min(4, curScale * factor));
      const w = rect.width / nextScale;
      const h = rect.height / nextScale;
      structCamera = {
        x: structCamera.x + structCamera.width * px - w * px,
        y: structCamera.y + structCamera.height * py - h * py,
        width: w,
        height: h,
        mode: 'manual',
      };
      applyStructCamera();
    }

    function fitRtZoom() {
      const view = host.querySelector('.csa-cmp-rt-viewport');
      if (!view || !view.clientWidth || !view.clientHeight) return;
      const sx = (view.clientWidth - 28) / 1100;
      const sy = (view.clientHeight - 28) / 914;
      state.rtZoom = Math.max(0.25, Math.min(1, Math.min(sx, sy)));
      const content = view.querySelector('.csa-compressor-zoom-content');
      if (content) content.style.setProperty('--compressor-zoom', state.rtZoom);
      const out = host.querySelector('.csa-cmp-rt-zoom-out');
      if (out) out.textContent = `${Math.round(state.rtZoom * 100)}%`;
    }

    // ---------- Main Render ----------
    function render() {
      const isBlock = state.level === 'block';
      const isRuntime = !isBlock && state.overviewMode === 'runtime';
      host.innerHTML = `<div class="csa-qp-shell csa-cmp-pipeline-shell">
        ${isBlock ? blockView() : macroContextBar() + (isRuntime ? runtimeTraceView() : laneMapView())}
      </div>`;
      host.dataset.level = isBlock ? 'block' : isRuntime ? 'runtime' : 'lane';

      if (isBlock) {
        const instHost = host.querySelector('.csa-qp-instance-host');
        if (state.intraScale === 'instances' && instHost && global.PtoCsaCompressorPipelineInstance) {
          const info = currentKernelInfo();
          const pos = lanePosition(info, state.block);
          instanceCtrl = global.PtoCsaCompressorPipelineInstance.mount(instHost, {
            onSource: options.onSource,
          });
          instanceCtrl.update({
            model,
            kernelInfo: info,
            block: state.block,
            rank: state.rank,
            lane: pos.lane,
            iter: pos.iter,
            localView: localView(),
          });
        } else {
          instanceCtrl = null;
          requestAnimationFrame(() => {
            if (structCamera?.mode === 'manual') applyStructCamera();
            else fitStructCamera();
          });
        }
      } else if (isRuntime) {
        instanceCtrl = null;
        requestAnimationFrame(() => {
          if (state.rtZoom === 1) fitRtZoom();
        });
      } else {
        instanceCtrl = null;
        applyMapCamera();
      }
    }

    // ---------- Events ----------
    host.addEventListener('click', event => {
      const target = event.target.closest('button,[data-cmp-block],[data-cmp-lane],[data-cmp-rt-slice],[data-cmp-rt-core],[data-cmp-struct-node]');
      if (!target) {
        if (suppressBlankClick) { suppressBlankClick = false; return; }
        if (event.target.closest('.csa-qp-stage-inner') && state.structSelection) {
          state.structSelection = null;
          render();
        }
        return;
      }

      if (target.dataset.cmpOverview) {
        const next = target.dataset.cmpOverview;
        if (next !== state.overviewMode) {
          state.overviewMode = next;
          if (next === 'lane' && kernelByName.has(state.filter)) setKernel(state.filter);
          render();
        }
        return;
      }
      if (target.hasAttribute('data-cmp-open-causal')) {
        if (state.overviewMode === 'runtime' && kernelByName.has(state.filter)) setKernel(state.filter);
        const info = currentKernelInfo();
        const laneTarget = state.selectedLane ?? state.lane;
        const blocks = laneBlocksFor(model, info, laneTarget, state.tokens);
        const blk = blocks[0] ?? 0;
        openBlock(blk, 'instances');
        return;
      }
      if (target.dataset.cmpScale) {
        const next = target.dataset.cmpScale;
        if (next === 'runtime') {
          openRuntimeTrace('all');
          return;
        }
        if (next !== state.intraScale) {
          state.intraScale = next;
          structCamera = null;
          render();
        }
        return;
      }
      if (target.hasAttribute('data-cmp-back-lane')) {
        openLaneMap(state.kernel, state.lane);
        return;
      }
      if (target.dataset.cmpDrillLane) {
        openLaneMap(target.dataset.cmpDrillLane, null);
        return;
      }
      if (target.dataset.cmpDrillBlock) {
        setKernel(target.dataset.cmpDrillBlock);
        openBlock(0, 'instances');
        return;
      }
      if (target.dataset.cmpGotoKernel) {
        setKernel(target.dataset.cmpGotoKernel);
        openBlock(0, state.intraScale);
        return;
      }
      if (target.dataset.cmpBlock != null) {
        openBlock(Number(target.dataset.cmpBlock));
        return;
      }
      if (target.dataset.cmpLane != null) {
        const clicked = Number(target.dataset.cmpLane);
        state.selectedLane = state.selectedLane === clicked ? null : clicked;
        if (state.selectedLane != null) state.lane = state.selectedLane;
        render();
        return;
      }
      if (target.hasAttribute('data-cmp-prev') || target.hasAttribute('data-cmp-next')) {
        const info = currentKernelInfo();
        const pos = lanePosition(info, state.block);
        const laneBlocks = laneBlocksFor(model, info, pos.lane, state.tokens);
        const idx = laneBlocks.indexOf(state.block);
        const nextIdx = idx + (target.hasAttribute('data-cmp-next') ? 1 : -1);
        if (nextIdx >= 0 && nextIdx < laneBlocks.length) openBlock(laneBlocks[nextIdx]);
        return;
      }
      if (target.dataset.cmpMapZoom) {
        zoomMap(Number(target.dataset.cmpMapZoom));
        return;
      }
      if (target.hasAttribute('data-cmp-map-fit')) {
        mapCamera = null;
        applyMapCamera();
        return;
      }
      if (target.dataset.cmpRtSlice != null) {
        const id = Number(target.dataset.cmpRtSlice);
        const rankData = currentRankData();
        const owner = Object.keys(rankData.kernels).find(name => rankData.kernels[name].some(s => s[0] === id));
        if (owner) setKernel(owner);
        state.sliceId = state.sliceId === id ? null : id;
        state.coreId = null;
        render();
        return;
      }
      if (target.dataset.cmpRtCore != null) {
        const c = Number(target.dataset.cmpRtCore);
        state.coreId = state.coreId === c ? null : c;
        state.sliceId = null;
        render();
        return;
      }
      if (target.hasAttribute('data-cmp-rt-close')) {
        state.sliceId = null;
        state.coreId = null;
        render();
        return;
      }
      if (target.hasAttribute('data-cmp-rt-all-cores')) {
        state.coreId = null;
        render();
        return;
      }
      if (target.dataset.cmpRtZoom) {
        const dir = target.dataset.cmpRtZoom;
        if (dir === 'fit') { fitRtZoom(); return; }
        state.rtZoom = Math.max(0.25, Math.min(2.4, state.rtZoom * (dir === 'in' ? 1.25 : 0.8)));
        const content = host.querySelector('.csa-cmp-rt-viewport .csa-compressor-zoom-content');
        if (content) content.style.setProperty('--compressor-zoom', state.rtZoom);
        const out = host.querySelector('.csa-cmp-rt-zoom-out');
        if (out) out.textContent = `${Math.round(state.rtZoom * 100)}%`;
        return;
      }
      if (target.dataset.cmpStructNode) {
        const id = target.dataset.cmpStructNode;
        const node = currentStructLayout?.nodes.find(n => n.id === id);
        if (node?.downstreamKernel && state.structSelection === id) {
          setKernel(node.downstreamKernel);
          openBlock(0, state.intraScale);
          return;
        }
        state.structSelection = state.structSelection === id ? null : id;
        render();
        return;
      }
      if (target.dataset.cmpStructJump) {
        state.structSelection = target.dataset.cmpStructJump;
        render();
        return;
      }
      if (target.hasAttribute('data-cmp-struct-close')) {
        state.structSelection = null;
        render();
        return;
      }
      if (target.dataset.cmpStructZoom) {
        zoomStructCamera(Number(target.dataset.cmpStructZoom));
        return;
      }
      if (target.hasAttribute('data-cmp-struct-fit')) {
        fitStructCamera();
        return;
      }
      if (target.dataset.cmpSource) {
        const [file, a, z] = target.dataset.cmpSource.split(':');
        options.onSource?.(file, Number(a), Number(z));
        return;
      }
      if (target.dataset.cmpTab) {
        const [tab, kName] = target.dataset.cmpTab.split(':');
        options.onNavigateTab?.(tab, { kernel: kName || state.kernel, block: state.block, rank: state.rank });
      }
    });

    host.addEventListener('change', event => {
      const el = event.target;
      if (el.matches('[data-cmp-rank]')) {
        const r = Number(el.value);
        if (r !== state.rank) {
          state.rank = r;
          state.sliceId = null;
          state.coreId = null;
          options.onRankChange?.(r);
          render();
        }
      } else if (el.matches('[data-cmp-rt-filter]')) {
        state.filter = el.value;
        state.sliceId = null;
        state.coreId = null;
        if (kernelByName.has(state.filter)) setKernel(state.filter);
        render();
      } else if (el.matches('[data-cmp-branch]')) {
        state.branch = el.value;
        if (state.branch !== 'all' && currentKernelInfo().branch !== state.branch) {
          const first = [...kernelByName.values()].find(k => k.branch === state.branch);
          if (first) setKernel(first.name);
        }
        render();
      } else if (el.matches('[data-cmp-kernel]')) {
        setKernel(el.value);
        render();
      } else if (el.matches('[data-cmp-map-group]')) {
        state.mapGroupFilter = el.value;
        render();
      } else if (el.matches('[data-cmp-map-part]')) {
        state.mapPartFilter = el.value;
        render();
      } else if (el.matches('[data-cmp-select-block]')) {
        openBlock(Number(el.value));
      } else if (el.matches('[data-cmp-case]')) {
        state.tokens = Number(el.value);
        const info = currentKernelInfo();
        const total = blockCountOf(info);
        state.block = Math.min(state.block, total - 1);
        state.lane = lanePosition(info, state.block).lane;
        const sh = model.blockShape(info, state.block, localView());
        if (sh.tiles) state.tileIndex = Math.min(state.tileIndex, sh.tiles.length - 1);
        state.structSelection = null;
        structCamera = null;
        render();
      } else if (el.matches('[data-cmp-tile]')) {
        state.tileIndex = Number(el.value);
        state.structSelection = null;
        structCamera = null;
        render();
      }
    });

    host.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const target = event.target;
      if (target.matches('[data-cmp-block],[data-cmp-lane],[data-cmp-rt-slice],[data-cmp-rt-core],[data-cmp-struct-node]')) {
        event.preventDefault();
        target.click();
      }
    });

    // Lane Map pan
    host.addEventListener('pointerdown', event => {
      const view = event.target.closest('.csa-qp-map-viewport');
      if (event.button !== 0 || !view || event.target.closest('[data-cmp-block],[data-cmp-lane]')) return;
      applyMapCamera();
      mapDrag = { x: event.clientX, y: event.clientY, camera: { ...mapCamera }, pid: event.pointerId };
      view.setPointerCapture(event.pointerId);
      view.classList.add('is-panning');
    });
    host.addEventListener('pointermove', event => {
      if (!mapDrag || !mapCamera) return;
      const view = host.querySelector('.csa-qp-map-viewport');
      const box = view?.getBoundingClientRect();
      if (!box?.width || !box?.height) return;
      mapCamera.x = mapDrag.camera.x - (event.clientX - mapDrag.x) * mapDrag.camera.width / box.width;
      mapCamera.y = mapDrag.camera.y - (event.clientY - mapDrag.y) * mapDrag.camera.height / box.height;
      applyMapCamera();
    });
    const endMapPan = () => { mapDrag = null; host.querySelector('.csa-qp-map-viewport')?.classList.remove('is-panning'); };
    host.addEventListener('pointerup', endMapPan);
    host.addEventListener('pointercancel', endMapPan);

    // Structure Pipeline pan & zoom
    host.addEventListener('pointerdown', event => {
      const view = event.target.closest('.csa-qp-stage-inner');
      if (event.button !== 0 || !view || event.target.closest('[data-cmp-struct-node]') || !structCamera) return;
      structDrag = { x: event.clientX, y: event.clientY, camera: { ...structCamera }, pid: event.pointerId, moved: false };
      view.setPointerCapture(event.pointerId);
      view.classList.add('is-panning');
    });
    host.addEventListener('pointermove', event => {
      if (!structDrag || !structCamera) return;
      const view = host.querySelector('.csa-qp-stage-inner');
      if (!view?.clientWidth || !view?.clientHeight) return;
      if (Math.abs(event.clientX - structDrag.x) + Math.abs(event.clientY - structDrag.y) > 4) structDrag.moved = true;
      structCamera.x = structDrag.camera.x - (event.clientX - structDrag.x) * structDrag.camera.width / view.clientWidth;
      structCamera.y = structDrag.camera.y - (event.clientY - structDrag.y) * structDrag.camera.height / view.clientHeight;
      structCamera.mode = 'manual';
      applyStructCamera();
    });
    const endStructPan = () => {
      suppressBlankClick = !!structDrag?.moved;
      structDrag = null;
      host.querySelector('.csa-qp-stage-inner')?.classList.remove('is-panning');
    };
    host.addEventListener('pointerup', endStructPan);
    host.addEventListener('pointercancel', endStructPan);

    // Runtime Trace pan
    host.addEventListener('pointerdown', event => {
      const view = event.target.closest('.csa-cmp-rt-viewport');
      if (event.button !== 0 || !view || event.target.closest('button,[role="button"]')) return;
      rtDrag = { view, pid: event.pointerId, x: event.clientX, y: event.clientY, left: view.scrollLeft, top: view.scrollTop };
      view.setPointerCapture(event.pointerId);
      view.classList.add('is-panning');
    });
    host.addEventListener('pointermove', event => {
      if (rtDrag?.pid !== event.pointerId) return;
      rtDrag.view.scrollTo(rtDrag.left + rtDrag.x - event.clientX, rtDrag.top + rtDrag.y - event.clientY);
    });
    const endRtPan = () => { rtDrag?.view?.classList.remove('is-panning'); rtDrag = null; };
    host.addEventListener('pointerup', endRtPan);
    host.addEventListener('pointercancel', endRtPan);

    host.addEventListener('wheel', event => {
      if (event.target.closest('.csa-qp-stage-inner') && structCamera) {
        event.preventDefault();
        zoomStructCamera(Math.exp(Math.max(-0.25, Math.min(0.25, -event.deltaY * 0.002))), event.clientX, event.clientY);
      }
    }, { passive: false });

    function update(next = {}) {
      if (next.rank != null && next.rank !== state.rank) {
        state.rank = next.rank;
        state.sliceId = null;
        state.coreId = null;
      }
      if (next.kernel && kernelByName.has(next.kernel)) {
        state.kernel = next.kernel;
      }
      if (next.tokens != null) state.tokens = next.tokens;
      if (next.tileIndex != null) state.tileIndex = next.tileIndex;
      if (next.block != null) {
        const info = currentKernelInfo();
        const clamped = Math.max(0, Math.min(blockCountOf(info) - 1, Number(next.block) || 0));
        state.block = clamped;
        state.lane = lanePosition(info, clamped).lane;
      }
      if (next.level) state.level = next.level;
      if (next.overviewMode) state.overviewMode = next.overviewMode;
      if (next.intraScale) state.intraScale = next.intraScale;
      if (next.filter) state.filter = next.filter;
      if (next.sliceId !== undefined) state.sliceId = next.sliceId;
      render();
    }

    return {
      update,
      getState: () => ({ ...state }),
    };
  }

  global.PtoCsaCompressorPipeline = Object.freeze({ create });
})(window);
