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

    const STEP_COLORS = ['#80b7f7', '#4fb899', '#f2b857', '#a87ef0', '#f47b8c', '#56c7d6', '#9be05a'];
    const PREP_KERNEL_TYPES = new Set(['qrope', 'kvrope', 'seed']);
    const isPrepKernel = name => PREP_KERNEL_TYPES.has(kernelByName.get(name)?.type);
    const hasPrepKernels = stage.kernels.some(k => isPrepKernel(k.name));
    const stageSourceFile = [...new Set(stage.kernels.map(k => kernelByName.get(k.name)?.file).filter(Boolean))].join(' · ');

    let state = {
      rank: options.rank ?? 0,
      tokens: caseOptions[0]?.tokens,
      tileIndex: 0,
      branch: 'all',
      kernel: options.kernel || model.defaultKernel || 'kv_score_proj',
      filter: options.filter || 'all', // Runtime Trace kernel/branch filter
      selectedGroup: null, // 0..23 Core Group selected in Runtime Trace
      sliceId: null,
      rtSortBy: 'group', // 'group' | 'duration'
      rtDensity: 'compact', // 'compact' (72-core panorama + insights) | 'expanded' (24 core-group rows)
      rtGroupBy: 'group', // 'group' (24 core groups: C+V0+V1) | 'type' (24 AIC / 48 AIV)
      rtShowPar: true, // parallelism width sparkline in compact mode
      rtShowExt: true, // cross-stage contention bands in compact mode
      rtInsightId: 'auto', // 'auto' | null | insight.id
      rtInsightKernel: null, // kernel explicitly selected via insight card click
      rtShowSetup: null, // null = auto (>=12% setupShare), true | false = user override
      rtShowBg: true, // show dimmed same-stage concurrent kernels when focused on a kernel/branch
      rtIncludePrep: false, // false = align with Perf Lane main chain (exclude early RoPE/Seed prep)
      rtHighlightStep: null, // kernel name highlighted via step chip
      rtHideIdle: false,
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
    };

    let mapCamera = null;
    let mapDrag = null;
    let structCamera = null;
    let structDrag = null;
    let suppressBlankClick = false;
    let currentStructLayout = null;
    let instanceCtrl = null;

    const currentRankData = () => data.ranks[String(state.rank)] || data.ranks['0'];
    const currentKernelInfo = () => kernelByName.get(state.kernel) || [...kernelByName.values()][0];
    const localView = () => ({ block: state.block, tokens: state.tokens, tileIndex: state.tileIndex });
    const branchLabelOf = id => layoutBranches[id] || id;
    const blockCountOf = info => blockTotalFor(model, info, state.tokens);

    function staticGroupCountFor(info, g) {
      if (!info) return 0;
      if (laneCountFor(info) === 24) {
        return laneBlocksFor(model, info, g, state.tokens).length;
      }
      return laneBlocksFor(model, info, g * 2, state.tokens).length
        + laneBlocksFor(model, info, g * 2 + 1, state.tokens).length;
    }

    function computeExternalBands(originUs, endUs) {
      const stages = global.PTO_CSA_RUNTIME_DATA?.stages;
      if (!stages?.length) return [];
      const rankKey = String(state.rank);
      const raw = [];
      for (const st of stages) {
        if (st.id === stage.id) continue;
        for (const k of st.kernels || []) {
          const r = k.ranks?.[rankKey] || k.ranks?.['0'];
          if (!r) continue;
          const s = Math.max(originUs, r.startUs);
          const e = Math.min(endUs, r.endUs);
          if (e - s < 14) continue;
          raw.push({
            stageId: st.id,
            stageLabel: st.label || st.id,
            name: k.name,
            coreType: k.coreType || 'AIC',
            startUs: s,
            endUs: e,
            durUs: e - s,
            slices: r.slices || 0,
            coresUsed: r.coresUsed || 0,
          });
        }
      }
      raw.sort((a, b) => (a.startUs - b.startUs) || (b.durUs - a.durUs));
      const merged = [];
      for (const item of raw) {
        const prev = merged[merged.length - 1];
        if (prev && prev.stageId === item.stageId && item.startUs <= prev.endUs + 6 && prev.coreType === item.coreType) {
          prev.endUs = Math.max(prev.endUs, item.endUs);
          prev.durUs = prev.endUs - prev.startUs;
          if (!prev.names.includes(item.name)) prev.names.push(item.name);
          prev.slices += item.slices;
          prev.coresUsed = Math.max(prev.coresUsed, item.coresUsed);
        } else {
          merged.push({
            id: `ext:${merged.length}`,
            stageId: item.stageId,
            stageLabel: item.stageLabel,
            names: [item.name],
            coreType: item.coreType,
            startUs: item.startUs,
            endUs: item.endUs,
            durUs: item.durUs,
            slices: item.slices,
            coresUsed: item.coresUsed,
          });
        }
      }
      return merged;
    }

    function computeParallelismProfile(slices, originUs, rawSpanUs, binCount = 120) {
      const bins = Array.from({ length: binCount }, () => ({ aic: 0, aiv: 0, total: 0 }));
      const stepUs = Math.max(1e-6, rawSpanUs / binCount);
      const aicSets = Array.from({ length: binCount }, () => new Set());
      const aivSets = Array.from({ length: binCount }, () => new Set());
      for (const s of slices) {
        const i0 = Math.max(0, Math.floor((s.startUs - originUs) / stepUs));
        const i1 = Math.min(binCount - 1, Math.floor((s.endUs - originUs) / stepUs));
        for (let i = i0; i <= i1; i++) {
          if (s.core < 24) aicSets[i].add(s.core);
          else aivSets[i].add(s.core);
        }
      }
      let peak = 0;
      let sumAic = 0;
      let sumAiv = 0;
      let lowBins = 0;
      for (let i = 0; i < binCount; i++) {
        const aic = aicSets[i].size;
        const aiv = aivSets[i].size;
        const total = aic + aiv;
        bins[i] = { aic, aiv, total };
        if (total > peak) peak = total;
        sumAic += aic;
        sumAiv += aiv;
        if (total <= 4) lowBins++;
      }
      return {
        bins,
        stepUs,
        peak,
        avgAicPct: (sumAic / (binCount * 24)) * 100,
        avgAivPct: (sumAiv / (binCount * 48)) * 100,
        avgTotalPct: ((sumAic + sumAiv) / (binCount * 72)) * 100,
        lowParPct: (lowBins / binCount) * 100,
      };
    }

    function collectCrossStageSlices(rankKey, originUs, endUs) {
      const out = [];
      const pushSlice = (stageId, stageLabel, name, core, startUs, durUs) => {
        const sEnd = startUs + durUs;
        if (sEnd <= originUs || startUs >= endUs) return;
        const cStart = Math.max(originUs, startUs);
        const cEnd = Math.min(endUs, sEnd);
        const overlapUs = cEnd - cStart;
        if (!(overlapUs > 0)) return;
        out.push({
          id: null,
          core,
          startUs,
          endUs: sEnd,
          durUs,
          overlapUs,
          clippedStart: cStart,
          clippedEnd: cEnd,
          name,
          stageId,
          stageLabel,
          crossStage: true,
          unit: core < 24 ? 'aic' : 'aiv',
        });
      };

      const stageTraces = global.PTO_CSA_STAGE_TRACES;
      if (stageTraces && typeof stageTraces === 'object') {
        for (const [sid, stDef] of Object.entries(stageTraces)) {
          if (sid === stage.id || !stDef?.ranks) continue;
          const rk = stDef.ranks[rankKey] || stDef.ranks['0'];
          if (!rk?.kernels) continue;
          const label = stDef.label || sid;
          for (const [kName, entries] of Object.entries(rk.kernels)) {
            for (const rec of entries) {
              // Tuple: [physicalCore, startUs, slotUs, computeUs, setupUs]
              pushSlice(sid, label, kName, rec[0], rec[1], rec[2]);
            }
          }
        }
      }

      const extraSources = [
        { sid: 'cache/main', label: 'Compressor', root: global.PTO_CSA_COMPRESSOR_DATA },
        { sid: 'input/q', label: 'Q Proj', root: global.PTO_CSA_Q_PROJECTION_DATA },
      ];
      for (const src of extraSources) {
        if (src.sid === stage.id || !src.root?.ranks) continue;
        const rk = src.root.ranks[rankKey] || src.root.ranks['0'];
        if (!rk?.kernels) continue;
        for (const [kName, entries] of Object.entries(rk.kernels)) {
          for (const rec of entries) {
            // Tuple: [eventId, physicalCore, startUs, slotUs, computeUs]
            pushSlice(src.sid, src.label, kName, rec[1], rec[2], rec[3]);
          }
        }
      }

      return out;
    }

    function analyzeKernelBalance(kInfo, kSlices, allStageSlices, rankKey) {
      if (!kInfo || !kSlices?.length) return null;
      const isAic = kInfo.coreType === 'AIC';
      const poolSize = isAic ? 24 : 48;
      const coreBase = isAic ? 0 : 24;
      let kOriginUs = Infinity;
      let kEndUs = -Infinity;
      const counts = new Array(poolSize).fill(0);
      const focusedDur = new Array(poolSize).fill(0);

      for (const s of kSlices) {
        if (s.startUs < kOriginUs) kOriginUs = s.startUs;
        if (s.endUs > kEndUs) kEndUs = s.endUs;
        const idx = s.core - coreBase;
        if (idx >= 0 && idx < poolSize) {
          counts[idx]++;
          focusedDur[idx] += s.durUs;
        }
      }

      const bgDur = new Array(poolSize).fill(0);
      const bgByCoreKernel = Array.from({ length: poolSize }, () => new Map());
      const bgGlobalKernel = new Map();
      const bgSlicesAll = [];

      const recordBg = (s, crossStage, stageId, stageLabel) => {
        if (s.endUs <= kOriginUs || s.startUs >= kEndUs) return;
        const cStart = Math.max(kOriginUs, s.startUs);
        const cEnd = Math.min(kEndUs, s.endUs);
        const overlapUs = cEnd - cStart;
        if (!(overlapUs > 0)) return;
        const item = crossStage ? s : {
          ...s,
          overlapUs,
          clippedStart: cStart,
          clippedEnd: cEnd,
          stageId,
          stageLabel,
          crossStage: false,
        };
        bgSlicesAll.push(item);
        const idx = s.core - coreBase;
        if (idx >= 0 && idx < poolSize) {
          bgDur[idx] += overlapUs;
          const map = bgByCoreKernel[idx];
          const cur = map.get(s.name) || { name: s.name, stageId, stageLabel, crossStage, count: 0, overlapUs: 0 };
          cur.count++;
          cur.overlapUs += overlapUs;
          map.set(s.name, cur);
          const gCur = bgGlobalKernel.get(s.name) || { name: s.name, stageId, stageLabel, crossStage, totalOverlapUs: 0, minCoreUs: Infinity, maxCoreUs: 0, perCore: new Array(poolSize).fill(0) };
          gCur.totalOverlapUs += overlapUs;
          gCur.perCore[idx] += overlapUs;
          bgGlobalKernel.set(s.name, gCur);
        }
      };

      for (const s of allStageSlices) {
        if (s.name === kInfo.name) continue;
        recordBg(s, false, stage.id, stageTitle);
      }
      const crossSlices = collectCrossStageSlices(rankKey, kOriginUs, kEndUs);
      for (const s of crossSlices) {
        recordBg(s, true, s.stageId, s.stageLabel);
      }

      const statics = Array.from({ length: poolSize }, (_, lane) => laneBlocksFor(model, kInfo, lane, state.tokens).length);
      const totalBusy = counts.map((_, i) => focusedDur[i] + bgDur[i]);

      let hotIdx = 0;
      let coldIdx = 0;
      let maxAbsDelta = 0;
      for (let i = 0; i < poolSize; i++) {
        const d = counts[i] - statics[i];
        if (Math.abs(d) > maxAbsDelta) maxAbsDelta = Math.abs(d);
        if (counts[i] > counts[hotIdx] || (counts[i] === counts[hotIdx] && d > counts[hotIdx] - statics[hotIdx])) hotIdx = i;
        if (counts[i] < counts[coldIdx] || (counts[i] === counts[coldIdx] && d < counts[coldIdx] - statics[coldIdx])) coldIdx = i;
      }

      const activeIndices = [];
      for (let i = 0; i < poolSize; i++) {
        if (counts[i] > 0) activeIndices.push(i);
      }
      const minAll = Math.min(...counts);
      const maxAll = Math.max(...counts);
      const minStatic = Math.min(...statics);
      const maxStatic = Math.max(...statics);
      const minActiveCount = activeIndices.length ? Math.min(...activeIndices.map(i => counts[i])) : 0;
      const blockSkew = minActiveCount > 0 ? maxAll / minActiveCount : 1;

      const activeFocused = activeIndices.map(i => focusedDur[i]);
      const minFocused = activeFocused.length ? Math.min(...activeFocused) : 0;
      const maxFocused = activeFocused.length ? Math.max(...activeFocused) : 0;
      const focusedDurSkew = minFocused > 0.1 ? maxFocused / minFocused : 1;

      const activeBusy = activeIndices.map(i => totalBusy[i]);
      const sortedBusy = [...activeBusy].sort((a, b) => a - b);
      const minTotalBusy = sortedBusy.length ? sortedBusy[0] : 0;
      const maxTotalBusy = sortedBusy.length ? sortedBusy[sortedBusy.length - 1] : 0;
      const medTotalBusy = sortedBusy.length ? sortedBusy[Math.floor(sortedBusy.length / 2)] : 0;
      const busySkew = minTotalBusy > 0.1 ? maxTotalBusy / minTotalBusy : 1;

      const sumFocused = focusedDur.reduce((a, b) => a + b, 0);
      const sumBg = bgDur.reduce((a, b) => a + b, 0);
      const minBgActive = activeIndices.length ? Math.min(...activeIndices.map(i => bgDur[i])) : 0;
      const maxBgActive = activeIndices.length ? Math.max(...activeIndices.map(i => bgDur[i])) : 0;
      const hasBg = sumBg > Math.max(8, sumFocused * 0.12) && (maxBgActive - minBgActive) >= Math.max(4, (maxFocused - minFocused) * 0.25);

      const isSkipPattern = maxAll <= 1 && activeIndices.length < poolSize;
      const singleSkew = Math.max(blockSkew, focusedDurSkew);

      let verdict = 'even';
      if (isSkipPattern) {
        verdict = 'skip';
      } else if (maxAbsDelta >= 1 || singleSkew >= 1.25) {
        if (hasBg && busySkew <= 1.32 && busySkew <= singleSkew * 0.86) {
          verdict = 'backfill';
        } else if (hasBg && busySkew < singleSkew * 0.75) {
          verdict = 'partial';
        } else {
          verdict = 'imbalance';
        }
      }

      // Find the concurrent background kernel with the largest per-core variance (best explains backfill)
      let driverBg = null;
      for (const item of bgGlobalKernel.values()) {
        const vals = activeIndices.map(i => item.perCore[i]);
        const lo = vals.length ? Math.min(...vals) : 0;
        const hi = vals.length ? Math.max(...vals) : 0;
        const spread = hi - lo;
        const hotVal = item.perCore[hotIdx] || 0;
        const coldVal = item.perCore[coldIdx] || 0;
        const explainScore = Math.max(0, coldVal - hotVal) * 2 + spread + item.totalOverlapUs * 0.05;
        if (!driverBg || explainScore > driverBg.explainScore) {
          driverBg = {
            name: item.name,
            stageId: item.stageId,
            stageLabel: item.stageLabel,
            crossStage: item.crossStage,
            hotVal,
            coldVal,
            lo,
            hi,
            spread,
            explainScore,
          };
        }
      }

      // Aggregate per-Core-Group (24 groups) for the kernel's active unit
      const groupFocusedUs = new Array(24).fill(0);
      const groupBgUs = new Array(24).fill(0);
      const groupTotalBusyUs = new Array(24).fill(0);
      for (let g = 0; g < 24; g++) {
        if (isAic) {
          groupFocusedUs[g] = focusedDur[g];
          groupBgUs[g] = bgDur[g];
          groupTotalBusyUs[g] = totalBusy[g];
        } else {
          const i0 = g * 2;
          const i1 = g * 2 + 1;
          groupFocusedUs[g] = focusedDur[i0] + focusedDur[i1];
          groupBgUs[g] = bgDur[i0] + bgDur[i1];
          groupTotalBusyUs[g] = totalBusy[i0] + totalBusy[i1];
        }
      }
      const maxGroupTotalBusyUs = Math.max(1, ...groupTotalBusyUs);
      const activeGroupBusy = groupTotalBusyUs.filter((_, g) => (isAic ? counts[g] > 0 : (counts[g * 2] + counts[g * 2 + 1]) > 0)).sort((a, b) => a - b);
      const medGroupBusyUs = activeGroupBusy.length ? activeGroupBusy[Math.floor(activeGroupBusy.length / 2)] : 0;

      return {
        kInfo,
        isAic,
        poolSize,
        coreBase,
        kOriginUs,
        kEndUs,
        counts,
        statics,
        focusedDur,
        bgDur,
        totalBusy,
        bgByCoreKernel,
        bgSlicesAll,
        hotIdx,
        coldIdx,
        hotCore: coreBase + hotIdx,
        coldCore: coreBase + coldIdx,
        hotCount: counts[hotIdx],
        coldCount: counts[coldIdx],
        hotDelta: counts[hotIdx] - statics[hotIdx],
        coldDelta: counts[coldIdx] - statics[coldIdx],
        hotFocusedUs: focusedDur[hotIdx],
        hotBgUs: bgDur[hotIdx],
        hotTotalBusyUs: totalBusy[hotIdx],
        coldFocusedUs: focusedDur[coldIdx],
        coldBgUs: bgDur[coldIdx],
        coldTotalBusyUs: totalBusy[coldIdx],
        maxAbsDelta,
        minAll,
        maxAll,
        minStatic,
        maxStatic,
        activeCoreCount: activeIndices.length,
        blockSkew,
        focusedDurSkew,
        busySkew,
        minTotalBusy,
        maxTotalBusy,
        medTotalBusy,
        hasBg,
        isSkipPattern,
        verdict,
        driverBg,
        groupFocusedUs,
        groupBgUs,
        groupTotalBusyUs,
        maxGroupTotalBusyUs,
        medGroupBusyUs,
      };
    }

    function buildStageInsights(visible, steps, extBands, parProfile, originUs, rawSpanUs, allStageSlices, rankKey) {
      const insights = [];

      // 1. Dynamic scheduling skew vs concurrent backfill (Static Lane Map vs Runtime + Concurrent Core Busy Time)
      let bestSkew = null;
      for (const st of steps) {
        const kInfo = kernelByName.get(st.name);
        if (!kInfo || st.count < 12) continue;
        const kSlices = visible.filter(s => s.name === st.name);
        const bal = analyzeKernelBalance(kInfo, kSlices, allStageSlices, rankKey);
        if (!bal) continue;
        if (bal.maxAbsDelta >= 1 && (bal.maxAll - bal.minAll >= 2 || bal.maxAbsDelta >= 1)) {
          // Prioritize true imbalance over explained backfill when comparing multiple kernels in a stage
          const severityMult = bal.verdict === 'imbalance' ? 1.35 : bal.verdict === 'partial' ? 1.15 : 0.72;
          const score = ((bal.maxAll - bal.minAll) * 3 + bal.maxAbsDelta * 5 + (st.spanUs / Math.max(1, rawSpanUs)) * 20) * severityMult;
          if (!bestSkew || score > bestSkew.score) {
            bestSkew = { score, st, bal };
          }
        }
      }
      if (bestSkew) {
        const { st, bal: b } = bestSkew;
        const unitLabel = b.isAic ? 'AIC' : 'AIV';
        const cPrefix = b.isAic ? 'C' : 'V';
        const staticText = b.minStatic === b.maxStatic ? `${b.minStatic}` : `${b.minStatic}–${b.maxStatic}`;
        const actualText = b.minAll === b.maxAll ? `${b.minAll}` : `${b.minAll}–${b.maxAll}`;
        const dHot = b.hotDelta > 0 ? `+${b.hotDelta}` : `${b.hotDelta}`;
        const dCold = b.coldDelta > 0 ? `+${b.coldDelta}` : `−${Math.abs(b.coldDelta)}`;
        const busyRangeText = `${Math.round(b.minTotalBusy)}–${Math.round(b.maxTotalBusy)} µs`;

        if (b.isSkipPattern) {
          insights.push({
            id: `skew:${st.name}`,
            priority: 100,
            kind: 'skew',
            verdict: 'skip',
            kernelName: st.name,
            unit: b.isAic ? 'aic' : 'aiv',
            tag: '物理跳核映射',
            title: st.name,
            subtitle: `${unitLabel} · ${b.activeCoreCount}/${b.poolSize} 核活跃 · 跳过 ${cPrefix}${b.coldCore} 等物理核`,
            detail: `Lane Map 按 0–${b.activeCoreCount - 1} 连续示意，Runtime 实际派发跳过 ${cPrefix}${b.coldCore} (${dCold})，落在 ${cPrefix}${b.hotCore} (${dHot})`,
            startUs: st.startUs,
            endUs: st.endUs,
            hotCore: b.hotCore,
            coldCore: b.coldCore,
            hotLabel: `${cPrefix}${b.hotCore} · ${b.hotCount}片 (${dHot})`,
            callout: `物理跳核 · ${b.activeCoreCount}/${b.poolSize} ${unitLabel} 活跃`,
          });
        } else if (b.verdict === 'backfill') {
          const drv = b.driverBg;
          const driverNote = drv && (drv.coldVal - drv.hotVal) >= 12
            ? `${cPrefix}${b.hotCore} 的并发任务 ${drv.name} 仅占 ${drv.hotVal.toFixed(0)} µs（vs ${cPrefix}${b.coldCore} ${drv.coldVal.toFixed(0)} µs），早空闲后由 Runtime 动态多派 ${dHot} 片补位`
            : `${cPrefix}${b.hotCore} 同窗并发背景较轻（${b.hotBgUs.toFixed(0)} µs vs ${cPrefix}${b.coldCore} ${b.coldBgUs.toFixed(0)} µs），Runtime 动态多派 ${dHot} 片拉平核忙时`;
          insights.push({
            id: `skew:${st.name}`,
            priority: steps.length === 1 ? 100 : 76,
            kind: 'skew',
            verdict: 'backfill',
            kernelName: st.name,
            unit: b.isAic ? 'aic' : 'aiv',
            tag: '动态补位',
            title: st.name,
            subtitle: `${unitLabel} · 单核 ${actualText} 片 (${b.blockSkew.toFixed(2)}×) · 总忙时 ${busyRangeText} (${b.busySkew.toFixed(2)}×)`,
            detail: `${driverNote}；含并发背景后总忙时收敛至 ${b.busySkew.toFixed(2)}×，属正常动态负载均衡`,
            startUs: st.startUs,
            endUs: st.endUs,
            hotCore: b.hotCore,
            coldCore: b.coldCore,
            hotLabel: `${cPrefix}${b.hotCore} · ${b.hotCount}片 (${dHot}) · 总 ${Math.round(b.hotTotalBusyUs)}µs`,
            callout: `动态补位 · 单核 ${actualText} 片 → 总忙时 ${b.busySkew.toFixed(2)}×（非失衡）`,
          });
        } else if (b.verdict === 'partial') {
          insights.push({
            id: `skew:${st.name}`,
            priority: 92,
            kind: 'skew',
            verdict: 'partial',
            kernelName: st.name,
            unit: b.isAic ? 'aic' : 'aiv',
            tag: '并发缓解 · 仍偏斜',
            title: st.name,
            subtitle: `${unitLabel} · 单核 ${actualText} 片 (${b.blockSkew.toFixed(2)}×) · 含并发总忙时 ${busyRangeText} (${b.busySkew.toFixed(2)}×)`,
            detail: `并发背景将核时极差从 ${b.focusedDurSkew.toFixed(2)}× 收敛至 ${b.busySkew.toFixed(2)}×，但 ${cPrefix}${b.hotCore}（总 ${Math.round(b.hotTotalBusyUs)} µs）与 ${cPrefix}${b.coldCore}（总 ${Math.round(b.coldTotalBusyUs)} µs）仍存在残余偏斜`,
            startUs: st.startUs,
            endUs: st.endUs,
            hotCore: b.hotCore,
            coldCore: b.coldCore,
            hotLabel: `${cPrefix}${b.hotCore} · ${b.hotCount}片 (${dHot}) · 总 ${Math.round(b.hotTotalBusyUs)}µs`,
            callout: `并发缓解 · 总忙时仍差 ${b.busySkew.toFixed(2)}× · ${cPrefix}${b.hotCore} (${b.hotCount}片 ${dHot})`,
          });
        } else {
          insights.push({
            id: `skew:${st.name}`,
            priority: 100,
            kind: 'skew',
            verdict: 'imbalance',
            kernelName: st.name,
            unit: b.isAic ? 'aic' : 'aiv',
            tag: '核级负载失衡',
            title: st.name,
            subtitle: `${unitLabel} · 静态 ${staticText} → 实测 ${actualText} 片 (${b.blockSkew.toFixed(2)}×) · 总忙时 ${b.busySkew.toFixed(2)}×`,
            detail: `计入同窗并发背景后仍存在真实失衡：${cPrefix}${b.hotCore} 承担 ${b.hotCount} 片 (${dHot}，总忙时 ${Math.round(b.hotTotalBusyUs)} µs)，${cPrefix}${b.coldCore} 仅 ${b.coldCount} 片 (${dCold}，总忙时 ${Math.round(b.coldTotalBusyUs)} µs)`,
            startUs: st.startUs,
            endUs: st.endUs,
            hotCore: b.hotCore,
            coldCore: b.coldCore,
            hotLabel: `${cPrefix}${b.hotCore} · ${b.hotCount}片 (${dHot}) · 总 ${Math.round(b.hotTotalBusyUs)}µs`,
            callout: `核级失衡 · 单核 ${actualText} 片 · 总忙时仍差 ${b.busySkew.toFixed(2)}×`,
          });
        }
      }

      // 2. High Setup / Barrier wait kernel
      let worstSetup = null;
      for (const st of steps) {
        const share = st.setupUs / Math.max(1e-6, st.slotUs);
        if (share >= 0.25 && st.spanUs >= 18) {
          if (!worstSetup || share * st.slotUs > worstSetup.weight) {
            worstSetup = { st, share, weight: share * st.slotUs };
          }
        }
      }
      if (worstSetup) {
        const { st, share } = worstSetup;
        const pct = Math.round(share * 100);
        const avgCompute = st.computeUs / Math.max(1, st.count);
        const avgSetup = st.setupUs / Math.max(1, st.count);
        insights.push({
          id: `setup:${st.name}`,
          priority: 95,
          kind: 'setup',
          kernelName: st.name,
          unit: st.coreType === 'AIC' ? 'aic' : 'aiv',
          tag: `Setup 占比 ${pct}%`,
          title: st.name,
          subtitle: `${st.coreType} · ${st.count} 块 · 均值等待 ${avgSetup.toFixed(1)} µs vs 计算 ${avgCompute.toFixed(1)} µs`,
          detail: `提前派发后在核内等待上游就绪，Setup 占槽位核时 ${pct}%`,
          startUs: st.startUs,
          endUs: st.endUs,
          setupPct: pct,
          callout: `Setup 等待占比 ${pct}% · 计算仅 ${avgCompute.toFixed(1)} µs`,
        });
      }

      // 3. Cross-stage contention / blank gap window
      let bestGap = null;
      let runStart = -1;
      for (let i = 0; i <= parProfile.bins.length; i++) {
        const isLow = i < parProfile.bins.length && parProfile.bins[i].total <= 2;
        if (isLow && runStart < 0) runStart = i;
        if (!isLow && runStart >= 0) {
          const gStart = originUs + runStart * parProfile.stepUs;
          const gEnd = originUs + i * parProfile.stepUs;
          const gDur = gEnd - gStart;
          if (gDur >= Math.max(24, rawSpanUs * 0.06)) {
            const matchedExt = extBands.filter(b => b.endUs > gStart + 2 && b.startUs < gEnd - 2);
            if (!bestGap || gDur > bestGap.durUs) {
              bestGap = { startUs: gStart, endUs: gEnd, durUs: gDur, matchedExt };
            }
          }
          runStart = -1;
        }
      }
      if (bestGap) {
        const ext = bestGap.matchedExt[0];
        const extTitle = ext ? `${ext.names[0]} (${ext.coreType})` : '低并行空窗';
        insights.push({
          id: `gap:${Math.round(bestGap.startUs)}`,
          priority: 90,
          kind: 'gap',
          kernelName: null,
          unit: 'all',
          tag: ext ? '跨阶段占核空窗' : '低并行空窗',
          title: ext ? `${ext.stageLabel} · ${ext.names[0]}` : `空窗 ${Math.round(bestGap.durUs)} µs`,
          subtitle: `+${Math.round(bestGap.startUs - originUs)}→+${Math.round(bestGap.endUs - originUs)} µs（${Math.round(bestGap.durUs)} µs · 占窗口 ${Math.round((bestGap.durUs / rawSpanUs) * 100)}%）`,
          detail: ext
            ? `${extTitle} 在该时段占核并发执行（${ext.coresUsed || ext.slices} 核/片），并非芯片空转`
            : `当前阶段在该时段活跃核数 ≤ 2`,
          startUs: bestGap.startUs,
          endUs: bestGap.endUs,
          callout: ext
            ? `跨阶段占核 · ${ext.names[0]} (${Math.round(bestGap.durUs)} µs)`
            : `低并行空窗 · ${Math.round(bestGap.durUs)} µs`,
        });
      } else if (extBands.length) {
        const topExt = [...extBands].sort((a, b) => b.durUs - a.durUs)[0];
        if (topExt && topExt.durUs >= 35) {
          insights.push({
            id: `ext:${topExt.names[0]}`,
            priority: 72,
            kind: 'gap',
            kernelName: null,
            unit: topExt.coreType === 'AIC' ? 'aic' : topExt.coreType === 'AIV' ? 'aiv' : 'all',
            tag: '跨阶段并发抢核',
            title: `${topExt.stageLabel} · ${topExt.names[0]}`,
            subtitle: `${topExt.coreType} · +${Math.round(topExt.startUs - originUs)}→+${Math.round(topExt.endUs - originUs)} µs (${Math.round(topExt.durUs)} µs)`,
            detail: `相邻阶段并发共享同一批物理核，导致本阶段切片呈现分段穿插`,
            startUs: topExt.startUs,
            endUs: topExt.endUs,
            callout: `跨阶段并发 · ${topExt.names[0]} (${Math.round(topExt.durUs)} µs)`,
          });
        }
      }

      // 4. Slowest single-block latency outlier
      let bestOutlier = null;
      for (const st of steps) {
        if (st.count < 8) continue;
        const kSlices = visible.filter(s => s.name === st.name);
        const durs = kSlices.map(s => s.durUs).sort((a, b) => a - b);
        const med = durs[Math.floor(durs.length / 2)] || 1;
        const worst = kSlices.reduce((a, b) => (b.durUs > a.durUs ? b : a), kSlices[0]);
        const ratio = worst.durUs / Math.max(0.1, med);
        if (ratio >= 1.75 && worst.durUs >= 15) {
          if (!bestOutlier || ratio * worst.durUs > bestOutlier.score) {
            bestOutlier = { st, slice: worst, med, ratio, score: ratio * worst.durUs };
          }
        }
      }
      if (bestOutlier) {
        const { st, slice, med, ratio } = bestOutlier;
        const coreTag = slice.core < 24 ? `C${slice.core}` : `V${slice.core}`;
        insights.push({
          id: `outlier:${slice.id}`,
          priority: 85,
          kind: 'outlier',
          kernelName: st.name,
          unit: slice.unit,
          tag: `${ratio.toFixed(1)}× 慢块`,
          title: `${st.name} · ${coreTag}`,
          subtitle: `最慢块 ${slice.durUs.toFixed(1)} µs vs 中位 ${med.toFixed(1)} µs（#${slice.id}）`,
          detail: `单块耗时显著高于同 Kernel 中位数，多由首波冷启动或访存带宽争用引起`,
          startUs: st.startUs,
          endUs: st.endUs,
          slice,
          callout: `最慢块 · ${coreTag} · ${slice.durUs.toFixed(1)} µs (${ratio.toFixed(1)}×) · 点击展开`,
        });
      }

      // 5. Underutilized core pool on long-running kernel
      let worstUnder = null;
      for (const st of steps) {
        if (st.spanUs < rawSpanUs * 0.18) continue;
        const poolSize = st.coreType === 'AIC' ? 24 : 48;
        const usedCores = new Set(visible.filter(s => s.name === st.name).map(s => s.core)).size;
        const util = usedCores / poolSize;
        if (util <= 0.7 && st.spanUs >= 45) {
          if (!worstUnder || (1 - util) * st.spanUs > worstUnder.score) {
            worstUnder = { st, usedCores, poolSize, util, score: (1 - util) * st.spanUs };
          }
        }
      }
      if (worstUnder) {
        const { st, usedCores, poolSize, util } = worstUnder;
        const pct = Math.round(util * 100);
        insights.push({
          id: `under:${st.name}`,
          priority: 80,
          kind: 'underutil',
          kernelName: st.name,
          unit: st.coreType === 'AIC' ? 'aic' : 'aiv',
          tag: `核利用率 ${pct}%`,
          title: st.name,
          subtitle: `仅占 ${usedCores}/${poolSize} ${st.coreType}（${poolSize - usedCores} 核闲置）· 跨度 ${st.spanUs.toFixed(0)} µs`,
          detail: `单块耗时较长而逻辑块数少于物理核池，尾段并行度受限`,
          startUs: st.startUs,
          endUs: st.endUs,
          callout: `${usedCores}/${poolSize} ${st.coreType} 活跃 · 跨度 ${Math.round(st.spanUs)} µs`,
        });
      }

      insights.sort((a, b) => b.priority - a.priority);
      return insights.slice(0, 5);
    }

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

    const isBranchFilter = value => Object.prototype.hasOwnProperty.call(layoutBranches, value);

    function resolveRuntimeKernel() {
      if (kernelByName.has(state.filter)) return state.filter;
      if (state.rtHighlightStep && kernelByName.has(state.rtHighlightStep)) return state.rtHighlightStep;
      if (state.sliceId != null) {
        const hit = findSliceById(currentRankData(), state.sliceId);
        if (hit && kernelByName.has(hit.owner)) return hit.owner;
      }
      if (state.rtInsightId && state.rtInsightId !== 'auto' && state.rtInsightKernel && kernelByName.has(state.rtInsightKernel)) {
        return state.rtInsightKernel;
      }
      return null;
    }

    function laneForGroup(info, groupIdx, sliceId) {
      if (sliceId != null) {
        const hit = findSliceById(currentRankData(), sliceId);
        if (hit && hit.owner === info.name) {
          const core = hit.rec[1];
          const laneIdx = info.coreType === 'AIC' ? core : core - 24;
          if (laneIdx >= 0 && laneIdx < laneCountFor(info)) return laneIdx;
        }
      }
      if (groupIdx == null) return null;
      return laneCountFor(info) === 24 ? groupIdx : groupIdx * 2;
    }

    function groupForLane(info, laneIdx) {
      if (laneIdx == null) return null;
      return laneCountFor(info) === 24 ? laneIdx : Math.floor(laneIdx / 2);
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
      state.selectedGroup = null;
      state.rtHighlightStep = null;
      state.rtShowSetup = null;
      render();
    }

    function coreGroupOf(core) {
      return core < 24 ? core : Math.floor((core - 24) / 2);
    }

    function findSliceById(rankData, sliceId) {
      if (sliceId == null || !rankData?.kernels) return null;
      for (const [name, entries] of Object.entries(rankData.kernels)) {
        const hit = entries.find(s => s[0] === sliceId);
        if (hit) return { owner: name, rec: hit };
      }
      return null;
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
      const rtScopeKernels = Object.keys(currentRankData().kernels).filter(name => {
        const kInfo = kernelByName.get(name);
        if (state.filter === 'all') return !hasPrepKernels || state.rtIncludePrep || !isPrepKernel(name);
        return state.filter === name || state.filter === kInfo?.branch;
      });
      const rtScopeSlices = rtScopeKernels.reduce((acc, name) => acc + (currentRankData().kernels[name]?.length || 0), 0);
      const focusedMetaInfo = isRt && kernelByName.has(state.filter) ? kernelByName.get(state.filter) : null;
      const meta = isRt
        ? (focusedMetaInfo
            ? `<span class="csa-qp-context-meta">${esc(focusedMetaInfo.name)} · 实测 ${rtScopeSlices} 块 · 理论均分 ${(blockCountOf(focusedMetaInfo) / 24).toFixed(1)} 块/组</span>`
            : `<span class="csa-qp-context-meta">${esc(hasPrepKernels && state.filter === 'all' && !state.rtIncludePrep ? '主计算链路' : branchSummary)} · ${rtScopeKernels.length} 阶段 · 24 核组 · ${rtScopeSlices} 块</span>`)
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
        extraTools = `<label>显示 <select data-cmp-rt-filter>${filterOpts}</select></label>`;
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

    // ---------- Macro View 1A: Runtime Trace (24 Core Groups, aligned with Perf Lane & qk_pv) ----------
    function runtimeTraceView() {
      const rankData = currentRankData();
      const fmtUs = v => (Number.isFinite(v) ? `${v.toFixed(1)} µs` : '—');
      const fmtShortUs = v => (!Number.isFinite(v) ? '—' : v >= 100 ? `${Math.round(v)} µs` : `${v.toFixed(1)} µs`);

      const allSlices = Object.entries(rankData.kernels).flatMap(([name, entries]) => {
        const kInfo = kernelByName.get(name);
        const branch = kInfo?.branch || 'main';
        return entries.map(([id, core, startUs, durUs, kdurUs]) => {
          const computeUs = Math.max(0, Math.min(durUs, Number.isFinite(kdurUs) ? kdurUs : durUs));
          const setupUs = Math.max(0, durUs - computeUs);
          return {
            id,
            core,
            startUs,
            endUs: startUs + durUs,
            durUs,
            computeUs,
            setupUs,
            name,
            branch,
            unit: core < 24 ? 'aic' : 'aiv',
          };
        });
      });

      const excludePrep = hasPrepKernels && state.filter === 'all' && !state.rtIncludePrep;
      const focusedKernel = kernelByName.get(state.filter) || null;
      const matchesFilter = s => state.filter === 'all'
        ? !(excludePrep && isPrepKernel(s.name))
        : (state.filter === s.name || state.filter === s.branch);
      const visible = allSlices.filter(matchesFilter);
      if (!visible.length) {
      return `<div class="csa-cmp-rt-host"><div class="csa-rt-empty">当前筛选条件下无实测切片。</div></div>`;
      }

      const rankKey = String(state.rank);
      const originUs = Math.min(...visible.map(s => s.startUs));
      const endUs = Math.max(...visible.map(s => s.endUs));
      const rawSpanUs = Math.max(0.5, endUs - originUs);
      const displaySpanUs = Math.max(rawSpanUs * 1.03, 1);
      const endPct = Math.min(100, (rawSpanUs / displaySpanUs) * 100);

      const focusedBalance = focusedKernel ? analyzeKernelBalance(focusedKernel, visible, allSlices, rankKey) : null;

      let bgSlices = [];
      if (focusedBalance) {
        bgSlices = focusedBalance.bgSlicesAll;
      } else if (state.filter !== 'all') {
        const sameStageBg = allSlices
          .filter(s => !matchesFilter(s) && s.endUs > originUs && s.startUs < endUs)
          .map(s => ({
            ...s,
            clippedStart: Math.max(originUs, s.startUs),
            clippedEnd: Math.min(endUs, s.endUs),
            overlapUs: Math.max(0, Math.min(endUs, s.endUs) - Math.max(originUs, s.startUs)),
            stageId: stage.id,
            stageLabel: stageTitle,
            crossStage: false,
          }));
        const crossStageBg = collectCrossStageSlices(rankKey, originUs, endUs);
        bgSlices = [...sameStageBg, ...crossStageBg];
      }
      const hasCrossBg = bgSlices.some(s => s.crossStage);
      const showBg = state.filter !== 'all' && bgSlices.length > 0 && state.rtShowBg;

      // Build ordered step metadata & Perf-Lane palette
      const stepMap = new Map();
      for (const s of visible) {
        let st = stepMap.get(s.name);
        if (!st) {
          const kInfo = kernelByName.get(s.name);
          st = {
            name: s.name,
            coreType: kInfo?.coreType || (s.unit === 'aic' ? 'AIC' : 'AIV'),
            branch: s.branch,
            startUs: Infinity,
            endUs: -Infinity,
            count: 0,
            slotUs: 0,
            computeUs: 0,
            setupUs: 0,
          };
          stepMap.set(s.name, st);
        }
        st.startUs = Math.min(st.startUs, s.startUs);
        st.endUs = Math.max(st.endUs, s.endUs);
        st.count++;
        st.slotUs += s.durUs;
        st.computeUs += s.computeUs;
        st.setupUs += s.setupUs;
      }
      const steps = [...stepMap.values()].sort((a, b) => (a.startUs - b.startUs) || (a.endUs - b.endUs));
      steps.forEach((st, idx) => {
        st.color = STEP_COLORS[idx % STEP_COLORS.length];
        st.spanUs = Math.max(0, st.endUs - st.startUs);
      });
      const stepByName = new Map(steps.map(st => [st.name, st]));
      const multiStep = steps.length > 1;
      if (state.rtHighlightStep && !stepByName.has(state.rtHighlightStep)) {
        state.rtHighlightStep = null;
      }

      const totalSlotUs = visible.reduce((a, s) => a + s.durUs, 0);
      const totalSetupUs = visible.reduce((a, s) => a + s.setupUs, 0);
      const stageSetupShare = totalSetupUs / Math.max(1e-6, totalSlotUs);
      const showSetup = state.rtShowSetup != null ? state.rtShowSetup : stageSetupShare >= 0.12;

      // Aggregate 72 cores into 24 Core Groups: Group g = AIC[g] + AIV0[24 + 2*g] + AIV1[24 + 2*g + 1]
      const groups = ticks(24).map(g => {
        const aicCore = g;
        const aiv0Core = 24 + 2 * g;
        const aiv1Core = 24 + 2 * g + 1;
        const aicSlices = visible.filter(s => s.core === aicCore).sort((a, b) => a.startUs - b.startUs);
        const aiv0Slices = visible.filter(s => s.core === aiv0Core).sort((a, b) => a.startUs - b.startUs);
        const aiv1Slices = visible.filter(s => s.core === aiv1Core).sort((a, b) => a.startUs - b.startUs);
        const bgAicSlices = bgSlices.filter(s => s.core === aicCore).sort((a, b) => a.startUs - b.startUs);
        const bgAiv0Slices = bgSlices.filter(s => s.core === aiv0Core).sort((a, b) => a.startUs - b.startUs);
        const bgAiv1Slices = bgSlices.filter(s => s.core === aiv1Core).sort((a, b) => a.startUs - b.startUs);
        const sumField = (arr, field) => arr.reduce((acc, s) => acc + s[field], 0);
        const aicDurUs = sumField(aicSlices, 'durUs');
        const aiv0DurUs = sumField(aiv0Slices, 'durUs');
        const aiv1DurUs = sumField(aiv1Slices, 'durUs');
        const aicComputeUs = sumField(aicSlices, 'computeUs');
        const aiv0ComputeUs = sumField(aiv0Slices, 'computeUs');
        const aiv1ComputeUs = sumField(aiv1Slices, 'computeUs');
        const aicSetupUs = sumField(aicSlices, 'setupUs');
        const aiv0SetupUs = sumField(aiv0Slices, 'setupUs');
        const aiv1SetupUs = sumField(aiv1Slices, 'setupUs');
        const allGroupSlices = [...aicSlices, ...aiv0Slices, ...aiv1Slices];
        const totalDurUs = aicDurUs + aiv0DurUs + aiv1DurUs;
        const totalComputeUs = aicComputeUs + aiv0ComputeUs + aiv1ComputeUs;
        const totalSetupUsGroup = aicSetupUs + aiv0SetupUs + aiv1SetupUs;
        const firstStartUs = allGroupSlices.length ? Math.min(...allGroupSlices.map(s => s.startUs)) : originUs;
        const lastEndUs = allGroupSlices.length ? Math.max(...allGroupSlices.map(s => s.endUs)) : originUs;
        const spanUs = allGroupSlices.length ? Math.max(0, lastEndUs - firstStartUs) : 0;
        const setupPct = (totalSetupUsGroup / Math.max(1e-6, totalDurUs)) * 100;
        const staticCount = focusedKernel ? staticGroupCountFor(focusedKernel, g) : null;
        const deltaCount = focusedKernel ? (allGroupSlices.length - staticCount) : null;
        const unitFocusedUs = focusedBalance ? focusedBalance.groupFocusedUs[g] : totalDurUs;
        const unitBgUs = focusedBalance ? focusedBalance.groupBgUs[g] : 0;
        const unitTotalBusyUs = focusedBalance ? focusedBalance.groupTotalBusyUs[g] : totalDurUs;
        return {
          group: g,
          aicCore,
          aiv0Core,
          aiv1Core,
          aicSlices,
          aiv0Slices,
          aiv1Slices,
          bgAicSlices,
          bgAiv0Slices,
          bgAiv1Slices,
          aicDurUs,
          aiv0DurUs,
          aiv1DurUs,
          aicComputeUs,
          aiv0ComputeUs,
          aiv1ComputeUs,
          aicSetupUs,
          aiv0SetupUs,
          aiv1SetupUs,
          totalDurUs,
          totalComputeUs,
          totalSetupUs: totalSetupUsGroup,
          unitFocusedUs,
          unitBgUs,
          unitTotalBusyUs,
          setupPct,
          sliceCount: allGroupSlices.length,
          staticCount,
          deltaCount,
          firstStartUs,
          lastEndUs,
          spanUs,
          tier: 'fast',
        };
      });

      const activeGroups = groups.filter(g => g.sliceCount > 0);
      const activeDurs = activeGroups.map(g => g.totalDurUs);
      const minActiveDur = activeDurs.length ? Math.min(...activeDurs) : 0;
      const maxActiveDur = activeDurs.length ? Math.max(...activeDurs) : 0;
      const ratio = minActiveDur > 0.01 ? maxActiveDur / minActiveDur : 1;
      const slowCut = minActiveDur + (maxActiveDur - minActiveDur) * 0.5;

      // In single-kernel backfill mode, evaluate row tier against total busy time rather than single-kernel duration alone
      const busyActive = activeGroups.map(g => g.unitTotalBusyUs);
      const minActiveBusy = busyActive.length ? Math.min(...busyActive) : 0;
      const maxActiveBusy = busyActive.length ? Math.max(...busyActive) : 0;
      const groupBusyRatio = minActiveBusy > 0.01 ? maxActiveBusy / minActiveBusy : 1;
      const busySlowCut = minActiveBusy + (maxActiveBusy - minActiveBusy) * 0.5;
      const useBusyTier = !!(focusedBalance && focusedBalance.hasBg);

      for (const g of groups) {
        if (g.sliceCount === 0) g.tier = 'idle';
        else if (useBusyTier) {
          g.tier = (groupBusyRatio >= 1.32 && g.unitTotalBusyUs >= busySlowCut) ? 'slow' : 'fast';
        } else if (ratio >= 1.25 && g.totalDurUs >= slowCut) g.tier = 'slow';
        else g.tier = 'fast';
      }

      const activeSortBy = (state.rtSortBy === 'totalBusy' && !focusedKernel) ? 'duration' : state.rtSortBy;
      const displayGroups = (state.rtHideIdle ? activeGroups : groups).slice();
      if (activeSortBy === 'totalBusy') {
        displayGroups.sort((a, b) => (b.unitTotalBusyUs - a.unitTotalBusyUs) || (a.group - b.group));
      } else if (activeSortBy === 'duration') {
        displayGroups.sort((a, b) => (b.totalDurUs - a.totalDurUs) || (a.group - b.group));
      } else {
        displayGroups.sort((a, b) => a.group - b.group);
      }

      const niceAxisStep = span => {
        const candidates = [1, 2, 5, 10, 20, 25, 50, 100, 150, 200, 250, 500, 1000];
        const target = span / 5.5;
        return candidates.find(c => c >= target) || Math.ceil(target / 500) * 500;
      };
      const tickStep = niceAxisStep(rawSpanUs);
      const axisTicks = [];
      for (let t = 0; t <= rawSpanUs + 1e-6; t += tickStep) {
        const pct = (t / displaySpanUs) * 100;
        if (endPct - pct < 11 && t > 0) continue;
        axisTicks.push(`<span style="left:${pct.toFixed(2)}%">${t >= 10 || t === 0 ? Math.round(t) : t.toFixed(1)} µs</span>`);
      }

      const barStyle = (s, e) => {
        const left = Math.max(0, ((s - originUs) / displaySpanUs) * 100);
        const width = Math.max(0.32, ((e - s) / displaySpanUs) * 100);
        return `left:${left.toFixed(3)}%;width:${width.toFixed(3)}%`;
      };

      const renderBgBars = list => {
        if (!showBg || !list.length) return '';
        return list.map(s => {
          const clippedStart = Math.max(originUs, s.startUs);
          const clippedEnd = Math.min(endUs, s.endUs);
          const overlapUs = Math.max(0, clippedEnd - clippedStart);
          const coreLabel = s.core < 24 ? `AIC C${s.core}` : `AIV V${s.core}`;
          const scopeTag = s.crossStage ? `跨阶段 · ${s.stageLabel}` : '同阶段并发';
          const idPart = s.id != null ? ` · #${s.id}` : '';
          const tip = `${scopeTag} · ${s.name}${idPart} · ${coreLabel}\n相对 +${(clippedStart - originUs).toFixed(1)} → +${(clippedEnd - originUs).toFixed(1)} µs（窗内 ${overlapUs.toFixed(1)} µs${Math.abs(overlapUs - s.durUs) > 0.2 ? ` / 切片总 ${s.durUs.toFixed(1)} µs` : ''}）`;
          const cls = s.crossStage ? 'csa-rt-bar is-bg-context is-bg-cross' : 'csa-rt-bar is-bg-context';
          return `<span class="${cls}" style="${barStyle(clippedStart, clippedEnd)}" title="${esc(tip)}"></span>`;
        }).join('');
      };

      const renderSublaneBars = (list, roleClass, bgList = []) => renderBgBars(bgList) + list.map(s => {
        const isPicked = state.sliceId === s.id;
        const isStepDim = state.rtHighlightStep && state.rtHighlightStep !== s.name;
        const isSliceDim = state.sliceId != null && !isPicked && state.selectedGroup === coreGroupOf(s.core);
        const dimCls = isStepDim ? ' is-dimmed' : isSliceDim ? ' is-dim-slice' : '';
        const pickCls = isPicked ? ' is-picked-slice' : '';
        const stepColor = stepByName.get(s.name)?.color;
        const bgInline = multiStep && stepColor ? `;background:${stepColor}` : '';
        const colorCls = multiStep ? '' : ` ${roleClass}`;
        const coreLabel = s.core < 24 ? `AIC C${s.core}` : `AIV V${s.core}`;
        const tip = `${s.name} · #${s.id} · ${coreLabel}\n相对 +${(s.startUs - originUs).toFixed(1)} → +${(s.endUs - originUs).toFixed(1)} µs\n计算 ${s.computeUs.toFixed(1)} µs · Setup ${s.setupUs.toFixed(1)} µs`;
        if (showSetup && s.setupUs > displaySpanUs * 0.002) {
          const setupEnd = Math.min(s.endUs, s.startUs + s.setupUs);
          return (
            `<span class="csa-rt-bar is-setup${pickCls}${dimCls}" data-cmp-rt-slice="${s.id}" style="${barStyle(s.startUs, setupEnd)}" title="${esc(tip)}"></span>` +
            `<span class="csa-rt-bar${colorCls}${pickCls}${dimCls}" data-cmp-rt-slice="${s.id}" style="${barStyle(setupEnd, s.endUs)}${bgInline}" title="${esc(tip)}"></span>`
          );
        }
        return `<span class="csa-rt-bar${colorCls}${pickCls}${dimCls}" data-cmp-rt-slice="${s.id}" style="${barStyle(s.startUs, s.endUs)}${bgInline}" title="${esc(tip)}"></span>`;
      }).join('');

      const verdictLabelOf = v => (
        v === 'backfill' ? '动态补位'
          : v === 'partial' ? '并发缓解 · 仍偏斜'
            : v === 'skip' ? '物理跳核'
              : v === 'imbalance' ? '核级失衡'
                : '均衡'
      );

      const idleCount = 24 - activeGroups.length;
      let metricsParts;
      if (focusedKernel) {
        const staticCounts = groups.map(g => g.staticCount);
        const actualCounts = groups.map(g => g.sliceCount);
        const minStatic = Math.min(...staticCounts);
        const maxStatic = Math.max(...staticCounts);
        const minActual = Math.min(...actualCounts);
        const maxActual = Math.max(...actualCounts);
        const staticRange = minStatic === maxStatic ? `${minStatic}` : `${minStatic}–${maxStatic}`;
        const actualRange = minActual === maxActual ? `${minActual}` : `${minActual}–${maxActual}`;
        const busyTag = focusedBalance
          ? `同窗总忙时 <b>${Math.round(focusedBalance.minTotalBusy)}–${Math.round(focusedBalance.maxTotalBusy)} µs</b> (${focusedBalance.busySkew.toFixed(2)}× · ${verdictLabelOf(focusedBalance.verdict)})`
          : `Setup <b>${(stageSetupShare * 100).toFixed(1)}%</b>`;
        metricsParts = [
          `<b>${visible.length}</b> 块`,
          `参考 <b>${staticRange}</b> → 实测 <b>${actualRange}</b> 片/组`,
          busyTag,
        ];
      } else {
        metricsParts = [
          `<b>${steps.length}</b> 阶段`,
          `<b>${visible.length}</b> 块`,
          `Setup <b>${(stageSetupShare * 100).toFixed(1)}%</b>`,
          ...(stageSourceFile ? [esc(stageSourceFile)] : []),
        ];
      }

      const bgLegendItem = showBg
        ? (hasCrossBg
            ? '<span><i class="is-bg-context"></i>同阶段并发</span><span><i class="is-bg-cross"></i>跨阶段并发</span>'
            : '<span><i class="is-bg-context"></i>其他 Kernel 并发</span>')
        : '';
      const legendHtml = multiStep
        ? `${showSetup ? '<span><i class="is-setup"></i>Setup 等待</span>' : ''}${bgLegendItem}<span>点击下方阶段胶囊可高亮单步</span>`
        : `<span><i class="is-aic"></i>AIC</span><span><i class="is-aiv0"></i>AIV 0</span><span><i class="is-aiv1"></i>AIV 1</span>${showSetup ? '<span><i class="is-setup"></i>Setup 等待</span>' : ''}${bgLegendItem}`;

      const densitySwitchHtml = `<div class="segmented-control segmented-control-muted csa-rt-seg" role="group" aria-label="视图密度">
        <button type="button" class="btn btn-sm${state.rtDensity === 'compact' ? ' is-selected' : ''}" data-cmp-rt-density="compact" aria-pressed="${state.rtDensity === 'compact'}" title="72 核一屏压缩全景，叠加热点洞察图层">紧凑全景</button>
        <button type="button" class="btn btn-sm${state.rtDensity === 'expanded' ? ' is-selected' : ''}" data-cmp-rt-density="expanded" aria-pressed="${state.rtDensity === 'expanded'}" title="展开 24 个核组（1 AIC + 2 AIV）逐行查看明细与同窗总忙时">展开核组</button>
      </div>`;

      const controlsHtml = `<div class="csa-rt-controls">
        <div class="csa-rt-controls-left">
          ${densitySwitchHtml}
          <div class="segmented-control segmented-control-muted csa-rt-seg" role="group" aria-label="核组排序">
            <button type="button" class="btn btn-sm${activeSortBy === 'group' ? ' is-selected' : ''}" data-cmp-rt-sort="group" aria-pressed="${activeSortBy === 'group'}">Group 序</button>
            <button type="button" class="btn btn-sm${activeSortBy === 'duration' ? ' is-selected' : ''}" data-cmp-rt-sort="duration" aria-pressed="${activeSortBy === 'duration'}">${focusedKernel ? '本核时降序' : '耗时降序'}</button>
            ${focusedKernel && focusedBalance?.hasBg ? `<button type="button" class="btn btn-sm${activeSortBy === 'totalBusy' ? ' is-selected' : ''}" data-cmp-rt-sort="totalBusy" aria-pressed="${activeSortBy === 'totalBusy'}" title="按本 Kernel + 同窗并发背景的总忙时降序排列，检验核池整体是否均衡">总忙时降序</button>` : ''}
          </div>
          <div class="segmented-control segmented-control-muted csa-rt-seg" role="group" aria-label="叠加信号">
            <button type="button" class="btn btn-sm${showSetup ? ' is-selected' : ''}" data-cmp-rt-toggle="setup" aria-pressed="${showSetup}">Setup 拆分</button>
            ${state.filter !== 'all' && bgSlices.length > 0 ? `<button type="button" class="btn btn-sm${showBg ? ' is-selected' : ''}" data-cmp-rt-toggle="bg" aria-pressed="${showBg}" title="叠加同阶段与跨阶段在同一物理核上并发执行的背景切片，直观看出动态补位与抢核原因">并发背景 (${bgSlices.length})</button>` : ''}
            ${hasPrepKernels && state.filter === 'all' ? `<button type="button" class="btn btn-sm${state.rtIncludePrep ? ' is-selected' : ''}" data-cmp-rt-toggle="prep" aria-pressed="${state.rtIncludePrep}">含前序准备</button>` : ''}
            ${idleCount > 0 ? `<button type="button" class="btn btn-sm${state.rtHideIdle ? ' is-selected' : ''}" data-cmp-rt-toggle="hideIdle" aria-pressed="${state.rtHideIdle}">隐藏空组 (${idleCount})</button>` : ''}
          </div>
          <span class="csa-rt-metrics-inline">${metricsParts.join('<i>·</i>')}</span>
        </div>
        <div class="csa-rt-legend" aria-label="泳道图例">${legendHtml}</div>
      </div>`;

      // Stage step chips strip with real overlap / gap handoffs (aligned with Perf Lane)
      const stepStripHtml = multiStep ? `<div class="csa-rt-steps" aria-label="阶段流水步骤与交接">${steps.map((st, idx) => {
        const isAct = state.rtHighlightStep === st.name;
        const isDim = state.rtHighlightStep && !isAct;
        let linkHtml = '';
        if (idx > 0) {
          const prev = steps[idx - 1];
          const delta = st.startUs - prev.endUs;
          if (delta < -1) {
            linkHtml = `<span class="csa-rt-step-link is-overlap" title="${esc(prev.name)} 与 ${esc(st.name)} 存在真实重叠 ${Math.abs(delta).toFixed(1)} µs">⇢ 重叠 ${Math.abs(delta).toFixed(0)}µs</span>`;
          } else if (delta > 2) {
            linkHtml = `<span class="csa-rt-step-link is-gap" title="${esc(prev.name)} 结束至 ${esc(st.name)} 启动之间存在 ${delta.toFixed(1)} µs 间隔">→ 空窗 +${delta.toFixed(0)}µs</span>`;
          } else {
            linkHtml = `<span class="csa-rt-step-link">→</span>`;
          }
        }
        const relWin = `${Math.round(st.startUs - originUs)}–${Math.round(st.endUs - originUs)}µs`;
        const tip = `点击仅高亮 ${st.name}（${st.coreType} · ${st.count} 块 · 跨度 ${fmtUs(st.spanUs)} · Setup ${(st.setupUs / Math.max(1e-6, st.slotUs) * 100).toFixed(1)}%）`;
        return `${linkHtml}<button type="button" class="csa-rt-step-chip${isAct ? ' is-active' : ''}${isDim ? ' is-dimmed' : ''}" data-cmp-rt-step="${esc(st.name)}" aria-pressed="${isAct}" title="${esc(tip)}"><i style="background:${st.color}"></i><span>${esc(st.name)}</span><small>${st.coreType} · ${relWin}</small></button>`;
      }).join('')}</div>` : '';

      const axisHtml = `<div class="csa-rt-axis">
        <div class="csa-rt-col-id">Group</div>
        <div class="csa-rt-axis-track">
          ${axisTicks.join('')}
          <span class="csa-rt-end-flag is-right-aligned" style="left:${endPct.toFixed(2)}%" title="当前阶段最慢完成时刻（相对起点 ${rawSpanUs.toFixed(1)} µs）">${fmtShortUs(rawSpanUs)} 完成线</span>
        </div>
        <div class="csa-rt-col-stat"${focusedKernel ? ' title="实测片数、相对 Lane Map 均分差值，以及下方迷你条（实色=本 Kernel，斜纹=同窗并发背景，总长=同窗核总忙时）"' : ''}>${focusedKernel ? '实测 / 差值 · 总忙时' : '组跨度'}</div>
      </div>`;

      const fmtDelta = d => (d > 0 ? `+${d}` : d < 0 ? `−${Math.abs(d)}` : '±0');
      const isBackfillCalm = focusedBalance?.verdict === 'backfill';
      const deltaClsOf = d => {
        if (d === 0) return 'is-even';
        if (isBackfillCalm) return d > 0 ? 'is-bf-over' : 'is-bf-under';
        return d > 0 ? 'is-over' : 'is-under';
      };

      const maxBusyRef = focusedBalance ? focusedBalance.maxGroupTotalBusyUs : Math.max(1, maxActiveDur);

      const rowsHtml = displayGroups.map(g => {
        const isSel = state.selectedGroup === g.group;
        const tierCls = g.tier === 'slow' ? 'is-slow' : g.tier === 'idle' ? 'is-idle' : 'is-fast';
        const setupPctInt = Math.round(g.setupPct);
        let statCellHtml;
        let rowLabel;
        if (focusedKernel) {
          const dText = fmtDelta(g.deltaCount);
          const dCls = deltaClsOf(g.deltaCount);
          const fUs = g.unitFocusedUs;
          const bUs = g.unitBgUs;
          const totUs = g.unitTotalBusyUs;
          const fPct = Math.min(100, (fUs / maxBusyRef) * 100);
          const bPct = Math.min(100 - fPct, (bUs / maxBusyRef) * 100);
          const unitTag = focusedKernel.coreType;
          const verdictHint = focusedBalance ? verdictLabelOf(focusedBalance.verdict) : '';
          const barTip = `${focusedKernel.name} ${fUs.toFixed(1)} µs + 同窗${unitTag}并发背景 ${bUs.toFixed(1)} µs = 总忙时 ${totUs.toFixed(1)} µs`
            + (focusedBalance ? `\n全池中位 ${focusedBalance.medGroupBusyUs.toFixed(0)} µs · ${verdictHint}` : '');
          const deltaTip = `Lane Map 理想参考 ${g.staticCount} 片，Runtime 实测 ${g.sliceCount} 片（差值 ${dText}）\n${barTip}`;
          const focusBarCls = focusedKernel.coreType === 'AIC' ? 'is-focused is-aic' : 'is-focused is-aiv';
          statCellHtml = `<div class="csa-rt-stat-stack">
              <div class="csa-rt-stat-top">
                <span class="csa-rt-stat-main">${g.sliceCount}片</span>
                <span class="csa-rt-delta ${dCls}" title="${esc(deltaTip)}">${dText}</span>
                <span class="csa-rt-stat-dur" title="${esc(barTip)}">${totUs > 0.05 ? fmtShortUs(totUs) : '—'}</span>
              </div>
              <div class="csa-rt-busy-bar" title="${esc(barTip)}" aria-hidden="true">
                ${fPct > 0.2 ? `<i class="${focusBarCls}" style="width:${fPct.toFixed(2)}%"></i>` : ''}
                ${bPct > 0.2 ? `<i class="is-bg" style="width:${bPct.toFixed(2)}%"></i>` : ''}
              </div>
            </div>`;
          rowLabel = `Group ${String(g.group).padStart(2, '0')} · 实测 ${g.sliceCount} 片（参考 ${g.staticCount} 片，差值 ${dText}） · 本 Kernel ${fUs.toFixed(1)} µs + 并发背景 ${bUs.toFixed(1)} µs = 总忙时 ${totUs.toFixed(1)} µs`;
        } else {
          const secondStat = !g.sliceCount
            ? `<span class="csa-rt-stat-merge">0 块</span>`
            : showSetup
              ? `<span class="csa-rt-stat-merge${setupPctInt >= 30 ? ' is-warn' : ''}">${setupPctInt}%</span>`
              : `<span class="csa-rt-stat-merge">${g.sliceCount} 块</span>`;
          statCellHtml = `<span class="csa-rt-stat-main">${g.sliceCount ? fmtShortUs(g.spanUs) : '—'}</span>${secondStat}`;
          rowLabel = `Group ${String(g.group).padStart(2, '0')} · 跨度 ${g.spanUs.toFixed(1)} µs · Setup ${setupPctInt}% · ${g.sliceCount} 块`;
        }
        return `<div class="csa-rt-row ${tierCls}${isSel ? ' is-selected' : ''}" role="button" tabindex="0" data-cmp-rt-group="${g.group}" aria-pressed="${isSel}" aria-label="${esc(rowLabel)}">
          <div class="csa-rt-col-id">
            <span class="csa-rt-dot-tier ${tierCls}" aria-hidden="true"></span>
            <strong>G${String(g.group).padStart(2, '0')}</strong>
            <small>C${g.aicCore}</small>
          </div>
          <div class="csa-rt-track">
            <div class="csa-rt-end-overlay" aria-hidden="true"><span class="csa-rt-end-line" style="left:${endPct.toFixed(2)}%"></span></div>
            <div class="csa-rt-sublane">
              <span class="csa-rt-subtag${g.aicSlices.length ? '' : ' is-muted'}">C</span>
              <div class="csa-rt-lane-bg">${renderSublaneBars(g.aicSlices, 'is-aic', g.bgAicSlices)}</div>
            </div>
            <div class="csa-rt-sublane">
              <span class="csa-rt-subtag${g.aiv0Slices.length ? '' : ' is-muted'}">V0</span>
              <div class="csa-rt-lane-bg">${renderSublaneBars(g.aiv0Slices, 'is-aiv0', g.bgAiv0Slices)}</div>
            </div>
            <div class="csa-rt-sublane">
              <span class="csa-rt-subtag${g.aiv1Slices.length ? '' : ' is-muted'}">V1</span>
              <div class="csa-rt-lane-bg">${renderSublaneBars(g.aiv1Slices, 'is-aiv1', g.bgAiv1Slices)}</div>
            </div>
          </div>
          <div class="csa-rt-col-stat">
            ${statCellHtml}
          </div>
        </div>`;
      }).join('');

      // On-demand compact right inspector (aligned with Perf Lane & qk_pv)
      let inspector = '';
      if (state.selectedGroup != null) {
        const g = groups[state.selectedGroup];
        if (g) {
          const pickedHit = state.sliceId != null ? findSliceById(rankData, state.sliceId) : null;
          const allGroupSlices = [...g.aicSlices, ...g.aiv0Slices, ...g.aiv1Slices];
          const byKernel = new Map();
          for (const s of allGroupSlices) {
            const cur = byKernel.get(s.name) || {
              name: s.name,
              count: 0,
              durUs: 0,
              computeUs: 0,
              setupUs: 0,
              startUs: Infinity,
              unit: s.unit,
              color: stepByName.get(s.name)?.color || (s.unit === 'aic' ? '#80b7f7' : '#a87ef0'),
            };
            cur.count++;
            cur.durUs += s.durUs;
            cur.computeUs += s.computeUs;
            cur.setupUs += s.setupUs;
            cur.startUs = Math.min(cur.startUs, s.startUs);
            byKernel.set(s.name, cur);
          }
          const kernelBreakdown = [...byKernel.values()].sort((a, b) => a.startUs - b.startUs);
          const primaryKernelName = (pickedHit && byKernel.has(pickedHit.owner) ? pickedHit.owner : null)
            || kernelBreakdown[0]?.name
            || (kernelByName.has(state.filter) ? state.filter : currentKernelInfo().name);
          const primaryInfo = kernelByName.get(primaryKernelName) || currentKernelInfo();
          const targetLane = primaryInfo.coreType === 'AIC'
            ? g.aicCore
            : (g.aiv0Slices.some(s => s.name === primaryInfo.name) ? g.group * 2 : g.group * 2 + (g.aiv1Slices.some(s => s.name === primaryInfo.name) ? 1 : 0));

          const tierLabel = focusedKernel
            ? `${g.sliceCount} 片 (${fmtDelta(g.deltaCount)})`
            : (g.tier === 'idle' ? '空载组' : `${g.sliceCount} 块`);
          const tierBadgeCls = g.tier === 'slow' ? 'is-slow' : g.tier === 'idle' ? 'is-idle' : 'is-fast';

          let staticCompareSec = '';
          if (focusedKernel) {
            const dText = fmtDelta(g.deltaCount);
            const dCls = deltaClsOf(g.deltaCount);
            const laneHint = laneCountFor(focusedKernel) === 24
              ? `AIC L${String(g.group).padStart(2, '0')}`
              : `AIV L${String(g.group * 2).padStart(2, '0')}/${String(g.group * 2 + 1).padStart(2, '0')}`;
            const medBusy = focusedBalance?.medGroupBusyUs || 0;
            const busyPctDiff = medBusy > 0.1 ? ((g.unitTotalBusyUs - medBusy) / medBusy) * 100 : 0;
            const busyDiffText = `${busyPctDiff >= 0 ? '+' : '−'}${Math.abs(busyPctDiff).toFixed(1)}%`;
            const vBadgeCls = focusedBalance?.verdict === 'backfill' ? 'is-bf' : focusedBalance?.verdict === 'imbalance' ? 'is-warn' : 'is-neutral';
            const vText = focusedBalance ? verdictLabelOf(focusedBalance.verdict) : '';
            staticCompareSec = `<section class="csa-rt-insp-sec">
              <h4>静态参考 vs Runtime 动态落核 · ${esc(focusedKernel.name)}</h4>
              <dl class="csa-rt-dl csa-rt-dl-wide">
                <dt>Lane Map 参考</dt><dd>${g.staticCount} 片（${laneHint} · 理想轮询）</dd>
                <dt>Runtime 实测</dt><dd>${g.sliceCount} 片 <span class="csa-rt-delta ${dCls}">${dText}</span> · ${fmtUs(g.unitFocusedUs)}</dd>
                <dt>同窗并发背景</dt><dd>${g.unitBgUs > 0.05 ? fmtUs(g.unitBgUs) : '0 µs'}（同 ${focusedKernel.coreType} 核池）</dd>
                <dt>同窗总忙时</dt><dd><b>${fmtUs(g.unitTotalBusyUs)}</b> · 较中位 ${busyDiffText}${vText ? ` <span class="csa-rt-verdict ${vBadgeCls}">${esc(vText)}</span>` : ''}</dd>
              </dl>
            </section>`;
          }

          let bgBreakdownSec = '';
          if (showBg || focusedBalance?.hasBg) {
            const unitOnlyBg = focusedKernel
              ? (focusedKernel.coreType === 'AIC' ? g.bgAicSlices : [...g.bgAiv0Slices, ...g.bgAiv1Slices])
              : [...g.bgAicSlices, ...g.bgAiv0Slices, ...g.bgAiv1Slices];
            if (unitOnlyBg.length) {
              const bgByK = new Map();
              for (const s of unitOnlyBg) {
                const key = `${s.stageId || stage.id}::${s.name}`;
                const cur = bgByK.get(key) || {
                  name: s.name,
                  stageLabel: s.stageLabel || stageTitle,
                  crossStage: !!s.crossStage,
                  count: 0,
                  overlapUs: 0,
                  unit: s.unit,
                };
                cur.count++;
                cur.overlapUs += (s.overlapUs != null ? s.overlapUs : s.durUs);
                bgByK.set(key, cur);
              }
              const bgList = [...bgByK.values()].sort((a, b) => b.overlapUs - a.overlapUs);
              const totalBgOverlap = bgList.reduce((a, b) => a + b.overlapUs, 0);
              bgBreakdownSec = `<section class="csa-rt-insp-sec">
                <h4>同窗并发背景（${unitOnlyBg.length} 块 · ${fmtShortUs(totalBgOverlap)}）</h4>
                <div class="csa-rt-klist">${bgList.map(bk => {
                  const dotCls = bk.crossStage ? 'is-bg-cross' : 'is-bg-context';
                  const tagHtml = bk.crossStage ? `<em class="csa-rt-stage-tag">${esc(bk.stageLabel)}</em>` : '';
                  const canFocus = !bk.crossStage && kernelByName.has(bk.name);
                  if (canFocus) {
                    return `<button type="button" class="csa-rt-krow" data-cmp-rt-focus-kernel="${esc(bk.name)}" title="在 Runtime Trace 中切换聚焦 ${esc(bk.name)}">
                      <span class="csa-rt-kname"><i class="csa-rt-dot ${dotCls}"></i>${esc(bk.name)}</span>
                      <span class="csa-rt-kstat">${fmtShortUs(bk.overlapUs)} · ${bk.count}块</span>
                    </button>`;
                  }
                  return `<div class="csa-rt-krow is-readonly" title="跨阶段并发任务 · ${esc(bk.stageLabel)} · 窗内重叠 ${bk.overlapUs.toFixed(1)} µs">
                    <span class="csa-rt-kname"><i class="csa-rt-dot ${dotCls}"></i>${esc(bk.name)}${tagHtml}</span>
                    <span class="csa-rt-kstat">${fmtShortUs(bk.overlapUs)} · ${bk.count}块</span>
                  </div>`;
                }).join('')}</div>
              </section>`;
            }
          }

          let pickedSliceSec = '';
          if (pickedHit && coreGroupOf(pickedHit.rec[1]) === g.group) {
            const [sid, score, stime, sdur, skdur] = pickedHit.rec;
            const skInfo = kernelByName.get(pickedHit.owner);
            const ssetup = Math.max(0, sdur - (Number.isFinite(skdur) ? skdur : sdur));
            pickedSliceSec = `<section class="csa-rt-insp-sec">
              <h4>选中切片 · ${esc(pickedHit.owner)}</h4>
              <dl class="csa-rt-dl">
                <dt>事件 / 核</dt><dd>#${sid} · ${score < 24 ? 'AIC C' : 'AIV V'}${score}</dd>
                <dt>相对窗口</dt><dd>+${(stime - originUs).toFixed(1)}→+${(stime + sdur - originUs).toFixed(1)} µs</dd>
                <dt>计算 / Setup</dt><dd>${(sdur - ssetup).toFixed(1)} / ${ssetup.toFixed(1)} µs (${Math.round(ssetup / Math.max(1e-6, sdur) * 100)}%)</dd>
                <dt>逻辑规模</dt><dd>${blockCountOf(skInfo)} Blocks · ${skInfo.coreType}</dd>
              </dl>
            </section>`;
          }

          const kernelRowsHtml = kernelBreakdown.length
            ? `<div class="csa-rt-klist">${kernelBreakdown.map(k => {
                const sPct = Math.round((k.setupUs / Math.max(1e-6, k.durUs)) * 100);
                return `<button type="button" class="csa-rt-krow" data-cmp-drill-lane="${esc(k.name)}" data-cmp-drill-lane-idx="${k.unit === 'aic' ? g.group : g.group * 2}" title="切换到 ${esc(k.name)} 的 Lane Map">
                  <span class="csa-rt-kname"><i class="csa-rt-dot" style="background:${k.color}"></i>${esc(k.name)}</span>
                  <span class="csa-rt-kstat">${fmtShortUs(k.durUs)} · ${k.count}块${sPct >= 15 ? ` · S${sPct}%` : ''}</span>
                </button>`;
              }).join('')}</div>`
            : `<p class="csa-rt-note">当前筛选条件下该核组无执行切片。</p>`;

          inspector = `<aside class="csa-rt-inspector" aria-label="Core Group 实测详情">
            <header class="csa-rt-insp-head">
              <div>
                <span class="csa-rt-insp-kicker">RANK ${state.rank} · ${g.sliceCount ? `+${(g.firstStartUs - originUs).toFixed(0)}→+${(g.lastEndUs - originUs).toFixed(0)} µs` : '无活跃切片'}</span>
                <h3>Group ${String(g.group).padStart(2, '0')} <span class="csa-rt-tier ${tierBadgeCls}">${tierLabel}</span></h3>
              </div>
              <button type="button" class="btn btn-ghost btn-sm csa-rt-close" data-cmp-rt-close aria-label="关闭详情">×</button>
            </header>
            <div class="csa-rt-insp-body">
              ${staticCompareSec}
              <section class="csa-rt-insp-sec">
                <h4>核组实测（1 AIC + 2 AIV）</h4>
                <table class="csa-rt-mini-table">
                  <thead><tr><th>核</th><th>槽位核时</th><th>计算 / Setup</th></tr></thead>
                  <tbody>
                    <tr><td><i class="csa-rt-dot is-aic"></i>AIC · C${g.aicCore}</td><td>${g.aicSlices.length ? fmtUs(g.aicDurUs) : '—'}</td><td>${g.aicSlices.length ? `${g.aicComputeUs.toFixed(0)} / ${g.aicSetupUs.toFixed(0)} µs` : '0 块'}</td></tr>
                    <tr><td><i class="csa-rt-dot is-aiv0"></i>AIV 0 · V${g.aiv0Core}</td><td>${g.aiv0Slices.length ? fmtUs(g.aiv0DurUs) : '—'}</td><td>${g.aiv0Slices.length ? `${g.aiv0ComputeUs.toFixed(0)} / ${g.aiv0SetupUs.toFixed(0)} µs` : '0 块'}</td></tr>
                    <tr><td><i class="csa-rt-dot is-aiv1"></i>AIV 1 · V${g.aiv1Core}</td><td>${g.aiv1Slices.length ? fmtUs(g.aiv1DurUs) : '—'}</td><td>${g.aiv1Slices.length ? `${g.aiv1ComputeUs.toFixed(0)} / ${g.aiv1SetupUs.toFixed(0)} µs` : '0 块'}</td></tr>
                  </tbody>
                </table>
              </section>
              ${pickedSliceSec}
              <section class="csa-rt-insp-sec">
                <h4>本核组执行的阶段任务（${kernelBreakdown.length}）</h4>
                ${kernelRowsHtml}
              </section>
              ${bgBreakdownSec}
              <div class="csa-rt-actions">
                <button type="button" class="btn btn-sm csa-rt-cta-primary" data-cmp-drill-block="${esc(primaryInfo.name)}" data-cmp-drill-lane-idx="${targetLane}">进入单 Block 因果泳道 →</button>
                <button type="button" class="btn btn-ghost btn-sm" data-cmp-drill-lane="${esc(primaryInfo.name)}" data-cmp-drill-lane-idx="${targetLane}">查看 ${esc(primaryInfo.name)} Lane ${String(targetLane).padStart(2, '0')} →</button>
                <button type="button" class="btn btn-ghost btn-sm" data-cmp-source="${esc(primaryInfo.file)}:${primaryInfo.first}:${primaryInfo.last}">${esc(primaryInfo.file)} : L${primaryInfo.first}–${primaryInfo.last} ↗</button>
                <button type="button" class="btn btn-ghost btn-sm" data-cmp-tab="assignment:${esc(primaryInfo.name)}">在 Assignment 中查看</button>
                <button type="button" class="btn btn-ghost btn-sm" data-cmp-tab="graph:${esc(primaryInfo.name)}">在 Graph 中查看</button>
              </div>
            </div>
          </aside>`;
        }
      }

      if (state.rtDensity === 'compact') {
        const extBands = computeExternalBands(originUs, endUs);
        const parProfile = computeParallelismProfile(visible, originUs, rawSpanUs, 120);
        const insights = buildStageInsights(visible, steps, extBands, parProfile, originUs, rawSpanUs, allSlices, rankKey);
        const activeInsight = state.rtInsightId === null
          ? null
          : (state.rtInsightId === 'auto'
              ? (insights[0] || null)
              : (insights.find(item => item.id === state.rtInsightId) || insights[0] || null));

        const isGroupCluster = state.rtGroupBy !== 'type';
        // Match Device Perf Lane ultra-fine hairline geometry (ROW=8, BAR_H=2.6)
        const BAR_H = 2.6;
        const GROUP_H = 21; // 3 tracks per group (AIC / AIV / AIV) at y+2.5, y+8.5, y+14.5
        const TYPE_ROW = 8;
        const AIC_H = 24 * TYPE_ROW; // 192
        const AIV_TOP = AIC_H + 20; // 212
        const AIV_H = 48 * TYPE_ROW; // 384
        const TOTAL_H = isGroupCluster ? 24 * GROUP_H : AIV_TOP + AIV_H; // 504 | 596
        const compactShowSetup = state.rtShowSetup === true;

        const coreBarGeom = c => {
          if (isGroupCluster) {
            if (c < 24) return { y: c * GROUP_H + 2.5, h: BAR_H, rowY: c * GROUP_H, rowH: GROUP_H };
            const g = Math.floor((c - 24) / 2);
            const sub = (c - 24) % 2;
            return { y: g * GROUP_H + (sub === 0 ? 8.5 : 14.5), h: BAR_H, rowY: g * GROUP_H, rowH: GROUP_H };
          }
          if (c < 24) return { y: c * TYPE_ROW + (TYPE_ROW - BAR_H) / 2, h: BAR_H, rowY: c * TYPE_ROW, rowH: TYPE_ROW };
          const base = AIV_TOP + (c - 24) * TYPE_ROW;
          return { y: base + (TYPE_ROW - BAR_H) / 2, h: BAR_H, rowY: base, rowH: TYPE_ROW };
        };
        const xPctOf = t => Math.max(0, Math.min(100, ((t - originUs) / displaySpanUs) * 100));
        const wPctOf = (s, e) => Math.max(0.28, ((e - s) / displaySpanUs) * 100);

        const groupBySwitchHtml = `<div class="segmented-control segmented-control-muted csa-rt-seg" role="group" aria-label="全景聚类方式">
          <button type="button" class="btn btn-sm${isGroupCluster ? ' is-selected' : ''}" data-cmp-rt-groupby="group" aria-pressed="${isGroupCluster}" title="按 24 个物理核组聚类，每组 3 条细轨默认对应 AIC / AIV / AIV">按核组</button>
          <button type="button" class="btn btn-sm${!isGroupCluster ? ' is-selected' : ''}" data-cmp-rt-groupby="type" aria-pressed="${!isGroupCluster}" title="按硬件核型聚类（上 24 AIC / 下 48 AIV）">按核型</button>
        </div>`;

        const compactControlsHtml = `<div class="csa-rt-controls">
          <div class="csa-rt-controls-left">
            ${densitySwitchHtml}
            ${groupBySwitchHtml}
            <div class="segmented-control segmented-control-muted csa-rt-seg" role="group" aria-label="全景叠加图层">
              <button type="button" class="btn btn-sm${state.rtShowPar ? ' is-selected' : ''}" data-cmp-rt-toggle="par" aria-pressed="${state.rtShowPar}">并行宽度</button>
              ${extBands.length ? `<button type="button" class="btn btn-sm${state.rtShowExt ? ' is-selected' : ''}" data-cmp-rt-toggle="ext" aria-pressed="${state.rtShowExt}" title="叠加同时间窗内其他阶段的占核区段，区分跨阶段并发与真实空转">跨阶段占核 (${extBands.length})</button>` : ''}
              <button type="button" class="btn btn-sm${compactShowSetup ? ' is-selected' : ''}" data-cmp-rt-toggle="setup" aria-pressed="${compactShowSetup}">耗时构成</button>
              ${state.filter !== 'all' && bgSlices.length > 0 ? `<button type="button" class="btn btn-sm${showBg ? ' is-selected' : ''}" data-cmp-rt-toggle="bg" aria-pressed="${showBg}">并发背景 (${bgSlices.length})</button>` : ''}
              ${hasPrepKernels && state.filter === 'all' ? `<button type="button" class="btn btn-sm${state.rtIncludePrep ? ' is-selected' : ''}" data-cmp-rt-toggle="prep" aria-pressed="${state.rtIncludePrep}">含前序准备</button>` : ''}
            </div>
          </div>
          <div class="csa-rt-legend" aria-label="全景图例">
            <span><i class="is-pano-terrain"></i>Task block（底层）</span>
            <span><i class="is-pano-cand"></i>洞察关联</span>
            ${compactShowSetup ? '<span><i class="is-pano-setup"></i>Setup / 计算</span>' : ''}
            ${state.rtShowExt && extBands.length ? '<span><i class="is-pano-ext"></i>跨阶段占核</span>' : ''}
          </div>
        </div>`;

        // 4 compact top KPI metrics (mirrors Perf Lane header)
        let fourthKpiLabel = 'Setup / 核时';
        let fourthKpiValue = `${(stageSetupShare * 100).toFixed(1)}%`;
        let fourthKpiSub = `均值 ${fmtUs(totalSetupUs / Math.max(1, visible.length))} / 块`;
        if (focusedKernel) {
          const sCounts = groups.map(g => g.staticCount);
          const aCounts = groups.map(g => g.sliceCount);
          const sRange = Math.min(...sCounts) === Math.max(...sCounts) ? `${sCounts[0]}` : `${Math.min(...sCounts)}–${Math.max(...sCounts)}`;
          const aRange = Math.min(...aCounts) === Math.max(...aCounts) ? `${aCounts[0]}` : `${Math.min(...aCounts)}–${Math.max(...aCounts)}`;
          fourthKpiLabel = '静态参考 → 动态落核';
          fourthKpiValue = `${aRange} 片/组`;
          if (focusedBalance && focusedBalance.hasBg) {
            fourthKpiSub = `总忙时 ${focusedBalance.busySkew.toFixed(2)}× · ${verdictLabelOf(focusedBalance.verdict)}（参考 ${sRange}）`;
          } else {
            fourthKpiSub = `理想均分 ${sRange} 片 · 极差 ${ratio.toFixed(2)}×`;
          }
        }
        const kpiStripHtml = `<div class="csa-pano-kpis">
          <div class="csa-pano-kpi">
            <span>执行窗口</span>
            <strong>${fmtShortUs(rawSpanUs)}</strong>
            <small>${visible.length} 块 · ${steps.length} Kernel · 72 核</small>
          </div>
          <div class="csa-pano-kpi">
            <span>平均核并行度</span>
            <strong>${parProfile.avgTotalPct.toFixed(1)}%</strong>
            <small>AIC ${parProfile.avgAicPct.toFixed(0)}% · AIV ${parProfile.avgAivPct.toFixed(0)}%</small>
          </div>
          <div class="csa-pano-kpi">
            <span>低并行时段占比</span>
            <strong>${parProfile.lowParPct.toFixed(1)}%</strong>
            <small>≤4 核运行 · 峰值 ${parProfile.peak} 核</small>
          </div>
          <div class="csa-pano-kpi">
            <span>${fourthKpiLabel}</span>
            <strong>${fourthKpiValue}</strong>
            <small>${fourthKpiSub}</small>
          </div>
        </div>`;

        // Parallelism sparkline SVG
        let parSparkHtml = '';
        if (state.rtShowPar) {
          const pts = parProfile.bins.map((b, i) => {
            const x = ((i + 0.5) / parProfile.bins.length) * (endPct * 10);
            const y = 34 - (b.total / 72) * 32;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          });
          const endX = (endPct * 10).toFixed(1);
          const areaPath = pts.length ? `M 0,34 L ${pts.join(' L ')} L ${endX},34 Z` : '';
          const linePath = pts.length ? `M ${pts.join(' L ')}` : '';
          const refY = (34 - (18 / 72) * 32).toFixed(1);
          parSparkHtml = `<div class="csa-pano-par">
            <div class="csa-pano-rail-label">
              <span>并行宽度</span>
              <small>25% · 18 核</small>
            </div>
            <div class="csa-pano-par-track">
              <svg viewBox="0 0 1000 36" preserveAspectRatio="none" aria-hidden="true">
                <line x1="0" y1="${refY}" x2="1000" y2="${refY}" stroke="rgba(196,196,196,0.34)" stroke-dasharray="4 4" stroke-width="1"/>
                <path d="${areaPath}" fill="rgba(110,110,110,0.48)"/>
                <path d="${linePath}" fill="none" stroke="#a6a6a6" stroke-width="1.1"/>
              </svg>
              <span class="csa-pano-par-peak">峰值 ${parProfile.peak} / 72 核</span>
            </div>
          </div>`;
        }

        // Build denoised monochrome 72-core SVG terrain (Device Perf Lane palette: #565656 / #a8a8a8 / #f0f0f0)
        const svgGridLines = isGroupCluster
          ? ticks(24).map(g => {
              const y = g * GROUP_H;
              const zebra = g % 2 === 1 ? `<rect x="0" y="${y}" width="1000" height="${GROUP_H}" fill="rgba(255,255,255,0.012)"/>` : '';
              const line = g > 0 ? `<line x1="0" y1="${y}" x2="1000" y2="${y}" stroke="rgba(255,255,255,0.045)" stroke-width="1"/>` : '';
              return zebra + line;
            }).join('')
          : [
              ...[0, 4, 8, 12, 16, 20].map(c => `<line x1="0" y1="${c * TYPE_ROW}" x2="1000" y2="${c * TYPE_ROW}" stroke="rgba(255,255,255,0.028)" stroke-width="1"/>`),
              `<line x1="0" y1="${AIC_H + 10}" x2="1000" y2="${AIC_H + 10}" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>`,
              ...[24, 32, 40, 48, 56, 64].map(c => `<line x1="0" y1="${AIV_TOP + (c - 24) * TYPE_ROW}" x2="1000" y2="${AIV_TOP + (c - 24) * TYPE_ROW}" stroke="rgba(255,255,255,0.028)" stroke-width="1"/>`),
            ].join('');

        const svgBgRects = showBg
          ? bgSlices.map(s => {
              const cs = Math.max(originUs, s.startUs);
              const ce = Math.min(endUs, s.endUs);
              const overlapUs = Math.max(0, ce - cs);
              const x = (xPctOf(cs) * 10).toFixed(2);
              const w = Math.max(1.2, (overlapUs / displaySpanUs) * 1000).toFixed(2);
              const geom = coreBarGeom(s.core);
              const grp = coreGroupOf(s.core);
              const role = s.core < 24 ? `AIC (C${s.core})` : `AIV ${(s.core - 24) % 2} (V${s.core})`;
              const scopeTag = s.crossStage ? `跨阶段 · ${s.stageLabel}` : '同阶段并发';
              const idPart = s.id != null ? ` #${s.id}` : '';
              const tip = `G${String(grp).padStart(2, '0')} · ${role} · ${scopeTag} · ${s.name}${idPart} · 窗内 ${overlapUs.toFixed(1)} µs`;
              const fill = s.crossStage ? '#4d4646' : '#3a3a3a';
              const opacity = s.crossStage ? '0.62' : '0.55';
              return `<rect class="csa-pano-bg-slice${s.crossStage ? ' is-cross' : ''}" data-tip="${esc(tip)}" x="${x}" y="${geom.y.toFixed(2)}" width="${w}" height="${geom.h.toFixed(2)}" rx="0.8" fill="${fill}" fill-opacity="${opacity}"/>`;
            }).join('')
          : '';

        const svgSliceRects = visible.map(s => {
          const x = (xPctOf(s.startUs) * 10).toFixed(2);
          const w = Math.max(1.4, ((s.endUs - s.startUs) / displaySpanUs) * 1000).toFixed(2);
          const geom = coreBarGeom(s.core);
          const y = geom.y.toFixed(2);
          const h = geom.h.toFixed(2);
          const isPicked = state.sliceId === s.id;

          let isCand = false;
          if (state.rtHighlightStep) {
            isCand = s.name === state.rtHighlightStep;
          } else if (activeInsight && activeInsight.kernelName) {
            isCand = s.name === activeInsight.kernelName;
          }
          // Monochrome neutral nodes: #565656 base, #a8a8a8 insight candidate, #f0f0f0 selected
          const fill = isPicked ? '#f0f0f0' : isCand ? '#a8a8a8' : '#565656';
          const opacity = isPicked || isCand ? '0.96' : '0.85';

          const grp = coreGroupOf(s.core);
          const role = s.core < 24 ? `AIC (C${s.core})` : `AIV ${(s.core - 24) % 2} (V${s.core})`;
          const setupNote = s.setupUs >= 0.8 ? ` · Setup ${s.setupUs.toFixed(1)} µs` : '';
          const tip = `G${String(grp).padStart(2, '0')} · ${role} · ${s.name} #${s.id} · ${s.durUs.toFixed(1)} µs (+${(s.startUs - originUs).toFixed(1)}→+${(s.endUs - originUs).toFixed(1)} µs)${setupNote}`;

          const wantsSetupSplit = (compactShowSetup || (activeInsight?.kind === 'setup' && isCand)) && s.setupUs > displaySpanUs * 0.002;
          if (wantsSetupSplit) {
            const setupEnd = Math.min(s.endUs, s.startUs + s.setupUs);
            const sw = Math.max(1.2, ((setupEnd - s.startUs) / displaySpanUs) * 1000).toFixed(2);
            const cx = (xPctOf(setupEnd) * 10).toFixed(2);
            const cw = Math.max(1.2, ((s.endUs - setupEnd) / displaySpanUs) * 1000).toFixed(2);
            const sFill = isCand || compactShowSetup ? '#9a7639' : '#4a4a4a';
            const cFill = isCand || compactShowSetup ? '#dfbd69' : '#565656';
            return `<g data-cmp-rt-slice="${s.id}" data-tip="${esc(tip)}" class="csa-pano-slice"><rect x="${x}" y="${y}" width="${sw}" height="${h}" rx="0.8" fill="${sFill}" fill-opacity="0.92"/><rect x="${cx}" y="${y}" width="${cw}" height="${h}" rx="0.8" fill="${cFill}" fill-opacity="0.96"/></g>`;
          }
          return `<rect class="csa-pano-slice" data-cmp-rt-slice="${s.id}" data-tip="${esc(tip)}" x="${x}" y="${y}" width="${w}" height="${h}" rx="0.8" fill="${fill}" fill-opacity="${opacity}"/>`;
        }).join('');

        // Background cross-stage contention bands (HTML layer)
        const extBandsHtml = state.rtShowExt
          ? extBands.map(b => {
              const l = xPctOf(b.startUs);
              const w = wPctOf(b.startUs, b.endUs);
              const topPct = isGroupCluster ? 0 : (b.coreType === 'AIC' ? 0 : b.coreType === 'AIV' ? (AIV_TOP / TOTAL_H) * 100 : 0);
              const hPct = isGroupCluster ? 100 : (b.coreType === 'AIC' ? (AIC_H / TOTAL_H) * 100 : b.coreType === 'AIV' ? (AIV_H / TOTAL_H) * 100 : 100);
              const tip = `跨阶段并发占核 · ${b.stageLabel} · ${b.names.join(', ')} (${b.coreType}) · +${Math.round(b.startUs - originUs)}→+${Math.round(b.endUs - originUs)} µs`;
              const showMicro = !activeInsight && w >= 12;
              return `<div class="csa-pano-ext-band" style="left:${l.toFixed(2)}%;width:${w.toFixed(2)}%;top:${topPct.toFixed(2)}%;height:${hPct.toFixed(2)}%" title="${esc(tip)}">${showMicro ? `<span>${esc(b.names[0])}</span>` : ''}</div>`;
            }).join('')
          : '';

        // Single Active Insight Semantic Overlay (strictly 1 active at a time => zero overlapping labels)
        let insightOverlayHtml = '';
        if (activeInsight) {
          const idxNum = String(insights.findIndex(x => x.id === activeInsight.id) + 1).padStart(2, '0');
          const l = xPctOf(activeInsight.startUs);
          const w = Math.max(1.5, wPctOf(activeInsight.startUs, activeInsight.endUs));
          const isGap = activeInsight.kind === 'gap';
          const topPct = isGroupCluster ? 0 : (activeInsight.unit === 'aic' ? 0 : activeInsight.unit === 'aiv' ? (AIV_TOP / TOTAL_H) * 100 : 0);
          const hPct = isGroupCluster ? 100 : (activeInsight.unit === 'aic' ? (AIC_H / TOTAL_H) * 100 : activeInsight.unit === 'aiv' ? (AIV_H / TOTAL_H) * 100 : 100);
          const alignRight = l > 56;
          const labelLeft = alignRight ? Math.min(98, l + w) : Math.max(1, l);
          const labelPosStyle = alignRight
            ? `right:${Math.max(1, 100 - labelLeft).toFixed(2)}%`
            : `left:${labelLeft.toFixed(2)}%`;

          let extraFocusHtml = '';
          if (activeInsight.kind === 'skew' && activeInsight.hotCore != null) {
            const hg = coreBarGeom(activeInsight.hotCore);
            const hy = (hg.rowY / TOTAL_H) * 100;
            const hh = (hg.rowH / TOTAL_H) * 100;
            const grpTag = isGroupCluster ? `G${String(coreGroupOf(activeInsight.hotCore)).padStart(2, '0')} · ` : '';
            const bfCls = activeInsight.verdict === 'backfill' ? ' is-backfill' : '';
            extraFocusHtml = `<div class="csa-pano-hot-row${bfCls}" style="top:${hy.toFixed(2)}%;height:${Math.max(2.2, hh).toFixed(2)}%" title="${esc(grpTag + activeInsight.hotLabel)}"><span>${esc(grpTag + activeInsight.hotLabel)}</span></div>`;
          } else if (activeInsight.kind === 'outlier' && activeInsight.slice) {
            const sl = activeInsight.slice;
            const sx = xPctOf(sl.startUs);
            const sw = Math.max(1.4, wPctOf(sl.startUs, sl.endUs));
            const sg = coreBarGeom(sl.core);
            const sy = (sg.y / TOTAL_H) * 100;
            const sh = (sg.h / TOTAL_H) * 100;
            const targetLane = sl.core < 24 ? sl.core : sl.core - 24;
            const pillSide = sx > 55 ? `right:${Math.max(1, 100 - sx - sw).toFixed(2)}%` : `left:${Math.max(1, sx).toFixed(2)}%`;
            const pillTop = sy > 78 ? Math.max(4, sy - 9) : Math.min(88, sy + Math.max(3, sh) + 1.5);
            extraFocusHtml = `<div class="csa-pano-outlier-box" style="left:${sx.toFixed(2)}%;width:${sw.toFixed(2)}%;top:${sy.toFixed(2)}%;height:${Math.max(2.2, sh).toFixed(2)}%"></div>`
              + `<button type="button" class="csa-pano-outlier-cta" style="${pillSide};top:${pillTop.toFixed(2)}%" data-cmp-drill-block="${esc(activeInsight.kernelName)}" data-cmp-drill-lane-idx="${targetLane}">${esc(activeInsight.callout)} →</button>`;
          }

          insightOverlayHtml = `<div class="csa-pano-insight-zone${isGap ? ' is-gap' : ''}" style="left:${l.toFixed(2)}%;width:${w.toFixed(2)}%;top:${topPct.toFixed(2)}%;height:${hPct.toFixed(2)}%"></div>`
            + `<div class="csa-pano-callout${isGap ? ' is-gap' : ''}" style="${labelPosStyle};top:${Math.min(82, topPct + 1.5).toFixed(2)}%"><strong>${idxNum} · ${esc(activeInsight.tag)}</strong><span>${esc(activeInsight.callout)}</span></div>`
            + extraFocusHtml;
        }

        const panoYAxisHtml = isGroupCluster
          ? [0, 3, 6, 9, 12, 15, 18, 21, 23].map(g =>
              `<span style="top:${((g * GROUP_H + GROUP_H * 0.45) / TOTAL_H * 100).toFixed(2)}%">G${String(g).padStart(2, '0')}</span>`
            ).join('')
          : `<em class="csa-pano-ygroup" style="top:1.2%">AIC</em>`
            + [0, 4, 8, 12, 16, 20, 22].map(c => `<span style="top:${((c * TYPE_ROW + TYPE_ROW * 0.35) / TOTAL_H * 100).toFixed(2)}%">${c}</span>`).join('')
            + `<em class="csa-pano-ygroup" style="top:${((AIV_TOP / TOTAL_H) * 100 - 1.5).toFixed(2)}%">AIV</em>`
            + [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68].map(c => `<span style="top:${((AIV_TOP + (c - 24) * TYPE_ROW + TYPE_ROW * 0.35) / TOTAL_H * 100).toFixed(2)}%">${c}</span>`).join('');

        // Right Insight Rail (or Group Inspector when a slice is clicked)
        let rightRailHtml = '';
        if (inspector) {
          rightRailHtml = inspector.replace(
            '<button type="button" class="btn btn-ghost btn-sm csa-rt-close" data-cmp-rt-close aria-label="关闭详情">×</button>',
            `<button type="button" class="btn btn-ghost btn-sm" data-cmp-rt-close>← 洞察 (${insights.length})</button>`
          );
        } else {
          const cardsHtml = insights.length
            ? insights.map((ins, idx) => {
                const isAct = activeInsight?.id === ins.id;
                const num = String(idx + 1).padStart(2, '0');
                const kInfo = ins.kernelName ? kernelByName.get(ins.kernelName) : null;
                const laneIdx = ins.hotCore != null
                  ? (ins.hotCore < 24 ? ins.hotCore : ins.hotCore - 24)
                  : (ins.slice ? (ins.slice.core < 24 ? ins.slice.core : ins.slice.core - 24) : 0);
                const actionsHtml = isAct
                  ? `<div class="csa-pano-card-actions">
                      ${kInfo && state.filter !== kInfo.name ? `<button type="button" class="btn btn-sm" data-cmp-rt-focus-kernel="${esc(kInfo.name)}">聚焦 ${esc(kInfo.name)}</button>` : ''}
                      ${kInfo ? `<button type="button" class="btn btn-ghost btn-sm" data-cmp-drill-lane="${esc(kInfo.name)}" data-cmp-drill-lane-idx="${laneIdx}">对照 Lane Map →</button>` : ''}
                      ${kInfo ? `<button type="button" class="btn btn-ghost btn-sm" data-cmp-drill-block="${esc(kInfo.name)}" data-cmp-drill-lane-idx="${laneIdx}">单 Block 流水 →</button>` : ''}
                      <button type="button" class="btn btn-ghost btn-sm" data-cmp-rt-density="expanded">展开 24 核组明细</button>
                    </div>`
                  : '';
                const tagCls = ins.kind === 'gap'
                  ? ' is-gap'
                  : (ins.kind === 'setup' || ins.verdict === 'imbalance' || ins.verdict === 'partial')
                    ? ' is-setup'
                    : ins.verdict === 'backfill'
                      ? ' is-backfill'
                      : '';
                return `<div class="csa-pano-card${isAct ? ' is-active' : ''}" role="button" tabindex="0" data-cmp-rt-insight="${esc(ins.id)}" data-cmp-rt-insight-kernel="${esc(ins.kernelName || '')}" aria-pressed="${isAct}">
                  <div class="csa-pano-card-head">
                    <span class="csa-pano-card-idx">${num}</span>
                    <b>${esc(ins.title)}</b>
                    <span class="csa-pano-card-tag${tagCls}">${esc(ins.tag)}</span>
                  </div>
                  <p class="csa-pano-card-sub">${esc(ins.subtitle)}</p>
                  ${isAct ? `<p class="csa-pano-card-detail">${esc(ins.detail)}</p>` : ''}
                  ${actionsHtml}
                </div>`;
              }).join('')
            : `<p class="csa-rt-note">当前窗口内 72 核负载均衡，未检测到显著瓶颈。</p>`;

          rightRailHtml = `<aside class="csa-pano-rail" aria-label="性能洞察">
            <header class="csa-pano-rail-head">
              <div>
                <h3>性能洞察</h3>
                <small>系统推荐 · ${insights.length} 处关注点</small>
              </div>
              ${activeInsight ? '<button type="button" class="btn btn-ghost btn-sm" data-cmp-rt-clear-insight>清除叠层</button>' : ''}
            </header>
            <div class="csa-pano-rail-body">${cardsHtml}</div>
          </aside>`;
        }

        return `<div class="csa-cmp-rt-host">
          <div class="csa-rt-shell has-inspector csa-pano-shell">
            <div class="csa-rt-main">
              ${compactControlsHtml}
              ${kpiStripHtml}
              <div class="csa-pano-board">
                ${parSparkHtml}
                <div class="csa-pano-axis">
                  <div class="csa-pano-rail-label"><span>${isGroupCluster ? 'Group' : 'Core'}</span></div>
                  <div class="csa-rt-axis-track">
                    ${axisTicks.join('')}
                    <span class="csa-rt-end-flag is-right-aligned" style="left:${endPct.toFixed(2)}%">${fmtShortUs(rawSpanUs)} 完成线</span>
                  </div>
                </div>
                <div class="csa-pano-canvas">
                  <div class="csa-pano-yaxis" aria-hidden="true">
                    ${panoYAxisHtml}
                  </div>
                  <div class="csa-pano-plot" data-cmp-pano-plot data-group-by="${isGroupCluster ? 'group' : 'type'}" data-total-h="${TOTAL_H}">
                    <div class="csa-rt-end-overlay" aria-hidden="true"><span class="csa-rt-end-line" style="left:${endPct.toFixed(2)}%"></span></div>
                    ${extBandsHtml}
                    <svg class="csa-pano-svg" viewBox="0 0 1000 ${TOTAL_H}" preserveAspectRatio="none">
                      ${svgGridLines}
                      ${svgBgRects}
                      ${svgSliceRects}
                    </svg>
                    ${insightOverlayHtml}
                    <div class="csa-pano-tip" data-cmp-pano-tip hidden></div>
                  </div>
                </div>
                <div class="csa-pano-foot">
                  <span>${isGroupCluster ? '按 24 个物理核组聚类（每组 3 条细轨，悬停提示核型与耗时）' : '按物理核型聚类（24 AIC + 48 AIV，悬停看耗时分解）'} · 点击色块查详情，或点右侧卡片切换洞察叠层</span>
                </div>
              </div>
            </div>
            ${rightRailHtml}
          </div>
        </div>`;
      }

      return `<div class="csa-cmp-rt-host">
        <div class="csa-rt-shell${inspector ? ' has-inspector' : ''}">
          <div class="csa-rt-main">
            ${controlsHtml}
            ${stepStripHtml}
            <div class="csa-rt-board${focusedKernel ? ' is-single-kernel' : ''}">
              ${axisHtml}
              <div class="csa-rt-rows">${rowsHtml}</div>
            </div>
          </div>
          ${inspector}
        </div>
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
        ? `逻辑 Lane ${String(state.selectedLane).padStart(2, '0')} · ${selBlocks.length} Blocks · block_idx = ${state.selectedLane} + iter × ${laneCount}${st ? ` · Rank ${state.rank} 实测中位 ${st.medDurUs} µs` : ''}（实际落核由 Runtime 动态派发）`
        : `${info.name}（${totalBlocks} Blocks）按 block_idx % ${laneCount} 映射至 ${laneCount} 条逻辑 ${info.coreType} Lanes（理想均分参考，非物理核绑定）；点击任意 Block 进入 Block Pipeline`;

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

    // ---------- Main Render ----------
    function render({ preserveRtScroll = false } = {}) {
      const prevRtScroll = preserveRtScroll ? host.querySelector('.csa-rt-rows')?.scrollTop : null;
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
        if (prevRtScroll != null) {
          const rowsEl = host.querySelector('.csa-rt-rows');
          if (rowsEl) rowsEl.scrollTop = prevRtScroll;
        }
      } else {
        instanceCtrl = null;
        applyMapCamera();
      }
    }

    // ---------- Events ----------
    host.addEventListener('click', event => {
      const target = event.target.closest('button,[data-cmp-block],[data-cmp-lane],[data-cmp-rt-slice],[data-cmp-rt-group],[data-cmp-rt-insight],[data-cmp-struct-node]');
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
          if (next === 'lane') {
            if (isBranchFilter(state.filter)) {
              state.branch = state.filter;
              if (currentKernelInfo().branch !== state.branch) {
                const firstInBranch = stage.kernels.find(k => kernelByName.get(k.name)?.branch === state.branch);
                if (firstInBranch) setKernel(firstInBranch.name);
              }
              options.onViewChange?.({ branch: state.branch });
            } else {
              const syncKernel = resolveRuntimeKernel();
              if (syncKernel) setKernel(syncKernel);
            }
            const targetLane = laneForGroup(currentKernelInfo(), state.selectedGroup, state.sliceId);
            if (targetLane != null) {
              state.selectedLane = targetLane;
              state.lane = targetLane;
            }
          } else if (next === 'runtime' && kernelByName.has(state.kernel)) {
            state.filter = state.kernel;
            state.rtHighlightStep = null;
            state.rtShowSetup = null;
            state.rtInsightId = 'auto';
            state.rtInsightKernel = null;
            state.sliceId = null;
            state.selectedGroup = groupForLane(currentKernelInfo(), state.selectedLane);
          }
          render();
        }
        return;
      }
      if (target.hasAttribute('data-cmp-open-causal')) {
        if (state.overviewMode === 'runtime') {
          const syncKernel = resolveRuntimeKernel();
          if (syncKernel) setKernel(syncKernel);
        }
        const info = currentKernelInfo();
        const laneTarget = state.overviewMode === 'runtime'
          ? (laneForGroup(info, state.selectedGroup, state.sliceId) ?? state.selectedLane ?? state.lane)
          : (state.selectedLane ?? state.lane);
        const blocks = laneBlocksFor(model, info, laneTarget, state.tokens);
        const blk = blocks[0] ?? 0;
        openBlock(blk, 'instances');
        return;
      }
      if (target.dataset.cmpScale) {
        const next = target.dataset.cmpScale;
        if (next === 'runtime') {
          openRuntimeTrace(kernelByName.has(state.kernel) ? state.kernel : 'all');
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
        const laneIdx = target.dataset.cmpDrillLaneIdx != null ? Number(target.dataset.cmpDrillLaneIdx) : null;
        openLaneMap(target.dataset.cmpDrillLane, Number.isFinite(laneIdx) ? laneIdx : null);
        return;
      }
      if (target.dataset.cmpDrillBlock) {
        setKernel(target.dataset.cmpDrillBlock);
        const info = currentKernelInfo();
        const laneIdx = target.dataset.cmpDrillLaneIdx != null ? Number(target.dataset.cmpDrillLaneIdx) : 0;
        const blocks = laneBlocksFor(model, info, Number.isFinite(laneIdx) ? laneIdx : 0, state.tokens);
        openBlock(blocks[0] ?? 0, 'instances');
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
      if (target.dataset.cmpRtSort) {
        state.rtSortBy = target.dataset.cmpRtSort;
        render();
        return;
      }
      if (target.dataset.cmpRtDensity) {
        const nextD = target.dataset.cmpRtDensity;
        if (nextD !== state.rtDensity) {
          state.rtDensity = nextD;
          render();
        }
        return;
      }
      if (target.dataset.cmpRtGroupby) {
        const nextG = target.dataset.cmpRtGroupby;
        if (nextG !== state.rtGroupBy) {
          state.rtGroupBy = nextG;
          render();
        }
        return;
      }
      if (target.dataset.cmpRtInsight) {
        const nextId = target.dataset.cmpRtInsight;
        const toggledOn = state.rtInsightId !== nextId;
        state.rtInsightId = toggledOn ? nextId : null;
        const insKernel = toggledOn ? (target.dataset.cmpRtInsightKernel || null) : null;
        state.rtInsightKernel = insKernel && kernelByName.has(insKernel) ? insKernel : null;
        if (state.rtInsightKernel) setKernel(state.rtInsightKernel);
        state.rtHighlightStep = null;
        state.selectedGroup = null;
        state.sliceId = null;
        render();
        return;
      }
      if (target.hasAttribute('data-cmp-rt-clear-insight')) {
        state.rtInsightId = null;
        state.rtInsightKernel = null;
        state.rtHighlightStep = null;
        render();
        return;
      }
      if (target.dataset.cmpRtStep) {
        const stepName = target.dataset.cmpRtStep;
        state.rtHighlightStep = state.rtHighlightStep === stepName ? null : stepName;
        if (state.rtHighlightStep && kernelByName.has(state.rtHighlightStep)) {
          setKernel(state.rtHighlightStep);
        }
        render({ preserveRtScroll: true });
        return;
      }
      if (target.dataset.cmpRtToggle) {
        const key = target.dataset.cmpRtToggle;
        if (key === 'setup') {
          const curPressed = target.getAttribute('aria-pressed') === 'true';
          state.rtShowSetup = !curPressed;
        } else if (key === 'par') {
          state.rtShowPar = !state.rtShowPar;
        } else if (key === 'ext') {
          state.rtShowExt = !state.rtShowExt;
        } else if (key === 'bg') {
          state.rtShowBg = !state.rtShowBg;
        } else if (key === 'prep') {
          state.rtIncludePrep = !state.rtIncludePrep;
          state.rtHighlightStep = null;
          state.rtShowSetup = null;
          state.rtInsightId = 'auto';
          state.sliceId = null;
          state.selectedGroup = null;
        } else if (key === 'hideIdle') {
          state.rtHideIdle = !state.rtHideIdle;
        }
        render({ preserveRtScroll: true });
        return;
      }
      if (target.dataset.cmpRtFocusKernel) {
        const kName = target.dataset.cmpRtFocusKernel;
        if (kernelByName.has(kName)) {
          setKernel(kName);
          state.filter = kName;
          state.sliceId = null;
          state.rtHighlightStep = null;
          state.rtShowSetup = null;
          state.rtInsightId = 'auto';
          render({ preserveRtScroll: true });
        }
        return;
      }
      if (target.dataset.cmpRtSlice != null) {
        const id = Number(target.dataset.cmpRtSlice);
        const hit = findSliceById(currentRankData(), id);
        if (hit) {
          setKernel(hit.owner);
          state.selectedGroup = coreGroupOf(hit.rec[1]);
          state.sliceId = state.sliceId === id ? null : id;
        }
        render({ preserveRtScroll: true });
        return;
      }
      if (target.dataset.cmpRtGroup != null) {
        const g = Number(target.dataset.cmpRtGroup);
        state.selectedGroup = state.selectedGroup === g ? null : g;
        state.sliceId = null;
        render({ preserveRtScroll: true });
        return;
      }
      if (target.hasAttribute('data-cmp-rt-close')) {
        state.sliceId = null;
        state.selectedGroup = null;
        render({ preserveRtScroll: true });
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
          state.selectedGroup = null;
          options.onRankChange?.(r);
          render();
        }
      } else if (el.matches('[data-cmp-rt-filter]')) {
        state.filter = el.value;
        state.sliceId = null;
        state.selectedGroup = null;
        state.rtHighlightStep = null;
        state.rtShowSetup = null;
        state.rtInsightId = 'auto';
        state.rtInsightKernel = null;
        if (kernelByName.has(state.filter)) {
          setKernel(state.filter);
        } else if (isBranchFilter(state.filter)) {
          state.branch = state.filter;
          if (currentKernelInfo().branch !== state.branch) {
            const first = stage.kernels.find(k => kernelByName.get(k.name)?.branch === state.branch);
            if (first) setKernel(first.name);
          }
          options.onViewChange?.({ branch: state.branch });
        } else if (state.filter === 'all') {
          state.branch = 'all';
          options.onViewChange?.({ branch: 'all' });
        }
        render();
      } else if (el.matches('[data-cmp-branch]')) {
        state.branch = el.value;
        if (state.branch !== 'all' && currentKernelInfo().branch !== state.branch) {
          const first = [...kernelByName.values()].find(k => k.branch === state.branch);
          if (first) setKernel(first.name);
        }
        options.onViewChange?.({ branch: state.branch });
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
        options.onViewChange?.({ tokens: state.tokens, tileIndex: state.tileIndex });
        render();
      } else if (el.matches('[data-cmp-tile]')) {
        state.tileIndex = Number(el.value);
        state.structSelection = null;
        structCamera = null;
        options.onViewChange?.({ tileIndex: state.tileIndex });
        render();
      }
    });

    host.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const target = event.target;
      if (target.matches('[data-cmp-block],[data-cmp-lane],[data-cmp-rt-slice],[data-cmp-rt-group],[data-cmp-struct-node]')) {
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

    host.addEventListener('wheel', event => {
      if (event.target.closest('.csa-qp-stage-inner') && structCamera) {
        event.preventDefault();
        zoomStructCamera(Math.exp(Math.max(-0.25, Math.min(0.25, -event.deltaY * 0.002))), event.clientX, event.clientY);
      }
    }, { passive: false });

    // Compact panorama hover micro-tooltip (reveals AIC/AIV core role & slice info on demand)
    host.addEventListener('mousemove', event => {
      const plot = event.target.closest?.('[data-cmp-pano-plot]');
      if (!plot) {
        const tipEl = host.querySelector('[data-cmp-pano-tip]');
        if (tipEl && !tipEl.hidden) tipEl.hidden = true;
        return;
      }
      const tipEl = plot.querySelector('[data-cmp-pano-tip]');
      if (!tipEl) return;
      const rect = plot.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const sliceEl = event.target.closest?.('[data-tip]');
      let text = sliceEl?.getAttribute('data-tip') || '';
      if (!text) {
        const yRatio = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
        const mode = plot.dataset.groupBy;
        if (mode === 'group') {
          const yUnit = yRatio * 504;
          const g = Math.max(0, Math.min(23, Math.floor(yUnit / 21)));
          const inG = yUnit - g * 21;
          const sub = inG < 6.8 ? 0 : inG < 12.8 ? 1 : 2;
          const role = sub === 0 ? `AIC · C${g}` : `AIV ${sub - 1} · V${24 + g * 2 + (sub - 1)}`;
          text = `G${String(g).padStart(2, '0')} · ${role}`;
        } else {
          const yUnit = yRatio * 596;
          if (yUnit < 192) {
            const c = Math.max(0, Math.min(23, Math.floor(yUnit / 8)));
            text = `AIC · C${c} (G${String(c).padStart(2, '0')})`;
          } else if (yUnit >= 212) {
            const idx = Math.max(0, Math.min(47, Math.floor((yUnit - 212) / 8)));
            const c = 24 + idx;
            text = `AIV ${idx % 2} · V${c} (G${String(Math.floor(idx / 2)).padStart(2, '0')})`;
          }
        }
      }
      if (!text) {
        tipEl.hidden = true;
        return;
      }
      tipEl.textContent = text;
      tipEl.hidden = false;
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const nearRight = px > rect.width * 0.62;
      const nearBottom = py > rect.height - 34;
      tipEl.style.left = nearRight ? 'auto' : `${Math.round(px + 12)}px`;
      tipEl.style.right = nearRight ? `${Math.round(rect.width - px + 12)}px` : 'auto';
      tipEl.style.top = `${Math.max(4, Math.round(nearBottom ? py - 26 : py - 10))}px`;
    });
    host.addEventListener('mouseleave', () => {
      const tipEl = host.querySelector('[data-cmp-pano-tip]');
      if (tipEl && !tipEl.hidden) tipEl.hidden = true;
    });

    function update(next = {}) {
      if (next.rank != null && next.rank !== state.rank) {
        state.rank = next.rank;
        state.sliceId = null;
        state.selectedGroup = null;
      }
      const kernelChanged = !!(next.kernel && kernelByName.has(next.kernel) && next.kernel !== state.kernel);
      if (next.kernel && kernelByName.has(next.kernel)) {
        state.kernel = next.kernel;
        const info = currentKernelInfo();
        if (state.branch !== 'all' && info.branch !== state.branch) state.branch = 'all';
        if (kernelChanged) {
          state.lastBlock = null;
          state.selectedLane = null;
          state.mapGroupFilter = '';
          state.mapPartFilter = '';
          state.structSelection = null;
          mapCamera = null;
          structCamera = null;
        }
      }
      if (next.branch != null && (next.branch === 'all' || currentKernelInfo().branch === next.branch)) {
        state.branch = next.branch;
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
      if (next.filter) {
        state.filter = next.filter;
        state.rtHighlightStep = null;
        state.rtShowSetup = null;
        state.rtInsightId = 'auto';
        state.rtInsightKernel = null;
      } else if (kernelChanged && (state.filter !== 'all' || next.syncFilter)) {
        state.filter = state.kernel;
        state.rtHighlightStep = null;
        state.rtShowSetup = null;
        state.rtInsightId = 'auto';
        state.rtInsightKernel = null;
        state.sliceId = null;
        state.selectedGroup = null;
      }
      if (next.sliceId !== undefined) {
        state.sliceId = next.sliceId;
        const hit = findSliceById(currentRankData(), next.sliceId);
        state.selectedGroup = hit ? coreGroupOf(hit.rec[1]) : null;
        if (hit && isPrepKernel(hit.owner)) state.rtIncludePrep = true;
      }
      render();
    }

    return {
      update,
      getState: () => ({ ...state }),
    };
  }

  global.PtoCsaCompressorPipeline = Object.freeze({ create });
})(window);
