/* Step 2: Causal serial hardware-core execution swimlane (AIC / AIV 0 / AIV 1).
 * - Strictly single-row serial per hardware lane / sub-pipe (never stacks 2 ops at the same time on 1 core).
 * - Horizontal axis = causal logical order by operation unit (not measured microseconds).
 * - 4 independent horizontal QK / PV page events per full Stack (1 for 1-page tail).
 * - Unified visual grammar: rounded pill + icon for OPs (no Tensor left border); squared blocks for DMA/Pipe transfers. */
(function (global) {
  'use strict';

  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);

  const PRESET_SEQ_LEN = 1537; // 4 Stacks · 13 Pages · Slot 0 reuse (S0 -> S3) + 1-page tail
  const HEADER_H = 96;
  const UNIT_W = 86; // 1 logical page-op unit width (px)
  const GAP_U = 0.24; // clean visual gap between serial blocks on the same lane

  const EVENT_META = {
    QK_READY_EVENT: {
      name: 'QK_READY',
      id: 'QK_READY_EVENT',
      color: '#80B7F7',
      producerCore: 'AIC',
      producerPipe: 'FIX',
      consumerCore: 'AIV 0 / AIV 1',
      consumerPipe: 'MTE2',
      buffer: 'score_transfer',
      lineSet: 357,
      lineWait: 762,
      debugTip: 'AIC 写完本 Stack 全部 Score Slice 到 score_transfer 后发 1 次 SET(QK_READY)，随后 AIV0([2,512]) 与 AIV1([3,512]) 并行读取各自切片。',
    },
    SOFTMAX_READY_EVENT: {
      name: 'SOFTMAX_READY',
      id: 'SOFTMAX_READY_EVENT',
      color: '#9EF04B',
      producerCore: 'AIV 0 / AIV 1',
      producerPipe: 'MTE3',
      consumerCore: 'AIC',
      consumerPipe: 'MTE2',
      buffer: 'probability_transfer',
      lineSet: 817,
      lineWait: 416,
      debugTip: '源码先发出本 Stack 的 V0–V3 预取，再执行 WAIT(SOFTMAX_READY)。静态因果序显示 PRE_LAUNCH=2 旨在掩盖等待；若真实泳道出现长气泡，优先核查 AIV1（3/5 行）耗时。',
    },
    PV_READY_EVENT: {
      name: 'PV_READY',
      id: 'PV_READY_EVENT',
      color: '#F2C45F',
      producerCore: 'AIC',
      producerPipe: 'FIX',
      consumerCore: 'AIV 0 / AIV 1',
      consumerPipe: 'MTE2',
      buffer: 'pv_transfer',
      lineSet: 719,
      lineWait: 831,
      debugTip: 'AIC 完成本 Stack 全部 PV 串行累加并写出 1 次 pv_transfer 后发 SET(PV_READY)，触发 AIV0([2,128]) 与 AIV1([3,128]) 的 Context Accumulate。',
    },
  };

  function opIcon(kind) {
    const entry = global.PtoOpIcons?.find(item => item.id === kind);
    if (!entry) return '';
    return `<span class="viz2-pi-op-icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="${entry.path}"/></svg></span>`;
  }

  function buildLayout(task, layers) {
    const showPipe = !!layers.pipe;
    const labelW = showPipe ? 214 : 168;
    const xOrigin = labelW + 24;
    const tToX = t => Math.round(xOrigin + t * UNIT_W);
    const uToW = u => Math.max(44, Math.round(u * UNIT_W));

    const transferSlots = task.slots?.length || 3;
    const qHeads = task.q?.validShape?.[0] || 5;
    const qBase = task.kvHead * qHeads;
    const qRangeAll = `Q${qBase}–Q${qBase + qHeads - 1}`;
    const qRange0 = `Q${qBase}–Q${qBase + 1}`;
    const qRange1 = `Q${qBase + 2}–Q${qBase + 4}`;
    const taskKey = `task-${task.id}`;

    // Single-row core tracks (or 3 single-row sub-pipes per core in Pipe mode)
    const aicTop = HEADER_H;
    const aicSubH = showPipe ? 54 : 0;
    const aicH = showPipe ? aicSubH * 3 + 22 : 108;
    const aicAivGap = 54; // Clean horizontal highway between AIC and AIV 0 for all cross-core sync arcs
    const aivSubH = showPipe ? 48 : 0;
    const aivH = showPipe ? aivSubH * 3 + 20 : 94;
    const aivBetween = 24;

    const aiv0Top = aicTop + aicH + aicAivGap;
    const aiv1Top = aiv0Top + aivH + aivBetween;
    const totalHeight = aiv1Top + aivH + 34;

    const cores = [
      {
        id: 'aic',
        title: 'AIC',
        subtitle: `[${qHeads}, 512] · ${qRangeAll}`,
        badge: '1 Core',
        top: aicTop,
        height: aicH,
        pipes: showPipe
          ? [
              { id: 'mte2', label: 'MTE2', desc: 'Load K/V/P', top: aicTop + 8, height: aicSubH },
              { id: 'cube', label: 'Cube', desc: 'QK / PV', top: aicTop + 8 + aicSubH + 3, height: aicSubH },
              { id: 'fix', label: 'FIX', desc: 'Store + SET', top: aicTop + 8 + (aicSubH + 3) * 2, height: aicSubH },
            ]
          : [],
      },
      {
        id: 'aiv0',
        lane: 0,
        title: 'AIV 0',
        subtitle: `${qRange0} · 2/5 行 [2, 512]`,
        badge: '2/5 Heads',
        top: aiv0Top,
        height: aivH,
        pipes: showPipe
          ? [
              { id: 'mte2', label: 'MTE2', desc: 'WAIT + Load', top: aiv0Top + 7, height: aivSubH },
              { id: 'vec', label: 'Vector', desc: 'Mask/SM/Acc', top: aiv0Top + 7 + aivSubH + 3, height: aivSubH },
              { id: 'mte3', label: 'MTE3', desc: 'Store + SET', top: aiv0Top + 7 + (aivSubH + 3) * 2, height: aivSubH },
            ]
          : [],
      },
      {
        id: 'aiv1',
        lane: 1,
        title: 'AIV 1',
        subtitle: `${qRange1} · 3/5 行 [3, 512]`,
        badge: '3/5 · +1',
        imbalance: true,
        top: aiv1Top,
        height: aivH,
        pipes: showPipe
          ? [
              { id: 'mte2', label: 'MTE2', desc: 'WAIT + Load', top: aiv1Top + 7, height: aivSubH },
              { id: 'vec', label: 'Vector', desc: 'Mask/SM/Acc', top: aiv1Top + 7 + aivSubH + 3, height: aivSubH },
              { id: 'mte3', label: 'MTE3', desc: 'Store + SET', top: aiv1Top + 7 + (aivSubH + 3) * 2, height: aivSubH },
            ]
          : [],
      },
    ];

    const blocks = [];
    const groups = [];
    const itemsById = new Map();
    const syncEdges = [];
    const register = item => {
      blocks.push(item);
      itemsById.set(item.uid, item);
      return item;
    };

    const qkReadyAt = new Map(); // stack -> { t, x, y }
    const smReadyAt = new Map(); // stack -> { t, aiv0: {x,y}, aiv1: {x,y} }
    const pvReadyAt = new Map(); // stack -> { t, x, y }
    const slotConsumedAt = new Map(); // stack -> x (end of PV on AIC)
    const slotProducedAt = new Map(); // stack -> x (start of QK on AIC)

    let maxT = 6;
    const highwayTop = aicTop + aicH + 10;

    if (!showPipe) {
      // ================== OP MODE: True Single-Row Causal Serial Schedule ==================
      let aicT = 0;
      let aivT = 0; // AIV0 & AIV1 run symmetric causal schedules on separate lanes
      const opY = core => core.top + 34;
      const opH = 54;

      task.ticks.forEach(tick => {
        const prodStack = tick.produce != null ? task.stacks[tick.produce] : null;
        const consStack = tick.consume != null ? task.stacks[tick.consume] : null;

        // 1) AIC Produce QK (4 independent horizontal page ops, or tail N ops)
        if (prodStack) {
          const s = prodStack;
          const isTail = s.pages.length < (task.pagesPerStack || 4);
          const groupStartT = aicT;
          slotProducedAt.set(s.index, tToX(groupStartT) + 28);

          s.pages.forEach((pIdx, offset) => {
            const pTokens = task.pages[pIdx].tokens;
            const startT = aicT;
            const dur = 1.0;
            register({
              uid: `aic:qk:s${s.index}:p${pIdx}`,
              core: 'aic', stage: 'qk', node: 'qk', stack: s.index, page: pIdx, slot: s.slot, tick: tick.index,
              executionInstanceId: `${taskKey}/qk/s${s.index}-p${pIdx}`,
              kind: 'op-pill', icon: 'matmul', isTail: pTokens < task.pageTokens,
              x: tToX(startT), y: opY(cores[0]), w: uToW(dur) - 4, h: opH,
              title: `QK P${pIdx}`, subtitle: `${pTokens} tok`,
              sourceLine: 293,
              debugTip: `独立 Cube Matmul 事件（S${s.index} 第 ${offset + 1}/${s.pages.length} 页）：Q[${qHeads},128] × K_P${pIdx}^T → score_transfer[:, ${offset * 128}:${(offset + 1) * 128}]。`,
            });
            aicT += dur + 0.08;
          });

          const setX = tToX(aicT - 0.08);
          groups.push({
            core: 'aic', stage: 'qk', stack: s.index, slot: s.slot, tick: tick.index, isTail,
            reusedFrom: s.index >= transferSlots ? s.index - transferSlots : null,
            x: tToX(groupStartT) - 5, y: cores[0].top + 8, w: setX - tToX(groupStartT) + 10, h: cores[0].height - 16,
            label: `QK · S${s.index} · [${qHeads}, 512]`,
            syncOut: 'QK_READY_EVENT',
          });
          qkReadyAt.set(s.index, { t: aicT + 0.12, x: setX, y: cores[0].top + cores[0].height - 8 });
          aicT += GAP_U + 0.16;
        }

        // 2) AIC Consume PV (Pre-load V -> WAIT SOFTMAX_READY -> 4 independent horizontal PV matmul/acc ops)
        if (consStack) {
          const s = consStack;
          const isTail = s.pages.length < (task.pagesPerStack || 4);
          const groupStartT = aicT;
          const vDur = isTail ? 0.85 : 1.15;
          const vRange = s.pages.length === 1 ? `V${s.pages[0]}` : `V${s.pages[0]}–V${s.pages.at(-1)}`;

          // Pre-load V block (explicitly BEFORE WAIT SOFTMAX_READY)
          register({
            uid: `aic:preload-v:s${s.index}`,
            core: 'aic', stage: 'pv', node: 'v-cache', stack: s.index, slot: s.slot, tick: tick.index,
            executionInstanceId: `${taskKey}/v-cache/s${s.index}-p${s.pages[0]}`,
            kind: 'dma-block', isTail,
            x: tToX(aicT), y: opY(cores[0]), w: uToW(vDur) - 4, h: opH,
            title: `Pre-load ${vRange}`, subtitle: 'MTE2 · 在 WAIT 前',
            sourceLine: 384,
            debugTip: '源码先发出本 Stack 全部 V Page 的 MTE2 预取，随后才执行 WAIT(SOFTMAX_READY)，用于掩盖访存延迟。',
          });
          aicT += vDur + 0.1;

          // Causal WAIT(SOFTMAX_READY)
          const smReady = smReadyAt.get(s.index);
          const waitStartT = Math.max(aicT, smReady ? smReady.t : aicT);
          const waitX = tToX(waitStartT);
          if (smReady) {
            ['aiv0', 'aiv1'].forEach(coreId => {
              syncEdges.push({
                id: `sync:SOFTMAX_READY:s${s.index}:${coreId}`,
                event: 'SOFTMAX_READY_EVENT', stack: s.index, slot: s.slot, sourceCore: coreId,
                from: smReady[coreId], to: { x: waitX + 6, y: opY(cores[0]) + opH },
                busY: highwayTop + 4 + (s.slot % 3) * 9,
                color: EVENT_META.SOFTMAX_READY_EVENT.color,
              });
            });
          }
          aicT = waitStartT;

          // 4 independent horizontal PV Matmul / Matmul Acc blocks
          s.pages.forEach((pIdx, offset) => {
            const pTokens = task.pages[pIdx].tokens;
            const dur = 1.0;
            register({
              uid: `aic:pv:s${s.index}:p${pIdx}`,
              core: 'aic', stage: 'pv', node: 'pv', stack: s.index, page: pIdx, slot: s.slot, tick: tick.index,
              executionInstanceId: `${taskKey}/pv/s${s.index}-p${pIdx}`,
              kind: 'op-pill', icon: 'matmul', isTail: pTokens < task.pageTokens,
              chainedFromPrev: offset > 0,
              x: tToX(aicT), y: opY(cores[0]), w: uToW(dur) - 4, h: opH,
              title: `PV P${pIdx}`, subtitle: offset ? 'matmul_acc' : 'matmul',
              sourceLine: offset ? 445 : 432,
              debugTip: offset
                ? `独立 Cube Matmul Acc 事件（S${s.index} 第 ${offset + 1} 页）：在片上累加到前一页 FP32 Acc Tile。`
                : `独立 Cube Matmul 事件（S${s.index} 首页）：读取 probability_transfer[:, 0:128] × V_P${pIdx}。`,
            });
            aicT += dur + 0.08;
          });

          const endX = tToX(aicT - 0.08);
          groups.push({
            core: 'aic', stage: 'pv', stack: s.index, slot: s.slot, tick: tick.index, isTail,
            x: tToX(groupStartT) - 5, y: cores[0].top + 8, w: endX - tToX(groupStartT) + 10, h: cores[0].height - 16,
            label: `PV · S${s.index} · [${qHeads}, 128]`,
            syncIn: 'SOFTMAX_READY_EVENT',
            syncOut: 'PV_READY_EVENT',
          });
          pvReadyAt.set(s.index, { t: aicT + 0.12, x: endX, y: cores[0].top + cores[0].height - 8 });
          slotConsumedAt.set(s.index, endX - 20);
          aicT += GAP_U + 0.16;
        }

        // 3) AIV 0 (2/5 rows) & AIV 1 (3/5 rows): Scale & Mask -> Online Softmax (starts AFTER QK_READY!)
        if (prodStack) {
          const s = prodStack;
          const qkReady = qkReadyAt.get(s.index);
          const startT = Math.max(aivT, qkReady ? qkReady.t : aivT);
          const maskDur = 1.45;
          const smDur = 2.05;
          const smStartT = startT + maskDur + 0.08;
          const endT = smStartT + smDur;
          const startX = tToX(startT);
          const endX = tToX(endT);

          const smEndpoints = { t: endT + 0.14 };
          [cores[1], cores[2]].forEach(core => {
            const lane = core.lane;
            const rows = lane === 0 ? 2 : 3;
            const qSub = lane === 0 ? qRange0 : qRange1;
            const y = core.top + 28;
            const h = 52;

            register({
              uid: `${core.id}:mask:s${s.index}`,
              core: core.id, lane, stage: 'scale-mask', node: 'scale-mask', stack: s.index, slot: s.slot, tick: tick.index,
              executionInstanceId: `${taskKey}/scale-mask/s${s.index}`,
              kind: 'op-pill', icon: 'softmax',
              x: tToX(startT), y, w: uToW(maskDur) - 4, h,
              title: `Scale & Mask`, subtitle: `S${s.index} · [${rows}, 512]`,
              syncIn: 'QK_READY_EVENT',
              sourceLine: 775,
              debugTip: `AIV ${lane}（${qSub}，占 ${rows}/5 行）等待 QK_READY 后读取 score_transfer[${lane === 0 ? '0:2' : '2:5'}, :]，执行 Scale 与 valid_cols fillpad。`,
            });
            register({
              uid: `${core.id}:softmax:s${s.index}`,
              core: core.id, lane, stage: 'softmax', node: 'softmax', stack: s.index, slot: s.slot, tick: tick.index,
              executionInstanceId: `${taskKey}/softmax/s${s.index}`,
              kind: 'op-pill', icon: 'softmax', chainedFromPrev: true,
              x: tToX(smStartT), y, w: uToW(smDur) - 4, h,
              title: `Online Softmax · S${s.index}`, subtitle: `[${rows}, 512] · ${rows}/5 行 · m/l`,
              syncOut: 'SOFTMAX_READY_EVENT',
              sourceLine: 789,
              debugTip: lane === 1
                ? `AIV 1 处理 3/5 行（${qSub}），比 AIV 0 多 1 行，真实泳道中常决定 SOFTMAX_READY 完成时刻。`
                : `AIV 0 处理 2/5 行（${qSub}），写回 probability_transfer[0:2, :] 后发 SET(SOFTMAX_READY)。`,
            });
            groups.push({
              core: core.id, stage: 'softmax', stack: s.index, slot: s.slot, tick: tick.index,
              x: startX - 4, y: core.top + 6, w: endX - startX + 8, h: core.height - 12,
              label: `S${s.index} · ${qSub} (${rows}/5)`,
            });

            if (qkReady) {
              syncEdges.push({
                id: `sync:QK_READY:s${s.index}:${core.id}`,
                event: 'QK_READY_EVENT', stack: s.index, slot: s.slot, targetCore: core.id,
                from: qkReady, to: { x: startX, y: y + h / 2 },
                busY: highwayTop + 34,
                color: EVENT_META.QK_READY_EVENT.color,
              });
            }
            smEndpoints[core.id] = { x: endX - 4, y: y + h / 2 };
          });

          smReadyAt.set(s.index, smEndpoints);
          aivT = endT + GAP_U + 0.12;
        }

        // 4) AIV 0 & AIV 1: Context Accumulate (starts AFTER PV_READY!)
        if (consStack) {
          const s = consStack;
          const pvReady = pvReadyAt.get(s.index);
          const startT = Math.max(aivT, pvReady ? pvReady.t : aivT);
          const accDur = 1.95;
          const endT = startT + accDur;
          const startX = tToX(startT);

          [cores[1], cores[2]].forEach(core => {
            const lane = core.lane;
            const rows = lane === 0 ? 2 : 3;
            const qSub = lane === 0 ? qRange0 : qRange1;
            const y = core.top + 28;
            const h = 52;

            register({
              uid: `${core.id}:acc:s${s.index}`,
              core: core.id, lane, stage: 'accumulate', node: 'context-acc', stack: s.index, slot: s.slot, tick: tick.index,
              executionInstanceId: `${taskKey}/context-acc/s${s.index}`,
              kind: 'op-pill', icon: 'reduce',
              x: startX, y, w: uToW(accDur) - 4, h,
              title: `Context Acc · S${s.index}`, subtitle: `[${rows}, 128] · o×p0+pv`,
              syncIn: 'PV_READY_EVENT',
              sourceLine: 827,
              debugTip: `AIV ${lane}（${qSub}）在 PV_READY 后读取 pv_transfer[${lane === 0 ? '0:2' : '2:5'}, :]，用延迟 2 拍的 pending0 重标定历史 o 并累加。`,
            });
            groups.push({
              core: core.id, stage: 'accumulate', stack: s.index, slot: s.slot, tick: tick.index,
              x: startX - 4, y: core.top + 6, w: uToW(accDur) + 4, h: core.height - 12,
              label: `S${s.index} · Acc (${rows}/5)`,
            });

            if (pvReady) {
              syncEdges.push({
                id: `sync:PV_READY:s${s.index}:${core.id}`,
                event: 'PV_READY_EVENT', stack: s.index, slot: s.slot, targetCore: core.id,
                from: pvReady, to: { x: startX, y: y + h / 2 },
                busY: highwayTop + 44,
                color: EVENT_META.PV_READY_EVENT.color,
              });
            }
          });
          aivT = endT + GAP_U + 0.12;
        }
      });

      // 5) Final Output Normalize & Store on AIV 0 / AIV 1
      const normStartT = aivT + 0.1;
      const normDur = 2.15;
      [cores[1], cores[2]].forEach(core => {
        const lane = core.lane;
        const rows = lane === 0 ? 2 : 3;
        const qSub = lane === 0 ? qRange0 : qRange1;
        register({
          uid: `${core.id}:output`,
          core: core.id, lane, stage: 'output', node: 'output-norm', tick: task.ticks.length,
          executionInstanceId: `${taskKey}/output-norm/task`,
          kind: 'op-pill', icon: 'reduce',
          x: tToX(normStartT), y: core.top + 26, w: uToW(normDur) - 4, h: 52,
          title: 'Normalize & Store', subtitle: `[${rows}, 128] → out2d`,
          sourceLine: 865,
          debugTip: `全部 Stack 累加完成后，AIV ${lane}（${qSub}）执行 o / l_sum 并转为 BF16 写入 out2d。`,
        });
      });
      maxT = Math.max(aicT, normStartT + normDur) + 0.8;
    } else {
      // ================== PIPE MODE: True Single-Row Staircase Per Sub-Pipe ==================
      // AIC: MTE2 / Cube / FIX; AIV0 & AIV1: MTE2 / Vector / MTE3 — zero vertical overlap on any sub-pipe
      const [aicMte2, aicCube, aicFix] = cores[0].pipes;
      let mte2T = 0, cubeT = 0, fixT = 0;
      let aivMte2T = 0, aivVecT = 0, aivMte3T = 0;

      task.ticks.forEach(tick => {
        const prodStack = tick.produce != null ? task.stacks[tick.produce] : null;
        const consStack = tick.consume != null ? task.stacks[tick.consume] : null;

        // AIC QK: pl.pipeline(STACK_PAGES, stage=2) staircase across MTE2 -> Cube -> FIX
        if (prodStack) {
          const s = prodStack;
          const isTail = s.pages.length < (task.pagesPerStack || 4);
          const baseStart = Math.max(mte2T, cubeT - 0.65);
          mte2T = baseStart;

          s.pages.forEach((pIdx, offset) => {
            const pTokens = task.pages[pIdx].tokens;
            const loadStart = mte2T;
            const loadDur = 0.92;
            register({
              uid: `aic:mte2:k:s${s.index}:p${pIdx}`, core: 'aic', pipe: 'mte2', stage: 'qk', node: 'k-cache',
              stack: s.index, page: pIdx, slot: s.slot, tick: tick.index, kind: 'dma-block',
              executionInstanceId: `${taskKey}/k-cache/s${s.index}-p${pIdx}`,
              x: tToX(loadStart), y: aicMte2.top + 6, w: uToW(loadDur) - 4, h: aicMte2.height - 12,
              title: `Load K${pIdx}`, subtitle: `S${s.index} · GM→Mat`,
              sourceLine: 287,
              debugTip: 'MTE2 预取 K Cache Page 到 Mat，与前序 Cube 形成 stage=2 阶梯流水。',
            });
            mte2T = loadStart + loadDur + 0.08;

            const qkStart = Math.max(cubeT, loadStart + 0.55);
            const qkDur = 0.92;
            register({
              uid: `aic:cube:qk:s${s.index}:p${pIdx}`, core: 'aic', pipe: 'cube', stage: 'qk', node: 'qk',
              stack: s.index, page: pIdx, slot: s.slot, tick: tick.index, kind: 'op-pill', icon: 'matmul',
              executionInstanceId: `${taskKey}/qk/s${s.index}-p${pIdx}`,
              x: tToX(qkStart), y: aicCube.top + 6, w: uToW(qkDur) - 4, h: aicCube.height - 12,
              title: `QK P${pIdx}`, subtitle: `${pTokens} tok`,
              sourceLine: 293,
              debugTip: 'Cube 侧独立 QK Matmul 事件。',
            });
            cubeT = qkStart + qkDur + 0.08;

            const stStart = Math.max(fixT, qkStart + 0.55);
            if (offset === 0) slotProducedAt.set(s.index, tToX(stStart) + 18);
            const stDur = 0.85;
            register({
              uid: `aic:fix:sc:s${s.index}:p${pIdx}`, core: 'aic', pipe: 'fix', stage: 'qk', node: 'scores',
              stack: s.index, page: pIdx, slot: s.slot, tick: tick.index, kind: 'dma-block',
              executionInstanceId: `${taskKey}/scores/s${s.index}`,
              x: tToX(stStart), y: aicFix.top + 6, w: uToW(stDur) - 4, h: aicFix.height - 12,
              title: `Store S${s.index}·P${pIdx}`, subtitle: offset === s.pages.length - 1 ? 'SET QK_READY' : `Slot ${s.slot}`,
              syncOut: offset === s.pages.length - 1 ? 'QK_READY_EVENT' : undefined,
              sourceLine: offset === s.pages.length - 1 ? 357 : 298,
              debugTip: offset === s.pages.length - 1
                ? '写完本 Stack 最后一页 Score Slice 到 score_transfer 后，FIX 发出 1 次 SET(QK_READY)。'
                : 'FIX 将当前页 Score 写入 score_transfer 对应列区间。',
            });
            fixT = stStart + stDur + 0.08;
          });

          qkReadyAt.set(s.index, { t: fixT + 0.08, x: tToX(fixT - 0.08), y: aicFix.top + aicFix.height - 6 });
          mte2T = Math.max(mte2T, cubeT - 0.3) + 0.2;
        }

        // AIC PV: Pre-load V0..V3 -> WAIT(SOFTMAX_READY) -> Load P0..P3 -> Cube PV0..PV3 -> 1 Store PV + SET PV_READY
        if (consStack) {
          const s = consStack;
          const vStart = mte2T;
          // Pre-load V pages BEFORE WAIT
          s.pages.forEach((pIdx, offset) => {
            const dur = 0.72;
            register({
              uid: `aic:mte2:v:s${s.index}:p${pIdx}`, core: 'aic', pipe: 'mte2', stage: 'pv', node: 'v-cache',
              stack: s.index, page: pIdx, slot: s.slot, tick: tick.index, kind: 'dma-block',
              executionInstanceId: `${taskKey}/v-cache/s${s.index}-p${pIdx}`,
              x: tToX(mte2T), y: aicMte2.top + 6, w: uToW(dur) - 4, h: aicMte2.height - 12,
              title: `Load V${pIdx}`, subtitle: `Pre-load`,
              sourceLine: 384,
              debugTip: '在 WAIT(SOFTMAX_READY) 之前预先加载 V Cache Page 到 L1（pv_v_l1）。',
            });
            mte2T += dur + 0.06;
          });

          // WAIT(SOFTMAX_READY) gate on MTE2 before Load P
          const smReady = smReadyAt.get(s.index);
          const loadPStart = Math.max(mte2T + 0.08, smReady ? smReady.t : mte2T + 0.08);
          if (smReady) {
            ['aiv0', 'aiv1'].forEach(coreId => {
              syncEdges.push({
                id: `sync:SOFTMAX_READY:s${s.index}:${coreId}`,
                event: 'SOFTMAX_READY_EVENT', stack: s.index, slot: s.slot, sourceCore: coreId,
                from: smReady[coreId], to: { x: tToX(loadPStart) + 4, y: aicMte2.top + aicMte2.height - 6 },
                busY: highwayTop + 4 + (s.slot % 3) * 9,
                color: EVENT_META.SOFTMAX_READY_EVENT.color,
              });
            });
          }
          mte2T = loadPStart;

          s.pages.forEach((pIdx, offset) => {
            const pDur = 0.85;
            const pStart = mte2T;
            register({
              uid: `aic:mte2:p:s${s.index}:p${pIdx}`, core: 'aic', pipe: 'mte2', stage: 'pv', node: 'probability',
              stack: s.index, page: pIdx, slot: s.slot, tick: tick.index, kind: 'dma-block',
              executionInstanceId: `${taskKey}/probability/s${s.index}`,
              x: tToX(pStart), y: aicMte2.top + 6, w: uToW(pDur) - 4, h: aicMte2.height - 12,
              title: offset === 0 ? `WAIT+P${pIdx}` : `Load P${pIdx}`, subtitle: `S${s.index} · Slot ${s.slot}`,
              syncIn: offset === 0 ? 'SOFTMAX_READY_EVENT' : undefined,
              sourceLine: offset === 0 ? 416 : 437,
              debugTip: '通过 pv_p_l1 双缓冲加载 BF16 Probability 分片。',
            });
            mte2T = pStart + pDur + 0.06;

            const pvStart = Math.max(cubeT, pStart + 0.5);
            const pvDur = 0.92;
            register({
              uid: `aic:cube:pv:s${s.index}:p${pIdx}`, core: 'aic', pipe: 'cube', stage: 'pv', node: 'pv',
              stack: s.index, page: pIdx, slot: s.slot, tick: tick.index, kind: 'op-pill', icon: 'matmul',
              executionInstanceId: `${taskKey}/pv/s${s.index}-p${pIdx}`,
              x: tToX(pvStart), y: aicCube.top + 6, w: uToW(pvDur) - 4, h: aicCube.height - 12,
              title: `PV P${pIdx}`, subtitle: offset ? 'acc' : 'mm',
              sourceLine: offset ? 445 : 432,
              debugTip: 'Cube 串行累加各 Page 的 PV 乘积。',
            });
            cubeT = pvStart + pvDur + 0.08;
          });

          // Only 1 Store PV + SET PV_READY after all pages finish accumulating
          const pvStoreStart = Math.max(fixT, cubeT - 0.15);
          const pvStoreDur = 1.1;
          register({
            uid: `aic:fix:pv:s${s.index}`, core: 'aic', pipe: 'fix', stage: 'pv', node: 'partial',
            stack: s.index, slot: s.slot, tick: tick.index, kind: 'dma-block',
            executionInstanceId: `${taskKey}/partial/s${s.index}`,
            x: tToX(pvStoreStart), y: aicFix.top + 6, w: uToW(pvStoreDur) - 4, h: aicFix.height - 12,
            title: `Store PV · S${s.index}`, subtitle: 'SET PV_READY',
            syncOut: 'PV_READY_EVENT',
            sourceLine: 719,
            debugTip: '整 Stack 累加完只做 1 次 pv_transfer 写回，并触发 SET(PV_READY)。',
          });
          fixT = pvStoreStart + pvStoreDur + 0.12;
          pvReadyAt.set(s.index, { t: fixT + 0.06, x: tToX(fixT - 0.12), y: aicFix.top + aicFix.height - 6 });
          slotConsumedAt.set(s.index, tToX(fixT - 0.25));
          mte2T = Math.max(mte2T, cubeT - 0.4) + 0.2;
        }

        // AIV 0 & AIV 1: MTE2 (WAIT QK + Load Score) -> Vector (Scale&Mask -> Softmax) -> MTE3 (Store Prob + SET SM)
        if (prodStack) {
          const s = prodStack;
          const qkReady = qkReadyAt.get(s.index);
          const loadStart = Math.max(aivMte2T, qkReady ? qkReady.t : aivMte2T);
          const loadDur = 1.05;
          const maskStart = Math.max(aivVecT, loadStart + loadDur + 0.06);
          const maskDur = 1.2;
          const smStart = maskStart + maskDur + 0.06;
          const smDur = 1.6;
          const storeStart = Math.max(aivMte3T, smStart + smDur + 0.06);
          const storeDur = 1.05;
          const smEndpoints = { t: storeStart + storeDur + 0.1 };

          [cores[1], cores[2]].forEach(core => {
            const lane = core.lane;
            const rows = lane === 0 ? 2 : 3;
            const [mte2, vec, mte3] = core.pipes;
            register({
              uid: `${core.id}:mte2:sm:s${s.index}`, core: core.id, lane, pipe: 'mte2', stage: 'scale-mask', node: 'scores',
              stack: s.index, slot: s.slot, tick: tick.index, kind: 'dma-block',
              executionInstanceId: `${taskKey}/scores/s${s.index}`,
              x: tToX(loadStart), y: mte2.top + 5, w: uToW(loadDur) - 4, h: mte2.height - 10,
              title: `Load Score`, subtitle: `S${s.index} · [${rows},512]`,
              syncIn: 'QK_READY_EVENT', sourceLine: 762,
              debugTip: 'MTE2 等待 QK_READY 后加载本 Lane 切片。',
            });
            register({
              uid: `${core.id}:vec:mask:s${s.index}`, core: core.id, lane, pipe: 'vec', stage: 'scale-mask', node: 'scale-mask',
              stack: s.index, slot: s.slot, tick: tick.index, kind: 'op-pill', icon: 'softmax',
              executionInstanceId: `${taskKey}/scale-mask/s${s.index}`,
              x: tToX(maskStart), y: vec.top + 5, w: uToW(maskDur) - 4, h: vec.height - 10,
              title: `Scale&Mask`, subtitle: `S${s.index}`, sourceLine: 775,
              debugTip: 'Vector 执行 Scale + fillpad Mask。',
            });
            register({
              uid: `${core.id}:vec:sm:s${s.index}`, core: core.id, lane, pipe: 'vec', stage: 'softmax', node: 'softmax',
              stack: s.index, slot: s.slot, tick: tick.index, kind: 'op-pill', icon: 'softmax',
              executionInstanceId: `${taskKey}/softmax/s${s.index}`,
              x: tToX(smStart), y: vec.top + 5, w: uToW(smDur) - 4, h: vec.height - 10,
              title: `Softmax · S${s.index}`, subtitle: `[${rows},512] · m/l`, sourceLine: 789,
              debugTip: 'Vector 执行 Online Softmax。',
            });
            register({
              uid: `${core.id}:mte3:sm:s${s.index}`, core: core.id, lane, pipe: 'mte3', stage: 'softmax', node: 'probability',
              stack: s.index, slot: s.slot, tick: tick.index, kind: 'dma-block',
              executionInstanceId: `${taskKey}/probability/s${s.index}`,
              x: tToX(storeStart), y: mte3.top + 5, w: uToW(storeDur) - 4, h: mte3.height - 10,
              title: `Store Prob`, subtitle: `SET SM_READY`,
              syncOut: 'SOFTMAX_READY_EVENT', sourceLine: 811,
              debugTip: 'MTE3 写回 probability_transfer 并触发 SET(SOFTMAX_READY)。',
            });
            if (qkReady) {
              syncEdges.push({
                id: `sync:QK_READY:s${s.index}:${core.id}`,
                event: 'QK_READY_EVENT', stack: s.index, slot: s.slot, targetCore: core.id,
                from: qkReady, to: { x: tToX(loadStart), y: mte2.top + mte2.height / 2 },
                busY: highwayTop + 34, color: EVENT_META.QK_READY_EVENT.color,
              });
            }
            smEndpoints[core.id] = { x: tToX(storeStart + storeDur) - 4, y: mte3.top + mte3.height / 2 };
          });
          smReadyAt.set(s.index, smEndpoints);
          aivMte2T = loadStart + loadDur + 0.14;
          aivVecT = smStart + smDur + 0.14;
          aivMte3T = storeStart + storeDur + 0.14;
        }

        // AIV 0 & AIV 1: MTE2 (WAIT PV + Load PV) -> Vector (Context Acc + FIFO)
        if (consStack) {
          const s = consStack;
          const pvReady = pvReadyAt.get(s.index);
          const loadStart = Math.max(aivMte2T, pvReady ? pvReady.t : aivMte2T);
          const loadDur = 0.95;
          const accStart = Math.max(aivVecT, loadStart + loadDur + 0.06);
          const accDur = 1.45;
          [cores[1], cores[2]].forEach(core => {
            const lane = core.lane;
            const rows = lane === 0 ? 2 : 3;
            const [mte2, vec] = core.pipes;
            register({
              uid: `${core.id}:mte2:acc:s${s.index}`, core: core.id, lane, pipe: 'mte2', stage: 'accumulate', node: 'partial',
              stack: s.index, slot: s.slot, tick: tick.index, kind: 'dma-block',
              executionInstanceId: `${taskKey}/partial/s${s.index}`,
              x: tToX(loadStart), y: mte2.top + 5, w: uToW(loadDur) - 4, h: mte2.height - 10,
              title: `Load PV`, subtitle: `S${s.index} · [${rows},128]`,
              syncIn: 'PV_READY_EVENT', sourceLine: 831,
              debugTip: 'MTE2 等待 PV_READY 后读取本 Lane 的 Partial Context。',
            });
            register({
              uid: `${core.id}:vec:acc:s${s.index}`, core: core.id, lane, pipe: 'vec', stage: 'accumulate', node: 'context-acc',
              stack: s.index, slot: s.slot, tick: tick.index, kind: 'op-pill', icon: 'reduce',
              executionInstanceId: `${taskKey}/context-acc/s${s.index}`,
              x: tToX(accStart), y: vec.top + 5, w: uToW(accDur) - 4, h: vec.height - 10,
              title: `Acc · S${s.index}`, subtitle: `o×p0+pv`, sourceLine: 842,
              debugTip: 'Vector 执行历史 o 缩放累加与 Rescale FIFO 移位。',
            });
            if (pvReady) {
              syncEdges.push({
                id: `sync:PV_READY:s${s.index}:${core.id}`,
                event: 'PV_READY_EVENT', stack: s.index, slot: s.slot, targetCore: core.id,
                from: pvReady, to: { x: tToX(loadStart), y: mte2.top + mte2.height / 2 },
                busY: highwayTop + 44, color: EVENT_META.PV_READY_EVENT.color,
              });
            }
          });
          aivMte2T = loadStart + loadDur + 0.14;
          aivVecT = accStart + accDur + 0.14;
        }
      });

      // Output Normalize (Vector) -> Store out2d (MTE3)
      const normStart = aivVecT + 0.12;
      const normDur = 1.3;
      const outStart = Math.max(aivMte3T, normStart + normDur + 0.06);
      const outDur = 1.1;
      [cores[1], cores[2]].forEach(core => {
        const lane = core.lane;
        const rows = lane === 0 ? 2 : 3;
        const [, vec, mte3] = core.pipes;
        register({
          uid: `${core.id}:vec:norm`, core: core.id, lane, pipe: 'vec', stage: 'output', node: 'output-norm',
          tick: task.ticks.length, kind: 'op-pill', icon: 'reduce',
          executionInstanceId: `${taskKey}/output-norm/task`,
          x: tToX(normStart), y: vec.top + 5, w: uToW(normDur) - 4, h: vec.height - 10,
          title: 'Normalize', subtitle: `[${rows},128] BF16`, sourceLine: 865,
          debugTip: 'Vector 执行最终归一化 o / l_sum 与 BF16 转换。',
        });
        register({
          uid: `${core.id}:mte3:out`, core: core.id, lane, pipe: 'mte3', stage: 'output', node: 'output',
          tick: task.ticks.length, kind: 'dma-block',
          executionInstanceId: `${taskKey}/output/task`,
          x: tToX(outStart), y: mte3.top + 5, w: uToW(outDur) - 4, h: mte3.height - 10,
          title: 'Store out2d', subtitle: `${rows} rows → GM`, sourceLine: 870,
          debugTip: 'MTE3 将最终结果写回全局内存 out2d。',
        });
      });
      maxT = Math.max(mte2T, cubeT, fixT, outStart + outDur) + 0.8;
    }

    // Slot reuse bridges: from S_{k-3} PV end to S_k QK start
    const reuses = task.stacks
      .filter(s => s.index >= transferSlots)
      .map(s => {
        const fromStack = task.stacks[s.index - transferSlots];
        const x1 = slotConsumedAt.get(fromStack.index);
        const x2 = slotProducedAt.get(s.index);
        return x1 != null && x2 != null
          ? { slot: s.slot, from: fromStack.index, to: s.index, fromTick: fromStack.consumeTick, toTick: s.produceTick, x1, x2 }
          : null;
      })
      .filter(Boolean);

    const totalWidth = Math.max(920, tToX(maxT) + 32);
    const stepMarks = Array.from({ length: Math.ceil(maxT) }, (_, i) => i).filter(i => i % 2 === 0);

    return { cores, blocks, groups, itemsById, syncEdges, reuses, labelW, xOrigin, tToX, stepMarks, width: totalWidth, height: totalHeight };
  }

  // Strictly forward-in-time orthogonal routing along dedicated inter-core highway (zero card crossings)
  function syncPath(edge) {
    const { from, to, event, busY } = edge;
    if (event === 'SOFTMAX_READY_EVENT') {
      const riseX = from.x + (edge.sourceCore === 'aiv1' ? 12 : 7);
      return `M${from.x} ${from.y} H${riseX} V${busY} H${to.x} V${to.y}`;
    }
    // QK_READY & PV_READY: AIC bottom -> drop to highway -> glide right -> drop in empty pre-start gap to AIV0/AIV1
    const dropX = to.x - (edge.targetCore === 'aiv1' ? 12 : 7);
    return `M${from.x} ${from.y} V${busY} H${dropX} V${to.y} H${to.x}`;
  }

  function mount(host, { scenario, source, onFocus, onNavigate, onClose }) {
    host.className = 'viz2-pipeline-instance';
    const layers = { op: true, pipe: false, sync: true, slot: true };
    host.innerHTML = `<div class="viz2-pi-toolbar">
      <div class="viz2-pi-toolbar__left">
        <span data-pi-summary></span>
        <div class="viz2-pi-layers" role="group" aria-label="执行泳道显示层">
          <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn is-selected" data-pi-layer="op" aria-pressed="true" title="按算子阶段展开独立 Page 事件（单行因果推进）">Op</button>
          <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn" data-pi-layer="pipe" aria-pressed="false" title="展开 AIC (MTE2/Cube/FIX) 与 AIV (MTE2/Vector/MTE3) 阶梯流水">Pipe</button>
          <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn is-selected" data-pi-layer="sync" aria-pressed="true" title="显示 QK_READY / SOFTMAX_READY / PV_READY 顺流同步箭头">Sync</button>
          <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn is-selected" data-pi-layer="slot" aria-pressed="true" title="显示 3-Slot 标记与跨 Stack 复用桥">Slot 复用</button>
        </div>
        <button type="button" class="btn btn-ghost btn-sm viz2-preset-btn" data-pi-preset="${PRESET_SEQ_LEN}" title="切换至 4 Stacks · 13 Pages (1537 tokens)，观察 Slot 0 复用与尾页分支">4 Stacks / 13 Pages · Slot 0 复用</button>
        <button type="button" class="btn btn-ghost btn-sm" data-pi-reset hidden>恢复原 Task</button>
      </div>
      <div class="viz2-instance-tools">
        <button type="button" class="btn btn-ghost btn-sm" data-pi-zoom="out" aria-label="缩小执行泳道">−</button>
        <output data-pi-scale aria-label="缩放比例"></output>
        <button type="button" class="btn btn-ghost btn-sm" data-pi-zoom="in" aria-label="放大执行泳道">+</button>
        <button type="button" class="btn btn-ghost btn-sm" data-pi-fit>Fit to View</button>
      </div>
    </div>
    <div class="viz2-pi-boundary-note" data-pi-note></div>
    <div class="viz2-pi-viewport" role="group" tabindex="0" aria-label="AIC / AIV 理论因果执行泳道，滚轮缩放、拖拽平移">
      <div class="viz2-pi-scene"></div>
    </div>
    <aside class="pto-ide-frame__pane viz2-anatomy-inspector viz2-pi-inspector" aria-label="执行实例详情" hidden>
      <header class="pto-ide-frame__pane-header">
        <div><b data-pi-title></b><small data-pi-context></small></div>
        <button type="button" class="btn btn-ghost btn-sm" data-pi-close aria-label="关闭执行实例详情">×</button>
      </header>
      <div class="pto-ide-frame__pane-body" data-pi-detail></div>
    </aside>`;

    const viewport = host.querySelector('.viz2-pi-viewport');
    const scene = host.querySelector('.viz2-pi-scene');
    const summary = host.querySelector('[data-pi-summary]');
    const noteEl = host.querySelector('[data-pi-note]');
    const presetBtn = host.querySelector('[data-pi-preset]');
    const resetBtn = host.querySelector('[data-pi-reset]');
    const scaleReadout = host.querySelector('[data-pi-scale]');
    const inspector = host.querySelector('.viz2-pi-inspector');
    const detailBody = inspector.querySelector('[data-pi-detail]');

    let task = null, layout = null, camera = null, drag = null, suppressBlankClick = false, fitMode = true;
    let currentFocus = null, inspectorOpen = false, hoverItem = null;

    function apply(next) {
      camera = next;
      scaleReadout.textContent = `${Math.round(camera.scale * 100)}%`;
      scene.style.transform = `translate(${camera.x}px,${camera.y}px) scale(${camera.scale})`;
    }
    function fit() {
      if (!layout || !viewport.clientWidth || !viewport.clientHeight) return;
      fitMode = true;
      const scale = Math.min(1, (viewport.clientWidth - 28) / layout.width, (viewport.clientHeight - 28) / layout.height);
      apply({ scale: Math.max(scale, 0.08), x: 14, y: 14 });
    }
    function zoom(factor, x, y) {
      if (!camera) return;
      fitMode = false;
      const { scale, x: left, y: top } = camera;
      const next = Math.min(2, Math.max(0.08, scale * factor));
      apply({ scale: next, x: x - (x - left) * next / scale, y: y - (y - top) * next / scale });
    }

    function updateLayerButtons() {
      host.querySelectorAll('[data-pi-layer]').forEach(btn => {
        const on = !!layers[btn.dataset.piLayer];
        btn.classList.toggle('is-selected', on);
        btn.setAttribute('aria-pressed', String(on));
      });
      host.dataset.showSync = String(layers.sync);
      host.dataset.showSlot = String(layers.slot);
      host.dataset.showPipe = String(layers.pipe);
      noteEl.textContent = layers.pipe
        ? '理论因果阶梯流水 · 展开 MTE2 / Cube / FIX / Vector / MTE3（同子泳道严格单行不重叠）；横向按算子单元推进，不代表实测微秒耗时，用于对照真实核内泳道。'
        : '理论因果执行序 · 同核严格单行串行（4 页 Stack = 4 个独立 QK/PV 事件，尾页 = 1 个）；按算子单元推进，不代表实测耗时，真实气泡需对比实测 Trace。';
    }

    function rebuild() {
      layout = buildLayout(task, layers);
      scene.style.width = `${layout.width}px`;
      scene.style.height = `${layout.height}px`;
      summary.textContent = `AIC / AIV 因果泳道 · ${task.stacks.length} Stacks · ${task.pages.length} Pages`;
      const isPreset = task.stacks.length === 4 && task.pages.length === 13;
      presetBtn.classList.toggle('is-selected', isPreset);
      presetBtn.setAttribute('aria-pressed', String(isPreset));
      resetBtn.hidden = !scenario?.get()?.isCustom;
      updateLayerButtons();

      // Causal logical time axis (replaces rigid Tick vertical buckets)
      const axisHtml = `<div class="viz2-pi-axis" style="left:${layout.xOrigin}px;width:${layout.width - layout.xOrigin - 20}px">
        <span class="viz2-pi-axis__title">理论因果逻辑时间（按算子单元推进，非实测耗时） →</span>
        ${layout.stepMarks.map(u => `<span class="viz2-pi-axis__mark" style="left:${layout.tToX(u) - layout.xOrigin}px">u${u}</span>`).join('')}
      </div>`
        + layout.stepMarks.map(u => `<div class="viz2-pi-gridline" style="left:${layout.tToX(u)}px;top:${HEADER_H - 12}px;height:${layout.height - HEADER_H}px"></div>`).join('');

      // Slot reuse bridges
      const reuseSvg = layout.reuses.map((r, idx) => {
        const yArc = 32 - (idx % 2) * 8;
        return `<g class="viz2-pi-reuse-group" style="--slot-color:var(--intra-slot-${r.slot % 3})">
          <path class="viz2-pi-reuse-arc" d="M${r.x1} 44 V${yArc} H${r.x2} V44" marker-end="url(#viz2-pi-arrow)"/>
        </g>`;
      }).join('');
      const reuseButtonsHtml = layout.reuses.map((r, idx) => {
        const midX = (r.x1 + r.x2) / 2;
        return `<button type="button" class="viz2-pi-reuse-pill" data-pi-reuse-btn="${r.slot}" data-pi-from="${r.from}" data-pi-to="${r.to}" style="left:${midX}px;top:${idx % 2 ? 2 : 8}px;--slot-color:var(--intra-slot-${r.slot % 3})">Slot ${r.slot} 复用 · S${r.from}(PV 消费完) → S${r.to}(QK 复用)</button>`;
      }).join('');

      // Core swimlanes
      const coresHtml = layout.cores.map(core => {
        const pipeRailHtml = core.pipes.length
          ? `<div class="viz2-pi-pipe-rail">${core.pipes.map(p => `<div class="viz2-pi-pipe-rail__item" style="top:${p.top - core.top}px;height:${p.height}px"><b>${p.label}</b><small>${p.desc}</small></div>`).join('')}</div>`
          : '';
        const subPipeLinesHtml = core.pipes.map(p => `<div class="viz2-pi-subpipe" style="top:${p.top - core.top}px;height:${p.height}px"></div>`).join('');
        return `<div class="viz2-pi-core-row" data-core="${core.id}" style="top:${core.top}px;height:${core.height}px;width:${layout.width - 16}px">
          <div class="viz2-pi-core-label" style="width:${layout.labelW - 8}px">
            <div class="viz2-pi-core-main">
              <div class="viz2-pi-core-title"><strong>${core.title}</strong><span class="viz2-pi-core-badge ${core.imbalance ? 'is-imbalance' : ''}">${core.badge}</span></div>
              <small>${core.subtitle}</small>
            </div>
            ${pipeRailHtml}
          </div>
          <div class="viz2-pi-core-track">${subPipeLinesHtml}</div>
        </div>`;
      }).join('');

      // Stage grouping brackets (Op mode: lightweight frame showing Stack / Slot / source loop Tick)
      const groupsHtml = layout.groups.map(g => {
        const slotStyle = `--slot-color:var(--intra-slot-${g.slot % 3});`;
        const reuseTag = g.reusedFrom != null ? `<span class="viz2-pi-reuse-mini" style="${slotStyle}">↻ S${g.reusedFrom}</span>` : '';
        const tailTag = g.isTail ? `<span class="viz2-pi-tail-tag">尾页</span>` : '';
        return `<div class="viz2-pi-stage-frame" data-stage="${g.stage}" data-stack="${g.stack}" style="left:${g.x}px;top:${g.y}px;width:${g.w}px;height:${g.h}px;${slotStyle}">
          <div class="viz2-pi-stage-frame__head">
            <span>${esc(g.label)}</span>
            <span class="viz2-pi-slot-tag" style="${slotStyle}">Slot ${g.slot}</span>
            ${reuseTag}${tailTag}
            <span class="viz2-pi-tick-tag" title="源码循环迭代序号">T${g.tick}</span>
          </div>
        </div>`;
      }).join('');

      // Individual horizontal event blocks (rounded OP pill vs squared DMA block — NO Tensor left stripe on OPs!)
      const blocksHtml = layout.blocks.map(b => {
        const slotStyle = b.slot != null ? `--slot-color:var(--intra-slot-${b.slot % 3});` : '';
        const iconHtml = b.icon ? opIcon(b.icon) : '';
        const syncBadge = b.syncOut
          ? `<span class="viz2-pi-ev-dot viz2-pi-ev-dot--out" style="--ev-color:${EVENT_META[b.syncOut].color}" title="SET ${EVENT_META[b.syncOut].name}"></span>`
          : '';
        const cls = b.kind === 'op-pill' ? 'viz2-pi-event viz2-pi-event--op' : 'viz2-pi-event viz2-pi-event--dma';
        return `<button type="button" class="${cls} ${b.isTail ? 'is-tail' : ''}" data-pi-block="${esc(b.uid)}" data-stage="${esc(b.stage)}" data-pipe="${esc(b.pipe || '')}" data-stack="${b.stack ?? ''}" data-slot="${b.slot ?? ''}"
          style="left:${b.x}px;top:${b.y}px;width:${b.w}px;height:${b.h}px;${slotStyle}" aria-pressed="false">
          <span class="viz2-pi-event__head">${iconHtml}<b>${esc(b.title)}</b>${syncBadge}</span>
          <small>${esc(b.subtitle)}</small>
        </button>`;
      }).join('');

      // Sync SVG arcs (strictly forward-in-time along inter-core highway)
      const syncSvg = layout.syncEdges.map(edge => `<path class="viz2-pi-sync-arc" data-sync-id="${esc(edge.id)}" data-sync-event="${edge.event}" data-sync-stack="${edge.stack}" stroke="${edge.color}" d="${syncPath(edge)}" marker-end="url(#viz2-pi-arrow)"/>`).join('');

      scene.innerHTML = axisHtml
        + coresHtml
        + groupsHtml
        + `<svg class="viz2-pi-overlay" width="${layout.width}" height="${layout.height}" aria-hidden="true">
            <defs><marker id="viz2-pi-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse"><path d="M0 0L8 4L0 8Z" fill="context-stroke"/></marker></defs>
            <g class="viz2-pi-reuse-layer">${reuseSvg}</g>
            <g class="viz2-pi-sync-layer">${syncSvg}</g>
          </svg>`
        + `<div class="viz2-pi-reuse-pills">${reuseButtonsHtml}</div>`
        + blocksHtml;
    }

    function matchesFocus(item, focus) {
      if (!focus) return false;
      if (focus.kind === 'slot') return item.slot === focus.index;
      if (focus.kind === 'event') return item.stack === focus.index && (item.syncIn === focus.event || item.syncOut === focus.event);
      if (focus.kind === 'output') return item.stage === 'output';
      if (focus.kind === 'stack') return item.stack === focus.index;
      if (focus.executionInstanceId && item.executionInstanceId === focus.executionInstanceId) {
        if (focus.lane != null && item.lane != null && item.lane !== focus.lane) return false;
        return true;
      }
      if (focus.stage && item.stage === focus.stage) {
        if (focus.stackIndex != null && item.stack !== focus.stackIndex) return false;
        if (Number.isInteger(focus.instancePage) && item.page != null && item.page !== focus.instancePage) return false;
        if (focus.lane != null && item.lane != null && item.lane !== focus.lane) return false;
        return true;
      }
      if (focus.node && item.node === focus.node && (focus.stackIndex == null || item.stack === focus.stackIndex)) return true;
      return false;
    }

    function paintSelection() {
      if (!layout) return;
      const f = hoverItem || currentFocus;
      scene.classList.toggle('has-focus', !!f);

      scene.querySelectorAll('[data-pi-block]').forEach(el => {
        const item = layout.itemsById.get(el.dataset.piBlock);
        const hit = matchesFocus(item, f);
        el.classList.toggle('is-selected', hit);
        el.setAttribute('aria-pressed', String(hit));
      });

      scene.querySelectorAll('.viz2-pi-stage-frame').forEach(el => {
        const st = Number(el.dataset.stack);
        const hit = f ? (f.kind === 'slot' ? task.stacks[st]?.slot === f.index : st === (f.stackIndex ?? f.index)) : false;
        el.classList.toggle('is-related', hit);
      });

      scene.querySelectorAll('.viz2-pi-sync-arc').forEach(arc => {
        const stack = Number(arc.dataset.syncStack);
        const ev = arc.dataset.syncEvent;
        const hit = !f ? false
          : f.kind === 'event' ? (stack === f.index && ev === f.event)
          : f.kind === 'slot' ? task.stacks[stack]?.slot === f.index
          : (stack === (f.stackIndex ?? f.index));
        arc.classList.toggle('is-related', hit);
      });

      scene.querySelectorAll('[data-pi-reuse-btn]').forEach(btn => {
        const slot = Number(btn.dataset.piReuseBtn);
        btn.classList.toggle('is-selected', f?.kind === 'slot' && f.index === slot);
      });
    }

    function renderInspector() {
      inspector.hidden = !inspectorOpen || !currentFocus;
      if (inspector.hidden) return;
      const f = currentFocus;
      const navHtml = onNavigate ? `<div class="viz2-inst-actions"><button type="button" class="btn btn-sm" data-pi-to-graph>在 Graph · Instance 中定位 ↗</button></div>` : '';

      let title = `Task ${task.id}`;
      let ctx = `Block ${String(task.block).padStart(2, '0')} · ${task.seqLen} tokens`;
      let rows = [];
      let tip = '';
      let line = 270;

      if (f.kind === 'slot') {
        const owners = task.stacks.filter(s => s.slot === f.index);
        title = `Slot ${f.index} · 缓冲区复用`;
        ctx = `TRANSFER_SLOTS = 3 · stack % 3`;
        rows = [
          ['复用序列', owners.map(s => `S${s.index} (T${s.produceTick}→T${s.consumeTick})`).join(' → ')],
          ['安全窗口', f.from != null ? `S${f.from} 的 PV 消费完 Slot ${f.index} 后，S${f.to} 的 QK 才写入同一 Slot` : '3 槽轮转隔离'],
          ['关联 Tensor', 'score_transfer / probability_transfer / pv_transfer'],
        ];
        tip = '若真实泳道出现 Slot 覆写，核查 S0 的 PV/Acc 消费是否在 S3 写入同一 Slot 前完成。';
        line = 282;
      } else {
        const matched = layout.blocks.find(b => matchesFocus(b, f));
        title = matched ? `${matched.title}${matched.stack != null ? ` · S${matched.stack}` : ''}` : `Stack ${f.stackIndex ?? f.index ?? 0}`;
        ctx = [
          matched?.core?.toUpperCase(),
          matched?.pipe ? `Pipe: ${matched.pipe.toUpperCase()}` : (matched?.kind === 'op-pill' ? 'OP 事件' : 'DMA / 搬运'),
          matched?.tick != null ? `源码循环 T${matched.tick}` : null,
        ].filter(Boolean).join(' · ');
        rows = [
          ['硬件单元', matched?.core === 'aic' ? 'AIC (1 Core · 全部 5 Q Heads)' : matched?.core === 'aiv0' ? 'AIV 0 (前 2/5 行)' : matched?.core === 'aiv1' ? 'AIV 1 (后 3/5 行 · +1 Head)' : 'AIC / AIV'],
          ['Stack / Slot', matched?.stack != null ? `Stack ${matched.stack} · Slot ${matched.slot}${matched.stack >= 3 ? ` (复用 S${matched.stack - 3})` : ''}` : null],
          ['Page 实例', matched?.page != null ? `Page ${matched.page} · ${task.pages[matched.page].tokens}/${task.pageTokens} tokens` : null],
          ['源码循环', matched?.tick != null ? `Tick ${matched.tick}` : null],
          ['关联同步', [matched?.syncIn ? `WAIT ${EVENT_META[matched.syncIn].name}` : '', matched?.syncOut ? `SET ${EVENT_META[matched.syncOut].name}` : ''].filter(Boolean).join(' / ') || '无'],
        ];
        tip = matched?.debugTip || '';
        line = matched?.sourceLine || 270;
      }

      const snippet = source && line
        ? source.split('\n').slice(Math.max(0, line - 2), line + 11).map((l, i) => `${Math.max(1, line - 1) + i}  ${l}`).join('\n')
        : '';

      inspector.querySelector('[data-pi-title]').textContent = title;
      inspector.querySelector('[data-pi-context]').textContent = ctx;
      detailBody.innerHTML = navHtml
        + `<section><dl>${rows.filter(([, v]) => v != null && v !== '').map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl></section>`
        + (tip ? `<section class="viz2-pi-debug-card"><h3>对照真实泳道排查要点</h3><p>${esc(tip)}</p></section>` : '')
        + (snippet ? `<details class="viz2-inspector-details"><summary>PyPTO 源码 · L${line}</summary><pre data-compute-code>${esc(snippet)}</pre></details>` : '');
    }

    host.addEventListener('click', event => {
      const btn = event.target.closest('button');
      if (!btn) {
        if (suppressBlankClick) { suppressBlankClick = false; return; }
        if (event.target.closest('.viz2-pi-viewport') && currentFocus) onFocus(null);
        return;
      }
      if (btn.dataset.piLayer) {
        const key = btn.dataset.piLayer;
        if (key === 'op') { layers.op = true; layers.pipe = false; }
        else if (key === 'pipe') { layers.pipe = !layers.pipe; layers.op = !layers.pipe; }
        else { layers[key] = !layers[key]; }
        rebuild();
        if (fitMode) fit();
        paintSelection();
        return;
      }
      if (btn.dataset.piPreset) { scenario?.preset?.(Number(btn.dataset.piPreset)); return; }
      if (btn.hasAttribute('data-pi-reset')) { scenario?.reset?.(); return; }
      if (btn.hasAttribute('data-pi-to-graph')) { onNavigate?.(); return; }
      if (btn.hasAttribute('data-pi-fit')) { fit(); return; }
      if (btn.hasAttribute('data-pi-close')) { onClose?.(); return; }
      if (btn.dataset.piZoom) {
        zoom(btn.dataset.piZoom === 'in' ? 1.28 : 1 / 1.28, viewport.clientWidth / 2, viewport.clientHeight / 2);
        return;
      }
      if (btn.dataset.piReuseBtn != null) {
        onFocus({ kind: 'slot', index: Number(btn.dataset.piReuseBtn), from: Number(btn.dataset.piFrom), to: Number(btn.dataset.piTo) });
        return;
      }
      if (btn.dataset.piBlock) {
        const item = layout.itemsById.get(btn.dataset.piBlock);
        if (!item) return;
        if (item.stage === 'output') {
          onFocus({ kind: 'output', node: item.node, index: 0, lane: item.lane, executionInstanceId: item.executionInstanceId });
        } else {
          onFocus({
            kind: 'stage', stage: item.stage, node: item.node,
            index: item.stack, stackIndex: item.stack,
            ...(item.page != null ? { instancePage: item.page } : {}),
            ...(item.lane != null ? { lane: item.lane } : {}),
            ...(item.executionInstanceId ? { executionInstanceId: item.executionInstanceId } : {}),
          });
        }
      }
    });

    scene.addEventListener('pointerover', event => {
      const blk = event.target.closest('[data-pi-block]');
      let next = null;
      if (blk) {
        const item = layout.itemsById.get(blk.dataset.piBlock);
        if (item) {
          next = item.stage === 'output'
            ? { kind: 'output', lane: item.lane, executionInstanceId: item.executionInstanceId }
            : { kind: 'stage', stage: item.stage, stackIndex: item.stack, instancePage: item.page, lane: item.lane, executionInstanceId: item.executionInstanceId };
        }
      }
      if (JSON.stringify(next) !== JSON.stringify(hoverItem)) { hoverItem = next; paintSelection(); }
    });
    scene.addEventListener('pointerleave', () => { if (hoverItem) { hoverItem = null; paintSelection(); } });

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
    viewport.addEventListener('keydown', event => { if (event.key === 'Escape' && currentFocus) onFocus(null); });

    const observer = new ResizeObserver(() => { if (!host.hidden && fitMode) fit(); });
    observer.observe(viewport);

    return {
      update(nextTask, focus, open, visible) {
        const entering = host.hidden && visible;
        host.hidden = !visible;
        if (!visible) return;
        currentFocus = focus;
        inspectorOpen = !!open;
        if (task !== nextTask) {
          task = nextTask;
          hoverItem = null;
          rebuild();
          fit();
        } else if (entering) {
          fit();
        }
        paintSelection();
        renderInspector();
      },
      getTransitionRect: () => viewport.getBoundingClientRect(),
    };
  }

  global.PtoPagedAttentionPipelineInstance = Object.freeze({ buildLayout, mount, PRESET_SEQ_LEN });
})(window);
