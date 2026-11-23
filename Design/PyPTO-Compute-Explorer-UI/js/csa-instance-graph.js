/* CSA qk_pv Unrolled Instance Computation Graph.
 * Mirrors Paged Attention's Instance Graph grammar (visualization2-graph-intra.css):
 *   - Tensors: solid cards with left semantic role bar (activation blue / aggregate gold).
 *   - Operations: rounded pills with op icon + hardware unit badge (AIC / AIV 0 / AIV ×48).
 *   - Scoped Bands:
 *       1) Work-Item Shared (Cols 0–1): 1× per Work Item (Query, KV Cache → KV Gather → Shared L1 KV Tile, sparse_bias).
 *       2) Head Batch 0 (H0–31 · Slot 0) & Head Batch 1 (H32–63 · Slot 1) (Cols 2–8): pl.pipeline(2, stage=2).
 *       3) Downstream Merge (Col 9): independent csa_merge_pack_publish task consuming all 4 Partials.
 *   - Orthogonal M/H/V routing with arrowheads, shared bus trunks & junction dots, direct mouse-wheel zoom. */
(function (global) {
  'use strict';

  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);

  const SUB_FILE = 'decode_sparse_attn_csa.py';
  const MAIN_FILE = 'decode_csa.py';
  const W = 190;
  const H = 48;
  const DX = 248;

  const VALID_STAGES = [
    'Work Inputs',
    'KV Gather & L1',
    'QK Matmul',
    'C2V Pipe · Scores',
    'Scale & Bias',
    'Softmax Stats',
    'V2C Pipe · Exp',
    'PV Matmul',
    'Partial m / l / o',
    'Downstream Merge',
  ];

  const INVALID_STAGES = [
    'Valid Mask',
    'Neutral Partial Write',
    'Partial m / l / o',
    'Downstream Merge',
  ];

  const validXAt = col => 26 + col * DX + (col >= 2 ? 42 : 0) + (col >= 9 ? 18 : 0);
  const invalidXAt = col => 26 + col * DX + (col >= 3 ? 22 : 0);

  function opIcon(kind) {
    const entry = global.PtoOpIcons?.find(item => item.id === kind);
    if (!entry) return '';
    return `<span class="viz2-inst-op-icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="${entry.path}"/></svg></span>`;
  }

  function buildGraph(item) {
    const nodes = new Map();
    const edges = [];
    const bands = [];
    const trunks = [];
    const dots = [];

    const xAt = item.valid ? validXAt : invalidXAt;

    function add(id, col, y, title, meta = {}) {
      const node = {
        id,
        title,
        col,
        x: xAt(col),
        y,
        w: meta.w || W,
        h: meta.h || H,
        role: meta.role || 'activation',
        unit: meta.unit || '',
        icon: meta.icon || null,
        note: meta.note || '',
        detail: meta.detail || '',
        file: meta.file || SUB_FILE,
        range: meta.range || [201, 314],
        upstream: [],
        downstream: [],
        ...meta,
      };
      nodes.set(id, node);
      return node;
    }

    function link(from, to, label, kind = 'data', route = {}) {
      from.downstream.push({ id: to.id, label, kind });
      to.upstream.push({ id: from.id, label, kind });
      edges.push({ from, to, label, kind, route });
    }

    if (!item.valid) {
      const bandTop = 46;
      const firstRowY = bandTop + 42; // 88 — clear of band header (46..74)
      const rowStep = 64;
      const yMid = firstRowY + Math.round((rowStep * 3) / 2); // 184

      const mask = add('mask', 0, yMid, 'valid_block_mask', {
        role: 'activation',
        note: `Q${item.token} · SB${item.sparseBlock} = 0`,
        detail: 'valid_block_mask[t, sb] == 0，跳过 KV Gather、QK Matmul、Softmax 与 PV Matmul。',
        range: [223, 226],
      });

      const maskSpineX = xAt(1) - 28;
      const neutralSpineX = xAt(2) - 28;
      const neutralOps = [];
      for (let h = 0; h < 4; h++) {
        const y = firstRowY + h * rowStep;
        const op = add(`neutral-${h}`, 1, y, `Neutral Write H${h * 16}–${h * 16 + 15}`, {
          role: 'operation',
          unit: 'AIV 0',
          icon: 'cache',
          note: 'mi=−3e38 · li=0 · oi=0',
          detail: `为 Head ${h * 16}–${h * 16 + 15}（H_TILE=16）写出中性 Partial，供下游 Merge 忽略。`,
          range: [305, 313],
        });
        link(mask, op, 'valid==0', 'state', { mode: 'elbow', spineX: maskSpineX });
        neutralOps.push(op);
      }

      const partial = add('partial', 2, yMid, 'Partial m / l / o', {
        role: 'aggregate',
        note: '4 × H_TILE=16 中性段',
        detail: 'sparse_blk_mi / li / oi 中该 Sparse Block 槽位写入中性值。',
        range: [305, 313],
      });
      neutralOps.forEach(op => link(op, partial, 'neutral store', 'data', { mode: 'elbow', spineX: neutralSpineX }));

      const merge = add('merge', 3, yMid, 'csa_merge_pack_publish', {
        role: 'operation',
        unit: 'AIV ×48',
        icon: 'reduce',
        note: '下游独立 Task · 合并 5 SB',
        detail: '读取同一 Query 的全部 5 个 Sparse Block Partial，Invalid 块在 alpha/beta 归约中自动权重归零。',
        file: MAIN_FILE,
        range: [395, 477],
      });
      link(partial, merge, 'qk_tid dep', 'accum', { mode: 'straight' });

      const bandHeight = firstRowY + 3 * rowStep + H + 20 - bandTop; // 302
      bands.push({
        id: 'invalid-band',
        kind: 'invalid',
        title: `Invalid 分支 · Work Item ${item.id} (Q${item.token} / SB${item.sparseBlock})`,
        badge: '跳过 AIC',
        meta: '0 / 128 valid KV rows · 仅写出 4 段中性 Partial',
        slot: 0,
        left: 10,
        width: xAt(2) + W + 16 - 10,
        top: bandTop,
        height: bandHeight,
      });

      const firstMidY = firstRowY + H / 2;
      const lastMidY = firstRowY + 3 * rowStep + H / 2;
      trunks.push({ kind: 'state', owner: 'mask', d: `M${mask.x + W} ${yMid + H / 2} H${maskSpineX} M${maskSpineX} ${firstMidY} V${lastMidY}` });
      dots.push({ kind: 'state', owner: 'mask', cx: maskSpineX, cy: yMid + H / 2 });
      dots.push({ kind: 'data', owner: 'partial', cx: neutralSpineX, cy: yMid + H / 2 });

      return {
        stages: INVALID_STAGES.map((label, idx) => ({ label, x: xAt(idx) })),
        nodes: [...nodes.values()],
        edges,
        bands,
        trunks,
        dots,
        width: xAt(3) + W + 36,
        height: bandTop + bandHeight + 36,
      };
    }

    // Valid Branch Geometry:
    //   - Left Shared Band (Cols 0–1): 1× Work-Item Preamble (Query, KV Cache → KV Gather → Shared L1 KV Tile, sparse_bias)
    //   - Head Batch 0 Band (Cols 2–8, top=46..222): Slot 0 · H0–31
    //   - Open Inter-Band Routing Corridor (y=222..312): Shared L1 KV 4× reuse highway (y=264) & sparse_bias highway (y=286)
    //   - Head Batch 1 Band (Cols 2–8, top=312..488): Slot 1 · H32–63
    //   - Downstream Merge (Col 9, outside HB bands): csa_merge_pack_publish
    const isWindow = item.sparseBlock === 0;
    const hbBandTop = [46, 312];
    const hbBandHeight = 176;
    const hbMainY = hbBandTop.map(t => t + 74); // 120, 386
    const hbPart0Y = hbBandTop.map(t => t + 42); // 88, 354 (>= 14px clear of header)
    const hbPart1Y = hbBandTop.map(t => t + 106); // 152, 418

    const sharedBandLeft = 10;
    const sharedBandRight = xAt(1) + W + 16; // 480
    const hbBandLeft = xAt(2) - 14; // 550
    const hbBandRight = xAt(8) + W + 14; // 2256

    // Dedicated orthogonal routing spines (zero collisions with nodes or band borders)
    const biasRiseX = sharedBandRight + 16; // 496
    const qBusX = sharedBandRight + 36; // 516
    const kvQkSpineX = xAt(2) - 28; // 536
    const biasSpineX = xAt(4) - 24; // 1036
    const kvPvSpineX = xAt(7) - 24; // 1780
    const pvOutSpineX = xAt(8) - 34; // 2018
    const mlOutSpineX = xAt(8) - 18; // 2034
    const mergeBusX = xAt(9) - 28; // 2290
    const kvBusY = 264;
    const biasBusY = 286;

    bands.push({
      id: 'work-shared',
      kind: 'shared',
      title: 'Work-Item Shared · 1× 执行',
      badge: 'L1 共享',
      meta: 'HB0 & HB1 复用输入与 L1 KV',
      slot: 2,
      left: sharedBandLeft,
      width: sharedBandRight - sharedBandLeft,
      top: hbBandTop[0],
      height: hbBandTop[1] + hbBandHeight - hbBandTop[0],
    });

    const qIn = add('q-in', 0, 112, `Query Q${item.token}`, {
      role: 'activation',
      note: 'BF16 [64, 512] · 2 × 32 Heads',
      detail: '同一 Query Token 的 64 个 Attention Heads，拆为 HB0（H0–31）与 HB1（H32–63）分别送入 AIC。',
      range: [278, 283],
    });
    const kvSrc = add('kv-src', 0, 178, isWindow ? 'Original KV Cache' : 'Compressed KV Cache', {
      role: 'activation',
      note: isWindow ? `SB0 滑窗 · ${item.validKvCount}/128` : `SB${item.sparseBlock} 压缩 Top-K · ${item.validKvCount}/128`,
      detail: isWindow
        ? 'SB0 从 Original KV Cache 按页连续段（SWA_RUNS）多行 Gather。'
        : 'SB1–4 根据 cmp_sparse_indices 与 cmp_block_table 从 Compressed KV Cache 逐行 Gather。',
      range: isWindow ? [235, 258] : [260, 276],
    });
    const kvGather = add('kv-gather', 1, 178, 'KV Gather', {
      role: 'operation',
      unit: 'AIC · MTE2',
      icon: 'cache',
      note: isWindow ? 'SWA 连续页多行 Gather' : 'Top-K 压缩索引逐行 Gather',
      detail: '每个 Work Item 仅执行 1 次 KV Gather，装载到共享 L1 Buffer。',
      range: [225, 276],
    });
    const kvL1 = add('kv-l1', 1, 248, 'Shared L1 KV Tile', {
      role: 'aggregate',
      note: 'BF16 [128, 512] · 4× 复用',
      detail: '驻留 AIC L1 的共享 [128, 512] BF16 Tile，被 HB0 与 HB1 的 QK（Kᵀ）和 PV（V）共复用 4 次。',
      range: [227, 296],
    });
    const biasIn = add('bias-in', 0, hbMainY[1], 'sparse_bias', {
      role: 'activation',
      note: `FP32 [1, 128] · SB${item.sparseBlock}`,
      detail: '由 qk_plan 生成的加性 Mask（有效槽位 0，无效槽位 −1e20），在 AIV 0 上直接广播相加。',
      range: [173, 184],
    });

    link(kvSrc, kvGather, 'GM → L1', 'data', { mode: 'straight' });
    link(kvGather, kvL1, 'L1 Tile', 'data', { mode: 'vertical-down' });

    const qPortY = hbMainY.map(y => y + 14); // 134, 400
    const kvPortY = hbMainY.map(y => y + 34); // 154, 420
    const partialNodes = [];

    for (const hb of [0, 1]) {
      const top = hbBandTop[hb];
      const yMain = hbMainY[hb];
      const yP0 = hbPart0Y[hb];
      const yP1 = hbPart1Y[hb];
      const hRange = hb === 0 ? 'H0–31' : 'H32–63';
      const slot = hb;
      const mlTrackY = yP1 + 34; // 186 (HB0) / 452 (HB1) — 18px clear below Exp & PV

      bands.push({
        id: `hb-${hb}`,
        kind: 'head-batch',
        title: `Head Batch ${hb} · ${hRange}`,
        badge: `Slot ${slot}`,
        meta: `pl.pipeline(2, stage=2) · C2V/V2C Slot ${slot} (16 KB)`,
        slot,
        left: hbBandLeft,
        width: hbBandRight - hbBandLeft,
        top,
        height: hbBandHeight,
      });

      const qk = add(`qk-hb${hb}`, 2, yMain, `QK Matmul · HB${hb}`, {
        role: 'operation',
        unit: 'AIC',
        icon: 'matmul',
        note: 'Q[32,512]×Kᵀ[512,128] · 4×K128',
        detail: `HB${hb}（${hRange}）执行 4 段 K=128 累加，产出 FP32 Scores [32, 128]。`,
        range: [280, 283],
      });
      const c2v = add(`c2v-hb${hb}`, 3, yMain, `Scores · C2V Slot ${slot}`, {
        role: 'aggregate',
        note: 'FP32 [32, 128] · 16 KB',
        detail: `通过 qk_pv_c2v_slot_buffer Slot ${slot}（tpush_to_aiv → tpop_from_aic）跨核传给 AIV 0。`,
        range: [282, 284],
      });
      const scaleBias = add(`sb-hb${hb}`, 4, yMain, 'Scale & Bias', {
        role: 'operation',
        unit: 'AIV 0',
        icon: 'softmax',
        note: `muls + col_expand_add · ${hRange}`,
        detail: 'AIV 0 消费 C2V Slot 后释放槽位，乘 SOFTMAX_SCALE 并加 sparse_bias。',
        range: [284, 287],
      });
      const sm = add(`sm-hb${hb}`, 5, yMain, `Softmax Stats · HB${hb}`, {
        role: 'operation',
        unit: 'AIV 0',
        icon: 'softmax',
        note: 'm[32,1] · exp[32,128] · l[32,1]',
        detail: '计算单 Sparse Block 局部最大 m、指数 exp 与行和 l，并把 exp 量化回 BF16。',
        range: [288, 294],
      });
      const v2c = add(`v2c-hb${hb}`, 6, yMain, `Exp · V2C Slot ${slot}`, {
        role: 'aggregate',
        note: 'BF16 [32, 128] · 16 KB',
        detail: `AIV 0 通过 qk_pv_v2c_slot_buffer Slot ${slot}（tpush_to_aic → tpop_from_aiv）把 BF16 exp 送回 AIC。`,
        range: [294, 295],
      });
      const pv = add(`pv-hb${hb}`, 7, yMain, `PV Matmul · HB${hb}`, {
        role: 'operation',
        unit: 'AIC',
        icon: 'matmul',
        note: 'exp[32,128]×V[128,512] · 4×K32',
        detail: '复用同一 L1 KV Tile 执行 4 段 K=32 累加，输出 FP32 Partial O [32, 512]。',
        range: [295, 296],
      });
      const pSeg0 = add(`p-hb${hb}-0`, 8, yP0, `Partial H${hb * 32}–${hb * 32 + 15}`, {
        role: 'aggregate',
        note: 'H_TILE=16 · m/l (AIV0) + o (AIC)',
        detail: `写出 sparse_blk_mi / li（AIV 0）与 sparse_blk_oi（AIC）第 ${hb * 2} 段。`,
        range: [297, 304],
      });
      const pSeg1 = add(`p-hb${hb}-1`, 8, yP1, `Partial H${hb * 32 + 16}–${hb * 32 + 31}`, {
        role: 'aggregate',
        note: 'H_TILE=16 · m/l (AIV0) + o (AIC)',
        detail: `写出 sparse_blk_mi / li（AIV 0）与 sparse_blk_oi（AIC）第 ${hb * 2 + 1} 段。`,
        range: [297, 304],
      });

      link(qIn, qk, `Q ${hRange}`, 'q-bus', { mode: 'q-bus', busX: qBusX, sy: qPortY[0], ty: qPortY[hb] });
      link(kvL1, qk, 'Kᵀ reuse', 'state', { mode: 'kv-bus', busY: kvBusY, spineX: kvQkSpineX, ty: kvPortY[hb] });
      link(qk, c2v, 'tpush C2V', 'data', { mode: 'straight' });
      link(c2v, scaleBias, 'tpop C2V', 'data', { mode: 'straight', sy: qPortY[hb], ty: qPortY[hb] });
      link(biasIn, scaleBias, 'bias', 'q-bus', { mode: 'bias-bus', riseX: biasRiseX, busY: biasBusY, spineX: biasSpineX, ty: kvPortY[hb] });
      link(scaleBias, sm, 'scores', 'data', { mode: 'straight' });
      link(sm, v2c, 'tpush V2C', 'data', { mode: 'straight', sy: qPortY[hb], ty: qPortY[hb] });
      link(v2c, pv, 'tpop V2C', 'data', { mode: 'straight', sy: qPortY[hb], ty: qPortY[hb] });
      link(kvL1, pv, 'V reuse', 'state', { mode: 'kv-bus', busY: kvBusY, spineX: kvPvSpineX, ty: kvPortY[hb] });
      link(pv, pSeg0, 'O[0:16]', 'data', { mode: 'elbow', spineX: pvOutSpineX, ty: yP0 + 16 });
      link(pv, pSeg1, 'O[16:32]', 'data', { mode: 'elbow', spineX: pvOutSpineX, ty: yP1 + 16 });
      link(sm, pSeg0, 'm/l[0:16]', 'accum', { mode: 'ml-bus', trackY: mlTrackY, spineX: mlOutSpineX, sy: yMain + 36, ty: yP0 + 34 });
      link(sm, pSeg1, 'm/l[16:32]', 'accum', { mode: 'ml-bus', trackY: mlTrackY, spineX: mlOutSpineX, sy: yMain + 36, ty: yP1 + 34 });

      dots.push({ kind: 'accum', owner: sm.id, cx: mlOutSpineX, cy: mlTrackY });
      dots.push({ kind: 'data', owner: pv.id, cx: pvOutSpineX, cy: yMain + H / 2 });
      partialNodes.push(pSeg0, pSeg1);
    }

    const merge = add('merge', 9, 244, 'csa_merge_pack_publish', {
      role: 'operation',
      unit: 'AIV ×48',
      icon: 'reduce',
      note: '下游独立 Task · 合并 5 SB',
      detail: '依赖 qk_tid + attn_rope_tid，将 5 个 Sparse Block 的 Partial m/l/o 做在线归一化、Sink 累加、逆 RoPE 并按 O-Group 发布。',
      file: MAIN_FILE,
      range: [395, 477],
    });
    partialNodes.forEach(p => link(p, merge, 'qk_tid', 'accum', { mode: 'elbow', spineX: mergeBusX }));

    // Shared visual bus trunks & junction dots (matches Paged Attention's q-bus grammar)
    trunks.push({
      kind: 'q-bus',
      owner: 'q-in',
      d: `M${qIn.x + W} ${qPortY[0]} H${qBusX} V${qPortY[1]}`,
    });
    dots.push({ kind: 'q-bus', owner: 'q-in', cx: qBusX, cy: qPortY[0] });

    trunks.push({
      kind: 'state',
      owner: 'kv-l1',
      d: `M${kvL1.x + W} ${kvBusY} H${kvPvSpineX}`,
    });
    dots.push({ kind: 'state', owner: 'kv-l1', cx: kvQkSpineX, cy: kvBusY });
    dots.push({ kind: 'state', owner: 'kv-l1', cx: kvPvSpineX, cy: kvBusY });

    trunks.push({
      kind: 'q-bus',
      owner: 'bias-in',
      d: `M${biasIn.x + W} ${biasIn.y + H / 2} H${biasRiseX} V${biasBusY} H${biasSpineX}`,
    });
    dots.push({ kind: 'q-bus', owner: 'bias-in', cx: biasSpineX, cy: biasBusY });

    dots.push({ kind: 'accum', owner: 'merge', cx: mergeBusX, cy: merge.y + H / 2 });

    return {
      stages: VALID_STAGES.map((label, idx) => ({ label, x: xAt(idx) })),
      nodes: [...nodes.values()],
      edges,
      bands,
      trunks,
      dots,
      width: xAt(9) + W + 36,
      height: hbBandTop[1] + hbBandHeight + 36,
    };
  }

  function edgePath(edge) {
    const { from, to, route = {} } = edge;
    const sx = from.x + from.w;
    const tx = to.x;
    const sy = route.sy != null ? route.sy : from.y + from.h / 2;
    const ty = route.ty != null ? route.ty : to.y + to.h / 2;

    if (route.mode === 'vertical-down') {
      const cx = from.x + from.w / 2;
      return `M${cx} ${from.y + from.h} V${to.y}`;
    }
    if (route.mode === 'q-bus') {
      return sy === ty
        ? `M${route.busX} ${sy} H${tx}`
        : `M${route.busX} ${sy} V${ty} H${tx}`;
    }
    if (route.mode === 'kv-bus') {
      return `M${route.spineX} ${route.busY} V${ty} H${tx}`;
    }
    if (route.mode === 'bias-bus') {
      return `M${route.spineX} ${route.busY} V${ty} H${tx}`;
    }
    if (route.mode === 'ml-bus') {
      const dropX = sx + 16;
      return ty === route.trackY
        ? `M${sx} ${sy} H${dropX} V${route.trackY} H${tx}`
        : `M${sx} ${sy} H${dropX} V${route.trackY} H${route.spineX} V${ty} H${tx}`;
    }
    if (route.mode === 'elbow') {
      const ch = route.spineX != null ? route.spineX : Math.round((sx + tx) / 2);
      return sy === ty ? `M${sx} ${sy} H${tx}` : `M${sx} ${sy} H${ch} V${ty} H${tx}`;
    }
    return sy === ty
      ? `M${sx} ${sy} H${tx}`
      : `M${sx} ${sy} H${Math.round((sx + tx) / 2)} V${ty} H${tx}`;
  }

  function renderGraphHtml(graph) {
    const ranksHtml = graph.stages.map(s =>
      `<b class="viz2-instance-rank" style="left:${s.x}px">${esc(s.label)}</b>`
    ).join('');

    const bandsHtml = graph.bands.map(b =>
      `<div class="viz2-instance-band" data-band-kind="${esc(b.kind || 'head-batch')}" style="left:${b.left}px;right:auto;width:${b.width}px;top:${b.top}px;height:${b.height}px;--slot-color:var(--intra-slot-${b.slot % 3})">
        <div class="viz2-instance-band__header">
          <strong>${esc(b.title)}</strong>
          ${b.badge ? `<span class="viz2-inst-badge viz2-inst-badge--slot">${esc(b.badge)}</span>` : ''}
          <span class="viz2-instance-band__ticks">${esc(b.meta)}</span>
        </div>
      </div>`
    ).join('');

    const trunksHtml = graph.trunks.map(t =>
      `<path class="csa-inst-trunk" data-kind="${esc(t.kind)}" data-owner="${esc(t.owner)}" d="${t.d}"/>`
    ).join('');

    const dotsHtml = graph.dots.map(d =>
      `<circle class="viz2-inst-bus-dot csa-inst-bus-dot" data-kind="${esc(d.kind)}" data-owner="${esc(d.owner)}" cx="${d.cx}" cy="${d.cy}" r="2.8"/>`
    ).join('');

    const edgesHtml = graph.edges.map((e, idx) =>
      `<path data-csa-edge-idx="${idx}" data-kind="${esc(e.kind)}" d="${edgePath(e)}" marker-end="url(#csa-instance-arrow)"/>`
    ).join('');

    const edgesSvg = `<svg class="viz2-instance-edges" width="${graph.width}" height="${graph.height}" viewBox="0 0 ${graph.width} ${graph.height}" aria-hidden="true">
      <defs>
        <marker id="csa-instance-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">
          <path d="M0 0L8 4L0 8Z" fill="context-stroke"/>
        </marker>
      </defs>
      ${trunksHtml}
      ${edgesHtml}
      ${dotsHtml}
    </svg>`;

    const nodesHtml = graph.nodes.map(n => {
      const unitAttr = n.unit ? (n.unit.startsWith('AIC') ? ' data-unit="aic"' : ' data-unit="aiv"') : '';
      return `<button type="button" class="viz2-instance-node" data-role="${esc(n.role)}" data-csa-ig-node="${esc(n.id)}" aria-pressed="false" style="left:${n.x}px;top:${n.y}px;width:${n.w}px;height:${n.h}px">
        <span class="viz2-inst-head">${n.icon ? opIcon(n.icon) : ''}<b>${esc(n.title)}</b>${n.unit ? `<em class="viz2-inst-unit"${unitAttr}>${esc(n.unit)}</em>` : ''}</span>
        <small>${esc(n.note)}</small>
      </button>`;
    }).join('');

    return `${ranksHtml}${bandsHtml}${edgesSvg}${nodesHtml}`;
  }

  function mount(host, options) {
    let scenario = null;
    let itemId = 11;
    let graph = null;
    let focusId = null;
    let hoverId = null;
    let fitMode = true;
    let cam = { x: 18, y: 12, scale: 0.72 };
    let drag = null;
    let suppressBlankClick = false;
    let nodeElements = new Map();
    let edgeElements = [];
    let trunkElements = [];
    let dotElements = [];

    host.innerHTML = `<div class="viz2-instance-graph">
      <div class="viz2-instance-toolbar">
        <div class="viz2-instance-toolbar__left">
          <span data-csa-ig-summary></span>
          <button type="button" class="viz2-preset-btn is-selected" data-csa-ig-preset="valid-window">SB0 滑窗 Valid</button>
          <button type="button" class="viz2-preset-btn" data-csa-ig-preset="valid-cmp">SB1 压缩 Valid</button>
          <button type="button" class="viz2-preset-btn" data-csa-ig-preset="invalid">Invalid 分支</button>
          <button type="button" class="viz2-preset-btn" data-csa-ig-open-pipeline>在因果泳道中查看 ↗</button>
        </div>
        <div class="viz2-instance-tools">
          <button type="button" class="btn btn-ghost btn-sm" data-csa-ig-zoom="out" aria-label="缩小">−</button>
          <output data-csa-ig-scale>72%</output>
          <button type="button" class="btn btn-ghost btn-sm" data-csa-ig-zoom="in" aria-label="放大">＋</button>
          <button type="button" class="btn btn-ghost btn-sm" data-csa-ig-fit>适应</button>
        </div>
      </div>
      <div class="viz2-instance-viewport" tabindex="0" role="region" aria-label="CSA qk_pv 单 Work Item 实例计算图，滚轮缩放、拖拽或方向键平移">
        <div class="viz2-instance-scene" data-csa-ig-scene></div>
      </div>
      <aside class="pto-ide-frame__pane viz2-pi-inspector" data-csa-ig-inspector hidden></aside>
    </div>`;

    const summaryEl = host.querySelector('[data-csa-ig-summary]');
    const viewport = host.querySelector('.viz2-instance-viewport');
    const scene = host.querySelector('[data-csa-ig-scene]');
    const scaleOut = host.querySelector('[data-csa-ig-scale]');
    const inspector = host.querySelector('[data-csa-ig-inspector]');

    function applyCamera() {
      scene.style.transform = `translate(${cam.x}px, ${cam.y}px) scale(${cam.scale})`;
      scaleOut.textContent = `${Math.round(cam.scale * 100)}%`;
    }

    function zoomAt(factor, mx, my) {
      if (!graph) return;
      fitMode = false;
      const next = Math.max(0.25, Math.min(2.4, cam.scale * factor));
      const ratio = next / cam.scale;
      cam = {
        scale: next,
        x: mx - (mx - cam.x) * ratio,
        y: my - (my - cam.y) * ratio,
      };
      applyCamera();
    }

    function fit() {
      if (!graph || !viewport.clientWidth || !viewport.clientHeight) return;
      fitMode = true;
      const sx = (viewport.clientWidth - 28) / graph.width;
      const sy = (viewport.clientHeight - 28) / graph.height;
      const s = Math.max(0.25, Math.min(1.05, Math.min(sx, sy)));
      cam = {
        scale: s,
        x: Math.max(12, Math.round((viewport.clientWidth - graph.width * s) / 2)),
        y: Math.max(12, Math.round((viewport.clientHeight - graph.height * s) / 2)),
      };
      applyCamera();
    }

    function paintHighlights() {
      if (!graph) return;
      const activeId = hoverId || focusId;
      const hasFocus = !!activeId;
      const relatedNodes = new Set();
      const relatedEdges = new Set();

      if (activeId) {
        relatedNodes.add(activeId);
        graph.edges.forEach((e, idx) => {
          if (e.from.id === activeId || e.to.id === activeId) {
            relatedEdges.add(idx);
            relatedNodes.add(e.from.id);
            relatedNodes.add(e.to.id);
          }
        });
      }

      scene.classList.toggle('has-focus', hasFocus);
      nodeElements.forEach((el, id) => {
        el.setAttribute('aria-pressed', String(focusId === id));
        el.classList.toggle('is-related', hasFocus && relatedNodes.has(id));
      });
      edgeElements.forEach((el, idx) => {
        el.classList.toggle('is-related', relatedEdges.has(idx));
      });
      trunkElements.forEach(el => {
        const owner = el.dataset.owner;
        const on = hasFocus && (owner === activeId || graph.edges.some((e, i) => relatedEdges.has(i) && e.from.id === owner));
        el.classList.toggle('is-related', on);
      });
      dotElements.forEach(el => {
        const owner = el.dataset.owner;
        const on = hasFocus && (owner === activeId || relatedNodes.has(owner));
        el.classList.toggle('is-related', on);
      });
    }

    function syncInspector() {
      const node = focusId ? graph?.nodes.find(n => n.id === focusId) : null;
      if (!node) {
        inspector.hidden = true;
        return;
      }
      const nodeById = new Map(graph.nodes.map(n => [n.id, n]));
      const renderLinks = list => list.length
        ? `<ul class="csa-ig-rel-list">${list.map(entry => {
            const target = nodeById.get(entry.id);
            const label = target ? `${target.title} (${entry.label})` : `${entry.id} (${entry.label})`;
            return `<li><button type="button" class="btn btn-ghost btn-sm" data-csa-ig-jump="${esc(entry.id)}">${esc(label)}</button></li>`;
          }).join('')}</ul>`
        : '外部输入 / 输出';

      inspector.innerHTML = `<header class="pto-ide-frame__pane-header">
        <div><b>${esc(node.title)}</b><small>${esc(node.unit || (node.role === 'aggregate' ? 'Intermediate / Buffer' : 'Tensor'))}</small></div>
        <button type="button" class="btn btn-ghost btn-sm" data-csa-ig-close aria-label="关闭详情">×</button>
      </header>
      <div class="pto-ide-frame__pane-body">
        <p>${esc(node.detail)}</p>
        <dl class="viz2-pi-dl">
          <dt>规格</dt><dd>${esc(node.note)}</dd>
          <dt>上游输入</dt><dd>${renderLinks(node.upstream)}</dd>
          <dt>下游消费</dt><dd>${renderLinks(node.downstream)}</dd>
        </dl>
        <button type="button" class="btn btn-ghost btn-sm" data-csa-ig-source="${esc(node.file)}:${node.range[0]}:${node.range[1]}">查看源码 · ${esc(node.file)} L${node.range[0]}–${node.range[1]} ↗</button>
      </div>`;
      inspector.hidden = false;
    }

    function currentItem() {
      if (!scenario) return null;
      return scenario.byId.get(itemId) || scenario.items[0];
    }

    function redraw(refit = false) {
      const item = currentItem();
      if (!item) return;
      const work = global.PtoCsaQkWork;
      const detail = work.getWorkItemDetails(item.id, scenario.queue, scenario.blockCount);
      graph = buildGraph(item);
      summaryEl.textContent = `qk_pv 实例计算图 · Work Item ${item.id} (Q${item.token} / SB${item.sparseBlock} · Lane ${detail.block} Iter ${detail.iteration}) · ${item.valid ? '2 Head Batches · 4×H_TILE=16' : 'Invalid 中性路径'}`;
      scene.style.width = `${graph.width}px`;
      scene.style.height = `${graph.height}px`;
      scene.innerHTML = renderGraphHtml(graph);

      nodeElements = new Map([...scene.querySelectorAll('[data-csa-ig-node]')].map(el => [el.dataset.csaIgNode, el]));
      edgeElements = [...scene.querySelectorAll('[data-csa-edge-idx]')];
      trunkElements = [...scene.querySelectorAll('.csa-inst-trunk')];
      dotElements = [...scene.querySelectorAll('.csa-inst-bus-dot')];

      paintHighlights();
      syncInspector();
      if (refit) requestAnimationFrame(fit);
      else applyCamera();
    }

    host.addEventListener('click', event => {
      const presetBtn = event.target.closest('[data-csa-ig-preset]');
      if (presetBtn && scenario) {
        const mode = presetBtn.dataset.csaIgPreset;
        const found = mode === 'valid-window'
          ? scenario.items.find(it => it.valid && it.sparseBlock === 0)
          : mode === 'valid-cmp'
            ? scenario.items.find(it => it.valid && it.sparseBlock > 0)
            : scenario.items.find(it => !it.valid);
        if (found) {
          itemId = found.id;
          focusId = null;
          hoverId = null;
          host.querySelectorAll('[data-csa-ig-preset]').forEach(b => b.classList.toggle('is-selected', b === presetBtn));
          redraw(true);
        }
        return;
      }
      if (event.target.closest('[data-csa-ig-open-pipeline]')) {
        options.onOpenPipeline?.(itemId);
        return;
      }
      const zoomBtn = event.target.closest('[data-csa-ig-zoom]');
      if (zoomBtn) {
        const factor = zoomBtn.dataset.csaIgZoom === 'in' ? 1.25 : 1 / 1.25;
        zoomAt(factor, viewport.clientWidth / 2, viewport.clientHeight / 2);
        return;
      }
      if (event.target.closest('[data-csa-ig-fit]')) { fit(); return; }
      if (event.target.closest('[data-csa-ig-close]')) {
        focusId = null;
        paintHighlights();
        syncInspector();
        return;
      }
      const srcBtn = event.target.closest('[data-csa-ig-source]');
      if (srcBtn) {
        const [f, a, z] = srcBtn.dataset.csaIgSource.split(':');
        options.onSource?.(f, Number(a), Number(z));
        return;
      }
      const jumpBtn = event.target.closest('[data-csa-ig-jump]');
      if (jumpBtn && graph) {
        const target = graph.nodes.find(n => n.id === jumpBtn.dataset.csaIgJump);
        if (target) {
          focusId = target.id;
          const nextScale = Math.max(cam.scale, 0.78);
          fitMode = false;
          cam = {
            scale: nextScale,
            x: viewport.clientWidth / 2 - (target.x + target.w / 2) * nextScale,
            y: viewport.clientHeight / 2 - (target.y + target.h / 2) * nextScale,
          };
          applyCamera();
          paintHighlights();
          syncInspector();
        }
        return;
      }
      const nodeBtn = event.target.closest('[data-csa-ig-node]');
      if (nodeBtn) {
        const id = nodeBtn.dataset.csaIgNode;
        focusId = focusId === id ? null : id;
        paintHighlights();
        syncInspector();
        return;
      }
      if (!event.target.closest('button') && event.target.closest('.viz2-instance-viewport')) {
        if (suppressBlankClick) { suppressBlankClick = false; return; }
        if (focusId) {
          focusId = null;
          paintHighlights();
          syncInspector();
        }
      }
    });

    scene.addEventListener('pointerover', event => {
      const btn = event.target.closest('[data-csa-ig-node]');
      const next = btn?.dataset.csaIgNode || null;
      if (next !== hoverId) {
        hoverId = next;
        paintHighlights();
      }
    });
    scene.addEventListener('pointerleave', () => {
      if (hoverId !== null) {
        hoverId = null;
        paintHighlights();
      }
    });

    viewport.addEventListener('pointerdown', event => {
      if (event.button !== 0 || event.target.closest('button')) return;
      suppressBlankClick = false;
      drag = {
        x: event.clientX,
        y: event.clientY,
        camX: cam.x,
        camY: cam.y,
        pid: event.pointerId,
        moved: false,
      };
      viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener('pointermove', event => {
      if (!drag || drag.pid !== event.pointerId) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
      if (!drag.moved) return;
      fitMode = false;
      cam.x = drag.camX + dx;
      cam.y = drag.camY + dy;
      applyCamera();
    });
    const endDrag = () => {
      if (drag?.moved) suppressBlankClick = true;
      drag = null;
    };
    viewport.addEventListener('pointerup', endDrag);
    viewport.addEventListener('pointercancel', endDrag);
    viewport.addEventListener('lostpointercapture', endDrag);

    // Direct mouse-wheel & trackpad pinch zoom centered on cursor (aligned with PA Instance Graph)
    viewport.addEventListener('wheel', event => {
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? rect.height : 1;
      const delta = Math.max(-0.28, Math.min(0.28, -event.deltaY * unit * 0.0022));
      zoomAt(Math.exp(delta), event.clientX - rect.left, event.clientY - rect.top);
    }, { passive: false });

    viewport.addEventListener('keydown', event => {
      if (event.key === 'Escape' && focusId) {
        focusId = null;
        paintHighlights();
        syncInspector();
        return;
      }
      if (event.target !== viewport) return;
      const step = { ArrowLeft: [48, 0], ArrowRight: [-48, 0], ArrowUp: [0, 48], ArrowDown: [0, -48] }[event.key];
      if (!step) return;
      event.preventDefault();
      fitMode = false;
      cam.x += step[0];
      cam.y += step[1];
      applyCamera();
    });

    const observer = new ResizeObserver(() => {
      if (!host.hidden && fitMode) fit();
    });
    observer.observe(viewport);

    return {
      update(nextScenario, preferredItemId) {
        scenario = nextScenario;
        if (preferredItemId != null && scenario?.byId.has(preferredItemId)) {
          itemId = preferredItemId;
        } else if (!scenario?.byId.has(itemId)) {
          const def = scenario?.items.find(it => it.valid && it.sparseBlock > 0) || scenario?.items[0];
          if (def) itemId = def.id;
        }
        const item = currentItem();
        if (item) {
          const activePreset = !item.valid ? 'invalid' : item.sparseBlock === 0 ? 'valid-window' : 'valid-cmp';
          host.querySelectorAll('[data-csa-ig-preset]').forEach(b =>
            b.classList.toggle('is-selected', b.dataset.csaIgPreset === activePreset)
          );
        }
        focusId = null;
        hoverId = null;
        redraw(true);
      },
      fit,
    };
  }

  global.PtoCsaInstanceGraph = Object.freeze({ mount });
})(window);
