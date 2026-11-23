/* Compressor Block Instance Causal Hardware Swimlane (AIC / AIV).
 * Reuses the exact viz2-pipeline-instance visual grammar of qk_pv:
 *   - Strictly single-row serial per hardware lane / sub-pipe.
 *   - Op mode: serial block-level execution with iteration stage frames and double-buffer slot tags.
 *   - Pipe mode: expanded hardware sub-pipes (AIC: MTE2 / MTE1 / Cube / FIX; AIV: Scalar / MTE2 / Vector / MTE3)
 *     showing pipeline overlap (e.g. pl.pipeline(0, 8, stage=2)), sync arcs, and buffer reuse arcs.
 *   - Grounded in decode_compressor_ratio4.py, decode_indexer_compressor.py, PTO IR, and DFX rank records. */
(function (global) {
  'use strict';

  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);

  const HEADER_H = 128;
  const UNIT_W = 96;

  const SYNC_META = {
    L1_READY: {
      name: 'MTE2 → MTE1 (L1 Ready)',
      color: '#80B7F7',
      producerPipe: 'MTE2 (GM → L1)',
      consumerPipe: 'MTE1 (L1 → L0A/L0B)',
      buffer: 'L1 Ping-Pong Slot (stage=2)',
      debugTip: 'MTE2 完成当前 K Tile 的 X / Wkv / Wgate 从 GM 到 L1（DN2ZN 转置加载）后释放槽位信号，MTE1 立即抽取至 L0A / L0B。',
    },
    L0_READY: {
      name: 'MTE1 → Cube (L0 Ready)',
      color: '#9EF04B',
      producerPipe: 'MTE1 (L1 → L0A/L0B)',
      consumerPipe: 'Cube (L0A × L0B → L0C)',
      buffer: 'L0A / L0B Cube Buffers',
      debugTip: 'MTE1 将子块送入 L0A / L0B 后触发 Cube 计算 KV 与 Score 矩阵乘累加，两路投影共用同一份 X Tile。',
    },
    L0C_FIX: {
      name: 'Cube → FIX (L0C Store)',
      color: '#F2C45F',
      producerPipe: 'Cube (L0C Accumulator)',
      consumerPipe: 'FIX (L0C → GM)',
      buffer: 'L0C FP32 Accumulator',
      debugTip: '全部 K 迭代累加完成后，FIX 单元将 L0C 中的 FP32 结果 Tile 写回 GM。',
    },
    SCALAR_GUARD: {
      name: 'Scalar → MTE2 (Guard / Index)',
      color: '#F2C45F',
      producerPipe: 'Scalar (Control / Index)',
      consumerPipe: 'MTE2 (GM → UB)',
      buffer: 'Scalar Register / Slot Index',
      debugTip: 'Scalar 单元读取 position_ids 或 slot_mapping 并计算边界/槽位索引，条件满足时才触发后续 MTE2 搬运与写回。',
    },
    UB_READY: {
      name: 'MTE2 → Vector (UB Ready)',
      color: '#80B7F7',
      producerPipe: 'MTE2 (GM → UB)',
      consumerPipe: 'Vector (UB Compute)',
      buffer: 'Unified Buffer (UB)',
      debugTip: 'MTE2 将输入 Tile 或历史 Ring 槽位加载至 UB，Vector 单元读取并执行归约、Softmax、RMSNorm、RoPE 或量化。',
    },
    VEC_STORE: {
      name: 'Vector → MTE3 (Writeback)',
      color: '#9EF04B',
      producerPipe: 'Vector (UB Compute)',
      consumerPipe: 'MTE3 (UB → GM)',
      buffer: 'UB Output Tile',
      debugTip: 'Vector 计算并转换输出精度（FP32 / BF16 / INT8）后，MTE3 将结果 Tile 写回 GM / KV Cache。',
    },
    STATE_CARRY: {
      name: 'Recurrent State Carry',
      color: '#D58BF7',
      producerPipe: 'Vector (Iter i)',
      consumerPipe: 'Vector (Iter i+1)',
      buffer: 'UB Accumulator (m/l/o 或 partial_sq)',
      debugTip: '跨迭代递归累加状态（Online Softmax 的 m/l/o 或 RMSNorm 的 partial_sq），前后迭代严格数据依赖。',
    },
  };

  function opIcon(kind) {
    const entry = global.PtoOpIcons?.find(item => item.id === kind);
    if (!entry) return '';
    return `<span class="viz2-pi-op-icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="${entry.path}"/></svg></span>`;
  }

  function measuredFor(model, kernelName, rank) {
    const k = model?.kernels?.get(kernelName);
    return k?.ranks?.[String(rank)] || k?.ranks?.['0'] || null;
  }

  function irForLines(model, kernelName, lineRange) {
    const all = model?.measured?.ir?.[kernelName] || [];
    if (!lineRange || !all.length) return all.slice(0, 8);
    const [a, b] = lineRange;
    const matched = all.filter(([, , py]) => py >= a && py <= b);
    return (matched.length ? matched : all).slice(0, 10);
  }

  function buildLayout(model, kernelInfo, block, rank, layers, localView) {
    const shape = model.blockShape(kernelInfo, block, localView);
    const showPipe = !!layers.pipe;
    const isAic = kernelInfo.coreType === 'AIC';
    const isMain = kernelInfo.branch === 'main';
    const srcLine = (mainRange, idxRange) => (isMain ? mainRange : idxRange);

    const labelW = showPipe ? 232 : 186;
    const xOrigin = labelW + 24;
    const tToX = t => Math.round(xOrigin + t * UNIT_W);
    const uToW = u => Math.max(52, Math.round(u * UNIT_W));

    const coreTop = HEADER_H;
    const mte1Desc = kernelInfo.type === 'qr' || kernelInfo.type === 'qmat' || isMain ? 'textract' : 'tmov L0';
    const fixDesc = kernelInfo.type === 'qr' ? 'Atomic Add' : 'L0C → GM';
    const pipeDefs = isAic
      ? [
          { id: 'mte2', label: 'MTE2', desc: 'GM → L1' },
          { id: 'mte1', label: 'MTE1', desc: mte1Desc },
          { id: 'cube', label: 'Cube', desc: 'Matmul Acc' },
          { id: 'fix', label: 'FIX', desc: fixDesc },
        ]
      : [
          { id: 'scalar', label: 'Scalar', desc: 'Control / Idx' },
          { id: 'mte2', label: 'MTE2', desc: 'GM → UB' },
          { id: 'vec', label: 'Vector', desc: 'UB Compute' },
          { id: 'mte3', label: 'MTE3', desc: 'UB → GM' },
        ];

    const subH = showPipe ? 52 : 0;
    const coreH = showPipe ? pipeDefs.length * (subH + 3) + 18 : 114;
    const totalHeight = coreTop + coreH + 46;

    const shortCoreSub = kernelInfo.type === 'qr'
      ? `N Group ${shape.group} · Split ${shape.split}`
      : kernelInfo.type === 'qmat'
        ? `N Group ${shape.group} · ${shape.path} M`
        : kernelInfo.type === 'post'
          ? `Head Group ${block} · 4 Heads`
          : kernelInfo.type === 'projection'
            ? `M ${model.range(shape.row, model.rows)} · N ${model.range(shape.col, shape.n)}`
            : `${kernelInfo.name} · Block ${block}`;

    const core = {
      id: isAic ? 'aic' : 'aiv0',
      title: isAic ? 'AIC' : 'AIV',
      subtitle: shortCoreSub,
      badge: isAic ? '1 Cube Core' : '1 Vector Core',
      top: coreTop,
      height: coreH,
      pipes: showPipe
        ? pipeDefs.map((p, idx) => ({ ...p, top: coreTop + 8 + idx * (subH + 3), height: subH }))
        : [],
    };

    const blocks = [];
    const groups = [];
    const syncEdges = [];
    const reuseArcs = [];

    const yFor = pipeId => {
      if (!showPipe) return coreTop + 32;
      const p = core.pipes.find(item => item.id === pipeId) || core.pipes[0];
      return p.top + 5;
    };
    const hFor = () => (showPipe ? 42 : 52);

    const addBlock = spec => {
      const item = {
        core: core.id,
        y: yFor(spec.pipe),
        h: hFor(),
        slot: 0,
        ...spec,
      };
      blocks.push(item);
      return item;
    };

    const addSync = (id, event, fromBlk, toBlk, slot = 0, label = '') => {
      const meta = SYNC_META[event] || SYNC_META.UB_READY;
      const fromDown = toBlk.y >= fromBlk.y;
      const fx = Math.min(fromBlk.x + fromBlk.w - 6, fromBlk.x + fromBlk.w * 0.82);
      const tx = Math.max(toBlk.x + 6, toBlk.x + Math.min(18, toBlk.w * 0.25));
      const fy = fromDown ? fromBlk.y + fromBlk.h : fromBlk.y;
      const ty = fromDown ? toBlk.y : toBlk.y + toBlk.h;
      const busY = Math.round((fy + ty) / 2);
      syncEdges.push({
        id,
        event,
        slot,
        label,
        color: meta.color,
        from: { x: Math.round(fx), y: Math.round(fy) },
        to: { x: Math.round(tx), y: Math.round(ty) },
        busY,
        sourceRange: fromBlk.sourceRange,
      });
    };

    let maxT = 6;

    switch (kernelInfo.type) {
      case 'projection': {
        const kCount = model.kCount; // 8 × K512
        const nSize = shape.n;
        const loadLines = srcLine([120, 127], [122, 130]);
        const extLines = srcLine([128, 133], [131, 136]);
        const kvLines = srcLine([128, 132], [131, 135]);
        const scLines = srcLine([129, 133], [132, 136]);
        const storeLines = srcLine([135, 136], [138, 139]);
        const extractLabel = isMain ? 'Extract 2×K256' : 'tmov L1→L0';

        if (!showPipe) {
          let t = 0.35;
          let prevKv = null;
          let prevSc = null;
          for (let k = 0; k < kCount; k++) {
            const slot = k % 2;
            const kRange = model.range(k * model.kTile, model.kTile);
            const gStart = t;
            const loadBlk = addBlock({
              uid: `load:k${k}`,
              pipe: 'mte2',
              slot,
              iter: k,
              kind: 'dma-block',
              x: tToX(t),
              w: uToW(1.15) - 4,
              title: `Load K${k}`,
              subtitle: `X/Wkv/Wgate · ${kRange}`,
              sourceRange: loadLines,
              syncOut: 'L1_READY',
              debugTip: `K 迭代 ${k}（${kRange}）：从 GM 搬运 X [16,512] 及转置权重 Wkv / Wgate [${nSize},512] 至 L1 Slot ${slot}（DN2ZN 长突发）。`,
            });
            t += 1.2;
            const extBlk = addBlock({
              uid: `ext:k${k}`,
              pipe: 'mte1',
              slot,
              iter: k,
              kind: 'dma-block',
              x: tToX(t),
              w: uToW(0.95) - 4,
              title: `${extractLabel} · K${k}`,
              subtitle: `Slot ${slot} → L0A/L0B`,
              sourceRange: extLines,
              syncOut: 'L0_READY',
              debugTip: isMain
                ? `Main 支路：pto.textract 将 K${k}（512）拆为 2×K256 送入 L0A / L0B，分别供给 KV 与 Score 投影。`
                : `Indexer 支路：pto.tmov 将 K${k} 从 L1 搬移至 L0A / L0B，供 KV 与 Score 矩阵乘使用。`,
            });
            t += 1.0;
            const kvBlk = addBlock({
              uid: `kv:k${k}`,
              pipe: 'cube',
              slot,
              iter: k,
              kind: 'op-pill',
              icon: 'matmul',
              x: tToX(t),
              w: uToW(1.1) - 4,
              title: k === 0 ? `KV Matmul K0` : `KV Acc K${k}`,
              subtitle: `[16,512]×[${nSize},512]ᵀ`,
              sourceRange: kvLines,
              debugTip: `Cube 计算 KV 投影 ${k === 0 ? '初始矩阵乘' : 'L0C 累加'}（FP32 [16,${nSize}]），b_trans=True。`,
            });
            t += 1.15;
            const scBlk = addBlock({
              uid: `score:k${k}`,
              pipe: 'cube',
              slot,
              iter: k,
              kind: 'op-pill',
              icon: 'matmul',
              x: tToX(t),
              w: uToW(1.1) - 4,
              title: k === 0 ? `Score Matmul K0` : `Score Acc K${k}`,
              subtitle: `[16,512]×[${nSize},512]ᵀ`,
              sourceRange: scLines,
              debugTip: `Cube 复用同一份 X Tile 计算 Score 门控投影（FP32 [16,${nSize}]），与 KV 投影共享 K 迭代。`,
            });
            t += 1.25;
            groups.push({
              slot,
              x: tToX(gStart) - 4,
              y: coreTop + 8,
              w: tToX(t - 0.1) - tToX(gStart) + 8,
              h: coreH - 16,
              label: `K${k} · ${kRange}`,
            });
            if (k < 2) {
              reuseArcs.push({
                id: `reuse:x:k${k}`,
                fromX: loadBlk.x + loadBlk.w / 2,
                toX: scBlk.x + scBlk.w / 2,
                y: coreTop - 6,
                label: k === 0 ? 'X Tile 复用 (KV & Score)' : `Slot ${slot} 复用`,
                color: slot === 0 ? '#80B7F7' : '#9EF04B',
              });
            }
            prevKv = kvBlk;
            prevSc = scBlk;
          }
          const storeKv = addBlock({
            uid: 'fix:kv',
            pipe: 'fix',
            slot: 0,
            kind: 'dma-block',
            x: tToX(t),
            w: uToW(1.15) - 4,
            title: 'Store KV Tile',
            subtitle: `FP32 [16,${nSize}] → GM`,
            sourceRange: [storeLines[0], storeLines[0]],
            debugTip: `FIX 单元将 L0C 中的 KV 投影结果写入 cmp4_kv_proj_pad[${shape.row}:${shape.row + 16}, ${shape.col}:${shape.col + nSize}]。`,
          });
          t += 1.22;
          const storeSc = addBlock({
            uid: 'fix:score',
            pipe: 'fix',
            slot: 1,
            kind: 'dma-block',
            x: tToX(t),
            w: uToW(1.15) - 4,
            title: 'Store Score Tile',
            subtitle: `FP32 [16,${nSize}] → GM`,
            sourceRange: [storeLines[1], storeLines[1]],
            debugTip: `FIX 单元将 L0C 中的 Score 投影结果写入 cmp4_score_proj_pad[${shape.row}:${shape.row + 16}, ${shape.col}:${shape.col + nSize}]。`,
          });
          if (prevKv) addSync('sync:fix:kv', 'L0C_FIX', prevKv, storeKv, 0);
          if (prevSc) addSync('sync:fix:sc', 'L0C_FIX', prevSc, storeSc, 1);
          maxT = t + 1.5;
        } else {
          // Pipe mode: pl.pipeline(0, 8, stage=2) double-buffering overlap across MTE2 / MTE1 / Cube / FIX
          let mte2T = 0.35;
          let mte1T = 0.35;
          let cubeT = 0.35;
          const l1End = new Map();
          let lastKv = null;
          let lastSc = null;
          const loadBlocks = [];

          for (let k = 0; k < kCount; k++) {
            const slot = k % 2;
            const kRange = model.range(k * model.kTile, model.kTile);
            // Stage=2 double-buffer: slot can be refilled once slot's previous MTE1 finishes
            const slotFreeT = k >= 2 ? l1End.get(k - 2) : 0.35;
            const lStart = Math.max(mte2T, slotFreeT);
            const loadBlk = addBlock({
              uid: `load:k${k}`,
              pipe: 'mte2',
              slot,
              iter: k,
              kind: 'dma-block',
              x: tToX(lStart),
              w: uToW(1.55) - 4,
              title: `tload K${k} · S${slot}`,
              subtitle: `X + Wkv/Wgate · ${kRange}`,
              sourceRange: loadLines,
              syncOut: 'L1_READY',
              debugTip: `pl.pipeline(stage=2) 预取 K${k}（${kRange}）至 L1 Slot ${slot}，与前一次迭代的 MTE1 / Cube 并行掩盖搬运延迟。`,
            });
            loadBlocks.push(loadBlk);
            mte2T = lStart + 1.62;

            const eStart = Math.max(mte1T, mte2T - 0.05);
            const extBlk = addBlock({
              uid: `ext:k${k}`,
              pipe: 'mte1',
              slot,
              iter: k,
              kind: 'dma-block',
              x: tToX(eStart),
              w: uToW(1.1) - 4,
              title: `${extractLabel} K${k}`,
              subtitle: `L1 S${slot} → L0A/L0B`,
              sourceRange: extLines,
              syncOut: 'L0_READY',
              debugTip: `从 L1 Slot ${slot} 抽取 K${k} 数据块送入 L0A / L0B，完成后释放 L1 Slot ${slot} 供 K${k + 2} 复用。`,
            });
            mte1T = eStart + 1.16;
            l1End.set(k, mte1T);

            const cStart = Math.max(cubeT, mte1T - 0.05);
            const kvBlk = addBlock({
              uid: `kv:k${k}`,
              pipe: 'cube',
              slot,
              iter: k,
              kind: 'op-pill',
              icon: 'matmul',
              x: tToX(cStart),
              w: uToW(1.0) - 4,
              title: k === 0 ? 'KV MM K0' : `KV Acc K${k}`,
              subtitle: `FP32 [16,${nSize}]`,
              sourceRange: kvLines,
              debugTip: `Cube 执行 KV ${k === 0 ? 'tmatmul' : 'tmatmul.acc'}（K${k}），结果留在 L0C。`,
            });
            const scBlk = addBlock({
              uid: `score:k${k}`,
              pipe: 'cube',
              slot,
              iter: k,
              kind: 'op-pill',
              icon: 'matmul',
              x: tToX(cStart + 1.04),
              w: uToW(1.0) - 4,
              title: k === 0 ? 'Score MM K0' : `Sc Acc K${k}`,
              subtitle: `FP32 [16,${nSize}]`,
              sourceRange: scLines,
              debugTip: `Cube 复用 L0A 中同一 X Tile 执行 Score ${k === 0 ? 'tmatmul' : 'tmatmul.acc'}（K${k}）。`,
            });
            cubeT = cStart + 2.1;
            lastKv = kvBlk;
            lastSc = scBlk;

            addSync(`sync:l1:k${k}`, 'L1_READY', loadBlk, extBlk, slot);
            addSync(`sync:l0:k${k}`, 'L0_READY', extBlk, kvBlk, slot);

            if (k === 2 || k === 5) {
              reuseArcs.push({
                id: `reuse:slot:${k}`,
                fromX: loadBlocks[k - 2].x + loadBlocks[k - 2].w / 2,
                toX: loadBlk.x + loadBlk.w / 2,
                y: coreTop - 6,
                label: `L1 Slot ${slot} 复用 (stage=2)`,
                color: slot === 0 ? '#80B7F7' : '#9EF04B',
              });
            }
          }

          const fixStart = cubeT + 0.1;
          const storeKv = addBlock({
            uid: 'fix:kv',
            pipe: 'fix',
            slot: 0,
            kind: 'dma-block',
            x: tToX(fixStart),
            w: uToW(1.25) - 4,
            title: 'tstore KV Tile',
            subtitle: `L0C → GM · [16,${nSize}]`,
            sourceRange: [storeLines[0], storeLines[0]],
            debugTip: `8 轮 K 累加结束后，FIX 将 L0C 中 KV 投影 Tile 写回 GM。`,
          });
          const storeSc = addBlock({
            uid: 'fix:score',
            pipe: 'fix',
            slot: 1,
            kind: 'dma-block',
            x: tToX(fixStart + 1.3),
            w: uToW(1.25) - 4,
            title: 'tstore Score Tile',
            subtitle: `L0C → GM · [16,${nSize}]`,
            sourceRange: [storeLines[1], storeLines[1]],
            debugTip: `FIX 将 L0C 中 Score 投影 Tile 写回 GM，供下游 scatter_softmax_pool 与 compress_state_commit 消费。`,
          });
          addSync('sync:fix:kv', 'L0C_FIX', lastKv, storeKv, 0);
          addSync('sync:fix:sc', 'L0C_FIX', lastSc, storeSc, 1);
          maxT = fixStart + 2.9;
        }
        break;
      }

      case 'pool': {
        const posLines = srcLine([146, 154], [149, 157]);
        const seedLines = srcLine([155, 170], [158, 173]);
        const winLines = srcLine([171, 208], [174, 214]);
        const carryLines = srcLine([209, 214], [215, 220]);
        const normLines = srcLine([215, 215], [221, 221]);
        const headTiles = kernelInfo.head / 64;

        let t = 0.35;
        const guardBlk = addBlock({
          uid: 'pool:guard',
          pipe: 'scalar',
          slot: 0,
          kind: 'dma-block',
          x: tToX(t),
          w: uToW(1.3) - 4,
          title: '边界检查 & 0 初始化',
          subtitle: `(pos+1)%4==0 · Req ${block}`,
          sourceRange: posLines,
          syncOut: 'SCALAR_GUARD',
          debugTip: `逐 Token（s_dim=8）读取 position_ids，先写 0 初始化 pooled_kv；仅在 Ratio-4 压缩边界进入 Online Softmax 池化（${headTiles} 个 Head Tiles）。`,
        });
        t += showPipe ? 1.15 : 1.36;

        const seedLoad = showPipe
          ? addBlock({
              uid: 'pool:seed-load',
              pipe: 'mte2',
              slot: 0,
              kind: 'dma-block',
              x: tToX(t),
              w: uToW(1.1) - 4,
              title: 'Load 本步 KV/Score',
              subtitle: `proj + APE · [1,64]`,
              sourceRange: seedLines,
              syncOut: 'UB_READY',
              debugTip: `加载当前压缩边界 Token 的 KV / Score 投影与 APE 位置偏置至 UB。`,
            })
          : null;
        if (seedLoad) {
          addSync('sync:guard:seed', 'SCALAR_GUARD', guardBlk, seedLoad, 0);
          t += 1.05;
        }

        const seedBlk = addBlock({
          uid: 'pool:seed',
          pipe: 'vec',
          slot: 0,
          kind: 'op-pill',
          icon: 'softmax',
          x: tToX(t),
          w: uToW(1.25) - 4,
          title: 'Seed m / l / o',
          subtitle: `mi=score+ape · li=1`,
          sourceRange: seedLines,
          syncOut: 'STATE_CARRY',
          debugTip: `用当前边界 Token 初始化 Online Softmax 状态 (m, l, o)，每个 Head Tile=64 独立归约。`,
        });
        if (seedLoad) addSync('sync:seed:vec', 'UB_READY', seedLoad, seedBlk, 0);
        else addSync('sync:guard:seed', 'SCALAR_GUARD', guardBlk, seedBlk, 0);
        t += 1.32;

        let prevCarry = seedBlk;
        for (let w = 0; w < 7; w++) {
          const slot = w % 2;
          const halfTag = w < 4 ? 'Front' : 'Back';
          const gStart = t;
          const gatherBlk = addBlock({
            uid: `pool:win:${w}`,
            pipe: 'mte2',
            slot,
            iter: w,
            kind: 'dma-block',
            x: tToX(t),
            w: uToW(1.05) - 4,
            title: `Gather W${w}`,
            subtitle: `${halfTag} · Ring/Overlay`,
            sourceRange: winLines,
            syncOut: 'UB_READY',
            debugTip: `历史窗口槽位 W${w}（state_idx=${w}，${halfTag} 半区）：按 logical_pos 判定从持久化 State Ring 或本步投影 Overlay 取值。`,
          });
          const vecStart = showPipe ? Math.max(t + 0.85, (prevCarry.x + prevCarry.w - xOrigin) / UNIT_W + 0.08) : t + 1.1;
          const carryBlk = addBlock({
            uid: `pool:carry:${w}`,
            pipe: 'vec',
            slot,
            iter: w,
            kind: 'op-pill',
            icon: 'reduce',
            x: tToX(vecStart),
            w: uToW(1.15) - 4,
            title: `Online Softmax W${w}`,
            subtitle: `tmax → texp → m/l/o`,
            sourceRange: carryLines,
            syncOut: 'STATE_CARRY',
            debugTip: `融合 W${w} 槽位至 Online Softmax 递归状态：更新 m_new = max(m, score)，缩放旧累加并合并新 KV。`,
          });
          addSync(`sync:win:ub:${w}`, 'UB_READY', gatherBlk, carryBlk, slot);
          addSync(`sync:carry:${w}`, 'STATE_CARRY', prevCarry, carryBlk, slot);
          t = vecStart + 1.22;
          if (!showPipe) {
            groups.push({
              slot,
              x: tToX(gStart) - 4,
              y: coreTop + 8,
              w: tToX(t - 0.08) - tToX(gStart) + 8,
              h: coreH - 16,
              label: `Window Slot ${w}`,
            });
          }
          prevCarry = carryBlk;
        }

        const divBlk = addBlock({
          uid: 'pool:norm',
          pipe: 'vec',
          slot: 0,
          kind: 'op-pill',
          icon: 'compress',
          x: tToX(t),
          w: uToW(1.1) - 4,
          title: 'Normalize o / l',
          subtitle: `pto.tdiv · FP32`,
          sourceRange: normLines,
          syncOut: 'VEC_STORE',
          debugTip: `全部 7 个历史槽位累加完成后，计算 o / l 得到该压缩边界的池化 KV 向量。`,
        });
        addSync('sync:carry:norm', 'STATE_CARRY', prevCarry, divBlk, 0);
        t += 1.18;

        const storeBlk = addBlock({
          uid: 'pool:store',
          pipe: 'mte3',
          slot: 0,
          kind: 'dma-block',
          x: tToX(t),
          w: uToW(1.25) - 4,
          title: 'Store Pooled KV',
          subtitle: `覆写边界 Token 行`,
          sourceRange: normLines,
          debugTip: `MTE3 将归一化后的 Pooled KV [1, ${kernelInfo.head}] 写回 GM，覆盖非边界的 0 初始化。`,
        });
        addSync('sync:norm:store', 'VEC_STORE', divBlk, storeBlk, 0);

        reuseArcs.push({
          id: 'reuse:mlo',
          fromX: seedBlk.x + seedBlk.w / 2,
          toX: divBlk.x + divBlk.w / 2,
          y: coreTop - 6,
          label: 'UB 驻留状态 m / l / o 贯穿 7 个历史窗口槽位',
          color: '#D58BF7',
        });
        maxT = t + 1.6;
        break;
      }

      case 'commit': {
        const guardLines = srcLine([223, 228], [229, 234]);
        const kvLines = srcLine([230, 231], [236, 237]);
        const apeLines = srcLine([232, 233], [238, 241]);
        const tokensPerReq = 8;
        let t = 0.35;

        for (let s = 0; s < tokensPerReq; s++) {
          const slot = s % 2;
          const tok = block * tokensPerReq + s;
          const gStart = t;
          const guardBlk = addBlock({
            uid: `commit:guard:${s}`,
            pipe: 'scalar',
            slot,
            iter: s,
            kind: 'dma-block',
            x: tToX(t),
            w: uToW(0.92) - 4,
            title: `Slot Guard s${s}`,
            subtitle: `T${tok} · row ≥ 0`,
            sourceRange: guardLines,
            syncOut: 'SCALAR_GUARD',
            debugTip: `Request ${block} / Token ${tok}：读取 state_slot_mapping 与 position_ids，仅 state_row ≥ 0 时执行提交。`,
          });
          const mteStart = showPipe ? t + 0.72 : t + 0.96;
          const loadBlk = addBlock({
            uid: `commit:load:${s}`,
            pipe: 'mte2',
            slot,
            iter: s,
            kind: 'dma-block',
            x: tToX(mteStart),
            w: uToW(0.98) - 4,
            title: `Load KV/Sc/APE`,
            subtitle: `s${s} · [1,${kernelInfo.head * 2}]`,
            sourceRange: kvLines,
            syncOut: 'UB_READY',
            debugTip: `加载 Token ${tok} 的 KV / Score 投影及对应 position % 4 的 APE 偏置行。`,
          });
          const vecStart = showPipe ? mteStart + 0.85 : mteStart + 1.02;
          const apeBlk = addBlock({
            uid: `commit:ape:${s}`,
            pipe: 'vec',
            slot,
            iter: s,
            kind: 'op-pill',
            icon: 'add',
            x: tToX(vecStart),
            w: uToW(0.95) - 4,
            title: `Score + APE`,
            subtitle: `pto.tadd · s${s}`,
            sourceRange: apeLines,
            syncOut: 'VEC_STORE',
            debugTip: `Vector 单元将位置偏置 APE 加到 Score 投影上，准备写入 State Ring 的 Score 分区。`,
          });
          const storeStart = showPipe ? vecStart + 0.85 : vecStart + 1.0;
          const storeBlk = addBlock({
            uid: `commit:store:${s}`,
            pipe: 'mte3',
            slot,
            iter: s,
            kind: 'dma-block',
            x: tToX(storeStart),
            w: uToW(1.02) - 4,
            title: `Commit Ring s${s}`,
            subtitle: `KV + Score 分区`,
            sourceRange: apeLines,
            debugTip: `写回 compress_state_flat[state_row]：前半区写入 KV 投影，后半区写入 Score+APE。`,
          });
          addSync(`sync:c:g:${s}`, 'SCALAR_GUARD', guardBlk, loadBlk, slot);
          addSync(`sync:c:u:${s}`, 'UB_READY', loadBlk, apeBlk, slot);
          addSync(`sync:c:s:${s}`, 'VEC_STORE', apeBlk, storeBlk, slot);

          if (!showPipe) {
            groups.push({
              slot,
              x: tToX(gStart) - 4,
              y: coreTop + 8,
              w: tToX(storeStart + 1.04) - tToX(gStart) + 8,
              h: coreH - 16,
              label: `Token s${s} (T${tok})`,
            });
            t = storeStart + 1.16;
          } else {
            t = mteStart + 1.15;
          }
          maxT = Math.max(maxT, storeStart + 1.4);
        }
        break;
      }

      case 'norm': {
        const headTiles = kernelInfo.head / 64; // 8 for Main, 2 for Indexer
        const nopeTiles = headTiles - 1;
        const sqLines = srcLine([245, 250], [254, 259]);
        const invLines = srcLine([252, 253], [261, 262]);
        const nopeLines = srcLine([254, 258], [263, 271]);
        const ropeLines = srcLine([260, 277], [273, 294]);
        const cacheLines = [279, 287];

        let t = 0.35;
        let prevSq = null;
        for (let h = 0; h < headTiles; h++) {
          const slot = h % 2;
          const hRange = model.range(h * 64, 64);
          const loadBlk = showPipe
            ? addBlock({
                uid: `norm:loadsq:${h}`,
                pipe: 'mte2',
                slot,
                iter: h,
                kind: 'dma-block',
                x: tToX(t),
                w: uToW(0.85) - 4,
                title: `Load H${h}`,
                subtitle: `Pooled ${hRange}`,
                sourceRange: sqLines,
                syncOut: 'UB_READY',
                debugTip: `加载 Pooled KV [16, 64]（Head ${hRange}）至 UB 计算局部平方和。`,
              })
            : null;
          const sqStart = showPipe ? Math.max(t + 0.72, prevSq ? (prevSq.x + prevSq.w - xOrigin) / UNIT_W + 0.06 : 0) : t;
          const sqBlk = addBlock({
            uid: `norm:sq:${h}`,
            pipe: 'vec',
            slot,
            iter: h,
            kind: 'op-pill',
            icon: 'reduce',
            x: tToX(sqStart),
            w: uToW(0.92) - 4,
            title: `SqSum H${h}`,
            subtitle: `${hRange}`,
            sourceRange: sqLines,
            syncOut: 'STATE_CARRY',
            debugTip: `Vector 计算 x² 并行归约（trowsum + tadd），累加至 partial_sq [1,16]。`,
          });
          if (loadBlk) addSync(`sync:n:ld:${h}`, 'UB_READY', loadBlk, sqBlk, slot);
          if (prevSq) addSync(`sync:n:sq:${h}`, 'STATE_CARRY', prevSq, sqBlk, slot);
          prevSq = sqBlk;
          t = showPipe ? t + 0.94 : sqStart + 0.98;
        }
        t = Math.max(t, (prevSq.x + prevSq.w - xOrigin) / UNIT_W + 0.14);

        const invBlk = addBlock({
          uid: 'norm:inv',
          pipe: 'vec',
          slot: 0,
          kind: 'op-pill',
          icon: 'norm',
          x: tToX(t),
          w: uToW(1.15) - 4,
          title: 'Inverse RMS',
          subtitle: `rsqrt(mean+ε)`,
          sourceRange: invLines,
          debugTip: `对累加平方和乘以 1/${kernelInfo.head} + ε，经 tsqrt + trecip 得到每行 inv_rms [16,1]。`,
        });
        addSync('sync:sq:inv', 'STATE_CARRY', prevSq, invBlk, 0);
        t += 1.24;

        let lastNopeStore = null;
        for (let h = 0; h < nopeTiles; h++) {
          const slot = h % 2;
          const hRange = model.range(h * 64, 64);
          const nopeBlk = addBlock({
            uid: `norm:nope:${h}`,
            pipe: 'vec',
            slot,
            iter: h,
            kind: 'op-pill',
            icon: 'norm',
            x: tToX(t),
            w: uToW(0.95) - 4,
            title: `NoPE H${h}`,
            subtitle: `${hRange}`,
            sourceRange: nopeLines,
            syncOut: 'VEC_STORE',
            debugTip: `NoPE 列归一化：pooled × inv_rms × gamma（${hRange}）${isMain ? '，保持 FP32' : '，转为 BF16'}。`,
          });
          const storeStart = showPipe ? t + 0.78 : t + 1.0;
          const storeBlk = showPipe
            ? addBlock({
                uid: `norm:nstore:${h}`,
                pipe: 'mte3',
                slot,
                iter: h,
                kind: 'dma-block',
                x: tToX(storeStart),
                w: uToW(0.82) - 4,
                title: `Store H${h}`,
                subtitle: isMain ? 'FP32' : 'BF16',
                sourceRange: nopeLines,
                debugTip: `写回归一化后的 NoPE Head Tile ${hRange} 至 normed_kv。`,
              })
            : null;
          if (storeBlk) addSync(`sync:nope:st:${h}`, 'VEC_STORE', nopeBlk, storeBlk, slot);
          lastNopeStore = storeBlk || nopeBlk;
          t = showPipe ? t + 1.02 : t + 1.02;
        }

        const ropeBlk = addBlock({
          uid: 'norm:rope',
          pipe: 'vec',
          slot: 1,
          kind: 'op-pill',
          icon: 'rope',
          x: tToX(t),
          w: uToW(1.45) - 4,
          title: 'RoPE Swap + Rotate',
          subtitle: `末 64 列 · j^1 gather`,
          sourceRange: ropeLines,
          syncOut: 'VEC_STORE',
          debugTip: `先融合 inv_rms 与 gamma，再构造 j^1 索引做 interleaved swap-gather，与预折叠 cos/sin 相乘相加完成旋转。`,
        });
        t += 1.52;

        const ropeStore = addBlock({
          uid: 'norm:rstore',
          pipe: 'mte3',
          slot: 1,
          kind: 'dma-block',
          x: tToX(t),
          w: uToW(1.15) - 4,
          title: isMain ? 'Store Normed KV' : 'Store BF16 KV',
          subtitle: `16 × ${kernelInfo.head}`,
          sourceRange: ropeLines,
          debugTip: isMain
            ? `写回 RoPE 列至 normed_kv（FP32），准备逐有效行写入 Main KV Cache。`
            : `转为 BF16 并写回 normed_kv [16, 128]，供下游 kv_hadamard 矩阵乘读取。`,
        });
        addSync('sync:rope:st', 'VEC_STORE', ropeBlk, ropeStore, 1);
        t += 1.25;

        reuseArcs.push({
          id: 'reuse:invrms',
          fromX: invBlk.x + invBlk.w / 2,
          toX: ropeBlk.x + ropeBlk.w / 2,
          y: coreTop - 6,
          label: `inv_rms [16,1] 广播复用至 ${nopeTiles}×NoPE + RoPE`,
          color: '#80B7F7',
        });

        if (isMain) {
          const slotGuard = addBlock({
            uid: 'norm:cache-guard',
            pipe: 'scalar',
            slot: 0,
            kind: 'dma-block',
            x: tToX(t),
            w: uToW(1.15) - 4,
            title: 'Cache Slot Guard',
            subtitle: `16 行 · slot ≥ 0`,
            sourceRange: cacheLines,
            syncOut: 'SCALAR_GUARD',
            debugTip: `逐 Token（最多 16 行）读取 cmp_slot_mapping，有效槽位才执行 BF16 转换与 Cache 写入。`,
          });
          t += 1.2;
          const cacheWrite = addBlock({
            uid: 'norm:cache-write',
            pipe: 'mte3',
            slot: 0,
            kind: 'dma-block',
            x: tToX(t),
            w: uToW(1.3) - 4,
            title: 'Write Main Cache',
            subtitle: `tcvt BF16 → Cache`,
            sourceRange: cacheLines,
            debugTip: `对有效槽位写出 kv_flat (FP32) 与 cmp_kv_cache_flat (BF16)。`,
          });
          addSync('sync:cache:guard', 'SCALAR_GUARD', slotGuard, cacheWrite, 0);
          t += 1.4;
        }
        maxT = Math.max(t + 0.5, lastNopeStore ? (lastNopeStore.x + lastNopeStore.w - xOrigin) / UNIT_W + 2.5 : 8);
        break;
      }

      case 'hadamard': {
        let t = 0.35;
        const loadKv = addBlock({
          uid: 'had:load-kv',
          pipe: 'mte2',
          slot: 0,
          kind: 'dma-block',
          x: tToX(t),
          w: uToW(1.3) - 4,
          title: 'Load Normed KV',
          subtitle: `BF16 [16,128] → L1`,
          sourceRange: [298, 300],
          syncOut: 'L1_READY',
          debugTip: `加载一次 normed_kv [${shape.row}:${shape.row + 16}, 0:128] BF16 至 L1，被两个 N=64 Hadamard 投影分块共享复用。`,
        });
        t += 1.38;

        let lastMat = null;
        for (let n = 0; n < 2; n++) {
          const slot = n;
          const nRange = model.range(n * 64, 64);
          const gStart = t;
          const loadW = addBlock({
            uid: `had:load-w:${n}`,
            pipe: 'mte2',
            slot,
            iter: n,
            kind: 'dma-block',
            x: tToX(t),
            w: uToW(1.1) - 4,
            title: `Load W_had N${n}`,
            subtitle: `BF16 [128,64] · ${nRange}`,
            sourceRange: [301, 302],
            syncOut: 'L1_READY',
            debugTip: `加载 Hadamard 正交变换矩阵块 [128, 64]（N ${nRange}）至 L1。`,
          });
          const mte1Start = showPipe ? t + 0.95 : t + 1.15;
          const movBlk = addBlock({
            uid: `had:mov:${n}`,
            pipe: 'mte1',
            slot,
            iter: n,
            kind: 'dma-block',
            x: tToX(mte1Start),
            w: uToW(0.95) - 4,
            title: `tmov L1→L0 N${n}`,
            subtitle: `KV & W_had`,
            sourceRange: [303, 303],
            syncOut: 'L0_READY',
            debugTip: `pto.tmov 将 KV Tile 与 Hadamard 权重搬运至 L0A / L0B。`,
          });
          const cubeStart = showPipe ? mte1Start + 0.95 : mte1Start + 1.0;
          const mmBlk = addBlock({
            uid: `had:mm:${n}`,
            pipe: 'cube',
            slot,
            iter: n,
            kind: 'op-pill',
            icon: 'matmul',
            x: tToX(cubeStart),
            w: uToW(1.15) - 4,
            title: `Hadamard MM N${n}`,
            subtitle: `FP32 [16,64] · ${nRange}`,
            sourceRange: [301, 303],
            syncOut: 'L0C_FIX',
            debugTip: `Cube 执行 Hadamard 矩阵乘：[16,128] × [128,64] → FP32 [16,64]。`,
          });
          const fixStart = showPipe ? cubeStart + 1.05 : cubeStart + 1.2;
          const storeBlk = addBlock({
            uid: `had:fix:${n}`,
            pipe: 'fix',
            slot,
            iter: n,
            kind: 'dma-block',
            x: tToX(fixStart),
            w: uToW(1.1) - 4,
            title: `Store KV N${n}`,
            subtitle: `kv_final ${nRange}`,
            sourceRange: [304, 304],
            debugTip: `FIX 将变换后的 FP32 Tile 写入 kv_final[${shape.row}:${shape.row + 16}, ${nRange}]。`,
          });
          addSync(`sync:had:l1:${n}`, 'L1_READY', loadW, movBlk, slot);
          addSync(`sync:had:l0:${n}`, 'L0_READY', movBlk, mmBlk, slot);
          addSync(`sync:had:fix:${n}`, 'L0C_FIX', mmBlk, storeBlk, slot);

          if (!showPipe) {
            groups.push({
              slot,
              x: tToX(gStart) - 4,
              y: coreTop + 8,
              w: tToX(fixStart + 1.15) - tToX(gStart) + 8,
              h: coreH - 16,
              label: `N Tile ${n} · ${nRange}`,
            });
            t = fixStart + 1.28;
          } else {
            t = mte1Start + 1.08;
          }
          lastMat = mmBlk;
          maxT = Math.max(maxT, fixStart + 1.5);
        }
        if (lastMat) {
          reuseArcs.push({
            id: 'reuse:had-kv',
            fromX: loadKv.x + loadKv.w / 2,
            toX: lastMat.x + lastMat.w / 2,
            y: coreTop - 6,
            label: 'L1 Normed KV [16,128] 跨 2×N Tiles 复用',
            color: '#F2C45F',
          });
        }
        break;
      }

      case 'write': {
        let t = 0.35;
        const loadKv = addBlock({
          uid: 'wr:load',
          pipe: 'mte2',
          slot: 0,
          kind: 'dma-block',
          x: tToX(t),
          w: uToW(1.25) - 4,
          title: 'Load Indexer KV',
          subtitle: `FP32 [16,128] → UB`,
          sourceRange: [310, 314],
          syncOut: 'UB_READY',
          debugTip: `加载 kv_final[${shape.row}:${shape.row + 16}, 0:128] FP32 至 UB。`,
        });
        t += showPipe ? 1.1 : 1.32;

        const roundBlk = addBlock({
          uid: 'wr:round',
          pipe: 'vec',
          slot: 0,
          kind: 'op-pill',
          icon: 'reshape',
          x: tToX(t),
          w: uToW(1.2) - 4,
          title: 'BF16 Round-trip',
          subtitle: `FP32→BF16→FP32`,
          sourceRange: [312, 314],
          debugTip: `先做 BF16 舍入再转回 FP32（pto.tcvt ×2），确保量化输入与 Golden 口径完全一致。`,
        });
        addSync('sync:wr:ub', 'UB_READY', loadKv, roundBlk, 0);
        t += 1.26;

        const scaleBlk = addBlock({
          uid: 'wr:scale',
          pipe: 'vec',
          slot: 0,
          kind: 'op-pill',
          icon: 'reduce',
          x: tToX(t),
          w: uToW(1.4) - 4,
          title: 'Per-row Amax & Scale',
          subtitle: `abs → row_max → scale`,
          sourceRange: [315, 320],
          debugTip: `计算每行 max(|x|)，clamp 到 INT8_AMAX_EPS 后求量化缩放系数 kv_scale_q 与反量化系数 kv_scale_dq。`,
        });
        t += 1.46;

        const quantBlk = addBlock({
          uid: 'wr:quant',
          pipe: 'vec',
          slot: 1,
          kind: 'op-pill',
          icon: 'compress',
          x: tToX(t),
          w: uToW(1.35) - 4,
          title: 'INT8 量化转换',
          subtitle: `mul → I32 → FP16 → I8`,
          sourceRange: [321, 324],
          syncOut: 'VEC_STORE',
          debugTip: `按行缩放后经 INT32 → FP16 → INT8 三步转换生成 16×128 INT8 量化块。`,
        });
        t += 1.42;

        const guardBlk = addBlock({
          uid: 'wr:guard',
          pipe: 'scalar',
          slot: 1,
          kind: 'dma-block',
          x: tToX(t),
          w: uToW(1.2) - 4,
          title: 'Slot Guard (×16)',
          subtitle: `idx_slot ≥ 0`,
          sourceRange: [325, 329],
          syncOut: 'SCALAR_GUARD',
          debugTip: `逐 Token 读取 idx_slot_mapping，仅有效缓存槽位执行写入。`,
        });
        t += showPipe ? 1.05 : 1.26;

        const cacheStore = addBlock({
          uid: 'wr:cache',
          pipe: 'mte3',
          slot: 1,
          kind: 'dma-block',
          x: tToX(t),
          w: uToW(1.35) - 4,
          title: 'Store INT8 Cache',
          subtitle: `kv_flat + idx_kv_cache`,
          sourceRange: [330, 331],
          debugTip: `MTE3 将有效行的 FP32 KV 与 INT8 量化 KV 写入 Indexer KV Cache。`,
        });
        addSync('sync:wr:qst', 'VEC_STORE', quantBlk, cacheStore, 1);
        addSync('sync:wr:gst', 'SCALAR_GUARD', guardBlk, cacheStore, 1);
        t += 1.42;

        const scaleStore = addBlock({
          uid: 'wr:scale-out',
          pipe: 'scalar',
          slot: 1,
          kind: 'dma-block',
          x: tToX(t),
          w: uToW(1.25) - 4,
          title: 'Scalar Store Scale',
          subtitle: `store_scalar · FP32`,
          sourceRange: [332, 333],
          debugTip: `单值 [1,1] 小于 32B 对齐粒度，通过 pto.tgetval + pto.store_scalar 写回每行反量化 Scale。`,
        });

        reuseArcs.push({
          id: 'reuse:wr-kv',
          fromX: roundBlk.x + roundBlk.w / 2,
          toX: quantBlk.x + quantBlk.w / 2,
          y: coreTop - 6,
          label: 'UB kv_blk_f32 [16,128] 供 Amax 与 Quant 复用',
          color: '#80B7F7',
        });
        maxT = t + 1.6;
        break;
      }

      // ---------- Q Projection kernels (qkv_proj_rope.py) ----------
      case 'qr':
      case 'qmat': {
        const isQr = kernelInfo.type === 'qr';
        const isTail = !isQr && shape.path === 'Tail';
        const kSteps = Math.round(shape.kLength / shape.kTile); // 8 K tiles
        const loadLines = isQr ? [284, 294] : isTail ? [395, 408] : [380, 387];
        const mulLines = isQr ? [295, 298] : isTail ? [409, 412] : [388, 391];
        const storeLines = isQr ? [299, 299] : isTail ? [413, 416] : [392, 392];
        const mSize = shape.size;
        const nSize = shape.n;
        const dtypeIn = isQr ? 'BF16' : 'INT8';
        const dtypeAcc = isQr ? 'FP32' : 'INT32';
        const aName = isQr ? 'X' : 'QR';
        const wName = isQr ? 'Wq_a' : 'Wq_b';

        if (!showPipe) {
          let t = 0.35;
          let lastCube = null;
          for (let k = 0; k < kSteps; k++) {
            const slot = k % 2;
            const kRange = model.range(shape.k + k * shape.kTile, shape.kTile);
            const gStart = t;
            addBlock({
              uid: `qmm:load:k${k}`,
              pipe: 'mte2',
              slot,
              iter: k,
              kind: 'dma-block',
              x: tToX(t),
              w: uToW(1.15) - 4,
              title: `Load K${k} · S${slot}`,
              subtitle: `${aName}/${wName} ${dtypeIn} · ${kRange}`,
              sourceRange: loadLines,
              syncOut: 'L1_READY',
              debugTip: `K 迭代 ${k}（${kRange}）：从 GM 加载 ${aName} [${mSize},${shape.kTile}] 与 ${wName} [${shape.kTile},${nSize}] (${dtypeIn}) 至 L1 Slot ${slot}。`,
            });
            t += 1.2;
            addBlock({
              uid: `qmm:ext:k${k}`,
              pipe: 'mte1',
              slot,
              iter: k,
              kind: 'dma-block',
              x: tToX(t),
              w: uToW(0.95) - 4,
              title: `textract K${k}`,
              subtitle: `L1 S${slot} → L0A/L0B`,
              sourceRange: mulLines,
              syncOut: 'L0_READY',
              debugTip: `pto.textract 将 L1 Slot ${slot} 的激活与权重分块搬入 L0A / L0B。`,
            });
            t += 1.0;
            lastCube = addBlock({
              uid: `qmm:cube:k${k}`,
              pipe: 'cube',
              slot,
              iter: k,
              kind: 'op-pill',
              icon: 'matmul',
              x: tToX(t),
              w: uToW(1.25) - 4,
              title: k === 0 ? `Matmul K0` : `Matmul Acc K${k}`,
              subtitle: `${dtypeAcc} [${mSize},${nSize}]`,
              sourceRange: mulLines,
              debugTip: `Cube 计算 ${aName} × ${wName} ${k === 0 ? '初始矩阵乘' : 'L0C 累加'}（${dtypeAcc} [${mSize},${nSize}]）${isQr ? ` · Split ${shape.split}` : ` · ${shape.path} M Tile`}。`,
            });
            t += 1.35;
            groups.push({
              slot,
              x: tToX(gStart) - 4,
              y: coreTop + 8,
              w: tToX(t - 0.1) - tToX(gStart) + 8,
              h: coreH - 16,
              label: `K${k} · ${kRange}`,
            });
          }
          const storeBlk = addBlock({
            uid: 'qmm:fix',
            pipe: 'fix',
            slot: isQr ? shape.split : 0,
            kind: 'dma-block',
            x: tToX(t),
            w: uToW(1.45) - 4,
            title: isQr ? `Atomic Add Split ${shape.split}` : `Store Q INT32`,
            subtitle: isQr ? `→ Shared QR FP32 [${mSize},${nSize}]` : `${shape.path} [${mSize},${nSize}]`,
            sourceRange: storeLines,
            debugTip: isQr
              ? `Split ${shape.split}（K ${model.range(shape.k, shape.kLength)}）累加完成后通过 pl.assemble(..., atomic=Add) 原子加汇入零初始化后的共享 qr_fp32。`
              : `8 轮 K128 累加完成后，FIX 将 ${shape.path} Q INT32 Tile [${mSize},${nSize}] 写回 GM，供 qproj_dequant_rms_nope_rope 消费。`,
          });
          if (lastCube) addSync('sync:qmm:fix', 'L0C_FIX', lastCube, storeBlk, 0);
          maxT = t + 1.8;
        } else {
          let mte2T = 0.35;
          let mte1T = 0.35;
          let cubeT = 0.35;
          const l1End = new Map();
          const loadBlocks = [];
          let lastCube = null;

          for (let k = 0; k < kSteps; k++) {
            const slot = k % 2;
            const kRange = model.range(shape.k + k * shape.kTile, shape.kTile);
            const slotFreeT = k >= 2 ? l1End.get(k - 2) : 0.35;
            const lStart = Math.max(mte2T, slotFreeT);
            const loadBlk = addBlock({
              uid: `qmm:load:k${k}`,
              pipe: 'mte2',
              slot,
              iter: k,
              kind: 'dma-block',
              x: tToX(lStart),
              w: uToW(1.45) - 4,
              title: `tload K${k} · S${slot}`,
              subtitle: `K ${kRange}`,
              sourceRange: loadLines,
              syncOut: 'L1_READY',
              debugTip: `pl.pipeline(stage=2) 预取 K${k}（${kRange}）至 L1 Slot ${slot}，掩盖 MTE2 搬运延迟。`,
            });
            loadBlocks.push(loadBlk);
            mte2T = lStart + 1.52;

            const eStart = Math.max(mte1T, mte2T - 0.05);
            const extBlk = addBlock({
              uid: `qmm:ext:k${k}`,
              pipe: 'mte1',
              slot,
              iter: k,
              kind: 'dma-block',
              x: tToX(eStart),
              w: uToW(1.08) - 4,
              title: `Extract K${k}`,
              subtitle: `L1 S${slot} → L0`,
              sourceRange: mulLines,
              syncOut: 'L0_READY',
              debugTip: `从 L1 Slot ${slot} 抽取 K${k} 数据送入 L0A / L0B，完成后释放 Slot ${slot} 给 K${k + 2} 复用。`,
            });
            mte1T = eStart + 1.14;
            l1End.set(k, mte1T);

            const cStart = Math.max(cubeT, mte1T - 0.05);
            lastCube = addBlock({
              uid: `qmm:cube:k${k}`,
              pipe: 'cube',
              slot,
              iter: k,
              kind: 'op-pill',
              icon: 'matmul',
              x: tToX(cStart),
              w: uToW(1.3) - 4,
              title: k === 0 ? 'Matmul K0' : `MM Acc K${k}`,
              subtitle: `${dtypeAcc} [${mSize},${nSize}]`,
              sourceRange: mulLines,
              debugTip: `Cube 执行 ${k === 0 ? 'pto.tmatmul' : 'pto.tmatmul.acc'}（K${k}），累加结果留在 L0C。`,
            });
            cubeT = cStart + 1.38;

            addSync(`sync:qmm:l1:${k}`, 'L1_READY', loadBlk, extBlk, slot);
            addSync(`sync:qmm:l0:${k}`, 'L0_READY', extBlk, lastCube, slot);

            if (k === 2 || k === 5) {
              reuseArcs.push({
                id: `reuse:qmm:slot:${k}`,
                fromX: loadBlocks[k - 2].x + loadBlocks[k - 2].w / 2,
                toX: loadBlk.x + loadBlk.w / 2,
                y: coreTop - 6,
                label: `L1 Slot ${slot} 复用 (stage=2)`,
                color: slot === 0 ? '#80B7F7' : '#9EF04B',
              });
            }
          }

          const fixStart = cubeT + 0.1;
          const storeBlk = addBlock({
            uid: 'qmm:fix',
            pipe: 'fix',
            slot: isQr ? shape.split : 0,
            kind: 'dma-block',
            x: tToX(fixStart),
            w: uToW(1.55) - 4,
            title: isQr ? `Atomic Add Split ${shape.split}` : 'tstore Q INT32',
            subtitle: isQr ? `L0C ⊕→ qr_fp32` : `L0C → q_proj_i32`,
            sourceRange: storeLines,
            debugTip: isQr
              ? `Split ${shape.split} 累加结果通过 Atomic Add 写回共享 qr_fp32，与另一 Split 汇合。`
              : `FIX 将完整 L0C 累加块 [${mSize},${nSize}] INT32 写回 GM。`,
          });
          if (lastCube) addSync('sync:qmm:fix', 'L0C_FIX', lastCube, storeBlk, 0);
          maxT = fixStart + 1.9;
        }
        break;
      }

      case 'seed': {
        const nTiles = 8; // 1024 / 128
        let t = 0.35;
        for (let n = 0; n < nTiles; n++) {
          const slot = n % 2;
          const nRange = model.range(n * 128, 128);
          const gStart = t;
          const zeroBlk = addBlock({
            uid: `seed:fill:${n}`,
            pipe: 'vec',
            slot,
            iter: n,
            kind: 'op-pill',
            icon: 'reshape',
            x: tToX(t),
            w: uToW(0.95) - 4,
            title: `Fill 0 · N${n}`,
            subtitle: `texpands · ${nRange}`,
            sourceRange: [271, 274],
            syncOut: 'VEC_STORE',
            debugTip: `AIV 核内生成 FP32 [16,128] 全零 Tile（N ${nRange}），用于 Split-K Atomic Add 前清零。`,
          });
          const storeStart = showPipe ? t + 0.78 : t + 1.0;
          const storeBlk = addBlock({
            uid: `seed:st:${n}`,
            pipe: 'mte3',
            slot,
            iter: n,
            kind: 'dma-block',
            x: tToX(storeStart),
            w: uToW(0.95) - 4,
            title: `Store Zero N${n}`,
            subtitle: `qr_fp32 ${nRange}`,
            sourceRange: [274, 275],
            debugTip: `MTE3 将全零块写入 qr_fp32[${shape.start}:${shape.start + 16}, ${nRange}]。`,
          });
          addSync(`sync:seed:${n}`, 'VEC_STORE', zeroBlk, storeBlk, slot);
          if (!showPipe) {
            groups.push({
              slot,
              x: tToX(gStart) - 4,
              y: coreTop + 8,
              w: tToX(storeStart + 1.0) - tToX(gStart) + 8,
              h: coreH - 16,
              label: `N${n} · ${nRange}`,
            });
            t = storeStart + 1.1;
          } else {
            t += 1.02;
          }
          maxT = Math.max(maxT, storeStart + 1.3);
        }
        break;
      }

      case 'quant': {
        const nCount = 4; // 1024 / 256
        let t = 0.35;
        let prevRed = null;
        for (let i = 0; i < nCount; i++) {
          const slot = i % 2;
          const nRange = model.range(i * 256, 256);
          const loadBlk = showPipe
            ? addBlock({
                uid: `qnt:p1:ld:${i}`,
                pipe: 'mte2',
                slot,
                iter: i,
                kind: 'dma-block',
                x: tToX(t),
                w: uToW(1.0) - 4,
                title: `P1 Load N${i}`,
                subtitle: `qr+γ · ${nRange}`,
                sourceRange: [315, 321],
                syncOut: 'UB_READY',
                debugTip: `Pass 1 pl.pipeline(stage=2)：加载 qr_fp32 [8,256] 与 gamma_cq [256] 至 UB。`,
              })
            : null;
          const redStart = showPipe ? Math.max(t + 0.8, prevRed ? (prevRed.x + prevRed.w - xOrigin) / UNIT_W + 0.08 : 0) : t;
          const redBlk = addBlock({
            uid: `qnt:p1:red:${i}`,
            pipe: 'vec',
            slot,
            iter: i,
            kind: 'op-pill',
            icon: 'reduce',
            x: tToX(redStart),
            w: uToW(1.2) - 4,
            title: `Σx² & amax N${i}`,
            subtitle: `Pass 1 · ${nRange}`,
            sourceRange: [315, 325],
            syncOut: 'STATE_CARRY',
            debugTip: `同步累加每行平方和 qr_sq_sum 与缩放后绝对值最大值 qr_amax_g（N ${nRange}）。`,
          });
          if (loadBlk) addSync(`sync:qnt:p1:ub:${i}`, 'UB_READY', loadBlk, redBlk, slot);
          if (prevRed) addSync(`sync:qnt:p1:c:${i}`, 'STATE_CARRY', prevRed, redBlk, slot);
          prevRed = redBlk;
          t = showPipe ? t + 1.15 : redStart + 1.28;
        }
        t = Math.max(t, (prevRed.x + prevRed.w - xOrigin) / UNIT_W + 0.14);

        const invBlk = addBlock({
          uid: 'qnt:inv',
          pipe: 'vec',
          slot: 0,
          kind: 'op-pill',
          icon: 'norm',
          x: tToX(t),
          w: uToW(1.4) - 4,
          title: 'Inv RMS & Quant Scale',
          subtitle: `rsqrt + INT8_MAX/amax`,
          sourceRange: [326, 333],
          syncOut: 'VEC_STORE',
          debugTip: `求高精度 rsqrt(mean(x²)+ε) 与每行量化/反量化系数（qr_scale_q / qr_scale_dq）。`,
        });
        addSync('sync:qnt:inv', 'STATE_CARRY', prevRed, invBlk, 0);
        t += 1.48;

        const scaleStore = addBlock({
          uid: 'qnt:scale-st',
          pipe: 'mte3',
          slot: 0,
          kind: 'dma-block',
          x: tToX(t),
          w: uToW(1.3) - 4,
          title: 'Store QR Scale',
          subtitle: `qr_scale + pad_store`,
          sourceRange: [334, 346],
          debugTip: `写回公开 qr_scale 与供下游 qproj_dequant 消费的 qr_scale_pad_store（manual_dep=True）。`,
        });
        addSync('sync:qnt:scst', 'VEC_STORE', invBlk, scaleStore, 0);
        t += showPipe ? 1.05 : 1.38;

        let lastP2 = null;
        for (let i = 0; i < nCount; i++) {
          const slot = i % 2;
          const nRange = model.range(i * 256, 256);
          const loadBlk = showPipe
            ? addBlock({
                uid: `qnt:p2:ld:${i}`,
                pipe: 'mte2',
                slot,
                iter: i,
                kind: 'dma-block',
                x: tToX(t),
                w: uToW(0.95) - 4,
                title: `P2 Load N${i}`,
                subtitle: `${nRange}`,
                sourceRange: [348, 353],
                syncOut: 'UB_READY',
                debugTip: `Pass 2 重新加载 qr_fp32 与 gamma_cq（N ${nRange}），利用已算好的 inv_rms / quant_scale 执行量化。`,
              })
            : null;
          const qStart = showPipe ? t + 0.8 : t;
          const qBlk = addBlock({
            uid: `qnt:p2:q:${i}`,
            pipe: 'vec',
            slot,
            iter: i,
            kind: 'op-pill',
            icon: 'compress',
            x: tToX(qStart),
            w: uToW(1.15) - 4,
            title: `Norm+Quant N${i}`,
            subtitle: `FP32 → INT8`,
            sourceRange: [348, 356],
            syncOut: 'VEC_STORE',
            debugTip: `归一化并缩放至 INT8（N ${nRange}）。`,
          });
          const stStart = showPipe ? qStart + 0.95 : qStart + 1.2;
          const stBlk = addBlock({
            uid: `qnt:p2:st:${i}`,
            pipe: 'mte3',
            slot,
            iter: i,
            kind: 'dma-block',
            x: tToX(stStart),
            w: uToW(1.0) - 4,
            title: `Store I8 N${i}`,
            subtitle: `qr_i8_matmul`,
            sourceRange: [357, 368],
            debugTip: `将 INT8 量化块写回 qr_i8_matmul（padded），供下游 qproj_matmul 读取。`,
          });
          if (loadBlk) addSync(`sync:qnt:p2:ub:${i}`, 'UB_READY', loadBlk, qBlk, slot);
          addSync(`sync:qnt:p2:st:${i}`, 'VEC_STORE', qBlk, stBlk, slot);
          lastP2 = qBlk;
          t = showPipe ? t + 1.25 : stStart + 1.08;
          maxT = Math.max(maxT, stStart + 1.3);
        }
        if (lastP2) {
          reuseArcs.push({
            id: 'reuse:qnt:inv',
            fromX: invBlk.x + invBlk.w / 2,
            toX: lastP2.x + lastP2.w / 2,
            y: coreTop - 6,
            label: 'inv_rms & quant_scale 广播复用至 Pass 2 (4×N256)',
            color: '#80B7F7',
          });
        }
        break;
      }

      case 'post': {
        const tail = shape.valid < 8;
        const headCount = 4;
        const sharedLines = tail ? [464, 502] : [427, 431];
        const dqLines = tail ? [513, 531] : [432, 439];
        const rmsLines = tail ? [533, 537] : [440, 446];
        const outLines = tail ? [539, 552] : [448, 459];

        let t = 0.35;
        const sharedLoad = addBlock({
          uid: 'post:shared',
          pipe: 'mte2',
          slot: 0,
          kind: 'dma-block',
          x: tToX(t),
          w: uToW(1.45) - 4,
          title: 'Load Scale & RoPE',
          subtitle: `qr_scale + cos/sin/swap`,
          sourceRange: sharedLines,
          syncOut: 'UB_READY',
          debugTip: `每个 Token Tile 加载一次 qr_scale_dq 与 Q-local RoPE 参数（cos / signed sin / swap_idx），随后在 Head Group 内跨 4 个 Head 共享复用。`,
        });
        t += showPipe ? 1.15 : 1.52;

        let lastVec = null;
        for (let h = 0; h < headCount; h++) {
          const slot = h % 2;
          const headId = block * headCount + h;
          const gStart = t;
          const ldBlk = showPipe
            ? addBlock({
                uid: `post:ld:${h}`,
                pipe: 'mte2',
                slot,
                iter: h,
                kind: 'dma-block',
                x: tToX(t),
                w: uToW(1.1) - 4,
                title: `Load H${headId}`,
                subtitle: `q_i32 + wqb_sc`,
                sourceRange: dqLines,
                syncOut: 'UB_READY',
                debugTip: `pl.pipeline(4, stage=2)：加载 Head ${headId} 的 Q INT32 累加块 [8,512] 与列缩放系数 wq_b_scale。`,
              })
            : null;
          const dqStart = showPipe ? t + 0.92 : t;
          const dqBlk = addBlock({
            uid: `post:dq:${h}`,
            pipe: 'vec',
            slot,
            iter: h,
            kind: 'op-pill',
            icon: 'norm',
            x: tToX(dqStart),
            w: uToW(1.25) - 4,
            title: `Dequant+RMS H${headId}`,
            subtitle: `I32→FP32 · 512 cols`,
            sourceRange: rmsLines,
            debugTip: `Head ${headId}：反量化（乘 qr_scale 与 wq_b_scale）后计算每 Head 512 列 RMSNorm。`,
          });
          const rotStart = dqStart + 1.3;
          const rotBlk = addBlock({
            uid: `post:rot:${h}`,
            pipe: 'vec',
            slot,
            iter: h,
            kind: 'op-pill',
            icon: 'rope',
            x: tToX(rotStart),
            w: uToW(1.3) - 4,
            title: `NoPE+RoPE H${headId}`,
            subtitle: `448 NoPE · 64 RoPE`,
            sourceRange: outLines,
            syncOut: 'VEC_STORE',
            debugTip: `前 448 列直接转 BF16（NoPE），末尾 64 列经 swap-gather 与 cos/sin 旋转并转 BF16。`,
          });
          const stStart = showPipe ? rotStart + 1.0 : rotStart + 1.35;
          const stBlk = addBlock({
            uid: `post:st:${h}`,
            pipe: 'mte3',
            slot,
            iter: h,
            kind: 'dma-block',
            x: tToX(stStart),
            w: uToW(1.1) - 4,
            title: `Store Q H${headId}`,
            subtitle: `BF16 [8,512]`,
            sourceRange: outLines,
            debugTip: `MTE3 写回 Head ${headId} 的最终 Q BF16（NoPE + RoPE），供 Sparse Attention qk_pv 使用。`,
          });
          if (ldBlk) addSync(`sync:post:ub:${h}`, 'UB_READY', ldBlk, dqBlk, slot);
          else addSync(`sync:post:sh:${h}`, 'UB_READY', sharedLoad, dqBlk, slot);
          addSync(`sync:post:st:${h}`, 'VEC_STORE', rotBlk, stBlk, slot);

          if (!showPipe) {
            groups.push({
              slot,
              x: tToX(gStart) - 4,
              y: coreTop + 8,
              w: tToX(stStart + 1.12) - tToX(gStart) + 8,
              h: coreH - 16,
              label: `Head ${headId} · stage=2`,
            });
            t = stStart + 1.24;
          } else {
            t = rotStart + 1.38;
          }
          lastVec = rotBlk;
          maxT = Math.max(maxT, stStart + 1.4);
        }
        if (lastVec) {
          reuseArcs.push({
            id: 'reuse:post:shared',
            fromX: sharedLoad.x + sharedLoad.w / 2,
            toX: lastVec.x + lastVec.w / 2,
            y: coreTop - 6,
            label: 'qr_scale & RoPE Params 跨 4 Heads 共享复用',
            color: '#F2C45F',
          });
        }
        break;
      }

      case 'qrope':
      case 'kvrope': {
        const tail = shape.valid < 8;
        const owner = kernelInfo.type === 'qrope' ? 'Rank-local Q' : '全组 KV';
        let t = 0.35;
        const idxBlk = addBlock({
          uid: 'rope:idx',
          pipe: 'vec',
          slot: 0,
          kind: 'op-pill',
          icon: 'reshape',
          x: tToX(t),
          w: uToW(1.45) - 4,
          title: 'Build Dup/Swap Index',
          subtitle: `j//2 · j^1 · sign`,
          sourceRange: [143, 156],
          debugTip: `在 UB 内构造交错索引 dup (j//2)、swap (j^1) 与符号掩码 2(j%2)−1。`,
        });
        const ldStart = showPipe ? t + 0.2 : t + 1.5;
        const ldBlk = addBlock({
          uid: 'rope:load',
          pipe: 'mte2',
          slot: 0,
          kind: 'dma-block',
          x: tToX(ldStart),
          w: uToW(1.3) - 4,
          title: 'Load Cos / Sin',
          subtitle: `${owner} BF16 [${shape.valid},64]`,
          sourceRange: [120, 130],
          syncOut: 'UB_READY',
          debugTip: `从 GM 加载 ${owner} 的原始 Cos / Sin 表（BF16）至 UB。`,
        });
        t = Math.max(t + 1.52, ldStart + 1.36);

        const gatherBlk = addBlock({
          uid: 'rope:gather',
          pipe: 'vec',
          slot: 1,
          kind: 'op-pill',
          icon: 'rope',
          x: tToX(t),
          w: uToW(1.6) - 4,
          title: 'Gather Cos & Fold Sin',
          subtitle: `FP32 cos_il / sin_signed`,
          sourceRange: tail ? [168, 221] : [157, 164],
          syncOut: 'VEC_STORE',
          debugTip: `按 dup 索引 Gather 并转为 FP32，同时把交错符号折叠进 sin_signed，下游 RoPE 可直接乘加。`,
        });
        addSync('sync:rope:ub', 'UB_READY', ldBlk, gatherBlk, 0);
        t += 1.68;

        const stBlk = addBlock({
          uid: 'rope:store',
          pipe: 'mte3',
          slot: 1,
          kind: 'dma-block',
          x: tToX(t),
          w: uToW(1.45) - 4,
          title: `Store ${owner} Params`,
          subtitle: `cos/sin FP32 + swap I32`,
          sourceRange: tail ? [222, 240] : [165, 167],
          debugTip: `写回预折叠的 RoPE 参数，供 ${kernelInfo.type === 'qrope' ? 'qproj_dequant_rms_nope_rope' : 'KV Projection / Compressor'} 直接消费。`,
        });
        addSync('sync:rope:st', 'VEC_STORE', gatherBlk, stBlk, 1);

        reuseArcs.push({
          id: 'reuse:rope:idx',
          fromX: idxBlk.x + idxBlk.w / 2,
          toX: gatherBlk.x + gatherBlk.w / 2,
          y: coreTop - 6,
          label: 'dup / sign / swap 索引同时用于 Gather 与写出',
          color: '#80B7F7',
        });
        maxT = t + 1.8;
        break;
      }
    }

    return {
      width: tToX(maxT) + 48,
      height: totalHeight,
      labelW,
      xOrigin,
      maxT,
      cores: [core],
      blocks,
      groups,
      syncEdges,
      reuseArcs,
      shape,
    };
  }

  function renderScene(layout, layers, selection) {
    const { width, height, labelW, xOrigin, maxT, cores, blocks, groups, syncEdges, reuseArcs } = layout;
    const marks = [];
    const gridlines = [];
    const step = maxT > 30 ? 4 : 2;
    for (let u = 0; u <= Math.ceil(maxT); u += step) {
      const x = Math.round(xOrigin + u * UNIT_W);
      marks.push(`<span class="viz2-pi-axis__mark" style="left:${x - xOrigin}px">u${u}</span>`);
      gridlines.push(`<i class="viz2-pi-gridline" style="left:${x}px;top:${HEADER_H - 12}px;height:${height - HEADER_H + 14}px"></i>`);
    }

    const axisHtml = `<div class="viz2-pi-axis" style="left:${xOrigin}px;width:${width - xOrigin - 24}px">
      <div class="viz2-pi-axis__title">理论因果逻辑时间（按核内硬件子单元推进，非实测周期） →</div>
      ${marks.join('')}
    </div>${gridlines.join('')}`;

    const lanesHtml = cores.map(core => {
      const pipeRail = core.pipes.length
        ? `<div class="viz2-pi-pipe-rail">${core.pipes.map(p =>
            `<div class="viz2-pi-pipe-rail__item" style="top:${p.top - core.top}px;height:${p.height}px"><b>${esc(p.label)}</b><small>${esc(p.desc)}</small></div>`
          ).join('')}</div>`
        : '';
      const subpipes = core.pipes.map(p =>
        `<div class="viz2-pi-subpipe" style="top:${p.top - core.top}px;height:${p.height}px"></div>`
      ).join('');
      return `<div class="viz2-pi-core-row" data-core="${core.id}" style="top:${core.top}px;height:${core.height}px;width:${width - 16}px">
        <div class="viz2-pi-core-label" style="width:${labelW}px">
          <div class="viz2-pi-core-main">
            <div class="viz2-pi-core-title"><strong>${esc(core.title)}</strong><span class="viz2-pi-core-badge">${esc(core.badge)}</span></div>
            <small>${esc(core.subtitle)}</small>
          </div>
          ${pipeRail}
        </div>
        <div class="viz2-pi-core-track">${subpipes}</div>
      </div>`;
    }).join('');

    const groupsHtml = groups.map(g => {
      const slotStyle = `--slot-color:var(--intra-slot-${(g.slot ?? 0) % 3})`;
      return `<div class="viz2-pi-stage-frame" style="left:${g.x}px;top:${g.y}px;width:${g.w}px;height:${g.h}px;${slotStyle}">
        <div class="viz2-pi-stage-frame__head">
          <span>${esc(g.label)}</span>
          ${layers.slot ? `<span class="viz2-pi-slot-tag" style="${slotStyle}">Slot ${g.slot ?? 0}</span>` : ''}
        </div>
      </div>`;
    }).join('');

    const selectedUid = selection?.kind === 'block' ? selection.uid : null;
    const selectedSync = selection?.kind === 'sync' ? selection.id : null;

    const blocksHtml = blocks.map(b => {
      const isSelected = selectedUid === b.uid;
      const cls = [
        'viz2-pi-event',
        b.kind === 'op-pill' ? 'viz2-pi-event--op' : 'viz2-pi-event--dma',
        isSelected ? 'is-selected' : '',
      ].filter(Boolean).join(' ');
      const dot = b.syncOut && layers.sync && SYNC_META[b.syncOut]
        ? `<i class="viz2-pi-ev-dot" style="--ev-color:${SYNC_META[b.syncOut].color}"></i>`
        : '';
      return `<button type="button" class="${cls}" data-cmp-pi-uid="${esc(b.uid)}" data-pipe="${esc(b.pipe || '')}" style="left:${b.x}px;top:${b.y}px;width:${b.w}px;height:${b.h}px" title="${esc(b.debugTip)}">
        <span class="viz2-pi-event__head">${b.icon ? opIcon(b.icon) : ''}<b>${esc(b.title)}</b>${dot}</span>
        <small>${esc(b.subtitle)}</small>
      </button>`;
    }).join('');

    let syncSvg = '';
    if (layers.sync && syncEdges.length) {
      const uniqueEvents = [...new Set(syncEdges.map(e => e.event))];
      const markers = uniqueEvents.map(ev => {
        const color = (SYNC_META[ev] || SYNC_META.UB_READY).color;
        return `<marker id="cmp-pi-arrow-${ev}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="${color}"/></marker>`;
      }).join('');
      const paths = syncEdges.map(edge => {
        const active = selectedSync === edge.id;
        const d = `M ${edge.from.x} ${edge.from.y} V ${edge.busY} H ${edge.to.x} V ${edge.to.y}`;
        return `<g class="viz2-pi-sync-layer" data-cmp-pi-sync="${esc(edge.id)}" style="pointer-events:auto;cursor:pointer">
          <path d="${d}" fill="none" stroke="transparent" stroke-width="12"/>
          <path class="viz2-pi-sync-arc${active ? ' is-related' : ''}" d="${d}" stroke="${edge.color}" marker-end="url(#cmp-pi-arrow-${edge.event})"/>
          <circle cx="${edge.from.x}" cy="${edge.from.y}" r="3.2" fill="${edge.color}"/>
        </g>`;
      }).join('');
      syncSvg = `<svg class="viz2-pi-overlay" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><defs>${markers}</defs>${paths}</svg>`;
    }

    let reuseSvg = '';
    if (layers.slot && reuseArcs.length) {
      const arcs = reuseArcs.map((arc, idx) => {
        const midX = (arc.fromX + arc.toX) / 2;
        const arcLift = 30 + (idx % 2) * 18;
        const topY = arc.y - arcLift;
        const d = `M ${arc.fromX} ${arc.y} C ${arc.fromX} ${topY}, ${arc.toX} ${topY}, ${arc.toX} ${arc.y}`;
        return `<g class="viz2-pi-reuse-layer" style="--slot-color:${arc.color}">
          <path class="viz2-pi-reuse-arc" d="${d}"/>
          <text x="${midX}" y="${topY - 6}" text-anchor="middle" fill="${arc.color}" font-size="11">${esc(arc.label)}</text>
        </g>`;
      }).join('');
      reuseSvg = `<svg class="viz2-pi-overlay" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${arcs}</svg>`;
    }

    return `${axisHtml}${lanesHtml}${groupsHtml}${blocksHtml}${syncSvg}${reuseSvg}`;
  }

  function inspectorHtml(selection, layout, state) {
    if (!selection) return '';
    const { model, kernelInfo, block, rank, lane, iter } = state;
    const ref = measuredFor(model, kernelInfo.name, rank);
    const measuredLine = ref
      ? `Rank ${rank} DFX 实测：${ref.slices} 切片 · 中位 ${ref.medDurUs} µs（${ref.minDurUs}–${ref.maxDurUs} µs）· ${ref.coresUsed} 核`
      : '无实测数据';

    if (selection.kind === 'sync') {
      const edge = layout.syncEdges.find(e => e.id === selection.id);
      const meta = edge ? SYNC_META[edge.event] : null;
      if (!meta) return '';
      const range = edge.sourceRange || [kernelInfo.first, kernelInfo.last];
      return `<header class="pto-ide-frame__pane-header">
        <div><b>${esc(meta.name)}</b><small>Slot ${edge.slot} · ${esc(kernelInfo.name)} · Block ${block}</small></div>
        <button class="btn btn-ghost btn-sm" data-cmp-pi-close aria-label="关闭详情">×</button>
      </header>
      <div class="pto-ide-frame__pane-body">
        <p>${esc(meta.debugTip)}</p>
        <dl class="viz2-pi-dl">
          <dt>Producer</dt><dd>${esc(meta.producerPipe)}</dd>
          <dt>Consumer</dt><dd>${esc(meta.consumerPipe)}</dd>
          <dt>Buffer / 资源</dt><dd>${esc(meta.buffer)}</dd>
          <dt>DFX 实测参考</dt><dd>${esc(measuredLine)}</dd>
        </dl>
        <button type="button" class="btn btn-ghost btn-sm" data-cmp-pi-source="${esc(kernelInfo.file)}:${range[0]}:${range[1]}">查看源码 · ${esc(kernelInfo.file)} L${range[0]}–${range[1]} ↗</button>
      </div>`;
    }

    const blk = layout.blocks.find(b => b.uid === selection.uid);
    if (!blk) return '';
    const range = blk.sourceRange || [kernelInfo.first, kernelInfo.last];
    const irOps = irForLines(model, kernelInfo.name, range);
    const irList = irOps.length
      ? `<div class="csa-cmp-pi-ir"><h4>关联 PTO IR 指令</h4><ul>${irOps.map(([op, irLine, pyLine]) =>
          `<li><code>${esc(op)}</code><span>IR L${irLine}${pyLine ? ` · Py L${pyLine}` : ''}</span></li>`
        ).join('')}</ul></div>`
      : '';

    return `<header class="pto-ide-frame__pane-header">
      <div><b>${esc(blk.title)}</b><small>${esc(kernelInfo.coreType)} · ${esc(kernelInfo.name)} · Block ${block}</small></div>
      <button class="btn btn-ghost btn-sm" data-cmp-pi-close aria-label="关闭详情">×</button>
    </header>
    <div class="pto-ide-frame__pane-body">
      <p>${esc(blk.debugTip)}</p>
      <dl class="viz2-pi-dl">
        <dt>硬件单元</dt><dd>${esc(kernelInfo.coreType)} · ${esc((blk.pipe || '').toUpperCase())}</dd>
        <dt>管道槽位</dt><dd>Slot ${blk.slot ?? 0}${blk.iter != null ? ` · 迭代 #${blk.iter}` : ''}</dd>
        <dt>逻辑归属</dt><dd>${esc(layout.shape.text)}</dd>
        <dt>Lane 位置</dt><dd>Lane ${String(lane).padStart(2, '0')} · Iter ${iter}</dd>
        <dt>DFX 实测参考</dt><dd>${esc(measuredLine)}</dd>
      </dl>
      ${irList}
      <button type="button" class="btn btn-ghost btn-sm" data-cmp-pi-source="${esc(kernelInfo.file)}:${range[0]}:${range[1]}">查看源码 · ${esc(kernelInfo.file)} L${range[0]}–${range[1]} ↗</button>
    </div>`;
  }

  function mount(host, options) {
    const layers = { op: false, pipe: true, sync: true, slot: true };
    let state = null;
    let layout = null;
    let selection = null;
    let cam = { x: 16, y: 12, scale: 0.85 };
    let drag = null;

    host.innerHTML = `<div class="viz2-pipeline-instance">
      <div class="viz2-pi-toolbar">
        <div class="viz2-pi-toolbar__left">
          <span class="viz2-pi-summary" data-cmp-pi-summary></span>
          <div class="viz2-pi-layers" role="group" aria-label="因果泳道图层">
            <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn" data-cmp-pi-mode="op" aria-pressed="false">Op</button>
            <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn is-selected" data-cmp-pi-mode="pipe" aria-pressed="true">Pipe</button>
            <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn is-selected" data-cmp-pi-toggle="sync" aria-pressed="true">Sync</button>
            <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn is-selected" data-cmp-pi-toggle="slot" aria-pressed="true">Slot 复用</button>
          </div>
        </div>
        <div class="viz2-pi-tools">
          <button type="button" class="btn btn-ghost btn-sm" data-cmp-pi-zoom="0.85" aria-label="缩小">−</button>
          <output data-cmp-pi-scale>85%</output>
          <button type="button" class="btn btn-ghost btn-sm" data-cmp-pi-zoom="1.18" aria-label="放大">＋</button>
          <button type="button" class="btn btn-ghost btn-sm" data-cmp-pi-fit>Fit to View</button>
        </div>
      </div>
      <div class="viz2-pi-boundary-note" data-cmp-pi-note></div>
      <div class="viz2-pi-viewport" tabindex="0" role="region" aria-label="Compressor 核内因果执行泳道">
        <div class="viz2-pi-scene" data-cmp-pi-scene></div>
      </div>
      <aside class="pto-ide-frame__pane viz2-pi-inspector" data-cmp-pi-inspector hidden></aside>
    </div>`;

    const summaryEl = host.querySelector('[data-cmp-pi-summary]');
    const noteEl = host.querySelector('[data-cmp-pi-note]');
    const viewport = host.querySelector('.viz2-pi-viewport');
    const scene = host.querySelector('[data-cmp-pi-scene]');
    const scaleOut = host.querySelector('[data-cmp-pi-scale]');
    const inspector = host.querySelector('[data-cmp-pi-inspector]');

    function applyCamera() {
      scene.style.transform = `translate(${cam.x}px, ${cam.y}px) scale(${cam.scale})`;
      scaleOut.textContent = `${Math.round(cam.scale * 100)}%`;
    }

    function fit() {
      if (!layout || !viewport.clientWidth || !viewport.clientHeight) return;
      const sx = (viewport.clientWidth - 28) / layout.width;
      const sy = (viewport.clientHeight - 28) / layout.height;
      const s = Math.max(0.32, Math.min(1.12, Math.min(sx, sy)));
      cam = {
        scale: s,
        x: Math.max(12, Math.round((viewport.clientWidth - layout.width * s) / 2)),
        y: Math.max(10, Math.round((viewport.clientHeight - layout.height * s) / 2)),
      };
      applyCamera();
    }

    function syncInspector() {
      if (!selection || !layout || !state) {
        inspector.hidden = true;
        return;
      }
      inspector.innerHTML = inspectorHtml(selection, layout, state);
      inspector.hidden = false;
    }

    function redraw(refit = false) {
      if (!state) return;
      const model = state.model || global.PtoCsaCompressorModel;
      const { kernelInfo, block, rank, localView } = state;
      layout = buildLayout(model, kernelInfo, block, rank, layers, localView);
      const ref = measuredFor(model, kernelInfo.name, rank);
      const branchLabel = model.branches?.[kernelInfo.branch] || (kernelInfo.branch === 'main' ? 'Main' : 'Indexer');
      const pipeNoteMap = {
        projection: '8 × K512 · pl.pipeline(stage=2) 双缓冲',
        pool: '7 历史窗口槽位 · Online Softmax',
        commit: '8 Token 槽位 · 条件 State Ring 提交',
        norm: `${(kernelInfo.head || 128) / 64} Head Tiles · RMSNorm + RoPE`,
        hadamard: '2 × N64 Tiles · L1 KV 复用',
        write: 'BF16 Round-trip → Per-row INT8 量化写回',
        qr: `8 × K256 · Split ${layout.shape.split ?? 0} · stage=2 → Atomic Add`,
        qmat: `8 × K128 · ${layout.shape.path || 'Full'} M Tile · stage=2 INT8 Matmul`,
        seed: '8 × N128 · On-core FP32 零初始化',
        quant: '2-Pass 4×N256 · stage=2 RMSNorm + INT8 量化',
        post: '4 Heads · stage=2 Dequant + Per-Head RMSNorm + RoPE',
        qrope: 'Rank-local Q · 预折叠 Cos / Signed Sin / Swap Index',
        kvrope: '全组 KV · 预折叠 Cos / Signed Sin / Swap Index',
      };
      const pipeNote = pipeNoteMap[kernelInfo.type] || kernelInfo.name;

      summaryEl.textContent = `${kernelInfo.coreType} 因果泳道 · ${kernelInfo.name} · Block ${block} · ${pipeNote}`;
      noteEl.textContent = `单 Block 理论因果执行序 · 同核/同子流水严格单行串行（Rank ${rank} DFX 实测：${ref ? `${ref.slices} 切片 · 中位 ${ref.medDurUs} µs / 区间 ${ref.minDurUs}–${ref.maxDurUs} µs · ${ref.coresUsed} 物理核` : '无实测'}；核内子操作按逻辑单元推进，真实气泡需对接核内仿真 Trace）。`;
      scene.style.width = `${layout.width}px`;
      scene.style.height = `${layout.height}px`;
      scene.innerHTML = renderScene(layout, layers, selection);
      syncInspector();
      if (refit) requestAnimationFrame(fit);
      else applyCamera();
    }

    host.addEventListener('click', event => {
      const modeBtn = event.target.closest('[data-cmp-pi-mode]');
      if (modeBtn) {
        layers.pipe = modeBtn.dataset.cmpPiMode === 'pipe';
        layers.op = !layers.pipe;
        host.querySelectorAll('[data-cmp-pi-mode]').forEach(b => {
          const on = (b.dataset.cmpPiMode === 'pipe') === layers.pipe;
          b.classList.toggle('is-selected', on);
          b.setAttribute('aria-pressed', String(on));
        });
        redraw(true);
        return;
      }
      const toggleBtn = event.target.closest('[data-cmp-pi-toggle]');
      if (toggleBtn) {
        const key = toggleBtn.dataset.cmpPiToggle;
        layers[key] = !layers[key];
        toggleBtn.classList.toggle('is-selected', layers[key]);
        toggleBtn.setAttribute('aria-pressed', String(layers[key]));
        redraw(false);
        return;
      }
      const zoomBtn = event.target.closest('[data-cmp-pi-zoom]');
      if (zoomBtn) {
        cam.scale = Math.max(0.28, Math.min(2.4, cam.scale * Number(zoomBtn.dataset.cmpPiZoom)));
        applyCamera();
        return;
      }
      if (event.target.closest('[data-cmp-pi-fit]')) {
        fit();
        return;
      }
      if (event.target.closest('[data-cmp-pi-close]')) {
        selection = null;
        redraw(false);
        return;
      }
      const srcBtn = event.target.closest('[data-cmp-pi-source]');
      if (srcBtn) {
        const [file, a, z] = srcBtn.dataset.cmpPiSource.split(':');
        options?.onSource?.(file, Number(a), Number(z));
        return;
      }
      const blk = event.target.closest('[data-cmp-pi-uid]');
      if (blk) {
        const uid = blk.dataset.cmpPiUid;
        selection = selection?.uid === uid ? null : { kind: 'block', uid };
        redraw(false);
        return;
      }
      const syncEl = event.target.closest('[data-cmp-pi-sync]');
      if (syncEl) {
        const id = syncEl.dataset.cmpPiSync;
        selection = selection?.id === id ? null : { kind: 'sync', id };
        redraw(false);
      }
    });

    viewport.addEventListener('pointerdown', event => {
      if (event.button !== 0 || event.target.closest('button,[data-cmp-pi-sync]')) return;
      drag = { x: event.clientX, y: event.clientY, camX: cam.x, camY: cam.y, pid: event.pointerId };
      viewport.setPointerCapture(event.pointerId);
    });
    viewport.addEventListener('pointermove', event => {
      if (!drag) return;
      cam.x = drag.camX + (event.clientX - drag.x);
      cam.y = drag.camY + (event.clientY - drag.y);
      applyCamera();
    });
    const endDrag = () => { drag = null; };
    viewport.addEventListener('pointerup', endDrag);
    viewport.addEventListener('pointercancel', endDrag);
    viewport.addEventListener('wheel', event => {
      if (!(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      const mx = event.clientX - rect.left;
      const my = event.clientY - rect.top;
      const next = Math.max(0.28, Math.min(2.4, cam.scale * Math.exp(Math.max(-0.25, Math.min(0.25, -event.deltaY * 0.002)))));
      const ratio = next / cam.scale;
      cam.x = mx - (mx - cam.x) * ratio;
      cam.y = my - (my - cam.y) * ratio;
      cam.scale = next;
      applyCamera();
    }, { passive: false });

    return {
      update(nextState) {
        const changed = !state || state.kernelInfo.name !== nextState.kernelInfo.name || state.block !== nextState.block || state.rank !== nextState.rank
          || state.localView?.tokens !== nextState.localView?.tokens || state.localView?.tileIndex !== nextState.localView?.tileIndex;
        state = nextState;
        if (changed) selection = null;
        redraw(changed);
      },
      fit,
    };
  }

  global.PtoCsaCompressorPipelineInstance = Object.freeze({ mount });
})(window);
