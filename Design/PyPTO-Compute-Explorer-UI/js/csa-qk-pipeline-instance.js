/* CSA qk_pv Instance Causal Hardware Swimlane (AIC / AIV 0 / AIV 1).
 * Reuses the exact visual grammar and CSS of Paged Attention's pipeline-instance view:
 *   - Strictly single-row serial per hardware lane / sub-pipe.
 *   - Grounded in passes_dump/25_after_SplitVectorKernel.py (dual_aiv_dispatch=True,
 *     C2V/V2C 2-slot 16KB pipes, AIV 1 zero-valid-row handshake companion).
 *   - Measured wall reference from _jit_l3_decode_csa_20260903_010617/dfx_outputs. */
(function (global) {
  'use strict';

  const esc = value => String(value ?? '').replace(/[&<>"']/g, ch =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);

  const SUB_FILE = 'decode_sparse_attn_csa.py';
  const HEADER_H = 92;
  const UNIT_W = 86;
  const GAP_U = 0.24;

  const EVENT_META = {
    C2V_PIPE: {
      name: 'C2V_PIPE (tpush→tpop)',
      id: 'C2V_PIPE',
      color: '#80B7F7',
      producerCore: 'AIC',
      producerPipe: 'FIX',
      consumerCore: 'AIV 0 (+ AIV 1 协议握手)',
      consumerPipe: 'MTE2',
      buffer: 'qk_pv_c2v_slot_buffer (32 KB = 2 × 16 KB)',
      lineSet: 282,
      lineWait: 284,
      debugTip: 'AIC 完成 32×128 FP32 QK Matmul 后调用 tpush_to_aiv(split=0) 写入 C2V 环形槽位（slot_num=2, slot_size=16 KB）；AIV 0 执行 tpop_from_aic 消费并立刻 tfree_to_aic 释放槽位。',
    },
    V2C_PIPE: {
      name: 'V2C_PIPE (tpush→tpop)',
      id: 'V2C_PIPE',
      color: '#9EF04B',
      producerCore: 'AIV 0 (+ AIV 1 协议握手)',
      producerPipe: 'MTE3',
      consumerCore: 'AIC',
      consumerPipe: 'MTE2',
      buffer: 'qk_pv_v2c_slot_buffer (32 KB = 2 × 16 KB)',
      lineSet: 294,
      lineWait: 295,
      debugTip: 'AIV 0 将 Softmax exp 转为 BF16 [32,128] 后通过 tpush_to_aic(split=0) 写回 V2C 双槽管道；AIC 通过 tpop_from_aiv 取出作为 PV Matmul 左矩阵，4×K=32 累加完成后 tfree_to_aiv。',
    },
  };

  function opIcon(kind) {
    const entry = global.PtoOpIcons?.find(item => item.id === kind);
    if (!entry) return '';
    return `<span class="viz2-pi-op-icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="${entry.path}"/></svg></span>`;
  }

  function measuredRef() {
    const rt = global.PTO_CSA_RUNTIME_DATA;
    const stage = rt?.stages?.find(s => s.id === 'sparse/attention/qk-pv');
    const aic = stage?.kernels?.find(k => k.name === 'qk_pv_aic')?.ranks?.[0];
    const aiv = stage?.kernels?.find(k => k.name === 'qk_pv_aiv')?.ranks?.[0];
    return { aic, aiv };
  }

  function buildLayout(item, detail, layers) {
    const showPipe = !!layers.pipe;
    const labelW = showPipe ? 226 : 182;
    const xOrigin = labelW + 24;
    const tToX = t => Math.round(xOrigin + t * UNIT_W);
    const uToW = u => Math.max(48, Math.round(u * UNIT_W));

    const aicTop = HEADER_H;
    const aicSubH = showPipe ? 54 : 0;
    const aicH = showPipe ? aicSubH * 3 + 22 : 108;
    const aicAivGap = 56;
    const aivSubH = showPipe ? 48 : 0;
    const aivH = showPipe ? aivSubH * 3 + 20 : 92;
    const aivBetween = 22;

    const aiv0Top = aicTop + aicH + aicAivGap;
    const aiv1Top = aiv0Top + aivH + aivBetween;
    const totalHeight = aiv1Top + aivH + 34;

    const cores = [
      {
        id: 'aic',
        title: 'AIC',
        subtitle: item.valid ? `L1 KV [128, 512] · 2 × HB` : `Valid=0 · 跳过矩阵计算`,
        badge: '1 Core',
        top: aicTop,
        height: aicH,
        pipes: showPipe
          ? [
              { id: 'mte2', label: 'MTE2', desc: 'Gather KV / Load Q', top: aicTop + 8, height: aicSubH },
              { id: 'cube', label: 'Cube', desc: 'QK / PV Acc', top: aicTop + 8 + aicSubH + 3, height: aicSubH },
              { id: 'fix', label: 'FIX', desc: 'tpush / Store O', top: aicTop + 8 + (aicSubH + 3) * 2, height: aicSubH },
            ]
          : [],
      },
      {
        id: 'aiv0',
        lane: 0,
        title: 'AIV 0',
        subtitle: item.valid ? 'subblock 0 · 32/32 Heads' : 'subblock 0 · 中性写入',
        badge: '32/32 Rows',
        top: aiv0Top,
        height: aivH,
        pipes: showPipe
          ? [
              { id: 'mte2', label: 'MTE2', desc: 'tpop / Bias', top: aiv0Top + 7, height: aivSubH },
              { id: 'vec', label: 'Vector', desc: 'Scale / Softmax', top: aiv0Top + 7 + aivSubH + 3, height: aivSubH },
              { id: 'mte3', label: 'MTE3', desc: 'tpush / Store m,l', top: aiv0Top + 7 + (aivSubH + 3) * 2, height: aivSubH },
            ]
          : [],
      },
      {
        id: 'aiv1',
        lane: 1,
        title: 'AIV 1',
        subtitle: 'subblock 1 · valid_shape=[0,0]',
        badge: '协议伴随核',
        top: aiv1Top,
        height: aivH,
        pipes: showPipe
          ? [
              { id: 'mte2', label: 'MTE2', desc: 'tpop (0行)', top: aiv1Top + 7, height: aivSubH },
              { id: 'vec', label: 'Vector', desc: 'valid=[0,0]', top: aiv1Top + 7 + aivSubH + 3, height: aivSubH },
              { id: 'mte3', label: 'MTE3', desc: 'tpush (0行)', top: aiv1Top + 7 + (aivSubH + 3) * 2, height: aivSubH },
            ]
          : [],
      },
    ];

    const blocks = [];
    const groups = [];
    const syncEdges = [];
    const reuseArcs = [];
    const highwayTop = aicTop + aicH + 8;

    const register = block => {
      blocks.push(block);
      return block;
    };

    let aicT = 0.35;
    let aiv0T = 0.35;

    if (!item.valid) {
      // Invalid branch: AIC skips Gather/QK/PV; AIV 0 writes neutral m/l/o for 4 H_TILE=16 segments.
      register({
        uid: 'aic:skip',
        core: 'aic',
        pipe: 'mte2',
        stage: 'mask',
        slot: 0,
        kind: 'dma-block',
        x: tToX(aicT),
        y: showPipe ? cores[0].pipes[0].top + 6 : cores[0].top + 30,
        w: uToW(2.2),
        h: showPipe ? 42 : 52,
        title: 'Skip Gather / QK / PV',
        subtitle: 'valid_block_mask == 0',
        sourceLine: 223,
        debugTip: 'Invalid Work Item 在 AIC 侧直接跳过 L1 KV Gather、QK Matmul 与 PV Matmul，不触发 C2V/V2C 管道交互。',
      });

      const neutralStart = 0.35;
      for (let hIdx = 0; hIdx < 4; hIdx++) {
        const y = showPipe ? cores[1].pipes[2].top + 5 : cores[1].top + 28;
        register({
          uid: `aiv0:neutral:${hIdx}`,
          core: 'aiv0',
          pipe: 'mte3',
          stage: 'neutral',
          slot: hIdx % 2,
          kind: 'op-pill',
          icon: 'cache',
          x: tToX(aiv0T),
          y,
          w: uToW(1.55) - 4,
          h: showPipe ? 38 : 50,
          title: `Neutral H${hIdx * 16}–${hIdx * 16 + 15}`,
          subtitle: 'mi=−3e38 · li=0 · oi=0',
          sourceLine: 305,
          debugTip: `AIV 0（subblock 0）为 Head ${hIdx * 16}–${hIdx * 16 + 15} 写入中性 Partial（mi=−3.0e38、li=0、oi=0），供下游 csa_merge_pack_publish 安全忽略。`,
        });
        aiv0T += 1.65;
      }
      groups.push({
        core: 'aiv0',
        stage: 'neutral',
        slot: 0,
        x: tToX(neutralStart) - 5,
        y: cores[1].top + 8,
        w: tToX(aiv0T - 0.1) - tToX(neutralStart) + 10,
        h: cores[1].height - 16,
        label: 'Invalid 分支 · 4 × H_TILE=16 中性写回',
      });
      register({
        uid: 'aiv1:idle',
        core: 'aiv1',
        pipe: 'vec',
        stage: 'neutral',
        slot: 0,
        kind: 'dma-block',
        x: tToX(0.35),
        y: showPipe ? cores[2].pipes[1].top + 5 : cores[2].top + 28,
        w: uToW(2.4),
        h: showPipe ? 38 : 50,
        title: 'AIV 1 空分支',
        subtitle: 'dual_aiv_dispatch · 0 行',
        sourceLine: 305,
        debugTip: '编译器 Pass 25 (SplitVectorKernel) 在 dual_aiv_dispatch=True 下对 Invalid 分支不向 AIV 1 分配存储任务。',
      });

      const maxT = Math.max(aicT + 2.8, aiv0T + 0.8);
      return {
        width: tToX(maxT) + 48,
        height: totalHeight,
        labelW,
        xOrigin,
        maxT,
        cores,
        blocks,
        groups,
        syncEdges,
        reuseArcs,
      };
    }

    // Valid Work Item: exact compiler-verified AIC / AIV 0 / AIV 1 causal pipeline
    const isWindow = item.sparseBlock === 0;
    const gatherTitle = isWindow ? 'Gather Window KV' : 'Gather Compressed KV';
    const gatherSub = isWindow
      ? `SB0 · ${item.validKvCount}/128 · 连续页`
      : `SB${item.sparseBlock} · ${item.validKvCount}/128 · 散列`;

    // 1. AIC: Gather shared L1 KV Tile [128, 512] BF16 (once per Work Item, reused by HB0 & HB1)
    const gatherDur = isWindow ? 1.55 : 2.1;
    const kvGatherBlock = register({
      uid: 'aic:gather-kv',
      core: 'aic',
      pipe: 'mte2',
      stage: 'gather',
      slot: 0,
      kind: 'dma-block',
      x: tToX(aicT),
      y: showPipe ? cores[0].pipes[0].top + 6 : cores[0].top + 30,
      w: uToW(gatherDur) - 4,
      h: showPipe ? 42 : 52,
      title: gatherTitle,
      subtitle: `${gatherSub} → L1`,
      sourceLine: isWindow ? 235 : 262,
      debugTip: isWindow
        ? 'SB0 滑窗 KV 按页连续段（SWA_RUNS）多行 Gather 到 L1 [128, 512] BF16，随后被 HB0 和 HB1 的 QK / PV 复用 4 次。'
        : 'SB1–SB4 压缩 KV 依据 cmp_sparse_indices + cmp_block_table 逐行 Gather 到 L1 [128, 512] BF16，无效槽位以有限值填充并由 sparse_bias 屏蔽。',
    });
    const kvReuseOriginX = kvGatherBlock.x + kvGatherBlock.w / 2;
    aicT += gatherDur + GAP_U;

    const c2vReady = new Map();
    const v2cReady = new Map();
    const qkGroupCenters = [];

    // Helper for strict single-row pipe-aware Y
    const yFor = (coreObj, pipeId) => {
      if (!showPipe) return coreObj.id === 'aic' ? coreObj.top + 30 : coreObj.top + 28;
      const p = coreObj.pipes.find(itemPipe => itemPipe.id === pipeId) || coreObj.pipes[0];
      return p.top + 5;
    };
    const hFor = coreObj => (showPipe ? (coreObj.id === 'aic' ? 42 : 38) : (coreObj.id === 'aic' ? 52 : 50));

    if (!showPipe) {
      // OP-LEVEL CAUSAL VIEW: strictly single-row serial on AIC, AIV 0, AIV 1
      // Stage A: HB0 QK -> C2V Slot 0, then HB1 QK -> C2V Slot 1 (pl.pipeline(2, stage=2) double-buffering)
      for (const hb of [0, 1]) {
        const slot = hb;
        const headRange = hb === 0 ? 'H0–31' : 'H32–63';
        const gStart = aicT;
        register({
          uid: `aic:load-q:hb${hb}`,
          core: 'aic',
          pipe: 'mte2',
          stage: 'query',
          hb,
          slot,
          kind: 'dma-block',
          x: tToX(aicT),
          y: yFor(cores[0], 'mte2'),
          w: uToW(0.95) - 4,
          h: hFor(cores[0]),
          title: `Load Q · HB${hb}`,
          subtitle: `[32, 512] · ${headRange}`,
          sourceLine: 280,
          debugTip: `加载 Head Batch ${hb}（${headRange}）的 Q Tile [32, 512] BF16 到 L1，准备与共享 L1 KV 做 QK Matmul。`,
        });
        aicT += 1.02;

        const qkStart = aicT;
        for (let k = 0; k < 4; k++) {
          register({
            uid: `aic:qk:hb${hb}:k${k}`,
            core: 'aic',
            pipe: 'cube',
            stage: 'qk',
            hb,
            slot,
            kind: 'op-pill',
            icon: 'matmul',
            chainedFromPrev: k > 0,
            x: tToX(aicT),
            y: yFor(cores[0], 'cube'),
            w: uToW(0.92) - 4,
            h: hFor(cores[0]),
            title: `QK K${k}`,
            subtitle: k === 0 ? 'matmul' : 'matmul_acc',
            sourceLine: 282,
            debugTip: `AutoTileMatmulL0 将 K=512 拆为 4 段 128（stage=2 流水）：K${k} 计算 [32,128]×[128,128] 累加到 FP32 L0C Acc。`,
          });
          aicT += 0.98;
        }
        const pushEndX = tToX(aicT - 0.06);
        groups.push({
          core: 'aic',
          stage: 'qk',
          hb,
          slot,
          x: tToX(gStart) - 5,
          y: cores[0].top + 8,
          w: pushEndX - tToX(gStart) + 10,
          h: cores[0].height - 16,
          label: `QK · HB${hb} (${headRange}) · C2V`,
          syncOut: 'C2V_PIPE',
        });
        qkGroupCenters.push({ hb, x: tToX(qkStart) + 40 });
        c2vReady.set(hb, { t: aicT + 0.14, x: pushEndX, y: cores[0].top + cores[0].height - 8 });
        aicT += GAP_U + 0.16;
      }

      // Stage B: AIV 0 computes HB0 & HB1 Softmax serially (triggered by C2V_PIPE); AIV 1 mirrors handshake with valid_shape=[0,0]
      for (const hb of [0, 1]) {
        const slot = hb;
        const headRange = hb === 0 ? 'H0–31' : 'H32–63';
        const c2v = c2vReady.get(hb);
        const startT = Math.max(aiv0T, c2v.t);
        const gStartX = tToX(startT);

        // AIV 0: Scale & Bias -> Softmax Stats -> Store m/l & tpush V2C
        const sbDur = 1.25;
        const smDur = 1.65;
        const stDur = 1.05;
        register({
          uid: `aiv0:bias:hb${hb}`,
          core: 'aiv0',
          pipe: 'vec',
          stage: 'stats',
          hb,
          slot,
          kind: 'op-pill',
          icon: 'softmax',
          x: tToX(startT),
          y: yFor(cores[1], 'vec'),
          w: uToW(sbDur) - 4,
          h: hFor(cores[1]),
          title: `Scale & Bias`,
          subtitle: `HB${hb} · [32, 128]`,
          syncIn: 'C2V_PIPE',
          sourceLine: 285,
          debugTip: `AIV 0 tpop_from_aic 取出 Scores [32,128] FP32 后立刻 tfree_to_aic，再执行 muls(scale) + col_expand_add(sparse_bias)。`,
        });
        register({
          uid: `aiv0:sm:hb${hb}`,
          core: 'aiv0',
          pipe: 'vec',
          stage: 'stats',
          hb,
          slot,
          kind: 'op-pill',
          icon: 'softmax',
          chainedFromPrev: true,
          x: tToX(startT + sbDur + 0.08),
          y: yFor(cores[1], 'vec'),
          w: uToW(smDur) - 4,
          h: hFor(cores[1]),
          title: `Softmax · HB${hb}`,
          subtitle: 'row_max · exp · row_sum',
          sourceLine: 289,
          debugTip: `对 HB${hb}（32 行）计算局部最大值 m [32,1]、exp [32,128]、行和 l [32,1]，并将 exp 转为 BF16 准备回推 AIC。`,
        });
        register({
          uid: `aiv0:store-ml:hb${hb}`,
          core: 'aiv0',
          pipe: 'mte3',
          stage: 'write',
          hb,
          slot,
          kind: 'dma-block',
          chainedFromPrev: true,
          x: tToX(startT + sbDur + smDur + 0.16),
          y: yFor(cores[1], 'mte3'),
          w: uToW(stDur) - 4,
          h: hFor(cores[1]),
          title: `Store m,l + tpush`,
          subtitle: '2×H_TILE=16 · V2C',
          syncOut: 'V2C_PIPE',
          sourceLine: 294,
          debugTip: `先通过 tpush_to_aic(split=0) 将 BF16 exp [32,128] 写入 V2C Slot ${slot}，再分两段 [16,1] 写出 sparse_blk_mi / li。`,
        });

        const endT = startT + sbDur + smDur + stDur + 0.16;
        const endX = tToX(endT);
        groups.push({
          core: 'aiv0',
          stage: 'stats',
          hb,
          slot,
          x: gStartX - 4,
          y: cores[1].top + 7,
          w: endX - gStartX + 8,
          h: cores[1].height - 14,
          label: `HB${hb} · ${headRange} (32/32)`,
          syncIn: 'C2V_PIPE',
          syncOut: 'V2C_PIPE',
        });

        // AIV 1 companion handshake block (truth from 25_after_SplitVectorKernel.py L1985-1998)
        register({
          uid: `aiv1:companion:hb${hb}`,
          core: 'aiv1',
          pipe: 'vec',
          stage: 'stats',
          hb,
          slot,
          kind: 'dma-block',
          x: tToX(startT),
          y: yFor(cores[2], 'vec'),
          w: endX - gStartX - 4,
          h: hFor(cores[2]),
          title: `tpop / tfree / tpush · HB${hb}`,
          subtitle: 'valid_shape = [0, 0] · 伴随同步',
          sourceLine: 284,
          debugTip: '编译产物 25_after_SplitVectorKernel.py 核实：dual_aiv_dispatch=True 时 AIV 1 走 else 分支，所有 Tile 携带 valid_shape=[0,0]，仅参与 C2V/V2C 管道握手而不分摊 32 行计算。',
        });

        syncEdges.push({
          id: `sync:C2V:hb${hb}`,
          event: 'C2V_PIPE',
          slot,
          sourceCore: 'aic',
          from: { x: c2v.x, y: c2v.y },
          to: { x: gStartX + 8, y: cores[1].top + 10 },
          busY: highwayTop + 6 + slot * 12,
          color: EVENT_META.C2V_PIPE.color,
        });

        v2cReady.set(hb, {
          t: endT + 0.14,
          x: endX - 10,
          y: cores[1].top + 10,
        });
        aiv0T = endT + GAP_U + 0.14;
      }

      // Stage C: AIC PV Matmul for HB0 & HB1 (waits on V2C_PIPE, reuses same L1 KV Tile!)
      for (const hb of [0, 1]) {
        const slot = hb;
        const headRange = hb === 0 ? 'H0–31' : 'H32–63';
        const v2c = v2cReady.get(hb);
        const startT = Math.max(aicT, v2c.t);
        const startX = tToX(startT);

        syncEdges.push({
          id: `sync:V2C:hb${hb}`,
          event: 'V2C_PIPE',
          slot,
          sourceCore: 'aiv0',
          from: { x: v2c.x, y: v2c.y },
          to: { x: startX + 8, y: cores[0].top + cores[0].height - 8 },
          busY: highwayTop + 28 + slot * 10,
          color: EVENT_META.V2C_PIPE.color,
        });

        aicT = startT;
        for (let k = 0; k < 4; k++) {
          register({
            uid: `aic:pv:hb${hb}:k${k}`,
            core: 'aic',
            pipe: 'cube',
            stage: 'pv',
            hb,
            slot,
            kind: 'op-pill',
            icon: 'matmul',
            chainedFromPrev: k > 0,
            x: tToX(aicT),
            y: yFor(cores[0], 'cube'),
            w: uToW(0.92) - 4,
            h: hFor(cores[0]),
            title: `PV K${k}`,
            subtitle: k === 0 ? 'matmul' : 'matmul_acc',
            sourceLine: 295,
            debugTip: `AutoTileMatmulL0 将 K=128 拆为 4 段 32：tpop_from_aiv 取出 exp BF16 [32,128] × 共享 L1 V [128,512]，K${k} 累加至 FP32 [32,512]。`,
          });
          aicT += 0.98;
        }

        register({
          uid: `aic:store-o:hb${hb}`,
          core: 'aic',
          pipe: 'fix',
          stage: 'write',
          hb,
          slot,
          kind: 'dma-block',
          chainedFromPrev: true,
          x: tToX(aicT),
          y: yFor(cores[0], 'fix'),
          w: uToW(1.15) - 4,
          h: hFor(cores[0]),
          title: `Store O · HB${hb}`,
          subtitle: '2×[16, 512] FP32',
          sourceLine: 297,
          debugTip: `tfree_to_aiv 释放 V2C Slot ${slot} 后，将 Partial O [32,512] 切为两段 [16,512] 写回 sparse_blk_oi。`,
        });
        aicT += 1.22;
        const endX = tToX(aicT - 0.06);
        groups.push({
          core: 'aic',
          stage: 'pv',
          hb,
          slot,
          x: startX - 5,
          y: cores[0].top + 8,
          w: endX - startX + 10,
          h: cores[0].height - 16,
          label: `PV · HB${hb} (${headRange}) · V2C`,
          syncIn: 'V2C_PIPE',
        });
        if (hb === 0) {
          reuseArcs.push({
            id: 'reuse:l1-kv',
            slot: 0,
            fromX: kvReuseOriginX,
            toX: startX + 26,
            y: cores[0].top - 6,
            label: 'L1 KV [128,512] 复用（HB0/HB1 QK & PV 共用）',
            color: '#F2C45F',
          });
        }
        aicT += GAP_U + 0.14;
      }
    } else {
      // PIPE-LEVEL CAUSAL VIEW: MTE2 / Cube / FIX on AIC; MTE2 / Vector / MTE3 on AIV 0 & AIV 1
      for (const hb of [0, 1]) {
        const slot = hb;
        const headRange = hb === 0 ? 'H0–31' : 'H32–63';
        const qLoad = register({
          uid: `aic:mte2-q:hb${hb}`,
          core: 'aic',
          pipe: 'mte2',
          stage: 'query',
          hb,
          slot,
          kind: 'dma-block',
          x: tToX(aicT),
          y: yFor(cores[0], 'mte2'),
          w: uToW(1.1) - 4,
          h: hFor(cores[0]),
          title: `MTE2 Load Q · HB${hb}`,
          subtitle: `[32, 512] · ${headRange}`,
          sourceLine: 280,
          debugTip: `MTE2 将 Q[${hb * 32}:${hb * 32 + 32}, 0:512] 载入 L1，准备喂给 Cube 4 次 K=128 切片。`,
        });
        const cubeStart = aicT + 0.55;
        register({
          uid: `aic:cube-qk:hb${hb}`,
          core: 'aic',
          pipe: 'cube',
          stage: 'qk',
          hb,
          slot,
          kind: 'op-pill',
          icon: 'matmul',
          x: tToX(cubeStart),
          y: yFor(cores[0], 'cube'),
          w: uToW(2.4) - 4,
          h: hFor(cores[0]),
          title: `Cube QK · HB${hb}`,
          subtitle: '4×K=128 · stage=2',
          sourceLine: 282,
          debugTip: `Cube 单元在 L0A/L0B 双缓冲下串行完成 4 段 K=128 矩阵乘累加，产出 FP32 Scores [32, 128]。`,
        });
        const fixStart = cubeStart + 2.48;
        const fixBlock = register({
          uid: `aic:fix-c2v:hb${hb}`,
          core: 'aic',
          pipe: 'fix',
          stage: 'qk',
          hb,
          slot,
          kind: 'dma-block',
          x: tToX(fixStart),
          y: yFor(cores[0], 'fix'),
          w: uToW(1.05) - 4,
          h: hFor(cores[0]),
          title: `FIX tpush C2V`,
          subtitle: `Slot ${slot} · 16 KB`,
          syncOut: 'C2V_PIPE',
          sourceLine: 282,
          debugTip: `FIX 通路将 L0C Scores [32, 128] FP32 写入 qk_pv_c2v_slot_buffer 的 Slot ${slot}（16 KB），触发 AIV 0 消费。`,
        });
        c2vReady.set(hb, { t: fixStart + 1.15, x: fixBlock.x + fixBlock.w, y: fixBlock.y + fixBlock.h });
        aicT = fixStart + 1.25;
      }

      for (const hb of [0, 1]) {
        const slot = hb;
        const c2v = c2vReady.get(hb);
        const startT = Math.max(aiv0T, c2v.t);
        const popBlock = register({
          uid: `aiv0:mte2-pop:hb${hb}`,
          core: 'aiv0',
          pipe: 'mte2',
          stage: 'stats',
          hb,
          slot,
          kind: 'dma-block',
          x: tToX(startT),
          y: yFor(cores[1], 'mte2'),
          w: uToW(0.95) - 4,
          h: hFor(cores[1]),
          title: `MTE2 tpop C2V`,
          subtitle: `Slot ${slot} + Bias`,
          syncIn: 'C2V_PIPE',
          sourceLine: 284,
          debugTip: `AIV 0 MTE2 从 C2V Slot ${slot} 读取 Scores [32, 128] 并立刻 tfree_to_aic，随后送入 Vector。`,
        });
        const vecStart = startT + 1.0;
        register({
          uid: `aiv0:vec-sm:hb${hb}`,
          core: 'aiv0',
          pipe: 'vec',
          stage: 'stats',
          hb,
          slot,
          kind: 'op-pill',
          icon: 'softmax',
          x: tToX(vecStart),
          y: yFor(cores[1], 'vec'),
          w: uToW(2.2) - 4,
          h: hFor(cores[1]),
          title: `Vec Scale/Softmax`,
          subtitle: `HB${hb} · m/exp/l/cast`,
          sourceLine: 285,
          debugTip: `Vector 单元完成 muls、col_expand_add、row_max、exp、row_sum 与 BF16 cast。`,
        });
        const mte3Start = vecStart + 2.26;
        const pushBlock = register({
          uid: `aiv0:mte3-push:hb${hb}`,
          core: 'aiv0',
          pipe: 'mte3',
          stage: 'write',
          hb,
          slot,
          kind: 'dma-block',
          x: tToX(mte3Start),
          y: yFor(cores[1], 'mte3'),
          w: uToW(1.25) - 4,
          h: hFor(cores[1]),
          title: `MTE3 tpush + m/l`,
          subtitle: `V2C Slot ${slot}`,
          syncOut: 'V2C_PIPE',
          sourceLine: 294,
          debugTip: `AIV 0 MTE3 将 BF16 exp 写入 V2C Slot ${slot}，同时写出两段 H_TILE=16 的 sparse_blk_mi / li。`,
        });

        // AIV 1 companion pipes
        register({
          uid: `aiv1:mte2:hb${hb}`,
          core: 'aiv1',
          pipe: 'mte2',
          stage: 'stats',
          hb,
          slot,
          kind: 'dma-block',
          x: tToX(startT),
          y: yFor(cores[2], 'mte2'),
          w: uToW(0.95) - 4,
          h: hFor(cores[2]),
          title: `tpop (0行)`,
          subtitle: `Slot ${slot}`,
          sourceLine: 284,
          debugTip: 'AIV 1 伴随核执行 valid_shape=[0,0] 的管道握手。',
        });
        register({
          uid: `aiv1:mte3:hb${hb}`,
          core: 'aiv1',
          pipe: 'mte3',
          stage: 'write',
          hb,
          slot,
          kind: 'dma-block',
          x: tToX(mte3Start),
          y: yFor(cores[2], 'mte3'),
          w: uToW(1.1) - 4,
          h: hFor(cores[2]),
          title: `tpush (0行)`,
          subtitle: `Slot ${slot}`,
          sourceLine: 294,
          debugTip: 'AIV 1 伴随核执行 valid_shape=[0,0] 的 V2C 管道握手。',
        });

        syncEdges.push({
          id: `sync:C2V:hb${hb}`,
          event: 'C2V_PIPE',
          slot,
          sourceCore: 'aic',
          from: { x: c2v.x, y: c2v.y },
          to: { x: popBlock.x + 8, y: popBlock.y },
          busY: highwayTop + 8 + slot * 12,
          color: EVENT_META.C2V_PIPE.color,
        });

        v2cReady.set(hb, {
          t: mte3Start + 1.32,
          x: pushBlock.x + pushBlock.w - 8,
          y: cores[1].top + 6,
        });
        aiv0T = mte3Start + 1.45;
      }

      for (const hb of [0, 1]) {
        const slot = hb;
        const v2c = v2cReady.get(hb);
        const startT = Math.max(aicT, v2c.t);
        const popV2c = register({
          uid: `aic:mte2-pv:hb${hb}`,
          core: 'aic',
          pipe: 'mte2',
          stage: 'pv',
          hb,
          slot,
          kind: 'dma-block',
          x: tToX(startT),
          y: yFor(cores[0], 'mte2'),
          w: uToW(1.0) - 4,
          h: hFor(cores[0]),
          title: `MTE2 tpop V2C`,
          subtitle: `Slot ${slot} · exp BF16`,
          syncIn: 'V2C_PIPE',
          sourceLine: 295,
          debugTip: `AIC MTE2 从 V2C Slot ${slot} 取出 BF16 exp [32,128]，与常驻 L1 的 KV [128,512] 一起喂给 Cube PV。`,
        });
        const cubeStart = startT + 1.06;
        register({
          uid: `aic:cube-pv:hb${hb}`,
          core: 'aic',
          pipe: 'cube',
          stage: 'pv',
          hb,
          slot,
          kind: 'op-pill',
          icon: 'matmul',
          x: tToX(cubeStart),
          y: yFor(cores[0], 'cube'),
          w: uToW(2.35) - 4,
          h: hFor(cores[0]),
          title: `Cube PV · HB${hb}`,
          subtitle: '4×K=32 · L1 V复用',
          sourceLine: 295,
          debugTip: `Cube 单元 4×K=32 累加产出 Partial O FP32 [32, 512]，结束后调用 tfree_to_aiv。`,
        });
        const fixStart = cubeStart + 2.42;
        register({
          uid: `aic:fix-o:hb${hb}`,
          core: 'aic',
          pipe: 'fix',
          stage: 'write',
          hb,
          slot,
          kind: 'dma-block',
          x: tToX(fixStart),
          y: yFor(cores[0], 'fix'),
          w: uToW(1.2) - 4,
          h: hFor(cores[0]),
          title: `FIX Store O`,
          subtitle: '2×[16, 512] FP32',
          sourceLine: 297,
          debugTip: `FIX 通路将 Partial O 拆为两段 [16, 512] 写入 GM sparse_blk_oi。`,
        });

        syncEdges.push({
          id: `sync:V2C:hb${hb}`,
          event: 'V2C_PIPE',
          slot,
          sourceCore: 'aiv0',
          from: { x: v2c.x, y: v2c.y },
          to: { x: popV2c.x + 8, y: cores[0].top + cores[0].height - 8 },
          busY: highwayTop + 28 + slot * 10,
          color: EVENT_META.V2C_PIPE.color,
        });

        if (hb === 0) {
          reuseArcs.push({
            id: 'reuse:l1-kv',
            slot: 0,
            fromX: kvReuseOriginX,
            toX: popV2c.x + 12,
            y: cores[0].top - 6,
            label: 'L1 KV [128,512] 复用',
            color: '#F2C45F',
          });
        }
        aicT = fixStart + 1.38;
      }
    }

    const maxT = Math.max(aicT, aiv0T) + 0.6;
    return {
      width: tToX(maxT) + 48,
      height: totalHeight,
      labelW,
      xOrigin,
      maxT,
      cores,
      blocks,
      groups,
      syncEdges,
      reuseArcs,
    };
  }

  function renderSvg(layout, layers, selection) {
    const { width, height, labelW, xOrigin, maxT, cores, blocks, groups, syncEdges, reuseArcs } = layout;
    const marks = [];
    const gridlines = [];
    for (let u = 0; u <= Math.ceil(maxT); u += 2) {
      const x = Math.round(xOrigin + u * UNIT_W);
      marks.push(`<span class="viz2-pi-axis__mark" style="left:${x - xOrigin}px">u${u}</span>`);
      gridlines.push(`<i class="viz2-pi-gridline" style="left:${x}px;top:${HEADER_H - 12}px;height:${height - HEADER_H + 14}px"></i>`);
    }

    const axisHtml = `<div class="viz2-pi-axis" style="left:${xOrigin}px;width:${width - xOrigin - 24}px">
      <div class="viz2-pi-axis__title">理论因果逻辑时间（按算子单元推进，非实测耗时） →</div>
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
      const badgeCls = core.id === 'aiv1' ? 'viz2-pi-core-badge is-imbalance' : 'viz2-pi-core-badge';
      return `<div class="viz2-pi-core-row" data-core="${core.id}" style="top:${core.top}px;height:${core.height}px;width:${width - 16}px">
        <div class="viz2-pi-core-label" style="width:${labelW}px">
          <div class="viz2-pi-core-main">
            <div class="viz2-pi-core-title"><strong>${esc(core.title)}</strong><span class="${badgeCls}">${esc(core.badge)}</span></div>
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
        b.core === 'aiv1' ? 'is-tail' : '',
      ].filter(Boolean).join(' ');
      const dot = b.syncOut && layers.sync && EVENT_META[b.syncOut]
        ? `<i class="viz2-pi-ev-dot" style="--ev-color:${EVENT_META[b.syncOut].color}"></i>`
        : '';
      return `<button type="button" class="${cls}" data-csa-pi-uid="${esc(b.uid)}" data-pipe="${esc(b.pipe || '')}" style="left:${b.x}px;top:${b.y}px;width:${b.w}px;height:${b.h}px" title="${esc(b.debugTip)}">
        <span class="viz2-pi-event__head">${b.icon ? opIcon(b.icon) : ''}<b>${esc(b.title)}</b>${dot}</span>
        <small>${esc(b.subtitle)}</small>
      </button>`;
    }).join('');

    let syncSvg = '';
    if (layers.sync && syncEdges.length) {
      const paths = syncEdges.map(edge => {
        const active = selectedSync === edge.id;
        const d = `M ${edge.from.x} ${edge.from.y} V ${edge.busY} H ${edge.to.x} V ${edge.to.y}`;
        return `<g class="viz2-pi-sync-layer" data-csa-pi-sync="${esc(edge.id)}" style="pointer-events:auto;cursor:pointer">
          <path d="${d}" fill="none" stroke="transparent" stroke-width="12"/>
          <path class="viz2-pi-sync-arc${active ? ' is-related' : ''}" d="${d}" stroke="${edge.color}" marker-end="url(#csa-pi-arrow-${edge.event})"/>
          <circle cx="${edge.from.x}" cy="${edge.from.y}" r="3.5" fill="${edge.color}"/>
        </g>`;
      }).join('');
      syncSvg = `<svg class="viz2-pi-overlay" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
        <defs>
          <marker id="csa-pi-arrow-C2V_PIPE" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#80B7F7"/></marker>
          <marker id="csa-pi-arrow-V2C_PIPE" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#9EF04B"/></marker>
        </defs>
        ${paths}
      </svg>`;
    }

    let reuseSvg = '';
    if (layers.slot && reuseArcs.length) {
      const arcs = reuseArcs.map(arc => {
        const midX = (arc.fromX + arc.toX) / 2;
        const topY = arc.y - 22;
        const d = `M ${arc.fromX} ${arc.y} C ${arc.fromX} ${topY}, ${arc.toX} ${topY}, ${arc.toX} ${arc.y}`;
        return `<g class="viz2-pi-reuse-layer" style="--slot-color:${arc.color}">
          <path class="viz2-pi-reuse-arc" d="${d}"/>
          <text x="${midX}" y="${topY - 4}" text-anchor="middle" fill="${arc.color}" font-size="11">${esc(arc.label)}</text>
        </g>`;
      }).join('');
      reuseSvg = `<svg class="viz2-pi-overlay" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">${arcs}</svg>`;
    }

    return `${axisHtml}${lanesHtml}${groupsHtml}${blocksHtml}${syncSvg}${reuseSvg}`;
  }

  function inspectorHtml(selection, layout, item, detail) {
    if (!selection) return '';
    if (selection.kind === 'sync') {
      const edge = layout.syncEdges.find(e => e.id === selection.id);
      const meta = edge ? EVENT_META[edge.event] : null;
      if (!meta) return '';
      return `<header class="pto-ide-frame__pane-header">
        <div><b>${esc(meta.name)}</b><small>Slot ${edge.slot} · 跨核环形管道</small></div>
        <button class="btn btn-ghost btn-sm" data-csa-pi-close aria-label="关闭详情">×</button>
      </header>
      <div class="pto-ide-frame__pane-body">
        <p>${esc(meta.debugTip)}</p>
        <dl class="viz2-pi-dl">
          <dt>Producer</dt><dd>${esc(meta.producerCore)} · ${esc(meta.producerPipe)}</dd>
          <dt>Consumer</dt><dd>${esc(meta.consumerCore)} · ${esc(meta.consumerPipe)}</dd>
          <dt>Pipe Buffer</dt><dd>${esc(meta.buffer)}</dd>
          <dt>编译依据</dt><dd>25_after_SplitVectorKernel.py · dir_mask=3, slot_num=2</dd>
        </dl>
        <button type="button" class="btn btn-ghost btn-sm" data-csa-pi-source="${SUB_FILE}:${meta.lineSet}:${meta.lineWait}">查看源码 · ${SUB_FILE} L${meta.lineSet}–${meta.lineWait} ↗</button>
      </div>`;
    }
    const block = layout.blocks.find(b => b.uid === selection.uid);
    if (!block) return '';
    const ref = measuredRef();
    const measuredLine = ref.aic
      ? `整核 24 组实测：AIC 中位 ${ref.aic.medDurUs} µs（最大 ${ref.aic.maxDurUs} µs） · AIV 中位 ${ref.aiv?.medDurUs ?? '—'} µs`
      : '无实测数据';
    return `<header class="pto-ide-frame__pane-header">
      <div><b>${esc(block.title)}</b><small>${esc(block.core.toUpperCase())} · Work Item ${item.id} · Q${item.token} / SB${item.sparseBlock}</small></div>
      <button class="btn btn-ghost btn-sm" data-csa-pi-close aria-label="关闭详情">×</button>
    </header>
    <div class="pto-ide-frame__pane-body">
      <p>${esc(block.debugTip)}</p>
      <dl class="viz2-pi-dl">
        <dt>硬件单元</dt><dd>${esc(block.core.toUpperCase())}${block.pipe ? ' · ' + esc(block.pipe.toUpperCase()) : ''}</dd>
        <dt>管道槽位</dt><dd>Slot ${block.slot ?? 0}${block.hb != null ? ` · Head Batch ${block.hb}` : ''}</dd>
        <dt>队列位置</dt><dd>qk_order[${detail.queuePosition}] · Lane ${detail.block} · Iter ${detail.iteration}</dd>
        <dt>DFX 实测参考</dt><dd>${esc(measuredLine)}</dd>
      </dl>
      ${block.sourceLine ? `<button type="button" class="btn btn-ghost btn-sm" data-csa-pi-source="${SUB_FILE}:${Math.max(1, block.sourceLine - 2)}:${block.sourceLine + 12}">查看源码 · ${SUB_FILE} L${block.sourceLine} ↗</button>` : ''}
    </div>`;
  }

  function mount(host, options) {
    const layers = { op: true, pipe: false, sync: true, slot: true };
    let currentItem = null;
    let currentDetail = null;
    let layout = null;
    let selection = null;
    let cam = { x: 16, y: 12, scale: 0.85 };
    let drag = null;

    host.innerHTML = `<div class="viz2-pipeline-instance">
      <div class="viz2-pi-toolbar">
        <div class="viz2-pi-toolbar__left">
          <span class="viz2-pi-summary" data-csa-pi-summary></span>
          <div class="viz2-pi-layers" role="group" aria-label="因果泳道图层">
            <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn is-selected" data-csa-pi-mode="op" aria-pressed="true">Op</button>
            <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn" data-csa-pi-mode="pipe" aria-pressed="false">Pipe</button>
            <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn is-selected" data-csa-pi-toggle="sync" aria-pressed="true">Sync</button>
            <button type="button" class="btn btn-ghost btn-sm viz2-pi-layer-btn is-selected" data-csa-pi-toggle="slot" aria-pressed="true">Slot 复用</button>
          </div>
        </div>
        <div class="viz2-pi-tools">
          <button type="button" class="btn btn-ghost btn-sm" data-csa-pi-zoom="0.85" aria-label="缩小">−</button>
          <output data-csa-pi-scale>85%</output>
          <button type="button" class="btn btn-ghost btn-sm" data-csa-pi-zoom="1.18" aria-label="放大">＋</button>
          <button type="button" class="btn btn-ghost btn-sm" data-csa-pi-fit>Fit to View</button>
        </div>
      </div>
      <div class="viz2-pi-boundary-note" data-csa-pi-note></div>
      <div class="viz2-pi-viewport" tabindex="0" role="region" aria-label="CSA qk_pv AIC/AIV 因果执行泳道">
        <div class="viz2-pi-scene" data-csa-pi-scene></div>
      </div>
      <aside class="pto-ide-frame__pane viz2-pi-inspector" data-csa-pi-inspector hidden></aside>
    </div>`;

    const summaryEl = host.querySelector('[data-csa-pi-summary]');
    const noteEl = host.querySelector('[data-csa-pi-note]');
    const viewport = host.querySelector('.viz2-pi-viewport');
    const scene = host.querySelector('[data-csa-pi-scene]');
    const scaleOut = host.querySelector('[data-csa-pi-scale]');
    const inspector = host.querySelector('[data-csa-pi-inspector]');

    function applyCamera() {
      scene.style.transform = `translate(${cam.x}px, ${cam.y}px) scale(${cam.scale})`;
      scaleOut.textContent = `${Math.round(cam.scale * 100)}%`;
    }

    function fit() {
      if (!layout || !viewport.clientWidth || !viewport.clientHeight) return;
      const sx = (viewport.clientWidth - 28) / layout.width;
      const sy = (viewport.clientHeight - 28) / layout.height;
      const s = Math.max(0.35, Math.min(1.15, Math.min(sx, sy)));
      cam = {
        scale: s,
        x: Math.max(12, Math.round((viewport.clientWidth - layout.width * s) / 2)),
        y: Math.max(10, Math.round((viewport.clientHeight - layout.height * s) / 2)),
      };
      applyCamera();
    }

    function syncInspector() {
      if (!selection || !layout || !currentItem) {
        inspector.hidden = true;
        return;
      }
      inspector.innerHTML = inspectorHtml(selection, layout, currentItem, currentDetail);
      inspector.hidden = false;
    }

    function redraw(refit = false) {
      if (!currentItem || !currentDetail) return;
      layout = buildLayout(currentItem, currentDetail, layers);
      const ref = measuredRef();
      summaryEl.textContent = `AIC / AIV 因果泳道 · Work Item ${currentItem.id} (Q${currentItem.token} / SB${currentItem.sparseBlock}) · ${currentItem.valid ? '2 Head Batches · C2V/V2C 双槽' : 'Invalid 中性分支'}`;
      noteEl.textContent = `理论因果执行序 · 同核严格单行串行（单次 qk_pv SPMD 24 组 DFX 实测：AIC 中位 ${ref.aic?.medDurUs ?? '783.2'} µs / AIV 中位 ${ref.aiv?.medDurUs ?? '782.3'} µs；核内子操作按逻辑单元推进，真实气泡需对接核内仿真 Trace）。`;
      scene.style.width = `${layout.width}px`;
      scene.style.height = `${layout.height}px`;
      scene.innerHTML = renderSvg(layout, layers, selection);
      syncInspector();
      if (refit) requestAnimationFrame(fit);
      else applyCamera();
    }

    host.addEventListener('click', event => {
      const modeBtn = event.target.closest('[data-csa-pi-mode]');
      if (modeBtn) {
        layers.pipe = modeBtn.dataset.csaPiMode === 'pipe';
        layers.op = !layers.pipe;
        host.querySelectorAll('[data-csa-pi-mode]').forEach(b => {
          const on = (b.dataset.csaPiMode === 'pipe') === layers.pipe;
          b.classList.toggle('is-selected', on);
          b.setAttribute('aria-pressed', String(on));
        });
        redraw(true);
        return;
      }
      const toggleBtn = event.target.closest('[data-csa-pi-toggle]');
      if (toggleBtn) {
        const key = toggleBtn.dataset.csaPiToggle;
        layers[key] = !layers[key];
        toggleBtn.classList.toggle('is-selected', layers[key]);
        toggleBtn.setAttribute('aria-pressed', String(layers[key]));
        redraw(false);
        return;
      }
      const zoomBtn = event.target.closest('[data-csa-pi-zoom]');
      if (zoomBtn) {
        cam.scale = Math.max(0.3, Math.min(2.4, cam.scale * Number(zoomBtn.dataset.csaPiZoom)));
        applyCamera();
        return;
      }
      if (event.target.closest('[data-csa-pi-fit]')) {
        fit();
        return;
      }
      if (event.target.closest('[data-csa-pi-close]')) {
        selection = null;
        redraw(false);
        return;
      }
      const srcBtn = event.target.closest('[data-csa-pi-source]');
      if (srcBtn) {
        const [file, a, z] = srcBtn.dataset.csaPiSource.split(':');
        options.onSource?.(file, Number(a), Number(z));
        return;
      }
      const blk = event.target.closest('[data-csa-pi-uid]');
      if (blk) {
        const uid = blk.dataset.csaPiUid;
        selection = selection?.uid === uid ? null : { kind: 'block', uid };
        redraw(false);
        return;
      }
      const syncEl = event.target.closest('[data-csa-pi-sync]');
      if (syncEl) {
        const id = syncEl.dataset.csaPiSync;
        selection = selection?.id === id ? null : { kind: 'sync', id };
        redraw(false);
      }
    });

    viewport.addEventListener('pointerdown', event => {
      if (event.button !== 0 || event.target.closest('button,[data-csa-pi-sync]')) return;
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
      const next = Math.max(0.3, Math.min(2.4, cam.scale * Math.exp(Math.max(-0.25, Math.min(0.25, -event.deltaY * 0.002)))));
      const ratio = next / cam.scale;
      cam.x = mx - (mx - cam.x) * ratio;
      cam.y = my - (my - cam.y) * ratio;
      cam.scale = next;
      applyCamera();
    }, { passive: false });

    return {
      update(item, detail) {
        const changed = !currentItem || currentItem.id !== item.id || currentItem.valid !== item.valid;
        currentItem = item;
        currentDetail = detail;
        if (changed) selection = null;
        redraw(true);
      },
      fit,
    };
  }

  global.PtoCsaQkPipelineInstance = Object.freeze({ mount });
})(window);
