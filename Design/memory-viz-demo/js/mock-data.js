/*
 * Mock data for the Ascend Memory Studio UX demo.
 *
 * Everything here is fabricated to exercise the UI. Shapes match:
 *   - the PtoMemoryReuseViewer schema (buffers / ticks / tensors / sourceFiles)
 *   - extra domain metadata the product page owns (diagnostics, runtime trace)
 *
 * Real integration would derive `memory` from AllocateMemoryAddr + LifetimeAnalyzer
 * and `runtime` from msprof / runtime DFX traces.
 */
(function (global) {
  'use strict';

  const KB = 1024;

  // On-chip buffer capacities (bytes) — realistic Ascend 910B-ish figures.
  const CAP = {
    UB: 256 * KB, // Vec / unified buffer
    L1: 512 * KB, // Mat
    L0A: 64 * KB, // Left
    L0B: 64 * KB, // Right
    L0C: 128 * KB, // Acc
  };

  const SRC_MATMUL = {
    name: 'matmul_add_relu.cce',
    content:
      /*  1 */ '// Fused kernel:  Y = ReLU( X @ W + bias )   (fp16 in / fp16 out)\n' +
      /*  2 */ '// Target: Ascend AI Core (Cube + Vector), mix aic/aiv pipeline\n' +
      /*  3 */ '#include "kernel_operator.h"\n' +
      /*  4 */ 'using namespace AscendC;\n' +
      /*  5 */ '\n' +
      /*  6 */ 'constexpr int M = 256, K = 512, N = 256;\n' +
      /*  7 */ 'constexpr int TILE_M = 128, TILE_N = 128, TILE_K = 256;\n' +
      /*  8 */ '\n' +
      /*  9 */ 'class KernelMatMulAddRelu {\n' +
      /* 10 */ '  __aicore__ inline void Process() {\n' +
      /* 11 */ '    // ---- stage weight tile GM -> L1 (wL1_stage, reused across loops) ----\n' +
      /* 12 */ '    DataCopy(wL1, wGm, {nBlk, blockLen, 0, 0});\n' +
      /* 13 */ '    // ---- stage next activation into SAME L1 region (aL1_stage) ----\n' +
      /* 14 */ '    DataCopy(aL1, aGm, {1, blockLen, 0, 0});\n' +
      /* 15 */ '    // ---- load activation tile GM -> UB (xGm_in) ----\n' +
      /* 16 */ '    LocalTensor<half> xUb = inQueueX.AllocTensor<half>();\n' +
      /* 17 */ '    DataCopy(xUb, xGm[offset], {1, blockLen, 0, 0});\n' +
      /* 18 */ '    xUb = inQueueX.DeQue<half>();\n' +
      /* 19 */ '    // ---- load weight tile GM -> UB (wGm_in) ----\n' +
      /* 20 */ '    LocalTensor<half> wUb = inQueueW.AllocTensor<half>();\n' +
      /* 21 */ '    DataCopy(wUb, wGm[woff], {nBlk, blockLen, 0, 0});\n' +
      /* 22 */ '    wUb = inQueueW.DeQue<half>();\n' +
      /* 23 */ '    // ---- MMAD: accumulate X @ W into L0C (cMatrix) ----\n' +
      /* 24 */ '    Mmad(cMatrix, xUb, wUb, {M, K, N, true});\n' +
      /* 25 */ '    // ---- load bias vector into UB (biasUb) ----\n' +
      /* 26 */ '    LocalTensor<float> biasUb = biasBuf.Get<float>();\n' +
      /* 27 */ '    DataCopy(biasUb, biasGm, n);\n' +
      /* 28 */ '    // ---- copy L0C -> UB, then vector Add bias (cUb_fp32) ----\n' +
      /* 29 */ '    LocalTensor<float> cUb = ubBuf.Get<float>();\n' +
      /* 30 */ '    DataCopy(cUb, cMatrix, {TILE_M, TILE_N, 0, 0});\n' +
      /* 31 */ '    Add(cUb, cUb, biasUb, mask);\n' +
      /* 32 */ '    // ---- in-place ReLU (reuses wGm_in weight region, reluUb) ----\n' +
      /* 33 */ '    Relu(reluUb, cUb);\n' +
      /* 34 */ '    // ---- cast fp32 -> fp16 and store to GM (yUb_out reuses xGm_in) ----\n' +
      /* 35 */ '    LocalTensor<half> yUb = outBuf.Get<half>();\n' +
      /* 36 */ '    Cast(yUb, reluUb, RoundMode::CAST_RINT, len);\n' +
      /* 37 */ '    DataCopy(yGm[offset], yUb, {1, blockLen, 0, 0});\n' +
      /* 38 */ '    // ---- per-loop scratch scaling (scratchUb, reuses cUb_fp32) ----\n' +
      /* 39 */ '    for (int i = 0; i < nLoop; ++i) { Muls(scratch, yUb, s); }\n' +
      /* 40 */ '  }\n' +
      /* 41 */ '};\n',
  };

  // Tensors for the healthy MatMulAddRelu kernel (fits comfortably in UB).
  function matmulTensors() {
    return [
      {
        id: 't1', name: 'xGm_in', buffer: 'UB', offset: 0, size: 64 * KB,
        allocTick: 2, freeTick: 40, kind: 'resident', reuseOf: null, reusedBy: ['t7'],
        srcFile: SRC_MATMUL.name, srcLineStart: 15, srcLineEnd: 18, srcHotLine: 17,
        code: 'LocalTensor<half> xUb = inQueueX.AllocTensor<half>();\nDataCopy(xUb, xGm[offset], {1, blockLen, 0, 0});',
        cce: '// xGm_in @UB+0x00000 64KB\nLD.global.b128 %ub0,[%gm_x+0]\nWAIT_FLAG MTE2->M\nMMAD %l0c0,%ub0,%ub_w',
      },
      {
        id: 't2', name: 'wGm_in', buffer: 'UB', offset: 64 * KB, size: 96 * KB,
        allocTick: 3, freeTick: 58, kind: 'resident', reuseOf: null, reusedBy: ['t6'],
        srcFile: SRC_MATMUL.name, srcLineStart: 19, srcLineEnd: 22, srcHotLine: 21,
        code: 'LocalTensor<half> wUb = inQueueW.AllocTensor<half>();\nDataCopy(wUb, wGm[woff], {nBlk, blockLen, 0, 0});',
        cce: '// wGm_in @UB+0x10000 96KB\nLD.global.b128 %ub_w,[%gm_w+0]',
      },
      {
        id: 't3', name: 'cMatrix', buffer: 'L0C', offset: 0, size: 128 * KB,
        allocTick: 6, freeTick: 74, kind: 'resident', reuseOf: null, reusedBy: [],
        srcFile: SRC_MATMUL.name, srcLineStart: 23, srcLineEnd: 24, srcHotLine: 24,
        code: 'Mmad(cMatrix, xUb, wUb, {M, K, N, true});',
        cce: '// cMatrix @L0C+0x0 128KB\nMMAD %l0c0,%ub0,%ub_w,init=1',
      },
      {
        id: 't4', name: 'biasUb', buffer: 'UB', offset: 160 * KB, size: 16 * KB,
        allocTick: 10, freeTick: 30, kind: 'temp', reuseOf: null, reusedBy: [],
        srcFile: SRC_MATMUL.name, srcLineStart: 25, srcLineEnd: 27, srcHotLine: 26,
        code: 'LocalTensor<float> biasUb = biasBuf.Get<float>();\nDataCopy(biasUb, biasGm, n);',
        cce: '// biasUb @UB+0x28000 16KB\nLD.global.b32 %ub_b,[%gm_bias]',
      },
      {
        id: 't5', name: 'cUb_fp32', buffer: 'UB', offset: 176 * KB, size: 64 * KB,
        allocTick: 62, freeTick: 92, kind: 'temp', reuseOf: null, reusedBy: ['t8'],
        srcFile: SRC_MATMUL.name, srcLineStart: 28, srcLineEnd: 31, srcHotLine: 30,
        code: 'LocalTensor<float> cUb = ubBuf.Get<float>();\nDataCopy(cUb, cMatrix, {TILE_M, TILE_N, 0, 0});\nAdd(cUb, cUb, biasUb, mask);',
        cce: '// cUb_fp32 @UB+0x2c000 64KB\nWAIT_FLAG M->V\nDataCopy %ub_c,%l0c0\nVADD %ub_c,%ub_c,%ub_b',
      },
      {
        id: 't6', name: 'reluUb', buffer: 'UB', offset: 64 * KB, size: 64 * KB,
        allocTick: 78, freeTick: 104, kind: 'temp', reuseOf: 't2', reusedBy: [],
        srcFile: SRC_MATMUL.name, srcLineStart: 32, srcLineEnd: 33, srcHotLine: 33,
        code: 'Relu(reluUb, cUb);\n// reluUb address reuses wGm_in (freed at tick #58)',
        cce: '// reluUb @UB+0x10000 64KB (reuse of wGm_in)\nVMAX %ub_r,%ub_c,#0',
      },
      {
        id: 't7', name: 'yUb_out', buffer: 'UB', offset: 0, size: 64 * KB,
        allocTick: 94, freeTick: 118, kind: 'temp', reuseOf: 't1', reusedBy: [],
        srcFile: SRC_MATMUL.name, srcLineStart: 34, srcLineEnd: 37, srcHotLine: 36,
        code: 'LocalTensor<half> yUb = outBuf.Get<half>();\nCast(yUb, reluUb, RoundMode::CAST_RINT, len);\nDataCopy(yGm[offset], yUb, {1, blockLen, 0, 0});',
        cce: '// yUb_out @UB+0x0 64KB (reuse of xGm_in)\nVCONV %ub_y,%ub_r,f32->f16\nST.global.b128 [%gm_y],%ub_y',
      },
      {
        id: 't8', name: 'scratchUb', buffer: 'UB', offset: 176 * KB, size: 32 * KB,
        allocTick: 96, freeTick: 112, kind: 'loop', reuseOf: 't5', reusedBy: [],
        srcFile: SRC_MATMUL.name, srcLineStart: 38, srcLineEnd: 39, srcHotLine: 39,
        code: 'for (int i = 0; i < nLoop; i++) { Muls(scratch, yUb, s); }',
        cce: '// scratchUb @UB+0x2c000 32KB (reuse cUb)\nLOOP i\n  VMULS %ub_s,%ub_y,%scale\nENDLOOP',
      },
      {
        id: 't9', name: 'wL1_stage', buffer: 'L1', offset: 0, size: 256 * KB,
        allocTick: 1, freeTick: 60, kind: 'resident', reuseOf: null, reusedBy: ['t10'],
        srcFile: SRC_MATMUL.name, srcLineStart: 11, srcLineEnd: 12, srcHotLine: 12,
        code: 'DataCopy(wL1, wGm, {...});  // stage weights GM -> L1',
        cce: '// wL1_stage @L1+0x0 256KB\nLD.global %l1_w,[%gm_w]',
      },
      {
        id: 't10', name: 'aL1_stage', buffer: 'L1', offset: 0, size: 256 * KB,
        allocTick: 64, freeTick: 110, kind: 'loop', reuseOf: 't9', reusedBy: [],
        srcFile: SRC_MATMUL.name, srcLineStart: 13, srcLineEnd: 14, srcHotLine: 14,
        code: 'DataCopy(aL1, aGm, {...});  // stage next activation into same L1 region',
        cce: '// aL1_stage @L1+0x0 256KB (reuse wL1_stage)\nLD.global %l1_a,[%gm_a]',
      },
    ];
  }

  // Baseline for the diff scenario: reluUb does NOT reuse wGm_in — it gets a
  // fresh UB slot beyond the current high-water, pushing the footprint past the
  // 256KB capacity. This is exactly the allocation footprint MemoryReuse saves:
  // the reuse turns a UB overflow (304KB) into a fit (240KB).
  function matmulBaselineTensors() {
    const ts = matmulTensors();
    const relu = ts.find((t) => t.id === 't6');
    relu.reuseOf = null;
    relu.offset = 240 * KB; // fresh slot @0x3C000 → end 0x4C000 (304KB) > 256KB cap
    relu.overflow = true;
    relu.code = 'Relu(reluUb, cUb);\n// no reuse: reluUb gets a fresh UB slot (overflow!)';
    relu.cce = '// reluUb @UB+0x3C000 64KB (fresh) << past 0x40000 cap\nVMAX %ub_r,%ub_c,#0';
    // wGm_in is no longer reused by t6
    ts.find((t) => t.id === 't2').reusedBy = [];
    return ts;
  }

  const SRC_ATTN = {
    name: 'fused_attention.cce',
    content:
      '// Fused attention:  O = softmax(Q @ K^T / sqrt(d)) @ V\n' +
      '#include "kernel_operator.h"\n' +
      'using namespace AscendC;\n' +
      '\n' +
      'class KernelFusedAttention {\n' +
      '  __aicore__ inline void Process() {\n' +
      '    // stage Q, K, V tiles into UB\n' +
      '    LocalTensor<half> qUb = qBuf.Get<half>();   // 64KB\n' +
      '    LocalTensor<half> kUb = kBuf.Get<half>();   // 64KB\n' +
      '    // QK^T scores accumulate into UB (128KB)\n' +
      '    Mmad(scores, qUb, kUb, {S, D, S, true});\n' +
      '    // softmax needs a full-row temp alongside scores\n' +
      '    LocalTensor<float> smTmp = smBuf.Get<float>();  // 48KB — overflows UB!\n' +
      '    Softmax(probs, scores, smTmp);\n' +
      '    // PV: probs @ V\n' +
      '    Mmad(out, probs, vUb, {S, S, D, true});\n' +
      '  }\n' +
      '};\n',
  };

  // Attention kernel: engineered so the UB high-water exceeds capacity, and the
  // L0B double-buffer depth is capped by capacity (PH-MR-001).
  function attentionTensors() {
    return [
      {
        id: 'a1', name: 'qUb', buffer: 'UB', offset: 0, size: 64 * KB,
        allocTick: 2, freeTick: 60, kind: 'resident', reuseOf: null, reusedBy: [],
        srcFile: SRC_ATTN.name, srcLineStart: 8, srcLineEnd: 8, srcHotLine: 8,
        code: 'LocalTensor<half> qUb = qBuf.Get<half>();  // 64KB',
        cce: '// qUb @UB+0x0 64KB\nLD.global %ub_q,[%gm_q]',
      },
      {
        id: 'a2', name: 'kUb', buffer: 'UB', offset: 64 * KB, size: 64 * KB,
        allocTick: 3, freeTick: 40, kind: 'resident', reuseOf: null, reusedBy: [],
        srcFile: SRC_ATTN.name, srcLineStart: 9, srcLineEnd: 9, srcHotLine: 9,
        code: 'LocalTensor<half> kUb = kBuf.Get<half>();  // 64KB',
        cce: '// kUb @UB+0x10000 64KB\nLD.global %ub_k,[%gm_k]',
      },
      {
        id: 'a3', name: 'scores', buffer: 'UB', offset: 128 * KB, size: 128 * KB,
        allocTick: 12, freeTick: 78, kind: 'resident', reuseOf: null, reusedBy: [],
        srcFile: SRC_ATTN.name, srcLineStart: 10, srcLineEnd: 11, srcHotLine: 11,
        code: 'Mmad(scores, qUb, kUb, {S, D, S, true});',
        cce: '// scores @UB+0x20000 128KB\nMMAD %ub_sc,%ub_q,%ub_k',
      },
      {
        // OVERFLOW: offset 224KB + 48KB = 272KB > 256KB cap.
        id: 'a4', name: 'smTmp', buffer: 'UB', offset: 224 * KB, size: 48 * KB,
        allocTick: 40, freeTick: 70, kind: 'temp', reuseOf: null, reusedBy: [], overflow: true,
        srcFile: SRC_ATTN.name, srcLineStart: 12, srcLineEnd: 14, srcHotLine: 13,
        code: 'LocalTensor<float> smTmp = smBuf.Get<float>();  // 48KB — overflows UB!\nSoftmax(probs, scores, smTmp);',
        cce: '// smTmp @UB+0x38000 48KB  << past 0x40000 capacity!\nVEXP %ub_t,%ub_sc',
      },
      {
        id: 'a5', name: 'qL0B', buffer: 'L0B', offset: 0, size: 32 * KB,
        allocTick: 4, freeTick: 52, kind: 'loop', reuseOf: null, reusedBy: [],
        srcFile: SRC_ATTN.name, srcLineStart: 11, srcLineEnd: 11, srcHotLine: 11,
        code: 'DataCopy(kL0B, kUb, {...});  // K operand into L0B (stage=2 requested)',
        cce: '// qL0B @L0B+0x0 32KB\nLD %l0b,%ub_k',
      },
      {
        id: 'a6', name: 'vL0B', buffer: 'L0B', offset: 32 * KB, size: 32 * KB,
        allocTick: 54, freeTick: 96, kind: 'loop', reuseOf: 'a5', reusedBy: [],
        srcFile: SRC_ATTN.name, srcLineStart: 15, srcLineEnd: 16, srcHotLine: 16,
        code: 'Mmad(out, probs, vUb, {S, S, D, true});  // V operand reuses L0B slot',
        cce: '// vL0B @L0B+0x8000 32KB (reuse qL0B)\nLD %l0b1,%ub_v',
      },
      {
        id: 'a7', name: 'outAcc', buffer: 'L0C', offset: 0, size: 64 * KB,
        allocTick: 58, freeTick: 110, kind: 'resident', reuseOf: null, reusedBy: [],
        srcFile: SRC_ATTN.name, srcLineStart: 16, srcLineEnd: 16, srcHotLine: 16,
        code: 'Mmad(out, probs, vUb, {S, S, D, true});',
        cce: '// outAcc @L0C+0x0 64KB\nMMAD %l0c,%ub_p,%l0b1',
      },
    ];
  }

  // ---- runtime execution trace (P4) ------------------------------------
  // Lanes model the hardware pipes. Tasks carry start/dur in cycle units.
  function matmulRuntime() {
    const lanes = [
      { id: 'mte2', label: 'MTE2 · GM→L1/UB' },
      { id: 'cube', label: 'Cube · MMAD' },
      { id: 'vec', label: 'Vector · VADD/RELU' },
      { id: 'mte3', label: 'MTE3 · UB→GM' },
    ];
    const tasks = [
      { lane: 'mte2', op: 'DataCopy wL1', start: 2, dur: 22, status: 'ok', clc: 20, total: 22, in: 1, out: 0 },
      { lane: 'mte2', op: 'DataCopy xUb', start: 6, dur: 16, status: 'ok', clc: 15, total: 16, in: 1, out: 0 },
      { lane: 'mte2', op: 'DataCopy wUb', start: 24, dur: 16, status: 'ok', clc: 15, total: 16, in: 1, out: 0 },
      { lane: 'cube', op: 'MMAD k0', start: 26, dur: 24, status: 'ok', clc: 24, total: 24 },
      { lane: 'cube', op: 'MMAD k1', start: 50, dur: 24, status: 'ok', clc: 24, total: 24 },
      { lane: 'mte2', op: 'DataCopy bias', start: 44, dur: 8, status: 'ok', clc: 7, total: 8, in: 1, out: 0 },
      { lane: 'vec', op: 'VADD bias', start: 76, dur: 14, status: 'ok', clc: 13, total: 14 },
      { lane: 'vec', op: 'RELU', start: 90, dur: 12, status: 'ok', clc: 11, total: 12 },
      { lane: 'vec', op: 'CAST f32→f16', start: 102, dur: 10, status: 'ok', clc: 9, total: 10 },
      { lane: 'mte3', op: 'DataCopy yGm', start: 110, dur: 14, status: 'ok', clc: 13, total: 14, in: 0, out: 1 },
    ];
    return { cycles: 128, lanes, tasks, realPeakKB: 224, staticPeakKB: 224, overlapPct: 71, stalls: 1 };
  }

  function attentionRuntime() {
    const lanes = [
      { id: 'mte2', label: 'MTE2 · GM→UB' },
      { id: 'cube', label: 'Cube · MMAD' },
      { id: 'vec', label: 'Vector · Softmax' },
      { id: 'mte3', label: 'MTE3 · UB→GM' },
    ];
    const tasks = [
      { lane: 'mte2', op: 'DataCopy qUb', start: 2, dur: 18, status: 'ok', clc: 17, total: 18, in: 1, out: 0 },
      { lane: 'mte2', op: 'DataCopy kUb', start: 4, dur: 18, status: 'ok', clc: 17, total: 18, in: 1, out: 0 },
      { lane: 'cube', op: 'MMAD QK^T', start: 22, dur: 30, status: 'ok', clc: 30, total: 30 },
      // Stall: L0B double-buffer capped to depth 1, so PV MMAD waits for QK operand free.
      { lane: 'cube', op: 'MMAD PV (stall)', start: 74, dur: 28, status: 'stall', clc: 22, total: 28, gap: 6, gapRatio: 0.21 },
      { lane: 'vec', op: 'Softmax exp', start: 52, dur: 20, status: 'ok', clc: 19, total: 20 },
      { lane: 'vec', op: 'Softmax norm', start: 72, dur: 14, status: 'ok', clc: 13, total: 14 },
      { lane: 'mte3', op: 'DataCopy oGm', start: 104, dur: 16, status: 'ok', clc: 15, total: 16, in: 0, out: 1 },
    ];
    return { cycles: 128, lanes, tasks, realPeakKB: 272, staticPeakKB: 272, overlapPct: 43, stalls: 2 };
  }

  // ---- kernels registry -------------------------------------------------
  const kernels = {
    matmul_add_relu: {
      id: 'matmul_add_relu',
      title: 'MatMulAddRelu_mix_aic',
      subtitle: 'Y = ReLU(X @ W + bias)',
      health: 'ok',
      memory: {
        kernel: 'MatMulAddRelu_mix_aic__kernel0',
        ticks: 120,
        buffers: [
          { name: 'UB', capacity: CAP.UB },
          { name: 'L1', capacity: CAP.L1 },
          { name: 'L0C', capacity: CAP.L0C },
        ],
        sourceFiles: [SRC_MATMUL],
        tensors: matmulTensors(),
      },
      diagnostics: {
        pipeline: [
          { space: 'L0C', slot: '128KB', requested: 2, achieved: 2, ok: true, note: 'Acc 双缓冲装得下（dbC=2）' },
          { space: 'UB', slot: '64KB', requested: 2, achieved: 2, ok: true, note: 'Vec ping-pong 正常' },
        ],
        hints: [
          { level: 'success', code: 'MR-OK', msg: 'reluUb 复用 wGm_in 空槽，UB 峰值省 64KB', srcFile: SRC_MATMUL.name, srcLine: 33 },
          { level: 'info', code: 'MR-INFO', msg: 'yUb_out 复用 xGm_in、scratchUb 复用 cUb_fp32', srcFile: SRC_MATMUL.name, srcLine: 36 },
        ],
        overflow: null,
      },
      runtime: matmulRuntime(),
    },

    fused_attention: {
      id: 'fused_attention',
      title: 'FusedAttention_mix',
      subtitle: 'O = softmax(QKᵀ/√d) · V',
      health: 'error',
      memory: {
        kernel: 'FusedAttention_mix__kernel0',
        ticks: 120,
        buffers: [
          { name: 'UB', capacity: CAP.UB },
          { name: 'L0B', capacity: CAP.L0B },
          { name: 'L0C', capacity: CAP.L0C },
        ],
        sourceFiles: [SRC_ATTN],
        tensors: attentionTensors(),
      },
      diagnostics: {
        pipeline: [
          { space: 'L0B', slot: '32KB', requested: 2, achieved: 1, ok: false, note: '⌊64KB / 32KB⌋ 仅够 1 份，双缓冲被降级为串行' },
          { space: 'L0C', slot: '64KB', requested: 2, achieved: 2, ok: true, note: 'Acc 正常' },
        ],
        hints: [
          { level: 'danger', code: 'ALLOC-OVF', msg: 'smTmp @UB+0x38000 48KB 越过 0x40000 容量边界，UB 溢出 16KB', srcFile: SRC_ATTN.name, srcLine: 13 },
          { level: 'warning', code: 'PH-MR-001', msg: 'L0B 请求 stage=2 但仅达成 depth=1：PV MMAD 被迫串行，流水停顿。修复：将 per-stage tile 缩到 ≤32KB 或降 stage=1', srcFile: SRC_ATTN.name, srcLine: 11 },
        ],
        overflow: { space: 'UB', byBytes: 16 * KB, tensor: 'smTmp' },
      },
      runtime: attentionRuntime(),
    },
  };

  // ---- diff scenario (P2): baseline (before MemoryReuse) vs current ------
  const diff = {
    label: 'MatMulAddRelu · origin/main → 当前分支',
    baseline: {
      ref: 'origin/main',
      kernel: 'MatMulAddRelu_mix_aic__kernel0',
      ticks: 120,
      buffers: [
        { name: 'UB', capacity: CAP.UB },
        { name: 'L1', capacity: CAP.L1 },
        { name: 'L0C', capacity: CAP.L0C },
      ],
      sourceFiles: [SRC_MATMUL],
      tensors: matmulBaselineTensors(),
    },
    current: {
      ref: 'feat/l0-reuse',
      kernel: 'MatMulAddRelu_mix_aic__kernel0',
      ticks: 120,
      buffers: [
        { name: 'UB', capacity: CAP.UB },
        { name: 'L1', capacity: CAP.L1 },
        { name: 'L0C', capacity: CAP.L0C },
      ],
      sourceFiles: [SRC_MATMUL],
      tensors: matmulTensors(),
    },
  };

  global.MEMVIZ = {
    KB,
    CAP,
    kernels,
    diff,
    order: ['matmul_add_relu', 'fused_attention'],
  };
})(window);
