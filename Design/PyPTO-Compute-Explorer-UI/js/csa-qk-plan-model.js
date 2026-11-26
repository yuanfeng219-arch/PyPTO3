/* qk_plan logical model for the archived TP=2 a2a3 build.
 * Grounded in decode_sparse_attn_csa.py:134–196 (csa_slots_build_valid_qk_plan)
 * and 315–347 (rope_cs). */
(function (global) {
  'use strict';

  const file = 'decode_sparse_attn_csa.py';
  const CAPACITY = 256, SMALL_CASE = 128;
  const BIAS_T_TILE = 8, ROPE_CS_T_TILE = 8;
  const WIN = 128, CMP_TOPK = 512, PADDED_TOPK = 640;
  const ATTN_K_TILE = 128, SPARSE_BLOCKS = 5, CMP_BLOCKS = SPARSE_BLOCKS - 1;
  const ROPE_DIM = 64, HALF_ROPE = 32, H_TILE = 16;

  const definitions = {
    csa_slots_build_valid_qk_plan: ['plan', 'qkplan', 138, 196],
    rope_cs: ['ropecs', 'ropecs', 317, 346],
  };
  const labels = {
    csa_slots_build_valid_qk_plan: 'Candidate Planning · qk_order',
    rope_cs: 'Inverse RoPE Metadata',
  };

  const sourceStage = global.PTO_CSA_RUNTIME_DATA.stages.find(stage => stage.id === 'sparse/attention/plan');
  const kernels = new Map(Object.entries(definitions).map(([name, [branch, type, first, last]]) => [
    name,
    { ...sourceStage.kernels.find(k => k.name === name), title: labels[name], branch, type, file, first, last },
  ]));
  const stage = { ...sourceStage, kernels: [...kernels.values()] };
  const measured = global.PTO_CSA_QK_PLAN_DATA;
  const stageEdges = (stage.internalEdges || []).filter(e => kernels.has(e.from) && kernels.has(e.to));

  const ticks = count => Array.from({ length: count }, (_, i) => i);
  const range = (start, size) => `[${start}, ${start + size})`;
  const ceil = (value, step) => Math.ceil(value / step);
  const isMatmul = () => false;
  const requestScoped = () => false;
  const groupSize = () => 1;
  const blockCount = () => 1;

  function tokenTiles(info, T) {
    const step = info.type === 'ropecs' ? ROPE_CS_T_TILE : BIAS_T_TILE;
    return ticks(ceil(T, step)).map(i => ({
      start: i * step,
      size: step,
      valid: Math.min(step, T - i * step),
      path: 'Token Tile',
    }));
  }

  function blockShape(info, block, { tokens: T = CAPACITY, tileIndex = 0 } = {}) {
    const tiles = tokenTiles(info, T);
    const m = tiles[tileIndex] || tiles[0];
    const mText = `M ${range(m.start, m.valid)}`;
    const mLabel = `${tiles.length} 个 Token Tiles · 当前 ${mText}`;
    if (info.type === 'qkplan') {
      return {
        ...m,
        tiles,
        group: block,
        text: `1 Core Group (pl.at) · ${mLabel} · ${T * SPARSE_BLOCKS} Work Items`,
      };
    }
    return {
      ...m,
      tiles,
      group: block,
      text: `1 Core Group (pl.at) · Swap [${H_TILE}, ${ROPE_DIM}] + ${mLabel}`,
    };
  }

  const objectId = (rank, info, block, op, view = {}) => {
    const parts = ['qk-plan', info.branch, info.name, `rank${rank}`];
    if (block != null) parts.push(`T${view.tokens ?? CAPACITY}`, `block${block}`);
    if (op != null) {
      if (block != null) parts.push(`m${view.tileIndex ?? 0}`);
      parts.push(op);
    }
    return parts.join('/');
  };

  function builder() {
    const nodes = [], edges = [], byId = new Map();
    function add(op, title, lane, lines, inputs = [], options = {}) {
      const id = options.id ?? (options.step == null ? op : `${op}:${options.step}`);
      const rank = Math.max(0, ...inputs.map(item => byId.get(typeof item === 'string' ? item : item.id).rank + 1));
      const node = { op, id, title, lane, lines, rank, role: 'operation', step: null, file, ...options };
      nodes.push(node);
      byId.set(id, node);
      for (const item of inputs) {
        edges.push(typeof item === 'string'
          ? { from: item, to: id, type: 'data' }
          : { from: item.id, to: id, type: item.type ?? 'data' });
      }
      return id;
    }
    function input(op, title, lines, detail, options = {}) {
      return add(op, title, 'Input', lines, [], { role: 'activation', detail, ...options });
    }
    function output(op, title, lines, inputs, detail, options = {}) {
      return add(op, title, 'Write', lines, inputs, { role: 'aggregate', detail, ...options });
    }
    return { add, input, output, nodes, edges };
  }
  const carry = id => ({ id, type: 'state' });

  function operationModel(info, expanded, view = { block: 0, tileIndex: 0, tokens: CAPACITY }) {
    const { add, input, output, nodes, edges } = builder();
    const T = view.tokens ?? CAPACITY;
    const shape = blockShape(info, view.block ?? 0, view);
    const m = `M ${range(shape.start, shape.valid)}`;
    const tileCount = shape.tiles.length;

    if (info.type === 'qkplan') {
      const idxTopk = input('idx-topk', 'idx_topk', [147, 147], `INT32 · ${m} × ${CMP_TOPK}`);
      const posIds = input('pos-ids', 'position_ids', [148, 148], `INT32 · ${m} × 1`);
      const winIdx = input('win-idx', 'window_swa_indices', [171, 171], `INT32 · ${m} × ${WIN}`);

      const slotMask = add('slot-mask', 'Slot Range Mask', 'Vector', [147, 158], [idxTopk, posIds], {
        detail: `×${tileCount} Token Tiles · 0 ≤ raw < ⌊(pos+1)/4⌋`,
      });
      output('cmp-indices', 'cmp_sparse_indices', [158, 158], [slotMask], `INT32 · ${m} × ${CMP_TOPK} · −1 无效`);

      const sb0Valid = add('sb0-valid', 'SB0 Constant Valid', 'Control', [160, 161], [], {
        detail: 'valid_block_mask[:, 0] = 1 · 滑窗恒有效',
      });

      const validSources = [sb0Valid];
      if (expanded) {
        for (const i of ticks(CMP_BLOCKS)) {
          const sb = i + 1;
          validSources.push(add('blk-valid', `Block Validity · SB${sb}`, 'Vector', [162, 167], [slotMask], {
            step: i,
            detail: `c_mask[:, ${range(i * ATTN_K_TILE, ATTN_K_TILE)}] row_max → 标量写`,
          }));
        }
      } else {
        validSources.push(add('blk-valid', 'Block Validity Reduce', 'Vector', [162, 167], [slotMask], {
          detail: `×${CMP_BLOCKS} CMP Blocks (SB1–4) · row_max → 标量写`,
        }));
      }
      const validMask = output('valid-mask', 'valid_block_mask', [160, 167], validSources, `INT32 · ${T} × ${SPARSE_BLOCKS}`);

      if (expanded) {
        const winBias = add('win-bias', 'Window Bias', 'Vector', [171, 176], [winIdx], {
          step: 0,
          detail: `Cols [0, ${WIN}) · valid 0 / invalid −1e20`,
        });
        const cmpBias = add('cmp-bias', 'Compressed Bias', 'Vector', [177, 177], [slotMask], {
          step: 1,
          detail: `Cols [${WIN}, ${PADDED_TOPK}) · 复用 c_out`,
        });
        output('sparse-bias', 'sparse_bias', [176, 177], [winBias, cmpBias], `FP32 · ${m} × ${PADDED_TOPK}`);
      } else {
        const biasBuild = add('bias-build', 'Sparse Bias Build', 'Vector', [171, 177], [winIdx, slotMask], {
          detail: `Win [0, ${WIN}) + CMP [${WIN}, ${PADDED_TOPK}) · 0 / −1e20`,
        });
        output('sparse-bias', 'sparse_bias', [176, 177], [biasBuild], `FP32 · ${m} × ${PADDED_TOPK}`);
      }

      const curInit = add('cursor-init', 'Reset qk_wcur = 0', 'Control', [182, 182], [], {
        detail: '标量写指针归零 · GM [1]',
      });
      const pass1 = add('pass1-valid', 'Pass 1 · Valid-First', 'Control', [184, 189], [validMask, curInit], {
        detail: `扫描 ${T}×${SPARSE_BLOCKS} · 非空块写入 qk_order 前段`,
      });
      const pass2 = add('pass2-invalid', 'Pass 2 · Invalid Tail', 'Control', [191, 196], [validMask, carry(pass1)], {
        detail: `扫描 ${T}×${SPARSE_BLOCKS} · 空块追加至尾部`,
      });
      output('qk-order', 'qk_order', [188, 196], [pass1, pass2], `INT32 · [${T * SPARSE_BLOCKS}] · valid-first 重排队列`);
    } else {
      const fCos = input('freqs-cos', 'freqs_cos', [342, 342], `BF16 · ${m} × ${HALF_ROPE}`);
      const fSin = input('freqs-sin', 'freqs_sin', [343, 343], `BF16 · ${m} × ${HALF_ROPE}`);

      const swapIdx = add('swap-idx', 'Build Swap Index', 'Vector', [323, 330], [], {
        detail: `j^1 · [${H_TILE}, ${ROPE_DIM}] · 供 Merge 旋转 Gather`,
      });
      output('rope-swap', 'rope_swap_idx', [330, 330], [swapIdx], `INT32 · ${H_TILE} × ${ROPE_DIM}`);

      const dupSign = add('dup-sign', 'Build Dup / Conjugate Sign', 'Vector', [332, 339], [], {
        detail: `j>>1 · [+1, −1, ...] · [${ROPE_CS_T_TILE}, ${ROPE_DIM}]`,
      });

      if (expanded) {
        const gathers = ticks(shape.valid).map(i =>
          add('gather-row', `Gather Cos / Sin · Row ${i}`, 'Gather', [342, 346], [fCos, fSin, dupSign], {
            step: i,
            detail: `Token ${shape.start + i} · tgather [1, 32]→[1, 64] + 共轭乘`,
          })
        );
        output('rope-cos', 'rope_cos_il', [344, 344], gathers, `FP32 · ${m} × ${ROPE_DIM}`);
        output('rope-sin', 'rope_sin_signed', [346, 346], gathers, `FP32 · ${m} × ${ROPE_DIM}`);
      } else {
        const gatherCos = add('gather-cos', 'Gather Interleaved Cos', 'Gather', [342, 344], [fCos, dupSign], {
          detail: `×${tileCount} Token Tiles · 8 行 tgather → FP32`,
        });
        const gatherSin = add('gather-sin', 'Gather & Fold Sin', 'Gather', [343, 346], [fSin, dupSign], {
          detail: `×${tileCount} Token Tiles · tgather × cs_sign`,
        });
        output('rope-cos', 'rope_cos_il', [344, 344], [gatherCos], `FP32 · ${T} × ${ROPE_DIM}`);
        output('rope-sin', 'rope_sin_signed', [346, 346], [gatherSin], `FP32 · ${T} × ${ROPE_DIM}`);
      }
    }

    return { nodes, edges };
  }

  global.PtoCsaQkPlanModel = Object.freeze({
    title: 'qk_plan',
    subtitle: 'Candidate Planning + Inverse RoPE',
    caseLabel: 'Token 行数',
    stage,
    measured,
    kernels,
    stageEdges,
    rows: BIAS_T_TILE,
    kTile: ATTN_K_TILE,
    kCount: CMP_BLOCKS,
    branches: { plan: 'Candidate Planning', ropecs: 'Inverse RoPE Metadata' },
    defaultKernel: 'csa_slots_build_valid_qk_plan',
    cases: [
      { tokens: CAPACITY, label: '256 · 实测形状' },
      { tokens: SMALL_CASE, label: '128 · 静态小 Batch 示例' },
    ],
    ticks,
    range,
    isMatmul,
    requestScoped,
    groupSize,
    blockCount,
    blockShape,
    objectId,
    operationModel,
  });
})(window);
