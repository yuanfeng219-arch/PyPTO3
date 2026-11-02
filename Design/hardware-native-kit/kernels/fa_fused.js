/* ============================================================================
 * kernel manifest · fa_fused
 * ----------------------------------------------------------------------------
 * 来源：Design/Toolkit Studio/decode_layer.py 的 fa_fused spmd 段，L728–L798
 *       （Qwen3-14B 融合 decode，online-softmax paged flash-attention）
 *
 * 说明两件事，都是口径而不是事实：
 *   · K 与 V 复用同一块 Right(L0B)，所以只声明 k_tile 一份；v_tile 节点有
 *     落点、没有独立声明。
 *   · online carry（o_i / m_i / l_i 累加量）在本节选之外，但它常驻 UB、
 *     参与高水位，所以作为一个无锚点的 resident buffer 声明。
 *
 * {{FA_STAGE}} 是参数代入位；原文件此处是字面量 2。
 * ========================================================================== */
(function (root) {
  'use strict';

  root.HNKitKernels = root.HNKitKernels || {};
  root.HNKitKernels.fa_fused = {
    id: 'fa_fused',
    entry: 'fa_fused',
    file: 'Design/Toolkit Studio/decode_layer.py',
    platform: 'a2a3',
    note: 'K/V 复用 Right(L0B)，只声明 k_tile；online carry 不在节选内，按常驻量计入 UB。',

    consts: {
      NUM_CORES: 24, NUM_HEADS: 40, NUM_KV_HEADS: 8,
      HEAD_DIM: 128, SEQ_TILE: 128, PAGE_SIZE: 128,
      FA_STAGE: 2, MAX_SEQ: 4096
    },
    knobs: [
      { id: 'SEQ_TILE', label: 'SEQ_TILE', options: [64, 128, 256] },
      { id: 'FA_STAGE', label: 'pipeline stage', options: [1, 2, 3, 4] },
      { id: 'NUM_KV_HEADS', label: 'NUM_KV_HEADS', options: [4, 8] },
      { id: 'HEAD_DIM', label: 'HEAD_DIM', options: [64, 128] }
    ],
    derive: function (c) {
      const qPerKv = c.NUM_HEADS / c.NUM_KV_HEADS;
      return {
        Q_PER_KV: qPerKv,
        Q_HEAD_BATCH: qPerKv,
        // CUBE 的 M 方向按 16 行分形，Q 行数要补齐到 16 的倍数
        Q_HEAD_PAD: Math.ceil(qPerKv / 16) * 16,
        // 一组 head 的 group pipeline 宽度跟 KV head 数一致
        GP_SIZE: c.NUM_KV_HEADS,
        // carry 一行放 o_i(HEAD_DIM) + m_i + l_i
        CARRY_W: c.HEAD_DIM + 2
      };
    },

    source: [
      { no: 728, t: 'with pl.spmd(' },
      { no: 729, t: '    NUM_CORES,' },
      { no: 730, t: '    name_hint="fa_fused",' },
      { no: 731, t: '    optimizations=[pl.split(pl.SplitMode.UP_DOWN)],' },
      { no: 732, t: '    deps=[work_tid, rope_grp_tids[0]],' },
      { no: 733, t: ') as fa_tid:' },
      { no: 734, t: '    fa_core = pl.get_block_idx()' },
      { no: 741, t: '    fa_total_blocks = pl.cast(pl.read(fa_total, [0, 0]), target_type=pl.INDEX)' },
      { no: 744, t: '    for fa_w in pl.range(fa_core, fa_total_blocks, NUM_CORES):' },
      { no: 745, t: '        fa_enc = pl.cast(pl.read(fa_work_table, [fa_w, 0]), target_type=pl.INDEX)' },
      { no: 746, t: '        fa_b = fa_enc // MAX_CTX_BLOCKS' },
      { no: 747, t: '        fa_p = fa_enc % MAX_CTX_BLOCKS' },
      { no: 748, t: '        fa_hg = 0' },
      { no: 749, t: '        fa_ctx_len = pl.read(seq_lens, [fa_b])' },
      { no: 751, t: '        sb = fa_p' },
      { no: 752, t: '        s0 = sb * SEQ_TILE' },
      { no: 753, t: '        valid_len = pl.min(SEQ_TILE, fa_ctx_len - s0)' },
      { no: 758, t: '        fa_pbid = pl.cast(pl.tensor.read(block_table, [fa_b * max_blocks_per_seq + sb]), pl.INDEX)' },
      { no: 764, t: '        for gp in pl.pipeline(GP_SIZE, stage={{FA_STAGE}}):' },
      { no: 765, t: '            gi = fa_hg * GP_SIZE + gp' },
      { no: 766, t: '            kvh = gi' },
      { no: 767, t: '            q_pad_row_g = fa_b * NUM_KV_HEADS * Q_HEAD_PAD + gi * Q_HEAD_PAD' },
      { no: 768, t: '            q_padded = all_q_padded[q_pad_row_g : q_pad_row_g + Q_HEAD_PAD, :]' },
      { no: 769, t: '            g_base = (fa_b * NUM_KV_HEADS + gi) * MAX_CTX_BLOCKS * Q_HEAD_PAD' },
      { no: 770, t: '            cache_row = layer_cache_base + (fa_pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE' },
      { no: 773, t: '            k_tile = k_cache[cache_row : cache_row + SEQ_TILE, :]' },
      { no: 774, t: '            raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)' },
      { no: 775, t: '            scores_scaled = pl.mul(raw_scores, ATTN_SCALE)' },
      { no: 781, t: '            scores_valid = pl.set_validshape(scores_scaled, Q_HEAD_BATCH, valid_len)' },
      { no: 782, t: '            scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)' },
      { no: 783, t: '            cur_mi = pl.row_max(scores)' },
      { no: 784, t: '            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))' },
      { no: 785, t: '            cur_li = pl.row_sum(exp_scores)' },
      { no: 786, t: '            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)' },
      { no: 789, t: '            v_tile = v_cache[cache_row : cache_row + SEQ_TILE, :]' },
      { no: 790, t: '            oi_tmp = pl.matmul(exp_scores_bf16, v_tile, out_dtype=pl.FP32)' },
      { no: 794, t: '            oi_tmp = pl.set_validshape(oi_tmp, Q_HEAD_BATCH, HEAD_DIM)' },
      { no: 796, t: '            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [g_base + sb * Q_HEAD_PAD, 0])' },
      { no: 797, t: '            all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [g_base + sb * Q_HEAD_PAD, 0])' },
      { no: 798, t: '            all_cur_li = pl.assemble(all_cur_li, cur_li, [g_base + sb * Q_HEAD_PAD, 0])' }
    ],

    tensors: [
      { id: 'k_cache', label: 'k_cache', note: 'paged · {{PAGE_SIZE}}/页' },
      { id: 'v_cache', label: 'v_cache', note: 'paged' },
      { id: 'work_table', label: 'work_table', note: '(b, page) 展平' },
      { id: 'out', label: 'all_oi_tmp', note: 'online 累加输出' }
    ],

    frames: [
      {
        id: 'f-spmd', scope: 'pl.spmd', label: 'PL.SPMD · {{NUM_CORES}} BLOCK',
        residency: 1, regions: ['cores']
      },
      {
        id: 'f-stride', call: 'pl.range', parent: 'f-spmd',
        label: 'PL.RANGE · grid-stride 领工作块', residency: 1
      },
      {
        // 这一层决定 UB 高水位：stage 份中间量同时在片上
        id: 'f-pipe', scope: 'pl.pipeline', parent: 'f-stride',
        label: 'PL.PIPELINE · stage={{FA_STAGE}}',
        bufs: 'all', regions: ['ub'],
        residency: function (c) { return Math.min(c.FA_STAGE, c.GP_SIZE); }
      }
    ],

    // 可选的展示布局；锚点、容量与联动仍由 manifest 数据驱动。
    graphLayout: {
      viewBox: '0 0 720 310',
      nodes: {
        tbl: [36, 62, 168, 'block_table → 物理页'],
        q: [58, 158, 80, 'Q'], k: [148, 158, 80, 'K 块'],
        qk: [58, 198, 80, 'Q · Kᵀ'], scale: [148, 198, 80, '× scale'],
        valid: [258, 158, 84, '有效区掩码'], fill: [352, 158, 84, '填充 −inf'],
        max: [446, 158, 84, '行最大'], exp: [258, 198, 84, 'exp(x − max)'],
        sum: [352, 198, 84, '行求和'], cast: [446, 198, 84, '转 BF16'],
        v: [572, 158, 92, 'V 块'], sv: [572, 198, 92, 'P · V'],
        acc: [572, 238, 92, '写回部分结果']
      },
      frames: { 'f-spmd': [12, 8, 696, 296], 'f-stride': [24, 36, 672, 260], 'f-pipe': [36, 104, 648, 184] },
      phases: { 'ph-score': [48, 132, 190, 142, '① 打分'], 'ph-soft': [248, 132, 296, 142, '② 局部 SOFTMAX'], 'ph-weight': [562, 132, 112, 142, '③ 加权求和'] }
    },

    phases: [
      { id: 'ph-page', label: '① 取页', nodes: ['tbl'] },
      { id: 'ph-score', label: '② 打分', nodes: ['q', 'k', 'qk', 'scale'] },
      {
        id: 'ph-soft', label: '③ online softmax', wrap: 3,
        nodes: ['valid', 'fill', 'max', 'exp', 'sum', 'cast']
      },
      { id: 'ph-weight', label: '④ 加权与累加', nodes: ['v', 'sv', 'acc'] }
    ],

    nodes: [
      {
        id: 'tbl', assigns: 'fa_pbid', also: ['fa_enc'], label: '查 block_table',
        frame: 'f-stride', regions: ['gm:work_table']
      },
      { id: 'q', assigns: 'q_padded', label: 'Q 补齐到 16 行', frame: 'f-pipe' },
      { id: 'k', assigns: 'k_tile', label: 'K 页 → L0B', frame: 'f-pipe', regions: ['gm:k_cache'] },
      {
        id: 'qk', assigns: 'raw_scores', label: 'Q·Kᵀ', frame: 'f-pipe',
        after: ['q', 'k'], regions: ['cube', 'ch-c2v']
      },
      { id: 'scale', assigns: 'scores_scaled', label: '× scale', frame: 'f-pipe', after: ['qk'] },
      {
        id: 'valid', assigns: 'scores_valid', also: ['valid_len'], label: '标有效列',
        frame: 'f-pipe', after: ['scale']
      },
      { id: 'fill', assigns: 'scores', label: '填 -inf', frame: 'f-pipe', after: ['valid'] },
      { id: 'max', assigns: 'cur_mi', label: 'row_max', frame: 'f-pipe', after: ['fill'] },
      { id: 'exp', assigns: 'exp_scores', label: 'exp(x − m)', frame: 'f-pipe', after: ['max'] },
      { id: 'sum', assigns: 'cur_li', label: 'row_sum', frame: 'f-pipe', after: ['exp'] },
      {
        id: 'cast', assigns: 'exp_scores_bf16', label: '→ BF16', frame: 'f-pipe',
        after: ['exp'], regions: ['ch-v2c']
      },
      {
        // V 与 K 复用 Right，所以这里只有落点、没有独立 buffer 声明
        id: 'v', assigns: 'v_tile', label: 'V 页 → L0B（与 K 复用）',
        frame: 'f-pipe', regions: ['gm:v_cache', 'l0b']
      },
      {
        id: 'sv', assigns: 'oi_tmp', label: 'P·V', frame: 'f-pipe',
        after: ['cast', 'v'], regions: ['cube', 'ch-v2c']
      },
      {
        id: 'acc', call: 'pl.assemble', label: '累加回 GM', frame: 'f-pipe',
        after: ['sv', 'sum'], regions: ['gm:out']
      }
    ],

    buffers: [
      // 常驻：跨 page 迭代的 online 累加量，不随 stage 翻倍
      {
        id: 'carry', label: 'online carry（o_i/m_i/l_i，不在本节选内）',
        space: 'vec', shape: ['Q_HEAD_PAD', 'CARRY_W'], dtype: 'fp32', resident: true
      },
      { assigns: 'scores_scaled', space: 'vec', shape: ['Q_HEAD_PAD', 'SEQ_TILE'], dtype: 'fp32', frame: 'f-pipe' },
      { assigns: 'scores_valid', space: 'vec', shape: ['Q_HEAD_PAD', 'SEQ_TILE'], dtype: 'fp32', frame: 'f-pipe' },
      { assigns: 'scores', space: 'vec', shape: ['Q_HEAD_PAD', 'SEQ_TILE'], dtype: 'fp32', frame: 'f-pipe' },
      {
        // 内联在 pl.exp(...) 里，没有赋值目标，用调用锚定
        id: 'rexp', label: 'row_expand_sub 临时量', calls: ['pl.row_expand_sub'],
        space: 'vec', shape: ['Q_HEAD_PAD', 'SEQ_TILE'], dtype: 'fp32', frame: 'f-pipe'
      },
      { assigns: 'exp_scores', space: 'vec', shape: ['Q_HEAD_PAD', 'SEQ_TILE'], dtype: 'fp32', frame: 'f-pipe' },
      { assigns: 'exp_scores_bf16', space: 'vec', shape: ['Q_HEAD_PAD', 'SEQ_TILE'], dtype: 'bf16', frame: 'f-pipe' },
      { assigns: 'cur_mi', space: 'vec', shape: ['Q_HEAD_PAD', 1], dtype: 'fp32', frame: 'f-pipe' },
      { assigns: 'cur_li', space: 'vec', shape: ['Q_HEAD_PAD', 1], dtype: 'fp32', frame: 'f-pipe' },

      { assigns: 'q_padded', space: 'left', shape: ['Q_HEAD_PAD', 'HEAD_DIM'], dtype: 'bf16', frame: 'f-pipe' },
      { assigns: 'k_tile', space: 'right', shape: ['SEQ_TILE', 'HEAD_DIM'], dtype: 'bf16', frame: 'f-pipe' },
      { assigns: 'raw_scores', space: 'acc', shape: ['Q_HEAD_PAD', 'SEQ_TILE'], dtype: 'fp32', frame: 'f-pipe' },
      { assigns: 'oi_tmp', space: 'acc', shape: ['Q_HEAD_PAD', 'HEAD_DIM'], dtype: 'fp32', frame: 'f-pipe' }
    ]
  };
})(typeof window !== 'undefined' ? window : this);
